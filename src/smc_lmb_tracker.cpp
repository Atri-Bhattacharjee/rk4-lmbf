#include "smc_lmb_tracker.h"
#include "assignment.h"
#include "fast_random.h"
#include "in_orbit_sensor_model.h"
#include "validation.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr double kMu = 3.986004418e14;               //!< Earth's gravitational parameter [m^3/s^2]
constexpr double kMinRadius = 6.371e6 + 100.0e3;    //!< The propagator's gravity floor [m]
//! Largest gravitational acceleration anywhere at or above the floor [m/s^2].
constexpr double kMaxGravity = kMu / (kMinRadius * kMinRadius);

//! Large cost for impossible assignments (unobservable pairs, other tracks' miss columns).
constexpr double INF_COST = 1e9;

//! Floor on every hypothesis factor eta before its log is taken. The smallest normal double: its
//! cost (~708) sits far above any realistic cost and far below INF_COST, so a floored factor ranks
//! below every genuine one without being mistaken for an impossible assignment. The floor only
//! matters when eta has underflowed to (or is exactly) zero.
constexpr double kEtaFloor = std::numeric_limits<double>::min();

double factor_cost(double eta) {
    return -std::log(std::max(eta, kEtaFloor));
}

//! Adds the scope's wall-clock duration to *sink; a null sink (profiling off) reads no clock.
class ScopedTimer {
public:
    explicit ScopedTimer(double* sink)
        : sink_(sink), start_(sink != nullptr ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{}) {}
    ~ScopedTimer() {
        if (sink_ != nullptr) {
            *sink_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        }
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double* sink_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace

void SMC_LMB_Tracker::ensure_models_configured() const {
    validation::require_models(propagator_, sensor_model_, birth_model_);
}

SMC_LMB_Tracker::SMC_LMB_Tracker(std::shared_ptr<IOrbitPropagator> propagator,
                                                                 std::shared_ptr<ISensorModel> sensor_model,
                                                                 std::shared_ptr<IBirthModel> birth_model,
                                                                 double survival_probability,
                                                                 int k_best,
                                                                 double prune_threshold,
                                                                 double clutter_intensity,
                                                                 double p_detection,
                                                                 double noise_decay_rate,
                                                                 double noise_min_scale,
                                                                 std::optional<uint64_t> seed)
        : current_state_(0.0, std::vector<Track>{}),
            propagator_(std::move(propagator)),
            sensor_model_(std::move(sensor_model)),
            birth_model_(std::move(birth_model)),
            survival_probability_(survival_probability),
            k_best_(k_best),
            prune_threshold_(prune_threshold),
            clutter_intensity_(clutter_intensity),
            p_detection_(p_detection),
            noise_decay_rate_(noise_decay_rate),
            noise_min_scale_(noise_min_scale),
            resample_rng_(seed.has_value() ? *seed : std::mt19937_64::result_type(std::random_device{}())),
            regularization_rng_(seed.has_value() ? (*seed ^ 0x9E3779B97F4A7C15ULL)
                                                 : std::mt19937_64::result_type(std::random_device{}())),
            fused_rng_(seed.has_value() ? (*seed ^ 0xD1B54A32D192ED03ULL)
                                        : std::mt19937_64::result_type(std::random_device{}())) {
    validation::require_models(propagator_, sensor_model_, birth_model_);
    validation::require_k_best(k_best_);
    validation::require_clutter_intensity(clutter_intensity_);
    stream_seed_ = fast_random::mix64((seed.has_value() ? *seed : std::random_device{}()) ^
                                      0xA54FF53A5F1D36F1ULL);
}

void SMC_LMB_Tracker::set_fast_mode(bool enabled) {
    if (enabled) {
        ensure_models_configured();
        if (!propagator_->supports_keyed_propagation()) {
            throw std::invalid_argument(
                "set_fast_mode: the propagator does not support keyed propagation "
                "(TwoBodyPropagator does)");
        }
    }
    fast_mode_ = enabled;
}

uint64_t SMC_LMB_Tracker::substep_key(const Track& track, double step_start, double dt) const {
    using fast_random::bits_of;
    using fast_random::combine;
    uint64_t key = combine(stream_seed_, 0x50524F50ULL);   // "PROP"
    key = combine(key, track.label().birth_time);
    key = combine(key, track.label().index);
    key = combine(key, bits_of(step_start));
    key = combine(key, bits_of(dt));
    // Labels need not be unique (two births in one integer second share birth_time and can share
    // an index), so the cloud itself goes into the key too: two different clouds never share noise.
    const std::vector<Particle>& particles = track.particles();
    if (!particles.empty()) {
        for (int k = 0; k < 6; ++k) {
            key = combine(key, bits_of(particles.front().state_vector(k)));
        }
    }
    return key;
}

double SMC_LMB_Tracker::noise_scale_at(const Track& track, double time) const {
    // alpha(age) = alpha_min + (1 - alpha_min) * exp(-lambda * age)
    const double age = time - static_cast<double>(track.label().birth_time);
    double noise_scale = 1.0;
    if (noise_decay_rate_ > 0.0 && age > 0.0) {
        noise_scale = noise_min_scale_ + (1.0 - noise_min_scale_) * std::exp(-noise_decay_rate_ * age);
    }
    return noise_scale;
}

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

void SMC_LMB_Tracker::predict(double dt) {
    ensure_models_configured();
    const ScopedTimer total_timer(profiling_ ? &profile_.predict : nullptr);
    if (profiling_) {
        ++profile_.predict_calls;
    }
    // Projects the filter state forward in time by operating in-place.
    const double previous_time = current_state_.timestamp();
    const double new_time = current_state_.timestamp() + dt;
    current_state_.set_timestamp(new_time);
    // Get a direct, MODIFIABLE reference to the internal track vector.
    std::vector<Track>& tracks = current_state_.tracks();

    if (lazy_propagation_) {
        // Only the clock and the survival probability advance for every track. A cloud is moved
        // here only once it has lagged by max_pending_; update() brings forward anything a sensor
        // might see, and synchronize() everything.
        for (Track& track : tracks) {
            track.set_existence_probability(track.existence_probability() * survival_probability_);
            if (std::isnan(track.propagated_time())) {
                track.set_propagated_time(previous_time);
            }
            if (new_time - track.propagated_time() >= max_pending_ - 1e-9) {
                const ScopedTimer timer(profiling_ ? &profile_.predict_propagate : nullptr);
                const uint64_t steps = propagate_track_to(track, new_time);
                if (profiling_) {
                    profile_.predict_particle_steps += steps;
                }
            }
        }
        return;
    }

    const ScopedTimer propagate_timer(profiling_ ? &profile_.predict_propagate : nullptr);
    for (Track& track : tracks) {
        if (profiling_) {
            profile_.predict_particle_steps += track.particles().size();
        }
        // Update existence probability in-place.
        track.set_existence_probability(track.existence_probability() * survival_probability_);

        // --- Process Noise Annealing ---
        const double noise_scale = noise_scale_at(track, new_time);

        // Propagate each particle in place. Overwriting the existing cloud avoids allocating a
        // second vector and the set_particles copy that used to follow it. Element-wise assignment
        // does not reallocate, so external aliases of the storage remain address-stable; values
        // change, as they would under any in-place rewrite.
        std::vector<Particle>& particles = track.mutable_particles();
        if (fast_mode_) {
            propagator_->propagate_cloud_keyed(particles, dt, previous_time, noise_scale,
                                               substep_key(track, previous_time, dt));
        } else {
            for (Particle& particle : particles) {
                particle = propagator_->propagate(particle, dt, previous_time, noise_scale);
            }
        }
        track.set_propagated_time(new_time);
    }
    // No need to call set_tracks(), as we have modified the state directly.
}

void SMC_LMB_Tracker::set_lazy_propagation(bool enabled, double max_pending) {
    if (!(std::isfinite(max_pending) && max_pending > 0.0)) {
        throw std::invalid_argument("set_lazy_propagation: max_pending must be finite and > 0, got " +
                                    std::to_string(max_pending));
    }
    if (enabled) {
        ensure_models_configured();
        if (!propagator_->noise_displacement_bound(max_pending).has_value()) {
            throw std::invalid_argument(
                "set_lazy_propagation: the propagator's process noise is not time-consistent, so a "
                "track stepped lazily would receive a different amount of noise than one stepped "
                "eagerly. Construct TwoBodyPropagator with noise_reference_dt.");
        }
    }
    lazy_propagation_ = enabled;
    max_pending_ = max_pending;
    stamp_unstamped_tracks();
    if (!enabled) {
        // Leaving lazy mode: every track must be current before eager predicts resume.
        synchronize();
    }
}

void SMC_LMB_Tracker::set_regularization(bool enabled, double bandwidth_scale, double ess_threshold) {
    if (!(std::isfinite(bandwidth_scale) && bandwidth_scale > 0.0)) {
        throw std::invalid_argument("set_regularization: bandwidth_scale must be finite and > 0, got " +
                                    std::to_string(bandwidth_scale));
    }
    if (!(ess_threshold >= 0.0 && ess_threshold <= 1.0)) {
        throw std::invalid_argument("set_regularization: ess_threshold must be in [0, 1], got " +
                                    std::to_string(ess_threshold));
    }
    regularization_ = enabled;
    regularization_bandwidth_scale_ = bandwidth_scale;
    regularization_ess_threshold_ = ess_threshold;
}

void SMC_LMB_Tracker::regularize(std::vector<Particle>& resampled, const std::vector<Particle>& predicted,
                                 const std::vector<double>& posterior_weights) {
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    const size_t n = resampled.size();

    StateVector mean = StateVector::Zero();
    for (size_t p = 0; p < predicted.size(); ++p) {
        mean += posterior_weights[p] * predicted[p].state_vector;
    }
    Matrix6d covariance = Matrix6d::Zero();
    for (size_t p = 0; p < predicted.size(); ++p) {
        const StateVector offset = predicted[p].state_vector - mean;
        covariance.noalias() += posterior_weights[p] * offset * offset.transpose();
    }
    // A degenerate posterior (a handful of distinct particles) gives a rank-deficient covariance;
    // an eigen-square-root handles that where a Cholesky factor would not.
    Eigen::SelfAdjointEigenSolver<Matrix6d> eigen(covariance);
    const Matrix6d root = eigen.eigenvectors() *
                          eigen.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();

    constexpr double kDim = 6.0;
    const double h_opt = std::pow(4.0 / (static_cast<double>(n) * (kDim + 2.0)), 1.0 / (kDim + 4.0));
    const double h = std::min(regularization_bandwidth_scale_ * h_opt, 1.0);
    const double shrink = std::sqrt(1.0 - h * h);

    for (Particle& particle : resampled) {
        StateVector eps;
        for (int k = 0; k < 6; ++k) {
            eps(k) = regularization_normal_(regularization_rng_);
        }
        particle.state_vector = mean + shrink * (particle.state_vector - mean) + h * (root * eps);
    }
}

std::vector<Particle> SMC_LMB_Tracker::systematic_resample(const std::vector<Particle>& source,
                                                          const std::vector<double>& weights, size_t n) {
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    std::vector<Particle> resampled_particles;
    resampled_particles.reserve(n);

    const size_t source_size = source.size();
    const double u = unit_dist(resample_rng_) / static_cast<double>(n);
    double cumsum = 0.0;
    size_t idx = 0;

    for (size_t p = 0; p < n; ++p) {
        const double threshold = u + static_cast<double>(p) / static_cast<double>(n);

        while (cumsum < threshold && idx < source_size) {
            cumsum += weights[idx];
            ++idx;
        }

        // The position in `weights` is the source index, so no side table is needed.
        const size_t chosen_index = (idx > 0) ? idx - 1 : 0;

        Particle resampled;
        resampled.state_vector = source[chosen_index].state_vector;
        resampled.weight = 1.0 / static_cast<double>(n);
        resampled_particles.push_back(resampled);
    }
    return resampled_particles;
}

void SMC_LMB_Tracker::set_fused_proposal(bool enabled, double ess_min, size_t neighbours,
                                         double fallback_ess_min) {
    if (!(std::isfinite(ess_min) && ess_min >= 0.0)) {
        throw std::invalid_argument("set_fused_proposal: ess_min must be finite and >= 0, got " +
                                    std::to_string(ess_min));
    }
    if (!(std::isfinite(fallback_ess_min) && fallback_ess_min >= 0.0)) {
        throw std::invalid_argument("set_fused_proposal: fallback_ess_min must be finite and >= 0, got " +
                                    std::to_string(fallback_ess_min));
    }
    if (neighbours != 0 && neighbours < 8) {
        throw std::invalid_argument("set_fused_proposal: neighbours must be 0 (automatic) or >= 8, got " +
                                    std::to_string(neighbours));
    }
    fused_proposal_ = enabled;
    fused_ess_min_ = ess_min;
    fused_fallback_ess_min_ = fallback_ess_min;
    fused_neighbours_ = neighbours;
}

namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

//! Lower-triangular factor of a symmetric positive semi-definite 6x6 matrix: Cholesky when it
//! succeeds, otherwise an eigen square root (which handles rank deficiency).
bool matrix_root(const Matrix6d& covariance, Matrix6d& root) {
    Eigen::LLT<Matrix6d> llt(covariance);
    if (llt.info() == Eigen::Success) {
        root = llt.matrixL();
        return true;
    }
    Eigen::SelfAdjointEigenSolver<Matrix6d> eigen(covariance);
    if (eigen.info() != Eigen::Success) {
        return false;
    }
    root = eigen.eigenvectors() * eigen.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
    return root.allFinite();
}

double log_sum_exp(const std::vector<double>& values) {
    double peak = -std::numeric_limits<double>::infinity();
    for (double v : values) {
        peak = std::max(peak, v);
    }
    if (!std::isfinite(peak)) {
        return peak;
    }
    double total = 0.0;
    for (double v : values) {
        total += std::exp(v - peak);
    }
    return peak + std::log(total);
}

}  // namespace

bool SMC_LMB_Tracker::build_fused_component(const Track& track, const Measurement& measurement,
                                            const MeasurementLikelihoodCache& cache, int sensor_index,
                                            const sensor::SensorArray* sensors, double& likelihood,
                                            FusedComponent& out) {
    constexpr double kDim = 6.0;
    constexpr double kLog2Pi = 1.8378770664093453;
    const std::vector<Particle>& particles = track.particles();
    const size_t n = particles.size();
    if (n < 8) {
        return false;
    }

    // 1. The measurement as a Gaussian in state space: x_z = h^-1(z), Sigma_z = J^-1 R J^-T with J the
    //    Jacobian of the local measurement function at x_z, by central differences. Linearised only
    //    over the measurement's own extent.
    const StateVector x_z = measurement.toCartesian();
    if (!x_z.allFinite()) {
        return false;
    }
    const StateVector& sensor_state = measurement.sensor_state_;
    auto residual = [&](const StateVector& x) {
        return LocalMeasVector(los::localResidual(cache.measured, los::observe(x, sensor_state),
                                                  cache.measured_basis));
    };
    Matrix6d jacobian;
    const double steps[6] = {1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2};
    for (int k = 0; k < 6; ++k) {
        StateVector forward = x_z;
        StateVector backward = x_z;
        forward(k) += steps[k];
        backward(k) -= steps[k];
        // residual = measured - predicted, so d(predicted)/dx = -d(residual)/dx.
        jacobian.col(k) = -(residual(forward) - residual(backward)) / (2.0 * steps[k]);
    }
    const double jacobian_det = jacobian.determinant();
    if (!std::isfinite(jacobian_det) || std::abs(jacobian_det) < 1e-300) {
        return false;
    }
    const Matrix6d jacobian_inv = jacobian.inverse();
    Matrix6d sigma_z = jacobian_inv * measurement.covariance_ * jacobian_inv.transpose();
    sigma_z = 0.5 * (sigma_z + sigma_z.transpose());

    // 2. Global moments of the cloud, and its particles nearest x_z in the metric (C + Sigma_z)^-1.
    double weight_total = 0.0;
    StateVector mean = StateVector::Zero();
    for (const Particle& p : particles) {
        weight_total += p.weight;
        mean += p.weight * p.state_vector;
    }
    if (!(weight_total > 0.0)) {
        return false;
    }
    mean /= weight_total;
    Matrix6d covariance = Matrix6d::Zero();
    for (const Particle& p : particles) {
        const StateVector d = p.state_vector - mean;
        covariance.noalias() += (p.weight / weight_total) * d * d.transpose();
    }
    const Eigen::LDLT<Matrix6d> metric(covariance + sigma_z);
    if (metric.info() != Eigen::Success) {
        return false;
    }
    std::vector<std::pair<double, size_t>> distance(n);
    for (size_t p = 0; p < n; ++p) {
        const StateVector d = particles[p].state_vector - x_z;
        distance[p] = {d.dot(metric.solve(d)), p};
    }
    size_t k_neighbours = fused_neighbours_ != 0
        ? fused_neighbours_
        : std::max<size_t>(30, static_cast<size_t>(std::ceil(0.05 * static_cast<double>(n))));
    k_neighbours = std::min(k_neighbours, n);
    std::nth_element(distance.begin(), distance.begin() + static_cast<std::ptrdiff_t>(k_neighbours - 1),
                     distance.end());

    // 3. Kernel density of the neighbourhood: p_hat(x) = sum_p (w_p / W) N(x; x_p, H), with
    //    H = h^2 P_L (Silverman/RPF bandwidth for K points) plus a ridge at 1e-6 of the measurement's
    //    own spread, so a collapsed neighbourhood stays a proper density.
    double local_weight = 0.0;
    StateVector local_mean = StateVector::Zero();
    for (size_t i = 0; i < k_neighbours; ++i) {
        const Particle& p = particles[distance[i].second];
        local_weight += p.weight;
        local_mean += p.weight * p.state_vector;
    }
    if (!(local_weight > 0.0)) {
        return false;
    }
    local_mean /= local_weight;
    Matrix6d local_cov = Matrix6d::Zero();
    for (size_t i = 0; i < k_neighbours; ++i) {
        const Particle& p = particles[distance[i].second];
        const StateVector d = p.state_vector - local_mean;
        local_cov.noalias() += (p.weight / local_weight) * d * d.transpose();
    }
    const double h = std::pow(4.0 / (static_cast<double>(k_neighbours) * (kDim + 2.0)), 1.0 / (kDim + 4.0));
    Matrix6d ridge = Matrix6d::Zero();
    ridge.diagonal() = 1e-6 * sigma_z.diagonal();
    const Matrix6d bandwidth = h * h * local_cov + ridge;
    Matrix6d bandwidth_root;
    if (!matrix_root(bandwidth, bandwidth_root)) {
        return false;
    }
    Eigen::LLT<Matrix6d> bandwidth_llt(bandwidth);
    if (bandwidth_llt.info() != Eigen::Success) {
        return false;
    }
    const Matrix6d bandwidth_l = bandwidth_llt.matrixL();
    const double bandwidth_log_det = 2.0 * bandwidth_l.diagonal().array().log().sum();

    // The kernel sum runs over every particle whose kernel can reach the measurement's footprint
    // (within 7 sigma in the metric (Sigma_z + H)^-1), plus the neighbourhood itself. The
    // neighbourhood only sets the kernel width; choosing the summed set by reach rather than by count
    // keeps the density whole wherever the likelihood is non-negligible, also when the measurement is
    // not much sharper than the cloud. Capped at kMaxKernels nearest, which only binds when the cloud
    // is so close to the measurement's size that the ordinary update would not have collapsed.
    constexpr double kReachSigmas = 7.0;
    constexpr size_t kMaxKernels = 4096;
    const Eigen::LDLT<Matrix6d> reach_metric(sigma_z + bandwidth);
    if (reach_metric.info() != Eigen::Success) {
        return false;
    }
    std::vector<std::pair<double, size_t>> reach(n);
    for (size_t p = 0; p < n; ++p) {
        const StateVector d = particles[p].state_vector - x_z;
        reach[p] = {d.dot(reach_metric.solve(d)), p};
    }
    std::vector<char> in_kernel_set(n, 0);
    for (size_t i = 0; i < k_neighbours; ++i) {
        in_kernel_set[distance[i].second] = 1;
    }
    std::sort(reach.begin(), reach.end());
    std::vector<size_t> kernel_set;
    kernel_set.reserve(std::min(n, kMaxKernels));
    for (const auto& entry : reach) {
        if (kernel_set.size() >= kMaxKernels) {
            break;
        }
        if (entry.first <= kReachSigmas * kReachSigmas || in_kernel_set[entry.second]) {
            kernel_set.push_back(entry.second);
        }
    }

    const size_t num_kernels = kernel_set.size();
    std::vector<StateVector> kernel_centres(num_kernels);
    std::vector<double> kernel_log_weight(num_kernels);
    for (size_t i = 0; i < num_kernels; ++i) {
        const Particle& p = particles[kernel_set[i]];
        kernel_centres[i] = bandwidth_l.triangularView<Eigen::Lower>().solve(p.state_vector);
        kernel_log_weight[i] = std::log(std::max(p.weight / weight_total, 1e-300)) -
                               0.5 * kDim * kLog2Pi - 0.5 * bandwidth_log_det;
    }
    auto log_prior_density = [&](const StateVector& x, std::vector<double>& scratch) {
        const StateVector y = bandwidth_l.triangularView<Eigen::Lower>().solve(x);
        scratch.resize(num_kernels);
        for (size_t i = 0; i < num_kernels; ++i) {
            scratch[i] = kernel_log_weight[i] - 0.5 * (y - kernel_centres[i]).squaredNorm();
        }
        return log_sum_exp(scratch);
    };

    // 4. Proposal: Gaussian product of the neighbourhood (moments of its kernel density) and the
    //    measurement, widened by 1.5 in standard deviation for defensive tails.
    const Matrix6d prior_cov = local_cov + bandwidth;
    const Eigen::LDLT<Matrix6d> innovation(prior_cov + sigma_z);
    if (innovation.info() != Eigen::Success) {
        return false;
    }
    const Matrix6d gain = innovation.solve(prior_cov).transpose();   // prior_cov (prior_cov + sigma_z)^-1
    const StateVector fused_mean = local_mean + gain * (x_z - local_mean);
    Matrix6d fused_cov = prior_cov - gain * prior_cov;
    fused_cov = 0.5 * (fused_cov + fused_cov.transpose());
    constexpr double kInflation = 1.5;
    Matrix6d proposal_cov = kInflation * kInflation * fused_cov + ridge;
    Matrix6d proposal_root;
    if (!matrix_root(proposal_cov, proposal_root)) {
        return false;
    }
    Eigen::LLT<Matrix6d> proposal_llt(proposal_cov);
    const bool proposal_has_density = proposal_llt.info() == Eigen::Success;
    const Matrix6d proposal_l = proposal_has_density ? Matrix6d(proposal_llt.matrixL()) : Matrix6d::Identity();
    const double proposal_log_det = proposal_has_density ? 2.0 * proposal_l.diagonal().array().log().sum() : 0.0;

    // 5. Draw, weight with prior density x exact likelihood x visibility / proposal density.
    std::vector<StateVector> draws(n);
    std::vector<double> log_weights(n, -std::numeric_limits<double>::infinity());
    if (proposal_has_density) {
        std::vector<double> scratch;
        if (profiling_) {
            profile_.fused_kernel_evaluations += static_cast<uint64_t>(n) * num_kernels;
        }
        for (size_t m = 0; m < n; ++m) {
            StateVector eps;
            for (int k = 0; k < 6; ++k) {
                eps(k) = fused_normal_(fused_rng_);
            }
            draws[m] = fused_mean + proposal_l * eps;
            const bool visible = sensors == nullptr || sensor_index < 0 ||
                                 sensors->sees(static_cast<size_t>(sensor_index), draws[m].head<3>());
            if (!visible) {
                continue;
            }
            Particle probe;
            probe.state_vector = draws[m];
            probe.weight = 1.0;
            const double g = sensor_model_->calculate_likelihood(probe, measurement, cache);
            if (!(g > 0.0)) {
                continue;
            }
            const double log_q = -0.5 * eps.squaredNorm() - 0.5 * kDim * kLog2Pi - 0.5 * proposal_log_det;
            log_weights[m] = log_prior_density(draws[m], scratch) + std::log(g) - log_q;
        }
    }
    const double log_total = log_sum_exp(log_weights);

    out.particles.clear();
    out.particles.reserve(n);
    out.fallback = false;
    out.ess = 0.0;
    if (std::isfinite(log_total)) {
        double sum_sq = 0.0;
        for (size_t m = 0; m < n; ++m) {
            const double w = std::exp(log_weights[m] - log_total);
            sum_sq += w * w;
        }
        out.ess = sum_sq > 0.0 ? 1.0 / sum_sq : 0.0;
    }

    if (std::isfinite(log_total) && out.ess >= fused_fallback_ess_min_) {
        likelihood = std::exp(log_total - std::log(static_cast<double>(n)));
        for (size_t m = 0; m < n; ++m) {
            Particle particle;
            particle.state_vector = draws[m];
            particle.weight = std::exp(log_weights[m] - log_total);
            out.particles.push_back(particle);
        }
        return std::isfinite(likelihood);
    }

    // 6. Fallback: the weights collapsed too (or the proposal had no density). Use the Gaussian
    //    product itself -- uniform draws from N(fused_mean, fused_cov) -- and the closed-form
    //    association likelihood (K/N of the cloud's mass) * N(x_z; local_mean, prior_cov + Sigma_z) / |det J|.
    Matrix6d fused_root;
    if (!matrix_root(fused_cov + ridge, fused_root)) {
        return false;
    }
    const StateVector innovation_vector = x_z - local_mean;
    const Matrix6d innovation_cov = prior_cov + sigma_z;
    const double log_det_innovation = std::log(std::max(innovation_cov.determinant(), 1e-300));
    const double log_closed_form = std::log(local_weight / weight_total) - 0.5 * kDim * kLog2Pi -
                                   0.5 * log_det_innovation -
                                   0.5 * innovation_vector.dot(innovation.solve(innovation_vector)) -
                                   std::log(std::abs(jacobian_det));
    likelihood = std::exp(log_closed_form);
    const double uniform = 1.0 / static_cast<double>(n);
    for (size_t m = 0; m < n; ++m) {
        StateVector eps;
        for (int k = 0; k < 6; ++k) {
            eps(k) = fused_normal_(fused_rng_);
        }
        Particle particle;
        particle.state_vector = fused_mean + fused_root * eps;
        particle.weight = uniform;
        out.particles.push_back(particle);
    }
    out.ess = static_cast<double>(n);
    out.fallback = true;
    return std::isfinite(likelihood);
}

void SMC_LMB_Tracker::stamp_unstamped_tracks() {
    const double now = current_state_.timestamp();
    for (Track& track : current_state_.tracks()) {
        if (std::isnan(track.propagated_time())) {
            track.set_propagated_time(now);
        }
    }
}

uint64_t SMC_LMB_Tracker::propagate_track_to(Track& track, double target_time) {
    const double start = track.propagated_time();
    const double span = target_time - start;
    if (span == 0.0) {
        return 0;
    }
    const int substeps = std::max(1, static_cast<int>(std::ceil(std::abs(span) / max_pending_ - 1e-9)));
    const double step = span / static_cast<double>(substeps);

    std::vector<Particle>& particles = track.mutable_particles();
    for (int k = 0; k < substeps; ++k) {
        const double step_start = start + static_cast<double>(k) * step;
        const double step_end = (k + 1 == substeps) ? target_time : start + static_cast<double>(k + 1) * step;
        const double noise_scale = noise_scale_at(track, step_end);
        const double dt = step_end - step_start;
        if (fast_mode_) {
            propagator_->propagate_cloud_keyed(particles, dt, step_start, noise_scale,
                                               substep_key(track, step_start, dt));
            continue;
        }
        for (Particle& particle : particles) {
            particle = propagator_->propagate(particle, dt, step_start, noise_scale);
        }
    }
    track.set_propagated_time(target_time);
    track.set_cloud_bound(compute_cloud_bound(track.particles()));
    return static_cast<uint64_t>(substeps) * static_cast<uint64_t>(particles.size());
}

bool SMC_LMB_Tracker::may_be_observable(Track& track, const sensor::SensorArray& sensors, double now,
                                        std::vector<size_t>& candidates, double& sleep_for) const {
    sleep_for = 0.0;
    const double tau = now - track.propagated_time();
    if (!(tau > 0.0)) {
        return true;
    }
    if (!track.cloud_bound().valid) {
        track.set_cloud_bound(compute_cloud_bound(track.particles()));
    }
    const CloudBound& bound = track.cloud_bound();
    const std::optional<double> noise = propagator_->noise_displacement_bound(tau);
    if (!noise.has_value()) {
        return true;
    }

    // Extrapolate a virtual particle at the cloud's centre with a second-order Taylor step, then
    // widen by everything that can separate a real particle's position from it after tau seconds:
    //   - the Taylor remainder of the virtual particle itself, |jerk| tau^3 / 6, with
    //     |d/dt (-mu r / |r|^3)| <= 4 mu |v| / |r|^3;
    //   - the cloud's spread, R_pos + R_vel tau, grown by the gravity gradient (norm <= 2 mu / r^3)
    //     acting over tau;
    //   - process noise, from the propagator's bound.
    // The sum is then scaled by 1.5 and padded by a kilometre. Over tau <= max_pending_ = 60 s the
    // gravity terms are metres to a kilometre; the margin is dominated by the cloud and the noise.
    const Eigen::Vector3d& position = bound.center_position;
    const Eigen::Vector3d& velocity = bound.center_velocity;
    const double radius = position.norm();
    const double radius_safe = std::max(radius, kMinRadius);
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    if (radius > 1e-6) {
        acceleration = -kMu * position / (radius * radius_safe * radius_safe);
    }
    const Eigen::Vector3d predicted = position + velocity * tau + 0.5 * acceleration * tau * tau;

    const double speed = velocity.norm() + bound.velocity_radius;
    const double radius_min = std::max(radius - bound.position_radius - speed * tau, kMinRadius);
    const double inv_radius_cubed = 1.0 / (radius_min * radius_min * radius_min);
    const double centre_error = 4.0 * kMu * speed * inv_radius_cubed * tau * tau * tau / 6.0;
    const double spread = (bound.position_radius + bound.velocity_radius * tau) *
                          (1.0 + 2.0 * kMu * inv_radius_cubed * tau * tau);
    const double margin = 1.5 * (centre_error + spread + *noise) + 1000.0;
    if (!particle_gate_) {
        return sensors.any_sensor_within_reach(predicted, margin);
    }

    // Particle gate. The sphere above is loose for a long curved cloud (its centre is off the arc,
    // its radius is half the arc), so it says "maybe" far more often than any particle is near a
    // sensor. Narrow it to the sensors the sphere flagged and test each particle against those.
    double nearest_other_sq = std::numeric_limits<double>::infinity();
    sensors.sensors_within_reach(predicted, margin, candidates, nearest_other_sq);

    // Sleep, chosen so that this same test is guaranteed to say "no" for every sensor it rules out
    // now, at every later time until the next forced propagation (tau <= max_pending_): the margin is
    // taken at its largest (every term grows with tau), the predicted centre moves at most
    // (|v| + a_max T) per second, and every sensor at most sleep_speed_bound_ (validate_sleeps holds
    // the sensors to it). Skipping only answers that are already known therefore changes nothing.
    const double max_range = sensors.fov().max_range;
    double sleep_others = 0.0;
    const double tau_cap = max_pending_;
    const double radius_min_cap = std::max(radius - bound.position_radius - speed * tau_cap, kMinRadius);
    const double inv_radius_cubed_cap = 1.0 / (radius_min_cap * radius_min_cap * radius_min_cap);
    const double centre_error_cap = 4.0 * kMu * speed * inv_radius_cubed_cap * tau_cap * tau_cap * tau_cap / 6.0;
    const double spread_cap = (bound.position_radius + bound.velocity_radius * tau_cap) *
                              (1.0 + 2.0 * kMu * inv_radius_cubed_cap * tau_cap * tau_cap);
    const double margin_cap = 1.5 * (centre_error_cap + spread_cap + sleep_noise_cap_unscaled_) + 1000.0;
    const double centre_speed = velocity.norm() + kMaxGravity * tau_cap;
    if (std::isfinite(max_range) && tau <= tau_cap) {
        const double gap = std::sqrt(nearest_other_sq) - max_range - margin_cap;
        sleep_others = gap / (centre_speed + sleep_speed_bound_);   // +inf with no other sensor
    }
    if (candidates.empty()) {
        sleep_for = sleep_others;
        return false;
    }
    // A sensor that reported a measurement this update keeps the sphere test: a cloud near it can
    // claim that measurement through the fused proposal even with no particle inside its volume,
    // and that pair is only scored if the cloud is current.
    for (size_t sensor_index : candidates) {
        if (std::find(meas_sensor_.begin(), meas_sensor_.end(), static_cast<int>(sensor_index)) !=
            meas_sensor_.end()) {
            return true;
        }
    }
    double sleep_particles = 0.0;
    const bool maybe = particles_may_be_observable(track, sensors, tau, *noise, candidates, sleep_particles,
                                                   predicted, std::sqrt(nearest_other_sq));
    if (!maybe) {
        // The particle bound covers every sensor (flagged ones per particle, the others through the
        // cloud's predicted centre), so it alone keeps this test's answer "no".
        sleep_for = sleep_particles;
        if (gate_audit_) {
            audit_gate_decision(track, sensors, now);
        }
    }
    return maybe;
}

void SMC_LMB_Tracker::validate_sleeps(const sensor::SensorArray& sensors, double now) {
    const size_t count = sensors.size();
    const double max_range = sensors.fov().max_range;
    double max_speed = 0.0;
    for (const sensor::Sensor& sensor : sensors.sensors()) {
        max_speed = std::max(max_speed, sensor.state.tail<3>().norm());
    }
    bool consistent = sleep_sensor_positions_.size() == count && now >= sleep_sensor_time_ &&
                      max_range == sleep_sensor_range_ && std::isfinite(max_speed);
    if (consistent) {
        // Every sensor must have moved no further than the speed bound the sleeps were computed
        // with allows. Displacements chain (triangle inequality), so checking each update interval
        // covers every sleep in progress.
        const double allowed = sleep_speed_bound_ * (now - sleep_sensor_time_) + 1.0;
        for (size_t k = 0; k < count && consistent; ++k) {
            const double moved = (sensors.sensors()[k].state.head<3>() - sleep_sensor_positions_[k]).norm();
            consistent = moved <= allowed;   // false for NaN as well
        }
        consistent = consistent && max_speed <= sleep_speed_bound_;
    }
    if (!consistent) {
        for (Track& track : current_state_.tracks()) {
            track.set_sleep_until(-std::numeric_limits<double>::infinity());
        }
        // Headroom so ordinary orbital speed changes do not wake everything every step.
        sleep_speed_bound_ = 1.25 * max_speed + 100.0;
        sleep_sensor_range_ = max_range;
        if (profiling_) {
            ++profile_.sleep_resets;
        }
    }
    sleep_sensor_positions_.resize(count);
    for (size_t k = 0; k < count; ++k) {
        sleep_sensor_positions_[k] = sensors.sensors()[k].state.head<3>();
    }
    sleep_sensor_time_ = now;
    const std::optional<double> cap = propagator_->noise_displacement_bound(max_pending_);
    sleep_noise_cap_ = cap.has_value() ? *cap * std::sqrt(std::max(1.0, noise_min_scale_))
                                       : std::numeric_limits<double>::infinity();
    sleep_noise_cap_unscaled_ = cap.has_value() ? *cap : std::numeric_limits<double>::infinity();
}

bool SMC_LMB_Tracker::particles_may_be_observable(const Track& track, const sensor::SensorArray& sensors,
                                                  double tau, double noise,
                                                  const std::vector<size_t>& candidates,
                                                  double& sleep_for,
                                                  const Eigen::Vector3d& cloud_predicted,
                                                  double nearest_other) const {
    // Per particle, the same second-order Taylor step as the sphere test with no cloud spread:
    //   predicted = x + v tau + a(x) tau^2 / 2,   |error| <= |jerk|_max tau^3 / 6 + noise,
    // with |jerk| <= 4 mu |v| / r^3 along the path. Along a particle's path over tau, its speed is
    // at most |v| + a_max tau (a_max = mu / r_floor^2 bounds gravity anywhere above the floor) and
    // its radius at least |x| - speed tau. Bounding per particle, not per cloud, matters: a needle
    // hundreds of km long has a sphere whose "lowest point" dips under the gravity floor even though
    // every particle stays near its own orbit radius.
    sleep_for = 0.0;
    const std::vector<Particle>& particles = track.particles();
    if (particles.empty()) {
        sleep_for = std::numeric_limits<double>::infinity();
        return false;   // nothing to see
    }
    const double max_range = sensors.fov().max_range;
    // Annealed noise can exceed the unit-scale bound when noise_min_scale > 1.
    const double noise_bound = noise * std::sqrt(std::max(1.0, noise_min_scale_));
    if (!std::isfinite(max_range) || !std::isfinite(noise_bound) || !std::isfinite(tau)) {
        // An unbounded sensor or noise bound leaves nothing to test against: stay conservative.
        return true;
    }
    const double fixed_margin = 1.5 * noise_bound + 1000.0;
    const double tau_cubed_sixth = tau * tau * tau / 6.0;
    const double half_tau_sq = 0.5 * tau * tau;

    // For the sleep (see may_be_observable): this test's largest margin over any later tau up to
    // max_pending_, the fastest a predicted particle can move, the closest a predicted particle
    // comes to a candidate, and how far any predicted particle sits from the cloud's predicted
    // centre (which bounds its distance to every sensor the sphere ruled out).
    const double tau_cap = max_pending_;
    const double tau_cap_cubed_sixth = tau_cap * tau_cap * tau_cap / 6.0;
    double nearest_sq = std::numeric_limits<double>::infinity();
    double worst_margin_cap = 0.0;
    double worst_speed_cap = 0.0;
    double worst_offset_sq = 0.0;
    bool can_sleep = tau <= tau_cap;
    for (const Particle& particle : particles) {
        const Eigen::Vector3d position = particle.state_vector.head<3>();
        const Eigen::Vector3d velocity = particle.state_vector.tail<3>();
        const double r = position.norm();
        const double v = velocity.norm();
        const double speed = v + kMaxGravity * tau;
        const double r_low = r - speed * tau;
        if (!(r_low > kMinRadius)) {
            // The path could dip under the gravity floor, where the acceleration model changes
            // form (or the state is not finite): treat as possibly visible.
            return true;
        }
        const double taylor_error = 4.0 * kMu * speed / (r_low * r_low * r_low) * tau_cubed_sixth;
        const double reach = max_range + 1.5 * taylor_error + fixed_margin;
        const double reach_sq = reach * reach;
        // r > kMinRadius here, so this is exactly the propagator's acceleration.
        const Eigen::Vector3d acceleration = -kMu * position / (r * r * r);
        const Eigen::Vector3d predicted = position + velocity * tau + acceleration * half_tau_sq;
        for (size_t sensor_index : candidates) {
            const double d2 = (predicted - sensors.sensors()[sensor_index].state.head<3>()).squaredNorm();
            if (!(d2 > reach_sq)) {   // negated: a NaN state counts as possibly visible
                return true;
            }
            nearest_sq = std::min(nearest_sq, d2);
        }
        // The same quantities at tau_cap. If this particle's path could reach the floor by then,
        // the test would answer "maybe" outright before then: no sleep.
        const double speed_cap = v + kMaxGravity * tau_cap;
        const double r_low_cap = r - speed_cap * tau_cap;
        if (!(r_low_cap > kMinRadius)) {
            can_sleep = false;
        } else {
            const double error_cap = 4.0 * kMu * speed_cap / (r_low_cap * r_low_cap * r_low_cap) * tau_cap_cubed_sixth;
            worst_margin_cap = std::max(worst_margin_cap, 1.5 * error_cap);
        }
        worst_speed_cap = std::max(worst_speed_cap, speed_cap);
        worst_offset_sq = std::max(worst_offset_sq, (predicted - cloud_predicted).squaredNorm());
    }
    if (can_sleep) {
        const double margin_cap = max_range + worst_margin_cap + 1.5 * sleep_noise_cap_ + 1000.0;
        const double closing = worst_speed_cap + sleep_speed_bound_;
        // Sensors the sphere flagged now: per-particle distances. Sensors it ruled out: at least
        // their distance to the cloud's predicted centre minus the farthest particle's offset.
        const double gap_candidates = std::sqrt(nearest_sq) - margin_cap;
        const double gap_others = nearest_other - std::sqrt(worst_offset_sq) - margin_cap;
        sleep_for = std::min(gap_candidates, gap_others) / closing;
    }
    return false;
}

void SMC_LMB_Tracker::audit_gate_decision(const Track& track, const sensor::SensorArray& sensors,
                                          double now) const {
    // Noise-free RK4 in substeps of at most max_pending_, the deterministic part of what
    // propagate_track_to would do. noise_scale = 0 draws no random number.
    const double start = track.propagated_time();
    const double span = now - start;
    const int substeps = std::max(1, static_cast<int>(std::ceil(std::abs(span) / max_pending_ - 1e-9)));
    const double step = span / static_cast<double>(substeps);
    bool violation = false;
    for (const Particle& original : track.particles()) {
        Particle particle = original;
        for (int k = 0; k < substeps; ++k) {
            particle = propagator_->propagate(particle, step, start + k * step, 0.0);
        }
        if (sensors.visible_sensor(particle.state_vector.head<3>()) >= 0) {
            violation = true;
            break;
        }
    }
    ++gate_audit_checks_;
    if (violation) {
        ++gate_audit_violations_;
    }
}


void SMC_LMB_Tracker::refresh_observable_tracks(const sensor::SensorArray& sensors) {
    const double now = current_state_.timestamp();
    const ScopedTimer timer(profiling_ ? &profile_.refresh : nullptr);
    std::vector<Track>& tracks = current_state_.tracks();
    const size_t count = tracks.size();
    // Per-track results, folded into the profile in track order after the loop.
    std::vector<char> checked(profiling_ ? count : 0, 0);
    std::vector<uint64_t> steps(profiling_ ? count : 0, 0);
    std::vector<double> seconds(profiling_ ? count : 0, 0.0);
    std::vector<signed char> bucket(profiling_ ? count : 0, -1);
    std::vector<double> check_seconds(profiling_ ? count : 0, 0.0);
    std::vector<char> slept(profiling_ ? count : 0, 0);
    std::vector<double> slept_for(profiling_ ? count : 0, 0.0);
    if (profiling_) {
        refreshed_scratch_.assign(count, 0);
    }
    if (particle_gate_) {
        validate_sleeps(sensors, now);
    }
    // A sleep guarantees the test's answer only for sensors that report nothing: a cloud near a
    // sensor that reported a measurement is refreshed on weaker grounds (its sphere reaches that
    // sensor, so the fused proposal can score it), so on such steps every lagging track is tested.
    const bool honour_sleep = particle_gate_ && gate_sleep_ && meas_sensor_.empty();
    // First pass: stamp new tracks, pass over current ones and sleepers. What is left -- usually a
    // few clouds near the ring -- is tested below.
    std::vector<size_t>& work = refresh_work_;
    work.clear();
    for (size_t i = 0; i < count; ++i) {
        Track& track = tracks[i];
        if (std::isnan(track.propagated_time())) {
            track.set_propagated_time(now);
            continue;
        }
        if (!(track.propagated_time() < now)) {
            continue;
        }
        if (honour_sleep && now < track.sleep_until()) {
            // The test is known to answer "not observable" until then (see may_be_observable).
            if (profiling_) {
                slept[i] = 1;
            }
            if (gate_audit_) {
                audit_gate_decision(track, sensors, now);
                // The sleep must only ever skip tests that would have answered "no".
                std::vector<size_t> scratch;
                double unused = 0.0;
                ++sleep_audit_checks_;
                if (may_be_observable(track, sensors, now, scratch, unused)) {
                    ++sleep_audit_violations_;
                    if (sleep_audit_first_.empty()) {
                        sleep_audit_first_ = "t=" + std::to_string(now) + " label=" +
                            std::to_string(track.label().birth_time) + "/" + std::to_string(track.label().index) +
                            " propagated=" + std::to_string(track.propagated_time()) +
                            " sleep_until=" + std::to_string(track.sleep_until());
                    }
                }
            }
            continue;
        }
        work.push_back(i);
    }
    std::vector<size_t>& candidates = gate_candidates_;
    for (size_t i : work) {
        Track& track = tracks[i];
        const auto check_start = profiling_ ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        double sleep_for = 0.0;
        const bool maybe = may_be_observable(track, sensors, now, candidates, sleep_for);
        if (profiling_) {
            checked[i] = 1;
            check_seconds[i] = seconds_since(check_start);
        }
        if (!maybe) {
            // Never past the next forced propagation, which re-derives everything; NaN, zero and
            // negative bounds mean no sleep.
            const double sleep = std::min(sleep_for, max_pending_);
            if (particle_gate_ && gate_sleep_ && sleep > 0.0) {
                track.set_sleep_until(now + sleep);
                if (profiling_) {
                    slept_for[i] = sleep;
                }
            }
            continue;
        }
        const auto start = profiling_ ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        const uint64_t done = propagate_track_to(track, now);
        if (profiling_) {
            steps[i] = done;
            seconds[i] = seconds_since(start);
            refreshed_scratch_[i] = 1;
            bucket[i] = static_cast<signed char>(refresh_gap_bucket(track, sensors));
        }
    }
    if (!profiling_) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        profile_.refresh_checks += checked[i];
        profile_.refresh_slept += slept[i];
        if (checked[i] && !refreshed_scratch_[i]) {
            const double amount = slept_for[i];
            ++profile_.sleep_histogram[amount <= 0.0 ? 0 : amount < 1.0 ? 1 : amount < 5.0 ? 2 : amount < 20.0 ? 3 : 4];
        }
        profile_.refresh_check += check_seconds[i];
        if (!refreshed_scratch_[i]) {
            continue;
        }
        ++profile_.tracks_refreshed;
        profile_.refresh_particle_steps += steps[i];
        profile_.refresh_propagate += seconds[i];
        switch (bucket[i]) {
            case 0: ++profile_.refresh_gap_inside; break;
            case 1: ++profile_.refresh_gap_under_5km; break;
            case 2: ++profile_.refresh_gap_5_to_50km; break;
            case 3: ++profile_.refresh_gap_over_50km; break;
            default: break;
        }
    }
}

int SMC_LMB_Tracker::refresh_gap_bucket(const Track& track, const sensor::SensorArray& sensors) const {
    constexpr double kNearGap = 5.0e3;
    constexpr double kFarGap = 50.0e3;
    const std::vector<Particle>& particles = track.particles();
    const double max_range = sensors.fov().max_range;
    if (particles.empty() || sensors.size() == 0) {
        return -1;
    }
    if (!std::isfinite(max_range)) {
        return 0;   // unbounded sensors see everything; never form inf - inf
    }
    const CloudBound bound = track.current_cloud_bound();
    std::vector<Eigen::Vector3d> candidates;
    for (const sensor::Sensor& sensor : sensors.sensors()) {
        const double sphere_gap =
            (bound.center_position - sensor.state.head<3>()).norm() - bound.position_radius - max_range;
        if (sphere_gap <= kFarGap) {
            candidates.push_back(sensor.state.head<3>());
        }
    }
    if (candidates.empty()) {
        return 3;
    }
    double best = std::numeric_limits<double>::infinity();
    for (const Particle& particle : particles) {
        const Eigen::Vector3d position = particle.state_vector.head<3>();
        for (const Eigen::Vector3d& sensor_position : candidates) {
            best = std::min(best, (position - sensor_position).squaredNorm());
        }
    }
    const double gap = std::sqrt(best) - max_range;
    if (!(gap > 0.0)) {
        return 0;
    }
    return gap < kNearGap ? 1 : (gap < kFarGap ? 2 : 3);
}

void SMC_LMB_Tracker::synchronize() {
    ensure_models_configured();
    const ScopedTimer timer(profiling_ ? &profile_.synchronize : nullptr);
    const double now = current_state_.timestamp();
    std::vector<Track>& tracks = current_state_.tracks();
    std::vector<uint64_t> steps(profiling_ ? tracks.size() : 0, 0);
    for (size_t i = 0; i < tracks.size(); ++i) {
        Track& track = tracks[i];
        if (std::isnan(track.propagated_time())) {
            track.set_propagated_time(now);
        } else if (track.propagated_time() != now) {
            const uint64_t done = propagate_track_to(track, now);
            if (profiling_) {
                steps[i] = done;
            }
        }
    }
    for (uint64_t done : steps) {
        profile_.synchronize_particle_steps += done;
    }
}

void SMC_LMB_Tracker::update(const std::vector<Measurement>& measurements) {
    update_impl(measurements, nullptr);
}

void SMC_LMB_Tracker::update(const std::vector<Measurement>& measurements,
                             const sensor::SensorArray& sensors) {
    update_impl(measurements, &sensors);
}

void SMC_LMB_Tracker::resolve_measurement_sensors(const std::vector<Measurement>& measurements,
                                                  const sensor::SensorArray* sensors) {
    const size_t num_meas = measurements.size();
    meas_sensor_.assign(num_meas, -1);
    sensor_array_ = sensors != nullptr;
    if (sensors == nullptr) {
        num_sensors_ = 0;
        return;
    }

    num_sensors_ = sensors->size();
    if (num_sensors_ == 0 && num_meas > 0) {
        throw std::invalid_argument("update(): got " + std::to_string(num_meas) +
                                    " measurement(s) but the SensorArray is empty");
    }
    for (size_t j = 0; j < num_meas; ++j) {
        const int sensor_index = sensors->index_of(measurements[j].sensor_id_);
        if (sensor_index < 0) {
            throw std::invalid_argument("update(): Measurement.sensor_id_ '" +
                                        measurements[j].sensor_id_ +
                                        "' is not in the SensorArray");
        }
        meas_sensor_[j] = sensor_index;
    }
}

void SMC_LMB_Tracker::compute_coverage(const sensor::SensorArray* sensors) {
    const ScopedTimer timer(profiling_ ? &profile_.coverage : nullptr);
    const std::vector<Track>& tracks = current_state_.tracks();
    const double now = current_state_.timestamp();
    const size_t num_tracks = tracks.size();

    active_tracks_.clear();
    coverage_per_sensor_.clear();
    coverage_union_.clear();
    particle_sensor_.clear();
    track_inv_weight_sum_.clear();
    reach_flags_.clear();
    track_particle_offsets_.assign(1, 0);

    // Sensors that produced a measurement this update: only they can make a reach-only pair.
    std::vector<size_t> reporting_sensors;
    const bool track_reach = fused_proposal_ && sensors != nullptr;
    if (track_reach) {
        for (int sensor_index : meas_sensor_) {
            if (sensor_index >= 0 &&
                std::find(reporting_sensors.begin(), reporting_sensors.end(),
                          static_cast<size_t>(sensor_index)) == reporting_sensors.end()) {
                reporting_sensors.push_back(static_cast<size_t>(sensor_index));
            }
        }
    }

    for (size_t i = 0; i < num_tracks; ++i) {
        const Track& track = tracks[i];
        double union_fraction = 1.0;
        if (sensors != nullptr) {
            // A track lagging the clock was judged unobservable by refresh_observable_tracks, so it
            // has no particle in any volume at `now`; its stale cloud must not be scored.
            if (track.propagated_time() < now) {
                continue;
            }
            union_fraction = sensors->coverage(track, coverage_scratch_, &particle_sensor_scratch_);

            // Fused proposal: a cloud whose bounding sphere reaches a reporting sensor is a candidate
            // for that sensor's measurements even with no particle inside the volume.
            bool reaches_any = false;
            if (track_reach) {
                const size_t row = reach_flags_.size();
                reach_flags_.resize(row + num_sensors_, 0);
                if (!reporting_sensors.empty()) {
                    const CloudBound bound = track.current_cloud_bound();
                    for (size_t sensor_index : reporting_sensors) {
                        if (coverage_scratch_[sensor_index] <= 0.0 &&
                            sensors->sphere_reaches(sensor_index, bound)) {
                            reach_flags_[row + sensor_index] = 1;
                            reaches_any = true;
                        }
                    }
                }
                if (!(union_fraction > 0.0) && !reaches_any) {
                    reach_flags_.resize(row);
                }
            }
            if (profiling_ && union_fraction > 0.0 && i < refreshed_scratch_.size() &&
                refreshed_scratch_[i]) {
                ++profile_.refreshed_tracks_covered;
            }
            if (!(union_fraction > 0.0) && !reaches_any) {
                continue;
            }
            coverage_per_sensor_.insert(coverage_per_sensor_.end(), coverage_scratch_.begin(),
                                        coverage_scratch_.end());
            particle_sensor_.insert(particle_sensor_.end(), particle_sensor_scratch_.begin(),
                                    particle_sensor_scratch_.end());
        }

        double weight_sum = 0.0;
        for (const Particle& particle : track.particles()) {
            weight_sum += particle.weight;
        }
        active_tracks_.push_back(i);
        if (profiling_) {
            ++profile_.active_tracks;
        }
        coverage_union_.push_back(union_fraction);
        track_inv_weight_sum_.push_back(weight_sum > 0.0 ? 1.0 / weight_sum : 0.0);
        track_particle_offsets_.push_back(track_particle_offsets_.back() + track.particles().size());
    }
}

void SMC_LMB_Tracker::prune_tracks(std::vector<Track>& tracks) {
    // erase-remove in place instead of copying survivors into a new vector.
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [this](const Track& track) {
                                    return track.existence_probability() < prune_threshold_;
                                }),
                 tracks.end());
}

void SMC_LMB_Tracker::apply_posterior(size_t a, size_t num_meas,
                                      const std::vector<double>& det_coefficients,
                                      const std::vector<char>& det_used, double miss_coefficient,
                                      bool miss_used, size_t contributing_hypotheses) {
    Track& track = current_state_.tracks()[active_tracks_[a]];
    const std::vector<Particle>& predicted_particles = track.particles();
    const size_t num_particles = predicted_particles.size();

    // LMB posterior (Reuter et al. 2014), with a state-dependent P_D(x) = P_D * visible(x):
    //   rho   = r (1 - <p, P_D>) / (1 - r <p, P_D>)        existence given "not detected"
    //   r'    = sum_j c_j + c_0 rho
    //   p'(x) ∝ sum_j c_j p(x) P_D(x) g_j(x) / <p, P_D g_j>  +  c_0 rho p(x) (1 - P_D(x)) / (1 - <p, P_D>)
    // Each bracket is a normalised density, so the mixture's mass is exactly r'. c_j and c_0 are the
    // hypothesis-weight marginals; they already carry the eta factors, and nothing multiplies them
    // by the likelihood again.
    const double r = track.existence_probability();
    const double pd_mass = detection_mass(a);
    const double not_detected = 1.0 - r * pd_mass;
    // not_detected <= 0 needs r == 1 and <p, P_D> == 1: the track certainly exists and would
    // certainly have been reported, so "not detected" is a contradiction and carries no existence.
    const double rho = not_detected > 0.0 ? r * (1.0 - pd_mass) / not_detected : 0.0;

    double detection_total = 0.0;
    for (size_t j = 0; j < num_meas; ++j) {
        if (det_used[j]) {
            detection_total += det_coefficients[j];
        }
    }
    const double miss_mass = miss_used ? miss_coefficient * rho : 0.0;
    const double r_new = std::min(1.0, std::max(0.0, detection_total + miss_mass));
    track.set_existence_probability(r_new);

    // Counts hypotheses that took a branch, not buckets that ended up non-zero.
    if (contributing_hypotheses == 0 || num_particles == 0) {
        return;
    }
    // A cloud nobody could see (reach-only under the fused proposal) that took no measurement: its
    // posterior is its prior, so there is nothing to resample.
    if (detection_total == 0.0 && pd_mass == 0.0) {
        return;
    }

    // Detection components drawn from the fused proposal live outside the track's own cloud; they
    // are mixed in below, after the components that reweight the track's own particles.
    const bool any_fused = !fused_index_.empty() && [&] {
        for (size_t j = 0; j < num_meas; ++j) {
            if (det_used[j] && fused_index_[a * num_meas + j] >= 0) {
                return true;
            }
        }
        return false;
    }();

    // Mixture, one pass per used association. Measurements ascending, then the miss term, so the
    // summation order is fixed run to run.
    mixture_weights_.assign(num_particles, 0.0);
    for (size_t j = 0; j < num_meas; ++j) {
        if (!det_used[j] || (any_fused && fused_index_[a * num_meas + j] >= 0)) {
            continue;
        }
        const double coefficient = det_coefficients[j];
        const double* assoc = association_weights_.data() + association_block_offset(a, j, num_meas);
        for (size_t p = 0; p < num_particles; ++p) {
            mixture_weights_[p] += coefficient * assoc[p];
        }
    }

    const double miss_normaliser = 1.0 - pd_mass;
    if (miss_mass > 0.0 && miss_normaliser > 0.0) {
        const double scale = miss_mass * track_inv_weight_sum_[a] / miss_normaliser;
        const double miss_visible = 1.0 - p_detection_;
        for (size_t p = 0; p < num_particles; ++p) {
            const double miss_probability = particle_visible_anywhere(a, p) ? miss_visible : 1.0;
            mixture_weights_[p] += scale * predicted_particles[p].weight * miss_probability;
        }
    }

    // The resampling source: the track's own particles, plus any fused components appended after
    // them with weight coefficient * (normalised component weight).
    const std::vector<Particle>* source = &predicted_particles;
    std::vector<Particle> union_particles;
    int fused_count = 0;
    int fallback_count = 0;
    double fused_ess = 0.0;
    if (any_fused) {
        union_particles = predicted_particles;
        for (size_t j = 0; j < num_meas; ++j) {
            const int index = det_used[j] ? fused_index_[a * num_meas + j] : -1;
            if (index < 0) {
                continue;
            }
            const FusedComponent& component = fused_components_[static_cast<size_t>(index)];
            for (const Particle& particle : component.particles) {
                union_particles.push_back(particle);
                mixture_weights_.push_back(det_coefficients[j] * particle.weight);
            }
            fused_ess = (fused_count == 0) ? component.ess : std::min(fused_ess, component.ess);
            ++fused_count;
            fallback_count += component.fallback ? 1 : 0;
        }
        source = &union_particles;
    }
    const size_t source_size = source->size();

    double sum_weights = 0.0;
    for (size_t p = 0; p < source_size; ++p) {
        sum_weights += mixture_weights_[p];
    }

    if (sum_weights > 0.0 && std::isfinite(sum_weights)) {
        const double inv_sum = 1.0 / sum_weights;
        for (size_t p = 0; p < source_size; ++p) {
            mixture_weights_[p] *= inv_sum;
        }
    } else {
        // No information to go on. Equal slots make the systematic walk keep every particle
        // exactly once, which is the sensible reading of that.
        const double uniform_weight = 1.0 / static_cast<double>(source_size);
        for (size_t p = 0; p < source_size; ++p) {
            mixture_weights_[p] = uniform_weight;
        }
    }

    double ess = 0.0;
    if (regularization_ || record_diagnostics_) {
        double sum_sq = 0.0;
        for (size_t p = 0; p < source_size; ++p) {
            sum_sq += mixture_weights_[p] * mixture_weights_[p];
        }
        ess = sum_sq > 0.0 ? 1.0 / sum_sq : 0.0;
    }
    const bool regularize_now = regularization_ && num_particles > 1 &&
                                ess < regularization_ess_threshold_ * static_cast<double>(num_particles);

    std::vector<Particle> resampled_particles = systematic_resample(*source, mixture_weights_, num_particles);

    if (regularize_now) {
        regularize(resampled_particles, *source, mixture_weights_);
    }
    if (profiling_) {
        ++profile_.posterior_updates;
        profile_.regularizations += regularize_now ? 1 : 0;
    }
    if (record_diagnostics_) {
        PosteriorRecord record;
        record.time = current_state_.timestamp();
        record.birth_time = track.label().birth_time;
        record.index = track.label().index;
        record.ess = ess;
        record.num_particles = num_particles;
        record.detection_mass = detection_total;
        record.regularized = regularize_now;
        record.fused_components = fused_count;
        record.fallback_components = fallback_count;
        record.fused_ess = fused_ess;
        for (size_t j = 0; j < num_meas; ++j) {
            if (det_used[j] && det_coefficients[j] > record.best_coefficient) {
                record.best_coefficient = det_coefficients[j];
                record.best_measurement = static_cast<int>(j);
            }
        }
        diagnostics_.push_back(record);
    }

    // Move the resampled cloud into the track. This reallocates particles_ and invalidates
    // any zero-copy NumPy views that still alias the previous storage.
    track.set_particles(std::move(resampled_particles));
}

void SMC_LMB_Tracker::update_impl(const std::vector<Measurement>& measurements,
                                  const sensor::SensorArray* sensors) {
    ensure_models_configured();
    const ScopedTimer total_timer(profiling_ ? &profile_.update : nullptr);
    if (profiling_) {
        ++profile_.update_calls;
        profile_.updates_with_measurements += measurements.empty() ? 0 : 1;
        refreshed_scratch_.clear();
    }

    for (const auto& measurement : measurements) {
        validation::require_measurement(measurement);
    }

    // Resolve every measurement to its sensor before anything mutates the filter, so an unknown
    // sensor_id_ is rejected on every path -- including a birth-only step -- and a rejected update
    // leaves the state exactly as it was.
    resolve_measurement_sensors(measurements, sensors);

    // Bring the clouds this update can score up to the clock. Without a sensor array there is no
    // telling what is observable, so everything is.
    if (lazy_propagation_) {
        if (sensors != nullptr) {
            refresh_observable_tracks(*sensors);
        } else {
            synchronize();
        }
    } else {
        stamp_unstamped_tracks();
    }

    std::vector<Track>& tracks = current_state_.tracks();
    const size_t num_meas = measurements.size();
    const double now = current_state_.timestamp();

    auto give_birth = [&](const std::vector<Measurement>& unused) {
        if (unused.empty() || !birth_model_) {
            return;
        }
        const ScopedTimer birth_timer(profiling_ ? &profile_.birth : nullptr);
        std::vector<Track> born_tracks = birth_model_->generate_new_tracks(unused, now);
        if (profiling_) {
            profile_.births += born_tracks.size();
        }
        for (auto& new_track : born_tracks) {
            new_track.set_propagated_time(now);
            new_track.set_cloud_bound(compute_cloud_bound(new_track.particles()));
            tracks.push_back(std::move(new_track));
        }
    };

    // Handle the case of no existing tracks - just create new ones from all measurements
    if (tracks.empty()) {
        give_birth(measurements);
        return;
    }

    if (num_meas == 0 && sensors == nullptr) {
        // Without a sensor array there is no notion of what was observable this step, so an
        // empty measurement list carries no information -- the historical behaviour.
        return;
    }

    compute_coverage(sensors);
    const size_t num_active = active_tracks_.size();

    if (num_meas == 0) {
        // Nobody reported anything. For a track some sensor could see that is still evidence: the
        // only hypothesis is "not detected", with marginal 1.
        const std::vector<double> no_coefficients;
        const std::vector<char> no_used;
        const ScopedTimer posterior_timer(profiling_ ? &profile_.posterior : nullptr);
        for (size_t a = 0; a < num_active; ++a) {
            apply_posterior(a, 0, no_coefficients, no_used, 1.0, true, 1);
        }
        prune_tracks(tracks);
        return;
    }

    if (num_active == 0) {
        // No track can explain any measurement: every one is unused, and no track learns anything.
        give_birth(measurements);
        return;
    }

    std::vector<MeasurementLikelihoodCache> meas_caches;
    meas_caches.reserve(num_meas);
    for (const auto& measurement : measurements) {
        meas_caches.push_back(InOrbitSensorModel::buildCache(measurement));
    }

    // Step 2: per-(track, measurement) association weights and likelihoods.
    //
    //   L_ij = <p_i, 1_{visible to s(j)} g_j> = sum_p (w_p / W_i) vis_{s(j)}(p) g_j(x_p)
    //
    // so P_D * L_ij is <p_i, P_D(x) g_j> for P_D(x) = P_D * visible(x). A particle the producing
    // sensor cannot see cannot have produced the measurement, so it gets no likelihood pass and a
    // zero association weight. The block stores the normalised posterior of that association.
    //
    // Storage is one flat buffer indexed through the prefix-sum offsets of the active tracks, so
    // per-track particle counts are free to differ.
    association_weights_.assign(track_particle_offsets_[num_active] * num_meas, 0.0);
    Eigen::MatrixXd likelihood_matrix = Eigen::MatrixXd::Zero(num_active, num_meas);
    fused_components_.clear();
    fused_index_.clear();
    if (fused_proposal_) {
        fused_index_.assign(num_active * num_meas, -1);
    }

    // The likelihood timer covers the whole loop; the fused builds inside it are timed separately
    // and taken back out afterwards.
    const double fused_before = profile_.fused;
    double likelihood_loop = 0.0;
    {
    const ScopedTimer likelihood_timer(profiling_ ? &likelihood_loop : nullptr);
    for (size_t a = 0; a < num_active; ++a) {
        const auto& current_particles = tracks[active_tracks_[a]].particles();
        const size_t num_particles = current_particles.size();
        const double inv_weight_sum = track_inv_weight_sum_[a];

        for (size_t j = 0; j < num_meas; ++j) {
            if (!pair_is_observable(a, j)) {
                // No particle of this track is inside the producing sensor's volume: the pair is
                // impossible and Step 3 prices it at INF_COST. The block stays zero.
                continue;
            }
            const auto& measurement = measurements[j];
            const int sensor_index = meas_sensor_[j];
            double* assoc = association_weights_.data() + association_block_offset(a, j, num_meas);

            double total_likelihood = 0.0;
            double total_squared = 0.0;
            size_t visible_count = 0;
            for (size_t p = 0; p < num_particles; ++p) {
                if (!particle_visible_to(a, p, sensor_index)) {
                    continue;
                }
                ++visible_count;
                if (profiling_) {
                    ++profile_.likelihood_evaluations;
                }
                const auto& current_particle = current_particles[p];
                const double particle_likelihood = sensor_model_->calculate_likelihood(
                    current_particle, measurement, meas_caches[j]);
                const double updated_weight = current_particle.weight * inv_weight_sum * particle_likelihood;
                assoc[p] = updated_weight;
                total_likelihood += updated_weight;
                total_squared += updated_weight * updated_weight;
            }

            // Raw L, with neither P_D nor kappa: Step 3 applies each exactly once, and nothing else
            // multiplies by them.
            likelihood_matrix(a, j) = total_likelihood;

            // Fused proposal: when the particle sum has collapsed onto a handful of particles (or
            // underflowed to zero), rebuild this pair's detection component where the cloud and the
            // measurement overlap. Its likelihood replaces the collapsed sum.
            if (fused_proposal_) {
                const double pair_ess = total_squared > 0.0
                    ? total_likelihood * total_likelihood / total_squared : 0.0;
                if (pair_ess < fused_ess_min_) {
                    FusedComponent component;
                    double fused_likelihood = 0.0;
                    bool built = false;
                    {
                        const ScopedTimer fused_timer(profiling_ ? &profile_.fused : nullptr);
                        if (profiling_) {
                            ++profile_.fused_builds;
                        }
                        built = build_fused_component(tracks[active_tracks_[a]], measurement, meas_caches[j],
                                                      sensor_index, sensors, fused_likelihood, component);
                    }
                    if (built) {
                        likelihood_matrix(a, j) = fused_likelihood;
                        fused_index_[a * num_meas + j] = static_cast<int>(fused_components_.size());
                        fused_components_.push_back(std::move(component));
                    }
                }
            }

            if (total_likelihood >= std::numeric_limits<double>::min()) {
                const double inv_total = 1.0 / total_likelihood;
                for (size_t p = 0; p < num_particles; ++p) {
                    assoc[p] *= inv_total;
                }
            } else if (visible_count > 0) {
                // Every visible likelihood underflowed. The association is then priced at the floor
                // and carries (almost) no weight; if it is used at all, spread it over the particles
                // that could have produced it.
                const double uniform_weight = 1.0 / static_cast<double>(visible_count);
                for (size_t p = 0; p < num_particles; ++p) {
                    assoc[p] = particle_visible_to(a, p, sensor_index) ? uniform_weight : 0.0;
                }
            }
        }
    }
    }
    if (profiling_) {
        profile_.likelihood += likelihood_loop - (profile_.fused - fused_before);
    }
    std::optional<ScopedTimer> assignment_timer;
    assignment_timer.emplace(profiling_ ? &profile_.assignment : nullptr);

    // Step 3: augmented cost matrix, N_active x (M + N_active), with the hypothesis factors of the
    // LMB update (Reuter et al. 2014, existence folded into the ranked assignment):
    //   detection  eta_ij = r_i P_D L_ij / kappa
    //   undetected eta_i0 = 1 - r_i <p_i, P_D>   (missed, or does not exist)
    // Left block [0, M): -ln eta_ij. Right block: diagonal -ln eta_i0, off-diagonal INF_COST (each
    // track owns one miss column).
    const size_t augmented_cols = num_meas + num_active;
    Eigen::MatrixXd cost_matrix(num_active, augmented_cols);

    for (size_t a = 0; a < num_active; ++a) {
        const double r = tracks[active_tracks_[a]].existence_probability();
        for (size_t j = 0; j < num_meas; ++j) {
            if (!pair_is_observable(a, j)) {
                cost_matrix(a, j) = INF_COST;
                continue;
            }
            cost_matrix(a, j) = factor_cost(r * p_detection_ * likelihood_matrix(a, j) / clutter_intensity_);
        }
        const double miss_cost = factor_cost(1.0 - r * detection_mass(a));
        for (size_t b = 0; b < num_active; ++b) {
            cost_matrix(a, num_meas + b) = (a == b) ? miss_cost : INF_COST;
        }
    }

    // Step 4: Solve assignment (K-best)
    std::vector<Hypothesis> hypotheses = solve_assignment(cost_matrix, k_best_);

    if (hypotheses.empty()) {
        return;
    }

    // Normalize hypothesis weights (log-sum-exp)
    std::vector<double> log_weights;
    for (const auto& h : hypotheses) log_weights.push_back(-h.weight);
    double max_logw = *std::max_element(log_weights.begin(), log_weights.end());
    std::vector<double> norm_weights;
    double sum_exp = 0.0;
    for (double lw : log_weights) sum_exp += std::exp(lw - max_logw);
    if (sum_exp <= 0.0) {
        return;
    }
    for (double lw : log_weights) norm_weights.push_back(std::exp(lw - max_logw) / sum_exp);
    assignment_timer.reset();

    // Step 5: marginals, posterior existence, mixture and resampling.
    //
    // The hypotheses are grouped by the association they give each track: every hypothesis that
    // assigns track a to measurement j contributes the same normalised per-particle vector, and
    // every miss column the same miss density, so at most num_meas + 1 per-particle passes are
    // needed however large k_best_ is.
    std::optional<ScopedTimer> posterior_timer;
    posterior_timer.emplace(profiling_ ? &profile_.posterior : nullptr);
    for (size_t a = 0; a < num_active; ++a) {
        assoc_coefficients_.assign(num_meas, 0.0);
        assoc_used_.assign(num_meas, 0);
        double miss_coefficient = 0.0;
        bool miss_used = false;
        size_t contributing_hypotheses = 0;

        for (size_t h = 0; h < hypotheses.size(); ++h) {
            const int assoc_idx = hypotheses[h].associations[a];
            const double hyp_weight = norm_weights[h];

            if (assoc_idx >= 0 && static_cast<size_t>(assoc_idx) < num_meas) {
                assoc_coefficients_[static_cast<size_t>(assoc_idx)] += hyp_weight;
                assoc_used_[static_cast<size_t>(assoc_idx)] = 1;
                ++contributing_hypotheses;
            } else if (assoc_idx == -1 ||
                       (static_cast<size_t>(assoc_idx) >= num_meas &&
                        static_cast<size_t>(assoc_idx) < num_meas + num_active)) {
                // All miss columns share one formula, so they collapse into a single bucket.
                miss_coefficient += hyp_weight;
                miss_used = true;
                ++contributing_hypotheses;
            }
            // Any other assoc_idx falls through without contributing.
        }

        apply_posterior(a, num_meas, assoc_coefficients_, assoc_used_, miss_coefficient, miss_used,
                        contributing_hypotheses);
    }
    posterior_timer.reset();

    // Step 6 reads the best hypothesis, which is indexed by active position; resolve it before
    // pruning reshuffles the track vector.
    std::vector<bool> measurement_used(num_meas, false);
    const Hypothesis& best_hypothesis = hypotheses[0];
    for (size_t a = 0; a < best_hypothesis.associations.size(); ++a) {
        const int meas_idx = best_hypothesis.associations[a];
        if (meas_idx >= 0 && static_cast<size_t>(meas_idx) < num_meas) {
            measurement_used[static_cast<size_t>(meas_idx)] = true;
        }
    }

    prune_tracks(tracks);

    // Step 6: Adaptive Birth - Create new tracks from unused measurements
    std::vector<Measurement> unused_measurements;
    unused_measurements.reserve(num_meas);
    for (size_t j = 0; j < num_meas; ++j) {
        if (!measurement_used[j]) {
            unused_measurements.push_back(measurements[j]);
        }
    }
    give_birth(unused_measurements);
}

// Returns the RAW weighted-average likelihood L = sum_p w_p * g(z|x_p), with no clutter
// intensity and no detection probability applied, over the whole cloud. update() uses the same
// quantity restricted to the particles the producing sensor can see; with no sensor array (or an
// unbounded one) the two coincide, up to the division by the weight sum. Callers reproducing the
// filter's cost matrix must apply r * P_D / kappa exactly once themselves (see
// tests/bench_engine.py::build_cost_matrix).
double SMC_LMB_Tracker::compute_association_likelihood(const Track& track, const Measurement& measurement) const {
    ensure_models_configured();
    validation::require_measurement(measurement);

    const MeasurementLikelihoodCache cache = InOrbitSensorModel::buildCache(measurement);
    double total_likelihood = 0.0;
    const auto& particles = track.particles();
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        double particle_likelihood = sensor_model_->calculate_likelihood(p, measurement, cache);
        double weighted_likelihood = particle_likelihood * p.weight;
        total_likelihood += weighted_likelihood;
    }
    return total_likelihood;
}

const std::vector<Track>& SMC_LMB_Tracker::get_tracks() const {
    return current_state_.tracks();
}

void SMC_LMB_Tracker::set_tracks(const std::vector<Track>& tracks) {
    current_state_.set_tracks(tracks);
    // A sleep is only valid against the sensor history this tracker validated it with.
    for (Track& track : current_state_.tracks()) {
        track.set_sleep_until(-std::numeric_limits<double>::infinity());
    }
    stamp_unstamped_tracks();
}
