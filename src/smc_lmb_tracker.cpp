#include "smc_lmb_tracker.h"
#include "assignment.h"
#include "in_orbit_sensor_model.h"
#include "validation.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr double kMu = 3.986004418e14;               //!< Earth's gravitational parameter [m^3/s^2]
constexpr double kMinRadius = 6.371e6 + 100.0e3;    //!< The propagator's gravity floor [m]

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
                                                 : std::mt19937_64::result_type(std::random_device{}())) {
    validation::require_models(propagator_, sensor_model_, birth_model_);
    validation::require_k_best(k_best_);
    validation::require_clutter_intensity(clutter_intensity_);
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

void SMC_LMB_Tracker::predict(double dt) {
    ensure_models_configured();
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
                propagate_track_to(track, new_time);
            }
        }
        return;
    }

    for (Track& track : tracks) {
        // Update existence probability in-place.
        track.set_existence_probability(track.existence_probability() * survival_probability_);

        // --- Process Noise Annealing ---
        const double noise_scale = noise_scale_at(track, new_time);

        // Propagate each particle in place. Overwriting the existing cloud avoids allocating a
        // second vector and the set_particles copy that used to follow it. Element-wise assignment
        // does not reallocate, so external aliases of the storage remain address-stable; values
        // change, as they would under any in-place rewrite.
        std::vector<Particle>& particles = track.mutable_particles();
        for (Particle& particle : particles) {
            particle = propagator_->propagate(particle, dt, previous_time, noise_scale);
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

void SMC_LMB_Tracker::stamp_unstamped_tracks() {
    const double now = current_state_.timestamp();
    for (Track& track : current_state_.tracks()) {
        if (std::isnan(track.propagated_time())) {
            track.set_propagated_time(now);
        }
    }
}

void SMC_LMB_Tracker::propagate_track_to(Track& track, double target_time) {
    const double start = track.propagated_time();
    const double span = target_time - start;
    if (span == 0.0) {
        return;
    }
    const int substeps = std::max(1, static_cast<int>(std::ceil(std::abs(span) / max_pending_ - 1e-9)));
    const double step = span / static_cast<double>(substeps);

    std::vector<Particle>& particles = track.mutable_particles();
    for (int k = 0; k < substeps; ++k) {
        const double step_start = start + static_cast<double>(k) * step;
        const double step_end = (k + 1 == substeps) ? target_time : start + static_cast<double>(k + 1) * step;
        const double noise_scale = noise_scale_at(track, step_end);
        const double dt = step_end - step_start;
        for (Particle& particle : particles) {
            particle = propagator_->propagate(particle, dt, step_start, noise_scale);
        }
    }
    track.set_propagated_time(target_time);
    track.set_cloud_bound(compute_cloud_bound(track.particles()));
}

bool SMC_LMB_Tracker::may_be_observable(Track& track, const sensor::SensorArray& sensors, double now) const {
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
    return sensors.any_sensor_within_reach(predicted, margin);
}

void SMC_LMB_Tracker::refresh_observable_tracks(const sensor::SensorArray& sensors) {
    const double now = current_state_.timestamp();
    for (Track& track : current_state_.tracks()) {
        if (std::isnan(track.propagated_time())) {
            track.set_propagated_time(now);
            continue;
        }
        if (track.propagated_time() < now && may_be_observable(track, sensors, now)) {
            propagate_track_to(track, now);
        }
    }
}

void SMC_LMB_Tracker::synchronize() {
    ensure_models_configured();
    const double now = current_state_.timestamp();
    for (Track& track : current_state_.tracks()) {
        if (std::isnan(track.propagated_time())) {
            track.set_propagated_time(now);
        } else if (track.propagated_time() != now) {
            propagate_track_to(track, now);
        }
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
    const std::vector<Track>& tracks = current_state_.tracks();
    const double now = current_state_.timestamp();
    const size_t num_tracks = tracks.size();

    active_tracks_.clear();
    coverage_per_sensor_.clear();
    coverage_union_.clear();
    particle_sensor_.clear();
    track_inv_weight_sum_.clear();
    track_particle_offsets_.assign(1, 0);

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
            if (!(union_fraction > 0.0)) {
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

    // Mixture, one pass per used association. Measurements ascending, then the miss term, so the
    // summation order is fixed run to run.
    mixture_weights_.assign(num_particles, 0.0);
    for (size_t j = 0; j < num_meas; ++j) {
        if (!det_used[j]) {
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

    double sum_weights = 0.0;
    for (size_t p = 0; p < num_particles; ++p) {
        sum_weights += mixture_weights_[p];
    }

    if (sum_weights > 0.0 && std::isfinite(sum_weights)) {
        const double inv_sum = 1.0 / sum_weights;
        for (size_t p = 0; p < num_particles; ++p) {
            mixture_weights_[p] *= inv_sum;
        }
    } else {
        // No information to go on. Equal slots make the systematic walk keep every particle
        // exactly once, which is the sensible reading of that.
        const double uniform_weight = 1.0 / static_cast<double>(num_particles);
        for (size_t p = 0; p < num_particles; ++p) {
            mixture_weights_[p] = uniform_weight;
        }
    }

    double ess = 0.0;
    if (regularization_ || record_diagnostics_) {
        double sum_sq = 0.0;
        for (size_t p = 0; p < num_particles; ++p) {
            sum_sq += mixture_weights_[p] * mixture_weights_[p];
        }
        ess = sum_sq > 0.0 ? 1.0 / sum_sq : 0.0;
    }
    const bool regularize_now = regularization_ && num_particles > 1 &&
                                ess < regularization_ess_threshold_ * static_cast<double>(num_particles);

    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    std::vector<Particle> resampled_particles;
    resampled_particles.reserve(num_particles);

    const double u = unit_dist(resample_rng_) / static_cast<double>(num_particles);
    double cumsum = 0.0;
    size_t idx = 0;

    for (size_t p = 0; p < num_particles; ++p) {
        const double threshold = u + static_cast<double>(p) / static_cast<double>(num_particles);

        while (cumsum < threshold && idx < num_particles) {
            cumsum += mixture_weights_[idx];
            ++idx;
        }

        // The position in mixture_weights_ is the particle index, so no side table is needed.
        const size_t chosen_index = (idx > 0) ? idx - 1 : 0;

        Particle resampled;
        resampled.state_vector = predicted_particles[chosen_index].state_vector;
        resampled.weight = 1.0 / static_cast<double>(num_particles);
        resampled_particles.push_back(resampled);
    }

    if (regularize_now) {
        regularize(resampled_particles, predicted_particles, mixture_weights_);
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
        diagnostics_.push_back(record);
    }

    // Move the resampled cloud into the track. This reallocates particles_ and invalidates
    // any zero-copy NumPy views that still alias the previous storage.
    track.set_particles(std::move(resampled_particles));
}

void SMC_LMB_Tracker::update_impl(const std::vector<Measurement>& measurements,
                                  const sensor::SensorArray* sensors) {
    ensure_models_configured();

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
        std::vector<Track> born_tracks = birth_model_->generate_new_tracks(unused, now);
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
            size_t visible_count = 0;
            for (size_t p = 0; p < num_particles; ++p) {
                if (!particle_visible_to(a, p, sensor_index)) {
                    continue;
                }
                ++visible_count;
                const auto& current_particle = current_particles[p];
                const double particle_likelihood = sensor_model_->calculate_likelihood(
                    current_particle, measurement, meas_caches[j]);
                const double updated_weight = current_particle.weight * inv_weight_sum * particle_likelihood;
                assoc[p] = updated_weight;
                total_likelihood += updated_weight;
            }

            // Raw L, with neither P_D nor kappa: Step 3 applies each exactly once, and nothing else
            // multiplies by them.
            likelihood_matrix(a, j) = total_likelihood;

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

    // Step 5: marginals, posterior existence, mixture and resampling.
    //
    // The hypotheses are grouped by the association they give each track: every hypothesis that
    // assigns track a to measurement j contributes the same normalised per-particle vector, and
    // every miss column the same miss density, so at most num_meas + 1 per-particle passes are
    // needed however large k_best_ is.
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
    stamp_unstamped_tracks();
}
