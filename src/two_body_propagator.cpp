#include "two_body_propagator.h"
#include "fast_random.h"
#include "validation.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

// Helper function: Compute state derivative for two-body problem
static StateVector calculate_state_derivative(const StateVector& state_6d) {
    constexpr double mu = 3.986004418e14; // Earth's gravitational parameter (m^3/s^2)
    constexpr double min_radius = 6.371e6 + 100.0e3; // Earth radius + 100 km floor

    // Fixed-size blocks: Eigen can emit three-wide loads/stores without a runtime size check.
    // Bit-identical to the former dynamic head(3) / segment(3, 3) on the reference toolchain.
    Eigen::Vector3d pos = state_6d.head<3>();
    Eigen::Vector3d vel = state_6d.tail<3>();
    double r_norm = pos.norm();
    Eigen::Vector3d radial_unit;
    double r_safe;
    if (r_norm > 1e-6) {
        radial_unit = pos / r_norm;
        r_safe = std::max(r_norm, min_radius);
    } else {
        radial_unit = Eigen::Vector3d::UnitX();
        r_safe = min_radius;
    }
    Eigen::Vector3d acc = -mu * radial_unit / (r_safe * r_safe);
    StateVector dydt;
    dydt.head<3>() = vel;
    dydt.tail<3>() = acc;
    return dydt;
}

void two_body_rk4_step(StateVector& state, double dt) {
    const StateVector k1 = calculate_state_derivative(state);
    const StateVector k2 = calculate_state_derivative(state + 0.5 * dt * k1);
    const StateVector k3 = calculate_state_derivative(state + 0.5 * dt * k2);
    const StateVector k4 = calculate_state_derivative(state + dt * k3);
    state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

namespace {

// Several states stepped side by side. One RK4 step is a long dependent chain (four square roots
// and twenty-four divisions), so a single state leaves the floating-point units mostly idle; laying
// kLanes states out as [component][lane] lets the CPU overlap their chains and the compiler pack
// lanes into vector registers. Every lane performs exactly the operations of
// calculate_state_derivative and two_body_rk4_step, in the same order, so each lane's result is
// bit-for-bit the scalar one (no FMA contraction: the build targets baseline x86-64).
constexpr int kLanes = 4;
constexpr double kLaneMu = 3.986004418e14;
constexpr double kLaneMinRadius = 6.371e6 + 100.0e3;

using LaneBlock = double[6][kLanes];

inline void derivative_lanes(const LaneBlock& s, LaneBlock& d) {
    for (int l = 0; l < kLanes; ++l) {
        const double x = s[0][l];
        const double y = s[1][l];
        const double z = s[2][l];
        const double r = std::sqrt(x * x + y * y + z * z);
        double ux = 1.0;
        double uy = 0.0;
        double uz = 0.0;
        double r_safe = kLaneMinRadius;
        if (r > 1e-6) {
            ux = x / r;
            uy = y / r;
            uz = z / r;
            r_safe = std::max(r, kLaneMinRadius);
        }
        const double r_safe_sq = r_safe * r_safe;
        d[0][l] = s[3][l];
        d[1][l] = s[4][l];
        d[2][l] = s[5][l];
        d[3][l] = (-kLaneMu * ux) / r_safe_sq;
        d[4][l] = (-kLaneMu * uy) / r_safe_sq;
        d[5][l] = (-kLaneMu * uz) / r_safe_sq;
    }
}

inline void rk4_lanes(LaneBlock& y, double dt) {
    LaneBlock k1, k2, k3, k4, probe;
    const double half = 0.5 * dt;
    derivative_lanes(y, k1);
    for (int c = 0; c < 6; ++c) {
        for (int l = 0; l < kLanes; ++l) {
            probe[c][l] = y[c][l] + half * k1[c][l];
        }
    }
    derivative_lanes(probe, k2);
    for (int c = 0; c < 6; ++c) {
        for (int l = 0; l < kLanes; ++l) {
            probe[c][l] = y[c][l] + half * k2[c][l];
        }
    }
    derivative_lanes(probe, k3);
    for (int c = 0; c < 6; ++c) {
        for (int l = 0; l < kLanes; ++l) {
            probe[c][l] = y[c][l] + dt * k3[c][l];
        }
    }
    derivative_lanes(probe, k4);
    const double sixth = dt / 6.0;
    for (int c = 0; c < 6; ++c) {
        for (int l = 0; l < kLanes; ++l) {
            y[c][l] = y[c][l] + sixth * (((k1[c][l] + 2.0 * k2[c][l]) + 2.0 * k3[c][l]) + k4[c][l]);
        }
    }
}

}  // namespace

void two_body_rk4_step_strided(double* first, size_t stride, size_t count, double dt) {
    size_t i = 0;
    LaneBlock block;
    for (; i + kLanes <= count; i += kLanes) {
        for (int l = 0; l < kLanes; ++l) {
            const double* state = first + (i + static_cast<size_t>(l)) * stride;
            for (int c = 0; c < 6; ++c) {
                block[c][l] = state[c];
            }
        }
        rk4_lanes(block, dt);
        for (int l = 0; l < kLanes; ++l) {
            double* state = first + (i + static_cast<size_t>(l)) * stride;
            for (int c = 0; c < 6; ++c) {
                state[c] = block[c][l];
            }
        }
    }
    // The last few states one at a time, through the scalar step.
    for (; i < count; ++i) {
        double* state = first + i * stride;
        Eigen::Map<StateVector> mapped(state);
        StateVector y = mapped;
        two_body_rk4_step(y, dt);
        mapped = y;
    }
}

namespace {

//! Multiple of the noise standard deviation taken as "cannot happen" by noise_displacement_bound.
//! The bound is on the norm of a 3-D Gaussian against the square root of its covariance trace, so
//! even a fully degenerate (one-direction) covariance leaves a one-sided tail of ~6e-16.
constexpr double kNoiseBoundSigmas = 8.0;

}  // namespace

TwoBodyPropagator::TwoBodyPropagator(const Eigen::MatrixXd& process_noise_covariance,
                                     std::optional<uint64_t> seed,
                                     std::optional<double> noise_reference_dt)
    : process_noise_covariance_(process_noise_covariance),
      rng_(seed.has_value() ? *seed : std::mt19937_64::result_type(std::random_device{}())),
      noise_reference_dt_(noise_reference_dt) {
    validation::require_covariance_6x6(process_noise_covariance_, "process_noise_covariance");
    if (noise_reference_dt_.has_value() &&
        !(std::isfinite(*noise_reference_dt_) && *noise_reference_dt_ > 0.0)) {
        throw std::invalid_argument("TwoBodyPropagator: noise_reference_dt must be finite and > 0, got " +
                                    std::to_string(*noise_reference_dt_));
    }

    has_process_noise_ = process_noise_covariance_.trace() > 1e-24;
    if (has_process_noise_) {
        Eigen::LLT<ProcessNoiseCov> llt(process_noise_covariance_);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("TwoBodyPropagator: process_noise_covariance is not positive definite");
        }
        noise_L_ = llt.matrixL();
        if (noise_reference_dt_.has_value()) {
            diffusion_ = process_noise_covariance_ / *noise_reference_dt_;
        }
    }
}

ProcessNoiseCov TwoBodyPropagator::step_noise_covariance(double dt) const {
    if (!noise_reference_dt_.has_value()) {
        return process_noise_covariance_;
    }
    const double h = std::abs(dt);
    const Eigen::Matrix3d d_pp = diffusion_.topLeftCorner<3, 3>();
    const Eigen::Matrix3d d_pv = diffusion_.topRightCorner<3, 3>();
    const Eigen::Matrix3d d_vp = diffusion_.bottomLeftCorner<3, 3>();
    const Eigen::Matrix3d d_vv = diffusion_.bottomRightCorner<3, 3>();

    // Integral over [0, h] of e^{Fs} D e^{F^T s} with F = [[0, I], [0, 0]], e^{Fs} = [[I, sI], [0, I]].
    ProcessNoiseCov q;
    q.topLeftCorner<3, 3>() = d_pp * h + (d_pv + d_vp) * (h * h / 2.0) + d_vv * (h * h * h / 3.0);
    q.topRightCorner<3, 3>() = d_pv * h + d_vv * (h * h / 2.0);
    q.bottomLeftCorner<3, 3>() = d_vp * h + d_vv * (h * h / 2.0);
    q.bottomRightCorner<3, 3>() = d_vv * h;
    return q;
}

const ProcessNoiseCov& TwoBodyPropagator::step_noise_factor(double dt) const {
    if (!noise_reference_dt_.has_value()) {
        return noise_L_;
    }
    if (dt != cached_dt_) {
        // Q positive definite makes the integrand, and so Qd(h), positive definite for any h > 0.
        Eigen::LLT<ProcessNoiseCov> llt(step_noise_covariance(dt));
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("TwoBodyPropagator: step noise covariance for dt = " +
                                     std::to_string(dt) + " s is not positive definite");
        }
        cached_L_ = llt.matrixL();
        cached_dt_ = dt;
    }
    return cached_L_;
}

std::optional<double> TwoBodyPropagator::noise_displacement_bound(double interval) const {
    if (!has_process_noise_) {
        return 0.0;
    }
    if (!noise_reference_dt_.has_value()) {
        // Per-call noise depends on how many calls cover the interval, not on its length.
        return std::nullopt;
    }
    const double position_variance = step_noise_covariance(interval).topLeftCorner<3, 3>().trace();
    return kNoiseBoundSigmas * std::sqrt(std::max(position_variance, 0.0));
}

Particle TwoBodyPropagator::propagate(const Particle& particle, double dt, double current_time, double noise_scale) const {
    (void)current_time;
    LMB_VALIDATION_ONLY(validation::require_state_vector(particle.state_vector, "particle.state_vector"));

    const StateVector y0 = particle.state_vector;
    // RK4 integration
    const StateVector k1 = calculate_state_derivative(y0);
    const StateVector k2 = calculate_state_derivative(y0 + 0.5 * dt * k1);
    const StateVector k3 = calculate_state_derivative(y0 + 0.5 * dt * k2);
    const StateVector k4 = calculate_state_derivative(y0 + dt * k3);
    const StateVector y1 = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    Particle propagated_particle = particle;
    propagated_particle.state_vector = y1;

    if (has_process_noise_ && noise_scale > 1e-12) {
        if (!noise_reference_dt_.has_value()) {
            // Per-call model: the historical path, kept verbatim (it is the hot loop of every
            // fixed-step driver).
            StateVector noise_vec;
            for (int i = 0; i < 6; ++i) {
                noise_vec(i) = unit_normal_(rng_);
            }
            const double std_dev_scale = std::sqrt(noise_scale);
            propagated_particle.state_vector +=
                noise_L_.triangularView<Eigen::Lower>() * noise_vec * std_dev_scale;
        } else if (dt != 0.0) {
            // Time-consistent model. A zero-length step accumulates no noise (and Qd(0) has no
            // Cholesky factor).
            add_time_consistent_noise(propagated_particle.state_vector, dt, noise_scale);
        }
    }
    return propagated_particle;
}

void TwoBodyPropagator::propagate_cloud_keyed(std::vector<Particle>& particles, double dt,
                                              double current_time, double noise_scale,
                                              uint64_t stream_key) const {
    (void)current_time;
    // The same noise guards as propagate(): no draws without process noise, at a vanishing scale,
    // or over a zero-length step of the time-consistent model (Qd(0) = 0 has no Cholesky factor).
    const bool add_noise = has_process_noise_ && noise_scale > 1e-12 &&
                           (!noise_reference_dt_.has_value() || dt != 0.0);
    ProcessNoiseCov factor = ProcessNoiseCov::Zero();
    if (add_noise) {
        if (noise_reference_dt_.has_value()) {
            factor = step_noise_factor(dt);
        } else {
            factor = noise_L_;
        }
        factor *= std::sqrt(noise_scale);
    }

#ifdef LMB_ENGINE_ENABLE_VALIDATION
    for (const Particle& particle : particles) {
        validation::require_state_vector(particle.state_vector, "particle.state_vector");
    }
#endif
    if (!particles.empty()) {
        static_assert(sizeof(Particle) % sizeof(double) == 0, "Particle must be a whole number of doubles");
        two_body_rk4_step_strided(particles.front().state_vector.data(), sizeof(Particle) / sizeof(double),
                                  particles.size(), dt);
    }
    if (!add_noise) {
        return;
    }
    for (size_t p = 0; p < particles.size(); ++p) {
        fast_random::Stream stream(fast_random::combine(stream_key, static_cast<uint64_t>(p)));
        StateVector eps;
        for (int i = 0; i < 6; ++i) {
            eps(i) = stream.normal();
        }
        particles[p].state_vector += factor.triangularView<Eigen::Lower>() * eps;
    }
}

// Out of line on purpose: keeping it out of propagate() keeps that function small enough for the
// compiler to go on inlining the four RK4 derivative evaluations, which is worth ~5% of predict().
__attribute__((noinline)) void TwoBodyPropagator::add_time_consistent_noise(StateVector& state, double dt,
                                                                         double noise_scale) const {
    StateVector noise_vec;
    for (int i = 0; i < 6; ++i) {
        noise_vec(i) = unit_normal_(rng_);
    }
    const double std_dev_scale = std::sqrt(noise_scale);
    const ProcessNoiseCov& noise_factor = step_noise_factor(dt);
    state += noise_factor.triangularView<Eigen::Lower>() * noise_vec * std_dev_scale;
}
