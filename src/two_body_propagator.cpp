#include "two_body_propagator.h"
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
