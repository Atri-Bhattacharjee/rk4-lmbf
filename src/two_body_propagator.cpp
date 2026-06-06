#include "two_body_propagator.h"
#include "validation.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <stdexcept>

// Helper function: Compute state derivative for two-body problem
static StateVector calculate_state_derivative(const StateVector& state_6d) {
    constexpr double mu = 3.986004418e14; // Earth's gravitational parameter (m^3/s^2)
    constexpr double min_radius = 6.371e6 + 100.0e3; // Earth radius + 100 km floor

    Eigen::Vector3d pos = state_6d.head(3);
    Eigen::Vector3d vel = state_6d.segment(3, 3);
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
    dydt.head(3) = vel;
    dydt.tail(3) = acc;
    return dydt;
}

TwoBodyPropagator::TwoBodyPropagator(const Eigen::MatrixXd& process_noise_covariance)
    : process_noise_covariance_(process_noise_covariance) {
    validation::require_covariance_6x6(process_noise_covariance_, "process_noise_covariance");

    has_process_noise_ = process_noise_covariance_.trace() > 1e-24;
    if (has_process_noise_) {
        Eigen::LLT<ProcessNoiseCov> llt(process_noise_covariance_);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("TwoBodyPropagator: process_noise_covariance is not positive definite");
        }
        noise_L_ = llt.matrixL();
    }
}

Particle TwoBodyPropagator::propagate(const Particle& particle, double dt, double current_time, double noise_scale) const {
    (void)current_time;
    validation::require_state_vector(particle.state_vector, "particle.state_vector");

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
        static thread_local std::mt19937 gen(std::random_device{}());
        std::normal_distribution<> dist(0.0, 1.0);
        StateVector noise_vec;
        for (int i = 0; i < 6; ++i) {
            noise_vec(i) = dist(gen);
        }
        const double std_dev_scale = std::sqrt(noise_scale);
        propagated_particle.state_vector += noise_L_ * noise_vec * std_dev_scale;
    }
    return propagated_particle;
}
