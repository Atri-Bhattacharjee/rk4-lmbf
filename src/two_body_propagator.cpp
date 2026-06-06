#include "two_body_propagator.h"
#include "validation.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <stdexcept>

// Helper function: Compute state derivative for two-body problem
Eigen::VectorXd calculate_state_derivative(const Eigen::VectorXd& state_6d) {
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
    Eigen::VectorXd dydt(6);
    dydt.head(3) = vel;
    dydt.tail(3) = acc;
    return dydt;
}

TwoBodyPropagator::TwoBodyPropagator(const Eigen::MatrixXd& process_noise_covariance)
    : process_noise_covariance_(process_noise_covariance) {
    validation::require_covariance_6x6(process_noise_covariance_, "process_noise_covariance");
}

Particle TwoBodyPropagator::propagate(const Particle& particle, double dt, double current_time, double noise_scale) const {
    validation::require_state_vector(particle.state_vector, "particle.state_vector");

    // Extract initial 6D state
    Eigen::VectorXd y0 = particle.state_vector.head(6);
    // RK4 integration
    Eigen::VectorXd k1 = calculate_state_derivative(y0);
    Eigen::VectorXd k2 = calculate_state_derivative(y0 + 0.5 * dt * k1);
    Eigen::VectorXd k3 = calculate_state_derivative(y0 + 0.5 * dt * k2);
    Eigen::VectorXd k4 = calculate_state_derivative(y0 + dt * k3);
    Eigen::VectorXd y1 = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    // Create propagated particle
    Particle propagated_particle = particle;
    propagated_particle.state_vector.head(6) = y1;
    // Add process noise with adaptive scaling
    // noise_scale affects variance (Q), so we apply sqrt(noise_scale) to std dev (L)
    const bool has_process_noise = process_noise_covariance_.trace() > 1e-24;
    if (has_process_noise && noise_scale > 1e-12) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::normal_distribution<> dist(0.0, 1.0);
        Eigen::VectorXd noise_vec(6);
        for (int i = 0; i < 6; ++i) noise_vec(i) = dist(gen);
        Eigen::LLT<Eigen::MatrixXd> llt(process_noise_covariance_);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("TwoBodyPropagator: process_noise_covariance is not positive definite");
        }
        Eigen::MatrixXd L = llt.matrixL();
        // Scale standard deviation by sqrt(noise_scale) since L represents std dev
        double std_dev_scale = std::sqrt(noise_scale);
        propagated_particle.state_vector += L * noise_vec * std_dev_scale;
    }
    return propagated_particle;
}
