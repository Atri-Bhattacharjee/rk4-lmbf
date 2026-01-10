#include "two_body_propagator.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

// Helper function: Compute state derivative for two-body problem
Eigen::VectorXd calculate_state_derivative(const Eigen::VectorXd& state_6d) {
    constexpr double mu = 3.986004418e14; // Earth's gravitational parameter (m^3/s^2)
    Eigen::Vector3d pos = state_6d.head(3);
    Eigen::Vector3d vel = state_6d.segment(3, 3);
    double r_norm = pos.norm();
    Eigen::Vector3d acc = -mu * pos / std::pow(r_norm, 3);
    Eigen::VectorXd dydt(6);
    dydt.head(3) = vel;
    dydt.tail(3) = acc;
    return dydt;
}

TwoBodyPropagator::TwoBodyPropagator(const Eigen::MatrixXd& process_noise_covariance)
    : process_noise_covariance_(process_noise_covariance) {}

Particle TwoBodyPropagator::propagate(const Particle& particle, double dt, double current_time, double noise_scale) const {
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
    if (process_noise_covariance_.rows() == 6 && process_noise_covariance_.cols() == 6 && noise_scale > 1e-12) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::normal_distribution<> dist(0.0, 1.0);
        Eigen::VectorXd noise_vec(6);
        for (int i = 0; i < 6; ++i) noise_vec(i) = dist(gen);
        Eigen::LLT<Eigen::MatrixXd> llt(process_noise_covariance_);
        Eigen::MatrixXd L = llt.matrixL();
        // Scale standard deviation by sqrt(noise_scale) since L represents std dev
        double std_dev_scale = std::sqrt(noise_scale);
        propagated_particle.state_vector += L * noise_vec * std_dev_scale;
    }
    return propagated_particle;
}
