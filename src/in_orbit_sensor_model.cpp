#include "in_orbit_sensor_model.h"
#include <cmath>

InOrbitSensorModel::InOrbitSensorModel()
    : range_var_(50.0 * 50.0),           // 50m standard deviation
      range_rate_var_(1.0 * 1.0),        // 1m/s standard deviation
      azimuth_var_(0.0),                 // No azimuth error initially
      elevation_var_(0.0) {              // No elevation error initially
}

InOrbitSensorModel::InOrbitSensorModel(double range_var, double range_rate_var, 
                                       double azimuth_var, double elevation_var)
    : range_var_(range_var),
      range_rate_var_(range_rate_var),
      azimuth_var_(azimuth_var),
      elevation_var_(elevation_var) {
}

Eigen::VectorXd InOrbitSensorModel::convertParticleToMeasurement(
    const Particle& particle, const Eigen::VectorXd& sensor_state) const {
    return Measurement::cartesianToMeasurement(particle.state_vector, sensor_state);
}

double InOrbitSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    // Check dimensions
    if (measurement.value_.size() != 4) {
        return 0.0;
    }
    if (measurement.covariance_.rows() != 4 || measurement.covariance_.cols() != 4) {
        return 0.0;
    }
    
    // Convert particle state to measurement space
    Eigen::VectorXd particle_meas = convertParticleToMeasurement(particle, measurement.sensor_state_);
    Eigen::VectorXd meas_value = measurement.value_;
    Eigen::MatrixXd cov = measurement.covariance_;
    
    // Residual
    Eigen::VectorXd residual = meas_value - particle_meas;
    
    constexpr double M_PI = 3.14159265358979323846;
    // Normalize angle differences for azimuth and elevation to handle wrap-around
    while (residual(2) > M_PI) residual(2) -= 2.0 * M_PI;
    while (residual(2) < -M_PI) residual(2) += 2.0 * M_PI;
    while (residual(3) > M_PI) residual(3) -= 2.0 * M_PI;
    while (residual(3) < -M_PI) residual(3) += 2.0 * M_PI;
    
    // Mahalanobis distance squared
    double mahalanobis_sq = residual.transpose() * cov.inverse() * residual;
    
    // Normalization factor for 4D Gaussian

    double norm_factor = std::pow(2.0 * M_PI, -2.0) * std::pow(cov.determinant(), -0.5);
    
    return norm_factor * std::exp(-0.5 * mahalanobis_sq);
}
