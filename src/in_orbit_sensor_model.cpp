#include "in_orbit_sensor_model.h"
#include "validation.h"
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;

void wrapAngleResidual(MeasVector& residual) {
    while (residual(2) > kPi) residual(2) -= 2.0 * kPi;
    while (residual(2) < -kPi) residual(2) += 2.0 * kPi;
    while (residual(3) > kPi) residual(3) -= 2.0 * kPi;
    while (residual(3) < -kPi) residual(3) += 2.0 * kPi;
}

bool isDiagonalCovariance(const MeasCovariance& cov, double tolerance = 1e-12) {
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (row != col && std::abs(cov(row, col)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

InOrbitSensorModel::InOrbitSensorModel()
    : range_var_(50.0 * 50.0),           // 50m standard deviation
      range_rate_var_(1.0 * 1.0),        // 1m/s standard deviation
      azimuth_var_(1.0e-9),              // Small epsilon for numerical stability during inversion
      elevation_var_(1.0e-9) {           // Small epsilon for numerical stability during inversion
}

InOrbitSensorModel::InOrbitSensorModel(double range_var, double range_rate_var, 
                                       double azimuth_var, double elevation_var)
    : range_var_(range_var),
      range_rate_var_(range_rate_var),
      azimuth_var_(azimuth_var),
      elevation_var_(elevation_var) {
}

MeasVector InOrbitSensorModel::convertParticleToMeasurement(
    const Particle& particle, const StateVector& sensor_state) const {
    return Measurement::cartesianToMeasurement(particle.state_vector, sensor_state);
}

MeasurementLikelihoodCache InOrbitSensorModel::buildCache(const Measurement& measurement) {
    MeasurementLikelihoodCache cache;
    const MeasCovariance& cov = measurement.covariance_;
    constexpr int dim = 4;

    if (isDiagonalCovariance(cov)) {
        cache.is_diagonal = true;
        cache.inv_var = cov.diagonal().cwiseInverse();
        cache.log_norm_factor = -0.5 * dim * std::log(2.0 * kPi);
        for (int i = 0; i < dim; ++i) {
            cache.log_norm_factor -= 0.5 * std::log(cov(i, i));
        }
    } else {
        cache.is_diagonal = false;
        cache.cov_inv = cov.inverse();
        cache.log_norm_factor = -0.5 * dim * std::log(2.0 * kPi) - 0.5 * std::log(cov.determinant());
    }

    return cache;
}

double InOrbitSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    return calculate_likelihood(particle, measurement, buildCache(measurement));
}

double InOrbitSensorModel::calculate_likelihood(const Particle& particle,
                                                const Measurement& measurement,
                                                const MeasurementLikelihoodCache& cache) const {
    validation::require_state_vector(particle.state_vector, "particle.state_vector");
    validation::require_measurement(measurement);

    const MeasVector particle_meas = convertParticleToMeasurement(particle, measurement.sensor_state_);
    MeasVector residual = measurement.value_ - particle_meas;
    wrapAngleResidual(residual);

    double mahalanobis_sq = 0.0;
    if (cache.is_diagonal) {
        mahalanobis_sq = residual.cwiseProduct(cache.inv_var).cwiseProduct(residual).sum();
    } else {
        mahalanobis_sq = residual.transpose() * cache.cov_inv * residual;
    }

    return std::exp(cache.log_norm_factor - 0.5 * mahalanobis_sq);
}
