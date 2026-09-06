#include "in_orbit_sensor_model.h"
#include "validation.h"
#include <cmath>
#include <stdexcept>

namespace {

constexpr int kDim = validation::MEAS_DIM;

bool isDiagonalCovariance(const MeasCovariance& cov, double tolerance = 1e-12) {
    for (int row = 0; row < kDim; ++row) {
        for (int col = 0; col < kDim; ++col) {
            if (row != col && std::abs(cov(row, col)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

InOrbitSensorModel::InOrbitSensorModel()
    : InOrbitSensorModel(50.0 * 50.0, 1.0 * 1.0, 1.0e-8, 1.0e-8, 1.0e-10, 1.0e-10) {
}

InOrbitSensorModel::InOrbitSensorModel(double range_var, double range_rate_var,
                                       double angle_var_1, double angle_var_2,
                                       double angle_rate_var_1, double angle_rate_var_2)
    : range_var_(range_var),
      range_rate_var_(range_rate_var),
      angle_var_1_(angle_var_1),
      angle_var_2_(angle_var_2),
      angle_rate_var_1_(angle_rate_var_1),
      angle_rate_var_2_(angle_rate_var_2) {
    const double variances[kDim] = {range_var_, range_rate_var_, angle_var_1_,
                                    angle_var_2_, angle_rate_var_1_, angle_rate_var_2_};
    for (double variance : variances) {
        if (!(std::isfinite(variance) && variance > 0.0)) {
            throw std::invalid_argument("InOrbitSensorModel: all six variances must be finite and > 0");
        }
    }
}

MeasCovariance InOrbitSensorModel::defaultCovariance() const {
    MeasCovariance covariance = MeasCovariance::Zero();
    covariance.diagonal() << range_var_, range_rate_var_, angle_var_1_,
                             angle_var_2_, angle_rate_var_1_, angle_rate_var_2_;
    return covariance;
}

los::LosObservation InOrbitSensorModel::predictObservation(const Particle& particle,
                                                           const StateVector& sensor_state) const {
    return los::observe(particle.state_vector, sensor_state);
}

MeasurementLikelihoodCache InOrbitSensorModel::buildCache(const Measurement& measurement) {
    MeasurementLikelihoodCache cache;
    const MeasCovariance& cov = measurement.covariance_;

    if (isDiagonalCovariance(cov)) {
        cache.is_diagonal = true;
        cache.inv_var = cov.diagonal().cwiseInverse();
        cache.log_norm_factor = -0.5 * kDim * std::log(2.0 * los::kPi);
        for (int i = 0; i < kDim; ++i) {
            cache.log_norm_factor -= 0.5 * std::log(cov(i, i));
        }
    } else {
        // Physical units differ by many orders of magnitude (m^2 vs (rad/s)^2), so factor the
        // correlation matrix D^-1 R D^-1 (D = sqrt(diag R)) instead of R itself: LLT on a
        // well-scaled matrix, then undo the scaling analytically.
        cache.is_diagonal = false;
        const LocalMeasVector scale = cov.diagonal().cwiseSqrt();
        const LocalMeasVector inv_scale = scale.cwiseInverse();
        const MeasCovariance correlation = inv_scale.asDiagonal() * cov * inv_scale.asDiagonal();
        Eigen::LLT<MeasCovariance> llt(correlation);
        if (llt.info() != Eigen::Success) {
            throw std::invalid_argument("Measurement.covariance_: must be positive definite");
        }
        const MeasCovariance correlation_inverse = llt.solve(MeasCovariance::Identity());
        cache.cov_inv = inv_scale.asDiagonal() * correlation_inverse * inv_scale.asDiagonal();
        double log_det = 0.0;
        const MeasCovariance cholesky = llt.matrixL();
        for (int i = 0; i < kDim; ++i) {
            log_det += 2.0 * std::log(cholesky(i, i)) + 2.0 * std::log(scale(i));
        }
        cache.log_norm_factor = -0.5 * kDim * std::log(2.0 * los::kPi) - 0.5 * log_det;
    }

    return cache;
}

double InOrbitSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    return calculate_likelihood(particle, measurement, buildCache(measurement));
}

double InOrbitSensorModel::calculate_likelihood(const Particle& particle,
                                                const Measurement& measurement,
                                                const MeasurementLikelihoodCache& cache) const {
    LMB_VALIDATION_ONLY(validation::require_state_vector(particle.state_vector, "particle.state_vector"));

    const los::LosObservation measured = measurement.observation();
    los::LosObservation predicted = predictObservation(particle, measurement.sensor_state_);
    if (predicted.range <= validation::RANGE_EPSILON) {
        // Direction undefined at the sensor: attribute no angular information to the particle.
        predicted.los = measured.los;
        predicted.los_rate = Eigen::Vector3d::Zero();
    }

    const LocalMeasVector residual = los::localResidual(measured, predicted);

    double mahalanobis_sq = 0.0;
    if (cache.is_diagonal) {
        mahalanobis_sq = residual.cwiseProduct(cache.inv_var).cwiseProduct(residual).sum();
    } else {
        mahalanobis_sq = residual.transpose() * cache.cov_inv * residual;
    }

    return std::exp(cache.log_norm_factor - 0.5 * mahalanobis_sq);
}
