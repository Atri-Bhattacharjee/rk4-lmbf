#pragma once

#include "datatypes.h"
#include "models.h"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(LMB_ENGINE_ENABLE_VALIDATION) || !defined(NDEBUG)
#define LMB_VALIDATION_ONLY(expr) (expr)
#define LMB_VALIDATION_ENABLED 1
#else
#define LMB_VALIDATION_ONLY(expr) ((void)0)
#define LMB_VALIDATION_ENABLED 0
#endif

namespace validation {

constexpr int STATE_DIM = 6;
constexpr int MEAS_DIM = 6;
constexpr double RANGE_EPSILON = los::kRangeEpsilon;
//! Relative tolerance on |los_| - 1 and on the tangency of los_rate_.
constexpr double LOS_TOLERANCE = 1e-6;
//! Relative tolerance on |R - R^T|_max / |R|_max for covariance symmetry.
constexpr double SYMMETRY_TOLERANCE = 1e-9;

template <typename Derived>
inline bool all_finite(const Eigen::MatrixBase<Derived>& matrix) {
    return matrix.allFinite();
}

template <typename Derived>
inline void require_symmetric(const Eigen::MatrixBase<Derived>& covariance, const char* context) {
    const double scale = covariance.cwiseAbs().maxCoeff();
    const double asymmetry = (covariance - covariance.transpose()).cwiseAbs().maxCoeff();
    if (!(asymmetry <= SYMMETRY_TOLERANCE * scale)) {
        throw std::invalid_argument(
            std::string(context) + ": covariance must be symmetric (|R - R^T|_max = " +
            std::to_string(asymmetry) + ", |R|_max = " + std::to_string(scale) + ")");
    }
}

inline void require_state_vector(const StateVector& vector, const char* context) {
    if (vector.size() != STATE_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": state vector must have size " +
            std::to_string(STATE_DIM) + ", got " + std::to_string(vector.size()));
    }
}

inline void require_state_vector(const Eigen::VectorXd& vector, const char* context) {
    if (vector.size() != STATE_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": state vector must have size " +
            std::to_string(STATE_DIM) + ", got " + std::to_string(vector.size()));
    }
}

inline void require_covariance_6x6(const Eigen::MatrixXd& covariance, const char* context) {
    if (covariance.rows() != STATE_DIM || covariance.cols() != STATE_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": covariance must be 6x6, got " +
            std::to_string(covariance.rows()) + "x" + std::to_string(covariance.cols()));
    }
}

inline void require_covariance_6x6(const Eigen::Matrix<double, 6, 6>& covariance, const char* context) {
    if (covariance.rows() != STATE_DIM || covariance.cols() != STATE_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": covariance must be 6x6, got " +
            std::to_string(covariance.rows()) + "x" + std::to_string(covariance.cols()));
    }
}

/**
 * @brief Require a finite, symmetric, positive-definite 6x6 covariance.
 * @param context Names the quantity and its frame in messages, e.g. "birth_covariance (local tangent frame)".
 */
template <typename Derived>
inline void require_covariance_6x6_positive_definite(const Eigen::MatrixBase<Derived>& covariance,
                                                     const char* context) {
    if (covariance.rows() != STATE_DIM || covariance.cols() != STATE_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": covariance must be 6x6, got " +
            std::to_string(covariance.rows()) + "x" + std::to_string(covariance.cols()));
    }
    if (!all_finite(covariance)) {
        throw std::invalid_argument(std::string(context) + ": covariance must be finite");
    }
    require_symmetric(covariance, context);
    Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(covariance);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument(
            std::string(context) + ": covariance matrix must be positive definite");
    }
}

/**
 * @brief Validate every field of a Measurement; messages name the offending field.
 *
 * Always: range_ finite and > 0; range_rate_, los_rate_, sensor_state_, timestamp_ finite;
 * ||los_| - 1| < 1e-6; |los_ . los_rate_| < 1e-6 max(1, |los_rate_|); covariance_ finite and symmetric.
 * Debug/validation builds additionally require covariance_ to be positive definite (LLT).
 */
inline void require_measurement(const Measurement& measurement) {
    if (!std::isfinite(measurement.timestamp_)) {
        throw std::invalid_argument("Measurement.timestamp_: must be finite");
    }
    if (!(std::isfinite(measurement.range_) && measurement.range_ > 0.0)) {
        throw std::invalid_argument(
            "Measurement.range_: must be finite and > 0, got " + std::to_string(measurement.range_));
    }
    if (!std::isfinite(measurement.range_rate_)) {
        throw std::invalid_argument("Measurement.range_rate_: must be finite");
    }
    if (!all_finite(measurement.los_)) {
        throw std::invalid_argument("Measurement.los_: must be finite");
    }
    const double los_norm_error = std::abs(measurement.los_.norm() - 1.0);
    if (!(los_norm_error < LOS_TOLERANCE)) {
        throw std::invalid_argument(
            "Measurement.los_: must be a unit vector (||los_| - 1| = " + std::to_string(los_norm_error) + ")");
    }
    if (!all_finite(measurement.los_rate_)) {
        throw std::invalid_argument("Measurement.los_rate_: must be finite");
    }
    const double tangency_error = std::abs(measurement.los_.dot(measurement.los_rate_));
    if (!(tangency_error < LOS_TOLERANCE * std::max(1.0, measurement.los_rate_.norm()))) {
        throw std::invalid_argument(
            "Measurement.los_rate_: must be orthogonal to los_ (|los_ . los_rate_| = " +
            std::to_string(tangency_error) + ")");
    }
    if (measurement.covariance_.rows() != MEAS_DIM || measurement.covariance_.cols() != MEAS_DIM) {
        throw std::invalid_argument(
            "Measurement.covariance_: expected " + std::to_string(MEAS_DIM) + "x" + std::to_string(MEAS_DIM) +
            " matrix, got " + std::to_string(measurement.covariance_.rows()) + "x" +
            std::to_string(measurement.covariance_.cols()));
    }
    if (!all_finite(measurement.covariance_)) {
        throw std::invalid_argument("Measurement.covariance_: must be finite");
    }
    require_symmetric(measurement.covariance_, "Measurement.covariance_");
    if (!(measurement.covariance_.diagonal().minCoeff() > 0.0)) {
        throw std::invalid_argument("Measurement.covariance_: all variances (diagonal) must be > 0");
    }
#if LMB_VALIDATION_ENABLED
    Eigen::LLT<MeasCovariance> llt(measurement.covariance_);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument("Measurement.covariance_: must be positive definite");
    }
#endif
    require_state_vector(measurement.sensor_state_, "Measurement.sensor_state_");
    if (!all_finite(measurement.sensor_state_)) {
        throw std::invalid_argument("Measurement.sensor_state_: must be finite");
    }
}

inline void require_particles_per_track(int particles_per_track) {
    if (particles_per_track <= 0) {
        throw std::invalid_argument(
            "particles_per_track must be positive, got " + std::to_string(particles_per_track));
    }
}

inline void require_k_best(int k_best) {
    if (k_best <= 0) {
        throw std::invalid_argument(
            "k_best must be positive, got " + std::to_string(k_best));
    }
}

inline void require_clutter_intensity(double clutter_intensity) {
    if (clutter_intensity <= 0.0) {
        throw std::invalid_argument(
            "clutter_intensity must be positive, got " + std::to_string(clutter_intensity));
    }
}

inline void require_models(const std::shared_ptr<IOrbitPropagator>& propagator,
                           const std::shared_ptr<ISensorModel>& sensor_model,
                           const std::shared_ptr<IBirthModel>& birth_model) {
    if (!propagator) {
        throw std::invalid_argument("propagator must not be null");
    }
    if (!sensor_model) {
        throw std::invalid_argument("sensor_model must not be null");
    }
    if (!birth_model) {
        throw std::invalid_argument("birth_model must not be null");
    }
}

}  // namespace validation
