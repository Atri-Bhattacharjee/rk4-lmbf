#pragma once

#include "datatypes.h"
#include "models.h"
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <string>

namespace validation {

constexpr int STATE_DIM = 6;
constexpr int MEAS_DIM = 4;
constexpr double RANGE_EPSILON = 1e-10;

inline void require_state_vector(const Eigen::VectorXd& vector, const char* context) {
    if (vector.size() != STATE_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": state vector must have size " +
            std::to_string(STATE_DIM) + ", got " + std::to_string(vector.size()));
    }
}

inline void require_measurement(const Measurement& measurement) {
    if (measurement.value_.size() != MEAS_DIM) {
        throw std::invalid_argument(
            "Measurement.value_: expected size " + std::to_string(MEAS_DIM) +
            ", got " + std::to_string(measurement.value_.size()));
    }
    if (measurement.covariance_.rows() != MEAS_DIM || measurement.covariance_.cols() != MEAS_DIM) {
        throw std::invalid_argument("Measurement.covariance_: expected 4x4 matrix");
    }
    require_state_vector(measurement.sensor_state_, "Measurement.sensor_state");
}

inline void require_covariance_6x6(const Eigen::MatrixXd& covariance, const char* context) {
    if (covariance.rows() != STATE_DIM || covariance.cols() != STATE_DIM) {
        throw std::invalid_argument(
            std::string(context) + ": covariance must be 6x6, got " +
            std::to_string(covariance.rows()) + "x" + std::to_string(covariance.cols()));
    }
}

inline void require_positive_definite(const Eigen::MatrixXd& covariance, const char* context) {
    require_covariance_6x6(covariance, context);
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument(
            std::string(context) + ": covariance matrix must be positive definite");
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
