#pragma once

#include "models.h"
#include "datatypes.h"
#include "los_geometry.h"
#include <Eigen/Dense>

/**
 * @brief Sensor model for an in-orbit observer measuring range, range rate, line-of-sight
 *        direction and line-of-sight angular rate.
 *
 * The likelihood is a 6-D Gaussian in the local tangent frame of the measured direction
 * (see Measurement for the frame and ordering). The residual between a measurement and a
 * particle's predicted observation is formed with los::localResidual, which uses the sphere
 * logarithmic map for the direction and parallel transport for the angular rate, so there is
 * no angle wrapping and no azimuth/elevation singularity.
 *
 * Measurement.covariance_ is authoritative for the likelihood. The six variances held by this
 * model define defaultCovariance(), a convenience for building measurement covariances in the
 * same frame and ordering.
 */
class InOrbitSensorModel : public ISensorModel {
public:
    //! Default variances: 50 m, 1 m/s, 1e-4 rad (both angles), 1e-5 rad/s (both angular rates).
    InOrbitSensorModel();

    /**
     * @param range_var        Range variance [m^2]
     * @param range_rate_var   Range-rate variance [(m/s)^2]
     * @param angle_var_1      Direction variance along tangent basis vector e1 [rad^2]
     * @param angle_var_2      Direction variance along tangent basis vector e2 [rad^2]
     * @param angle_rate_var_1 Angular-rate variance along e1 [(rad/s)^2]
     * @param angle_rate_var_2 Angular-rate variance along e2 [(rad/s)^2]
     */
    InOrbitSensorModel(double range_var, double range_rate_var,
                       double angle_var_1, double angle_var_2,
                       double angle_rate_var_1, double angle_rate_var_2);

    double calculate_likelihood(const Particle& particle, const Measurement& measurement) const override;
    double calculate_likelihood(const Particle& particle,
                                const Measurement& measurement,
                                const MeasurementLikelihoodCache& cache) const override;

    static MeasurementLikelihoodCache buildCache(const Measurement& measurement);

    /**
     * @brief Noise-free observation of a particle from the given sensor state.
     *
     * If the particle coincides with the sensor (range <= RANGE_EPSILON) the direction is
     * undefined; the caller (calculate_likelihood) substitutes the measured direction with zero rate.
     */
    los::LosObservation predictObservation(const Particle& particle, const StateVector& sensor_state) const;

    //! diag(range_var, range_rate_var, angle_var_1, angle_var_2, angle_rate_var_1, angle_rate_var_2)
    MeasCovariance defaultCovariance() const;

private:
    double range_var_;
    double range_rate_var_;
    double angle_var_1_;
    double angle_var_2_;
    double angle_rate_var_1_;
    double angle_rate_var_2_;
};
