#pragma once

/**
 * @file adaptive_birth_model.h
 * @brief Measurement-driven track birth by sampling in the sensor's local tangent frame.
 *
 * Each unassociated measurement fully determines a 6D ECI state
 * (r = r_s + rho u, v = v_s + rhodot u + rho udot). New-track particles are drawn by adding
 * Gaussian noise to the measurement in its own local tangent frame
 * [d_range, d_range_rate, d_theta1, d_theta2, d_omega1, d_omega2] and mapping each sample
 * exactly to ECI. The spread of the birth cloud is therefore range-wise, angle-wise and
 * rate-wise, as the sensor noise is, and never expressed in ECI x/y/z.
 */

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <random>

/**
 * @brief Birth model: one track per unused measurement, particles ~ measurement (+) N(0, R_birth)
 *        in the local tangent frame of the measured direction.
 *
 * Frame and ordering of R_birth (6x6, symmetric positive definite):
 *   index 0  d_range       [m]
 *   index 1  d_range_rate  [m/s]
 *   index 2  d_theta1      [rad]    direction offset along e1 = tangentBasis(los)[:,0]
 *   index 3  d_theta2      [rad]    direction offset along e2 = tangentBasis(los)[:,1]
 *   index 4  d_omega1      [rad/s]  angular-rate offset along e1
 *   index 5  d_omega2      [rad/s]  angular-rate offset along e2
 *
 * Sampling: xi ~ N(0, I6), eps = L xi with L L^T = R_birth, obs_i = perturbed(measurement, eps),
 * state_i = toCartesian(obs_i, sensor_state), weight 1/N. A sample whose range would be
 * <= RANGE_EPSILON is redrawn (at most 100 times, then the range is clamped to RANGE_EPSILON).
 */
class AdaptiveBirthModel : public IBirthModel {
private:
    int particles_per_track_;                    //!< Number of particles per new track
    double initial_existence_probability_;       //!< Existence probability assigned to a new track
    MeasCovariance birth_covariance_local_;      //!< R_birth in the local tangent frame (see class docs)
    MeasCovariance birth_noise_L_;               //!< Lower Cholesky factor of R_birth
    mutable std::mt19937_64 rng_;                //!< Seeded from the ctor argument or std::random_device

public:
    /**
     * @param particles_per_track Number of particles to generate for each new track (> 0)
     * @param initial_existence_probability Existence probability of new tracks
     * @param birth_covariance_local 6x6 SPD covariance in the local tangent frame (ordering above)
     * @param seed Optional RNG seed for reproducible births; unseeded uses std::random_device
     */
    AdaptiveBirthModel(int particles_per_track,
                       double initial_existence_probability,
                       const Eigen::MatrixXd& birth_covariance_local,
                       std::optional<uint64_t> seed = std::nullopt);

    ~AdaptiveBirthModel() override = default;

    //! The birth covariance in the local tangent frame.
    const MeasCovariance& birthCovarianceLocal() const { return birth_covariance_local_; }

    /**
     * @brief Generate one new track per unused measurement (see class docs for the sampling scheme).
     *
     * label.birth_time = current_time, label.index = index of the measurement in the input vector.
     */
    std::vector<Track> generate_new_tracks(const std::vector<Measurement>& unused_measurements,
                                           double current_time) const override;
};
