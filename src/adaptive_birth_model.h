#pragma once

/**
 * @file adaptive_birth_model.h
 * @brief Implementation of adaptive birth model for track initialization
 * 
 * This class implements the IBirthModel interface to create new track hypotheses
 * from sensor measurements that were not associated with existing tracks. It uses
 * a physics-based tangent fan approach for particle velocity initialization that
 * respects orbital mechanics.
 */

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>

/**
 * @brief Adaptive birth model using tangent fan velocity sampling
 * 
 * This class implements track birth by generating particles with physically
 * plausible orbital velocities. For each measurement:
 * 
 * - Position = sensor position + range × line-of-sight
 * - Relative velocity = range-rate × LOS + circular-speed tangent fan
 * - Absolute ECI velocity = sensor velocity + relative velocity
 * - Gaussian noise from the birth covariance captures eccentricity variation
 * 
 * This approach ensures particles start with realistic orbital velocities
 * rather than assuming zero cross-range motion, and remain consistent with
 * the relative range-rate measurement model when the sensor is moving.
 */
class AdaptiveBirthModel : public IBirthModel {
private:
    int particles_per_track_;                    //!< The number of particles to generate for each new track
    double initial_existence_probability_;       //!< The low probability assigned to a brand new track
    ProcessNoiseCov initial_covariance_;         //!< A 6x6 covariance matrix defining the initial uncertainty
    ProcessNoiseCov birth_noise_L_;              //!< Precomputed Cholesky factor of initial covariance
    
    // Physical constants for orbital mechanics
    static constexpr double MU_EARTH = 3.986004418e14;   //!< Earth gravitational parameter (m³/s²)
    static constexpr double R_EARTH = 6.371e6;           //!< Earth radius (m)
    
    /**
     * @brief Compute circular orbital velocity at given radius from Earth center
     * 
     * Uses the vis-viva equation for circular orbits: v = sqrt(mu/r)
     * 
     * @param radius Distance from Earth center (m)
     * @return Circular orbital velocity (m/s)
     */
    double computeCircularVelocity(double radius) const;

public:
    /**
     * @brief Construct a new Adaptive Birth Model object
     * 
     * @param particles_per_track Number of particles to generate for each new track
     * @param initial_existence_probability The initial existence probability for new tracks
     * @param initial_covariance 6x6 covariance matrix for initial state uncertainty
     */
    AdaptiveBirthModel(int particles_per_track, 
                      double initial_existence_probability, 
                      const Eigen::MatrixXd& initial_covariance);

    /**
     * @brief Default destructor
     */
    ~AdaptiveBirthModel() override = default;

    /**
     * @brief Generate new tracks from unused measurements using tangent fan sampling
     * 
     * For each unassociated measurement, creates a new track with particles sampled
     * using physics-based velocity initialization:
     * 
     * 1. Position = sensor position + range × line-of-sight
     * 2. Relative velocity = range-rate × LOS + circular-speed tangent fan
     *    (tangent direction sampled uniformly in [0, 2π); magnitude = circular
     *    orbital velocity at target altitude)
     * 3. Absolute ECI velocity = sensor velocity + relative velocity
     * 4. Gaussian noise from birth covariance added to capture eccentricity
     * 
     * @param unused_measurements Vector of measurements not associated with existing tracks
     * @param current_time The current simulation time for track initialization
     * @return std::vector<Track> Vector of newly created tracks with initial particle clouds
     */
    std::vector<Track> generate_new_tracks(const std::vector<Measurement>& unused_measurements, 
                                          double current_time) const override;
};
