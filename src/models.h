#pragma once

/**
 * @file models.h
 * @brief Abstract interfaces for pluggable physics models in the high-performance simulation engine
 * 
 * This header defines the abstract base classes that serve as contracts for various
 * physics models including orbit propagation, sensor modeling, and track birth modeling.
 * These interfaces enable polymorphism and pluggable architecture for different
 * implementation strategies.
 */

#include "datatypes.h"
#include <vector>

/**
 * @brief Abstract base class that defines the interface for all orbit propagation models
 * 
 * This interface provides a contract for any orbit propagation implementation,
 * allowing different propagation algorithms (e.g., Two-Body, SGP4, high-fidelity)
 * to be used interchangeably within the simulation engine.
 */
class IOrbitPropagator {
public:
    /**
     * @brief Virtual destructor to enable proper polymorphic cleanup
     */
    virtual ~IOrbitPropagator() = default;

    /**
     * @brief Propagate a particle's state forward in time
     * 
     * This method takes a particle with a current state and propagates it
     * forward by the specified time interval using the implemented propagation model.
     * 
     * @param particle The input particle with current state [x, y, z, vx, vy, vz, bc]
     * @param dt The time step in seconds to propagate forward
     * @param current_time The current simulation timestamp (seconds since epoch)
     * @param noise_scale Scale factor for process noise (1.0 = full noise, 0.0 = no noise)
     * @return Particle The propagated particle with updated state
     */
    virtual Particle propagate(const Particle& particle, double dt, double current_time, double noise_scale = 1.0) const = 0;
};

/**
 * @brief Abstract base class that defines the interface for all sensor models
 * 
 * This interface provides a contract for sensor modeling implementations,
 * allowing different sensor types (e.g., radar, optical, RF) to calculate
 * measurement likelihoods in a consistent manner.
 */
class ISensorModel {
public:
    /**
     * @brief Virtual destructor to enable proper polymorphic cleanup
     */
    virtual ~ISensorModel() = default;

    /**
     * @brief Calculate the likelihood of a measurement given a particle state
     * 
     * This method computes the probability density of observing the given
     * measurement if the true object state matches the provided particle.
     * This is a core component of the particle filter update step.
     * 
     * @param particle The particle representing a possible object state
     * @param measurement The sensor measurement [azimuth, elevation, range] with covariance
     * @return double The likelihood value (probability density)
     */
    virtual double calculate_likelihood(const Particle& particle, const Measurement& measurement) const = 0;
};

/**
 * @brief Abstract base class that defines the interface for track birth models
 * 
 * This interface provides a contract for track initialization strategies,
 * allowing different approaches for creating new tracks from unused measurements
 * (e.g., single-measurement birth, multiple-measurement confirmation).
 */
class IBirthModel {
public:
    /**
     * @brief Virtual destructor to enable proper polymorphic cleanup
     */
    virtual ~IBirthModel() = default;

    /**
     * @brief Generate new tracks from unused measurements
     * 
     * This method analyzes measurements that were not associated with existing
     * tracks and creates new track hypotheses. The implementation determines
     * the birth strategy and initial particle distributions.
     * 
     * @param unused_measurements Vector of measurements not associated with existing tracks
     * @param current_time The current simulation time for track initialization
     * @return std::vector<Track> Vector of newly created tracks with initial particle clouds
     */
    virtual std::vector<Track> generate_new_tracks(const std::vector<Measurement>& unused_measurements, double current_time) const = 0;
};