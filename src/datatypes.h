#pragma once


/**
 * @file datatypes.h
 * @brief Core data structures for space debris tracking filter
 * 
 * This header defines the fundamental data structures for the space debris tracker.
 * These structures prioritize data locality and performance for the core C++ 
 * computational engine and are designed to be exposed to Python via pybind11.
 */

#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>

#include "los_geometry.h"

using StateVector = Eigen::Matrix<double, 6, 1>;
//! Local tangent-frame measurement perturbation/residual [d_range, d_range_rate, d_theta1, d_theta2, d_omega1, d_omega2]
using LocalMeasVector = Eigen::Matrix<double, 6, 1>;
//! 6x6 covariance in the local tangent frame of the measured direction (same ordering as LocalMeasVector)
using MeasCovariance = Eigen::Matrix<double, 6, 6>;
using ProcessNoiseCov = Eigen::Matrix<double, 6, 6>;

struct MeasurementLikelihoodCache {
    MeasCovariance cov_inv;
    double log_norm_factor = 0.0;
    bool is_diagonal = false;
    LocalMeasVector inv_var;
    //! Geometric part of the measurement, cached once per update so calculate_likelihood need not rebuild it.
    los::LosObservation measured;
    //! tangentBasis(measured.los), identical for every particle of this measurement.
    los::TangentBasis measured_basis = los::TangentBasis::Zero();
};

/**
 * @brief A simple POD structure for creating unique, persistent track identities
 * 
 * This structure provides a unique identifier for each track, combining
 * temporal and sequential information for complete uniqueness.
 */
struct TrackLabel {
    uint64_t birth_time;  //!< The simulation time or epoch when the track was created
    uint32_t index;       //!< A unique index assigned at the time of birth
};

/**
 * @brief A POD structure representing a single, weighted state hypothesis
 * 
 * Each particle represents one possible state of the tracked object,
 * with an associated probability weight.
 */
struct Particle {
    //! [x, y, z, vx, vy, vz] where position is ECI in m, velocity is ECI in m/s
    StateVector state_vector;
    double weight;  //!< The probability weight of this particle
    Particle() : state_vector(StateVector::Zero()), weight(0.0) {}
};

/**
 * @brief Represents a single tracked object, its identity, and state uncertainty distribution
 * 
 * A track contains the unique identifier for the object and a cloud of weighted 
 * particles representing the probability density function of the object's state.
 */
class Track {
private:
    TrackLabel label_;                     //!< The unique, persistent label for this track
    double existence_probability_;         //!< The probability r that this track corresponds to a real object
    std::vector<Particle> particles_;     //!< The cloud of weighted particles representing the state probability density p(x)

public:
    /**
     * @brief Default constructor: label birth_time=0, index=0; existence_probability=0; empty particles
     */
    Track() : label_{0, 0}, existence_probability_(0.0), particles_{} {}

    /**
     * @brief Construct a new Track object
     * 
     * @param label The unique track label
     * @param existence_probability Initial existence probability
     * @param particles Initial particle cloud
     */
    Track(const TrackLabel& label, double existence_probability, const std::vector<Particle>& particles)
        : label_(label), existence_probability_(existence_probability), particles_(particles) {}

    /**
     * @brief Get the track label
     * @return const TrackLabel& Reference to the track label
     */
    const TrackLabel& label() const { return label_; }

    /**
     * @brief Get the existence probability
     * @return double The existence probability
     */
    double existence_probability() const { return existence_probability_; }

    /**
     * @brief Get the particles
     * @return const std::vector<Particle>& Reference to the particle vector
     */
    const std::vector<Particle>& particles() const { return particles_; }

    /**
     * @brief Set the existence probability
     * @param probability New existence probability
     */
    void set_existence_probability(double probability) { existence_probability_ = probability; }

    /**
     * @brief Set the particles
     * @param particles New particle cloud
     */
    void set_particles(const std::vector<Particle>& particles) { particles_ = particles; }
};

/**
 * @brief A single sensor detection of an object: range, range rate, line-of-sight direction and
 *        line-of-sight angular rate, with a 6x6 noise covariance.
 *
 * Representation (see los_geometry.h):
 *   range_      rho     [m]      |r_target - r_sensor|
 *   range_rate_ rhodot  [m/s]    u . (v_target - v_sensor)
 *   los_        u       [unit]   ECI unit vector from the sensor to the object
 *   los_rate_   udot    [rad/s]  time derivative of u (always orthogonal to u)
 *
 * Covariance frame convention: covariance_ is expressed in the local tangent frame at los_,
 * with the deterministic basis (e1, e2) = los::tangentBasis(los_), and ordered as
 *   [ d_range (m), d_range_rate (m/s), d_theta1 (rad), d_theta2 (rad), d_omega1 (rad/s), d_omega2 (rad/s) ]
 * where d_theta = e_k . log_u(u') is the angular offset of a direction u' and d_omega = e_k . (udot' - udot)
 * is the angular-rate offset (after parallel transport into the tangent plane at los_). Residuals
 * produced by los::localResidual and perturbations consumed by perturbed() use the same frame and
 * ordering. The representation has no azimuth/elevation singularity anywhere on the sphere.
 */
struct Measurement {
    double timestamp_ = 0.0;                                     //!< Epoch timestamp of the measurement [s]
    double range_ = 0.0;                                         //!< Range [m]
    double range_rate_ = 0.0;                                    //!< Range rate [m/s]
    Eigen::Vector3d los_ = Eigen::Vector3d::UnitX();             //!< Unit line-of-sight direction (ECI)
    Eigen::Vector3d los_rate_ = Eigen::Vector3d::Zero();         //!< Line-of-sight angular rate [rad/s] (ECI, orthogonal to los_)
    MeasCovariance covariance_ = MeasCovariance::Zero();         //!< 6x6 noise covariance in the local tangent frame at los_
    std::string sensor_id_;                                      //!< Identifier for the sensor that produced the measurement
    StateVector sensor_state_ = StateVector::Zero();             //!< 6D ECI state of the sensor [x, y, z, vx, vy, vz]

    Measurement() = default;

    //! The geometric part of the measurement as a LosObservation.
    los::LosObservation observation() const {
        los::LosObservation obs;
        obs.range = range_;
        obs.range_rate = range_rate_;
        obs.los = los_;
        obs.los_rate = los_rate_;
        return obs;
    }

    //! Overwrite the geometric part from a LosObservation (covariance, sensor and timestamp unchanged).
    void setObservation(const los::LosObservation& obs) {
        range_ = obs.range;
        range_rate_ = obs.range_rate;
        los_ = obs.los;
        los_rate_ = obs.los_rate;
    }

    /**
     * @brief Noise-free measurement of a 6D ECI target state as seen from a 6D ECI sensor state.
     *
     * covariance_ is left at zero and sensor_id_ empty; sensor_state_ is stored.
     */
    static Measurement fromCartesian(const StateVector& target_state, const StateVector& sensor_state) {
        Measurement measurement;
        measurement.setObservation(los::observe(target_state, sensor_state));
        measurement.sensor_state_ = sensor_state;
        return measurement;
    }

    /**
     * @brief Measurement from ECI-axis spherical angles and their rates (display/interop convention).
     *
     * Angles are relative to the ECI axes: azimuth = atan2(u_y, u_x), elevation = asin(u_z).
     * This convention is singular on the ECI z-axis and is offered for interoperability only.
     */
    static Measurement fromAnglesAndRates(double range, double range_rate,
                                          double azimuth, double elevation,
                                          double azimuth_rate, double elevation_rate,
                                          const StateVector& sensor_state) {
        Measurement measurement;
        measurement.setObservation(los::fromAnglesAndRates(range, range_rate, azimuth, elevation,
                                                           azimuth_rate, elevation_rate));
        measurement.sensor_state_ = sensor_state;
        return measurement;
    }

    //! Exact 6D ECI state implied by the measurement: r = r_s + rho u, v = v_s + rhodot u + rho udot.
    StateVector toCartesian() const {
        return los::toCartesian(observation(), sensor_state_);
    }

    /**
     * @brief Copy of this measurement with a local tangent-frame perturbation applied to its geometry.
     * @param eps [d_range, d_range_rate, d_theta1, d_theta2, d_omega1, d_omega2] in the frame at los_.
     */
    Measurement perturbed(const LocalMeasVector& eps) const {
        Measurement result = *this;
        result.setObservation(los::perturbed(observation(), eps));
        return result;
    }

    //! Derived ECI-axis [azimuth, elevation, azimuth_rate, elevation_rate] (display/interop only).
    Eigen::Vector4d angularCoordinates() const {
        return los::angularCoordinates(observation());
    }
};

/**
 * @brief A container for the complete state of the LMB filter at a single point in time
 * 
 * This class holds the complete filter state, including all active tracks
 * and the timestamp of the current state.
 */
class FilterState {
private:
    double timestamp_;              //!< The timestamp of this filter state
    std::vector<Track> tracks_;     //!< The list of all current tracks

public:
    /**
     * @brief Default constructor: timestamp=0, empty tracks
     */
    FilterState() : timestamp_(0.0), tracks_{} {}

    FilterState(double timestamp, const std::vector<Track>& tracks)
        : timestamp_(timestamp), tracks_(tracks) {}

    /**
     * @brief Get the timestamp
     * @return double The filter state timestamp
     */
    double timestamp() const { return timestamp_; }

    // Non-const getter for direct modification
    std::vector<Track>& tracks() { return tracks_; }

    /**
     * @brief Get the tracks
     * @return const std::vector<Track>& Reference to the tracks vector
     */
    const std::vector<Track>& tracks() const { return tracks_; }

    /**
     * @brief Set the timestamp
     * @param timestamp New timestamp
     */
    void set_timestamp(double timestamp) { timestamp_ = timestamp; }

    /**
     * @brief Set the tracks
     * @param tracks New tracks vector
     */
    void set_tracks(const std::vector<Track>& tracks) { tracks_ = tracks; }
};
