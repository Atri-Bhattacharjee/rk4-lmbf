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
#include <Eigen/Dense>
#include <iostream>

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
    Eigen::VectorXd state_vector;
    double weight;  //!< The probability weight of this particle
    Particle() : state_vector(6), weight(0.0) {}
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
 * @brief A POD structure for a single sensor detection
 * 
 * Contains all information about a measurement from a sensor,
 * including the measurement values, uncertainty, and source.
 */
struct Measurement {
    double timestamp_;                    //!< The epoch timestamp of the measurement
    Eigen::VectorXd value_;              //!< The 4D measurement vector [range, range_rate, azimuth, elevation]
    Eigen::MatrixXd covariance_;         //!< The 4x4 measurement noise covariance matrix
    std::string sensor_id_;              //!< Identifier for the sensor that produced the measurement
    Eigen::VectorXd sensor_state_;       //!< The 6D ECI state of the sensor satellite [x, y, z, vx, vy, vz]

    Measurement()
        : timestamp_(0.0), value_(Eigen::VectorXd::Zero(4)), covariance_(Eigen::MatrixXd::Zero(4,4)), sensor_id_(), sensor_state_(Eigen::VectorXd::Zero(6)) {}
        
    /**
     * @brief Convert Cartesian state to measurement space
     * 
     * Converts a 6D Cartesian state [x, y, z, vx, vy, vz] to a 4D measurement space
     * [range, range_rate, azimuth, elevation] relative to the sensor state.
     * 
     * @param cartesian_state 6D Cartesian state vector [x, y, z, vx, vy, vz]
     * @param sensor_state 6D Cartesian state vector of sensor [x, y, z, vx, vy, vz]
     * @return Eigen::VectorXd 4D measurement vector [range, range_rate, azimuth, elevation]
     */
    static Eigen::VectorXd cartesianToMeasurement(const Eigen::VectorXd& cartesian_state, const Eigen::VectorXd& sensor_state) {
        // Relative position and velocity
        Eigen::Vector3d rel_pos = cartesian_state.head(3) - sensor_state.head(3);
        Eigen::Vector3d rel_vel = cartesian_state.tail(3) - sensor_state.tail(3);
        
        // Calculate range
        double range = rel_pos.norm();
        
        // Calculate range rate (dot product of unit vector and relative velocity)
        double range_rate = 0.0;
        if (range > 1e-10) {
            Eigen::Vector3d unit_vector = rel_pos / range;
            range_rate = rel_vel.dot(unit_vector);
        }
        
        // Calculate azimuth and elevation
        double azimuth = std::atan2(rel_pos(1), rel_pos(0));
        double elevation = std::asin(rel_pos(2) / range);
        
        Eigen::VectorXd measurement(4);
        measurement << range, range_rate, azimuth, elevation;
        
        return measurement;
    }
    
    /**
     * @brief Convert measurement space to Cartesian state
     * 
     * Converts a 4D measurement [range, range_rate, azimuth, elevation] to a 6D Cartesian state
     * [x, y, z, vx, vy, vz] relative to the sensor state.
     * 
     * @param measurement 4D measurement vector [range, range_rate, azimuth, elevation]
     * @param sensor_state 6D Cartesian state vector of sensor [x, y, z, vx, vy, vz]
     * @return Eigen::VectorXd 6D Cartesian state vector [x, y, z, vx, vy, vz]
     */
    static Eigen::VectorXd measurementToCartesian(const Eigen::VectorXd& measurement, const Eigen::VectorXd& sensor_state) {
        double range = measurement(0);
        double range_rate = measurement(1);
        double azimuth = measurement(2);
        double elevation = measurement(3);
        
        // Convert from spherical to Cartesian coordinates
        double cos_el = std::cos(elevation);
        Eigen::Vector3d rel_pos;
        rel_pos << range * cos_el * std::cos(azimuth),
                   range * cos_el * std::sin(azimuth),
                   range * std::sin(elevation);
                   
        // Convert range_rate to Cartesian velocity (simplified model)
        // This is a simplification - we only have the radial component
        // A more accurate model would need additional info or assumptions
        Eigen::Vector3d unit_vector = rel_pos.normalized();
        Eigen::Vector3d rel_vel = unit_vector * range_rate;
        
        // Convert to absolute coordinates
        Eigen::VectorXd cartesian_state(6);
        cartesian_state.head(3) = rel_pos + sensor_state.head(3);
        cartesian_state.tail(3) = rel_vel + sensor_state.tail(3);
        
        return cartesian_state;
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