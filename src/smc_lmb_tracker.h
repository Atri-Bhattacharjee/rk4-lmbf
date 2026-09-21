#pragma once

/**
 * @file smc_lmb_tracker.h
 * @brief Central class for Sequential Monte Carlo Labeled Multi-Bernoulli tracking filter
 * 
 * This header defines the SMC_LMB_Tracker class which orchestrates the entire 
 * filtering process using pluggable model interfaces for orbit propagation,
 * sensor modeling, and track birth modeling.
 */

#include <vector>
#include <memory>
#include <cstdint>
#include <optional>
#include <random>
#include "datatypes.h"
#include "models.h"
#include "assignment.h"
#include "sensor_fov.h"

/**
 * @brief Sequential Monte Carlo Labeled Multi-Bernoulli Tracker
 * 
 * This class implements the central orchestration logic for a Bayesian tracking
 * filter using the Labeled Multi-Bernoulli framework. It coordinates between
 * different pluggable model components to perform prediction and update steps
 * for multiple target tracking.
 */
class SMC_LMB_Tracker {
private:
    FilterState current_state_;                         //!< Stores the current list of tracks and timestamp
    std::shared_ptr<IOrbitPropagator> propagator_;     //!< Shared pointer to the propagator model
    std::shared_ptr<ISensorModel> sensor_model_;       //!< Shared pointer to the sensor model
    std::shared_ptr<IBirthModel> birth_model_;         //!< Shared pointer to the birth model
    double survival_probability_;                       //!< Configuration parameter for track survival probability
    int k_best_;                                        //!< Number of K-best assignment hypotheses
    double prune_threshold_;                            //!< Existence probability threshold for track pruning
    double clutter_intensity_;                          //!< Clutter intensity (false alarms per unit measurement volume)
    double p_detection_;                                //!< Detection probability (P_D) per Reuter LMB formulation
    double noise_decay_rate_;                           //!< Process noise annealing decay rate (lambda)
    double noise_min_scale_;                            //!< Process noise annealing minimum scale factor (alpha_min)
    mutable std::mt19937_64 resample_rng_;              //!< Resampler stream; seeded from the ctor or std::random_device

    // Scratch buffers reused across update() calls so the per-step allocation count does not
    // scale with the track or measurement count. They carry no state between calls; every one is
    // assigned before it is read.
    std::vector<double> association_weights_;           //!< Flat per-(track, measurement) particle weights
    std::vector<size_t> track_particle_offsets_;        //!< Prefix sums of per-track particle counts, size num_tracks + 1
    std::vector<double> assoc_coefficients_;            //!< Per-measurement hypothesis-weight totals
    std::vector<char> assoc_used_;                      //!< Whether any hypothesis chose that measurement
    std::vector<double> mixture_weights_;               //!< Per-particle posterior mixture weights

    // Field-of-view coverage for the current update, on the same reuse-across-calls footing as the
    // buffers above. They are rewritten by prepare_coverage() before anything reads them.
    std::vector<double> coverage_per_sensor_;           //!< q[track][sensor], flat, num_tracks * num_sensors_
    std::vector<double> coverage_union_;                //!< Per-track weight fraction inside ANY sensor's volume
    std::vector<double> coverage_scratch_;              //!< One track's row, as SensorArray::coverage fills it
    std::vector<int> meas_sensor_;                      //!< Sensor behind each measurement; -1 = no sensor array
    size_t num_sensors_ = 0;                            //!< Sensors in the array driving the current update

    void ensure_models_configured() const;

    //! Shared body of both update() overloads. sensors == nullptr selects the historical
    //! omniscient-sensor behaviour, in which every coverage fraction is exactly 1.0.
    void update_impl(const std::vector<Measurement>& measurements, const sensor::SensorArray* sensors);

    /**
     * @brief Fill coverage_* and meas_sensor_ for one update step.
     *
     * Throws when a measurement's sensor_id_ is not in the array: guessing would silently score
     * the association with the wrong sensor's coverage fraction.
     */
    void prepare_coverage(const std::vector<Track>& tracks,
                          const std::vector<Measurement>& measurements,
                          const sensor::SensorArray* sensors);

    //! Existence update for a step in which no sensor reported anything. Every particle weight is
    //! scaled by the same (1 - P_D_eff), so the normalised cloud does not move and nothing is
    //! resampled -- which also leaves resample_rng_ untouched.
    void apply_missed_detection_only(std::vector<Track>& tracks);

    //! Drop tracks whose existence probability fell below prune_threshold_, in place.
    void prune_tracks(std::vector<Track>& tracks);

    /**
     * @brief Effective P_D for track i being detected by the sensor that produced measurement j.
     *
     * The configured P_D scaled by the fraction of the track's cloud inside that sensor's volume,
     * so a track that cannot be where that sensor is looking cannot claim its measurement.
     */
    double detection_probability(size_t track_index, size_t meas_index) const {
        const int sensor_index = meas_sensor_[meas_index];
        if (sensor_index < 0) {
            return p_detection_;
        }
        return p_detection_ *
               coverage_per_sensor_[track_index * num_sensors_ + static_cast<size_t>(sensor_index)];
    }

    /**
     * @brief Whether track i could have produced measurement j at all.
     *
     * False exactly when a sensor array is in play and none of the track's particles are inside the
     * volume of the sensor that produced j, i.e. detection_probability(i, j) == 0. Such a pair is
     * impossible, so update_impl skips its likelihood pass and writes INF_COST into the cost matrix.
     * This is the single predicate for both, so the two cannot disagree. Always true without a
     * sensor array (num_sensors_ == 0), where nothing is ever skipped.
     */
    bool pair_is_observable(size_t track_index, size_t meas_index) const {
        return num_sensors_ == 0 || detection_probability(track_index, meas_index) > 0.0;
    }

    /**
     * @brief Effective P_D for track i being detected by anybody, used by the missed-detection branch.
     *
     * Uses the union coverage, so a track outside every field of view has P_D_eff = 0 and its
     * existence probability is left exactly where it was rather than decaying while unobservable.
     */
    double miss_detection_probability(size_t track_index) const {
        return p_detection_ * coverage_union_[track_index];
    }

    /**
     * @brief Offset of one (track, measurement) block inside association_weights_
     *
     * Particle counts are deliberately not assumed uniform across tracks, so blocks are located
     * through the prefix-sum table rather than a fixed stride. This keeps the layout correct if
     * cloud sizes become adaptive (for example, scaled by track confidence).
     *
     * @param track_index Track whose block is wanted
     * @param meas_index Measurement whose block is wanted
     * @param num_meas Number of measurements in the current update
     * @return size_t Index of the first particle weight of that block
     */
    size_t association_block_offset(size_t track_index, size_t meas_index, size_t num_meas) const {
        const size_t start = track_particle_offsets_[track_index];
        const size_t num_particles = track_particle_offsets_[track_index + 1] - start;
        return start * num_meas + meas_index * num_particles;
    }

public:
    /**
     * @brief Default constructor
     */
    SMC_LMB_Tracker() : current_state_(0.0, std::vector<Track>{}), propagator_(nullptr), sensor_model_(nullptr), birth_model_(nullptr), survival_probability_(0.0), k_best_(100), prune_threshold_(0.01), clutter_intensity_(1.0e-6), p_detection_(0.99), noise_decay_rate_(0.0), noise_min_scale_(1.0), resample_rng_(std::mt19937_64::result_type(std::random_device{}())) {}

    /**
     * @brief Construct a new SMC_LMB_Tracker object
     * 
     * @param propagator Shared pointer to orbit propagation model
     * @param sensor_model Shared pointer to sensor model
     * @param birth_model Shared pointer to birth model
     * @param survival_probability Probability of a track surviving a time step
     * @param k_best Number of K-best assignment hypotheses to generate
     * @param prune_threshold Existence probability threshold for track pruning
     * @param clutter_intensity Clutter intensity (false alarms per unit measurement volume)
     * @param p_detection Detection probability (P_D) for the sensor model
     * @param noise_decay_rate Process noise annealing decay rate (lambda, per second)
     * @param noise_min_scale Process noise annealing minimum scale factor (alpha_min)
     * @param seed Optional seed for the resampler's random stream. When omitted the stream is
     *             seeded from std::random_device, which is the historical behavior.
     */
    SMC_LMB_Tracker(std::shared_ptr<IOrbitPropagator> propagator,
                   std::shared_ptr<ISensorModel> sensor_model,
                   std::shared_ptr<IBirthModel> birth_model,
                   double survival_probability,
                   int k_best,
                   double prune_threshold,
                   double clutter_intensity,
                   double p_detection,
                   double noise_decay_rate,
                   double noise_min_scale,
                   std::optional<uint64_t> seed = std::nullopt);

    /**
     * @brief Run the predict step of the filter
     * 
     * This method propagates all existing tracks forward in time using the
     * configured orbit propagator model.
     * 
     * @param dt Time step in seconds to propagate forward
     */
    void predict(double dt);

    /**
     * @brief Run the update step of the filter
     * 
     * This method updates track existence probabilities and particle weights
     * based on new sensor measurements, and creates new tracks from unused
     * measurements.
     * 
     * This overload keeps the historical behaviour in which the sensor sees everything: every
     * track is fully observable, so the effective P_D is the configured P_D, and a step with no
     * measurements changes nothing.
     *
     * @param measurements Vector of new sensor measurements to process
     */
    void update(const std::vector<Measurement>& measurements);

    /**
     * @brief Run the update step against a configured set of sensors
     *
     * Each measurement is traced back through its sensor_id_ to the sensor that produced it, and
     * the effective detection probability of every track is the configured P_D scaled by the
     * fraction of that track's particle cloud inside the relevant sensor volume. A step in which
     * no sensor reported anything is still informative and is applied as a pure missed detection.
     *
     * @param measurements Vector of new sensor measurements to process
     * @param sensors The sensors, with the pointing they had for this step
     */
    void update(const std::vector<Measurement>& measurements, const sensor::SensorArray& sensors);

    /**
     * @brief Get the current tracks
     * 
     * Returns a const reference to the current track list for efficiency.
     * 
     * @return const std::vector<Track>& Reference to the current tracks
     */
    const std::vector<Track>& get_tracks() const;

    /**
     * @brief Set the tracks for testing purposes
     * 
     * This method allows initialization of the filter's state for testing
     * and debugging purposes.
     * 
     * @param tracks Vector of tracks to set as the current filter state
     */
    void set_tracks(const std::vector<Track>& tracks);

    // Helper to compute the likelihood of a track-measurement pair by averaging over its particles.
    double compute_association_likelihood(const Track& track, const Measurement& measurement) const;
};