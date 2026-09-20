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

    void ensure_models_configured() const;

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
     * @param measurements Vector of new sensor measurements to process
     */
    void update(const std::vector<Measurement>& measurements);

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