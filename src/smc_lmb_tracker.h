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
#include <utility>
#include <string>
#include <tuple>
#include <cstdint>
#include <limits>
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
    //! Regularization jitter stream, separate from resample_rng_ so switching regularization on or
    //! off never shifts the resampler's draws.
    std::mt19937_64 regularization_rng_;
    std::normal_distribution<double> regularization_normal_{0.0, 1.0};
    //! Fused-proposal sampling stream, separate for the same reason.
    std::mt19937_64 fused_rng_;
    std::normal_distribution<double> fused_normal_{0.0, 1.0};

    // Scratch buffers reused across update() calls so the per-step allocation count does not
    // scale with the track or measurement count. They carry no state between calls; every one is
    // assigned before it is read.
    std::vector<double> association_weights_;           //!< Flat per-(track, measurement) particle weights
    std::vector<size_t> track_particle_offsets_;        //!< Prefix sums of per-track particle counts, size num_tracks + 1
    std::vector<double> assoc_coefficients_;            //!< Per-measurement hypothesis-weight totals
    std::vector<char> assoc_used_;                      //!< Whether any hypothesis chose that measurement
    std::vector<double> mixture_weights_;               //!< Per-particle posterior mixture weights

    // Field-of-view coverage for the current update, on the same reuse-across-calls footing as the
    // buffers above. Rewritten by compute_coverage() before anything reads them. Every per-track
    // buffer is indexed by the track's position in active_tracks_, not by its index in the filter:
    // only tracks some sensor can see take part in the assignment (see compute_coverage).
    std::vector<size_t> active_tracks_;                 //!< Filter indices of the tracks in this update's assignment
    std::vector<double> coverage_per_sensor_;           //!< q[active track][sensor], flat, num_active * num_sensors_
    std::vector<double> coverage_union_;                //!< Per active track: weight fraction inside ANY sensor's volume
    std::vector<double> coverage_scratch_;              //!< One track's row, as SensorArray::coverage fills it
    std::vector<int> particle_sensor_;                  //!< Per particle of the active tracks: sensor that sees it, or -1
    std::vector<int> particle_sensor_scratch_;          //!< One track's mask, as SensorArray::coverage fills it
    std::vector<double> track_inv_weight_sum_;          //!< Per active track: 1 / sum of particle weights
    std::vector<int> meas_sensor_;                      //!< Sensor behind each measurement; -1 = no sensor array
    size_t num_sensors_ = 0;                            //!< Sensors in the array driving the current update
    bool sensor_array_ = false;                         //!< Whether the current update has a sensor array at all

    // Fast mode (off by default; see set_fast_mode).
    bool fast_mode_ = false;
    uint64_t stream_seed_ = 0;                          //!< Root of every keyed stream in fast mode

    //! Key of the noise stream for one substep of one track (fast mode).
    uint64_t substep_key(const Track& track, double step_start, double dt) const;

    // Particle gate (off by default; see set_particle_gate).
    bool particle_gate_ = false;
    bool gate_audit_ = false;
    bool gate_sleep_ = true;
    // Sleep bookkeeping for the particle gate (see validate_sleeps).
    std::vector<Eigen::Vector3d> sleep_sensor_positions_;
    double sleep_sensor_time_ = std::numeric_limits<double>::quiet_NaN();
    double sleep_sensor_range_ = std::numeric_limits<double>::quiet_NaN();
    double sleep_speed_bound_ = 0.0;     //!< Bound on every sensor's speed that current sleeps assume
    double sleep_noise_cap_ = 0.0;       //!< Noise displacement bound over max_pending_, annealing-scaled
    double sleep_noise_cap_unscaled_ = 0.0;  //!< ... at unit scale, as the sphere test uses it
    mutable uint64_t gate_audit_checks_ = 0;
    mutable uint64_t gate_audit_violations_ = 0;
    uint64_t sleep_audit_checks_ = 0;
    uint64_t sleep_audit_violations_ = 0;
    std::string sleep_audit_first_;
    //! Audit one "not observable" decision (see set_gate_audit).
    void audit_gate_decision(const Track& track, const sensor::SensorArray& sensors, double now) const;
    std::vector<size_t> refresh_work_;                //!< Lagging, awake tracks to test this update
    std::vector<size_t> gate_candidates_;             //!< Sensors the sphere test flagged (scratch)

    // Lazy propagation (off by default; see set_lazy_propagation).
    bool lazy_propagation_ = false;
    double max_pending_ = 60.0;                         //!< Longest a track may lag the clock, and the longest substep [s]

    // Regularization (off by default; see set_regularization).
    bool regularization_ = false;
    double regularization_bandwidth_scale_ = 1.0;
    double regularization_ess_threshold_ = 0.5;

    // Fused proposal (off by default; see set_fused_proposal).
    bool fused_proposal_ = false;
    double fused_ess_min_ = 20.0;                       //!< Ordinary-update ESS below which a pair switches
    double fused_fallback_ess_min_ = 20.0;              //!< Fused ESS below which the Gaussian fallback is used
    size_t fused_neighbours_ = 0;                       //!< 0 = max(30, 5% of the cloud)

    //! A detection component drawn from the fused proposal rather than from the track's own cloud.
    struct FusedComponent {
        std::vector<Particle> particles;                //!< States, with normalised weights
        double ess = 0.0;                               //!< ESS of those weights
        bool fallback = false;                          //!< Gaussian fallback used (weights uniform)
    };
    std::vector<int> fused_index_;                      //!< Per (active track, measurement): index into fused_components_, or -1
    std::vector<FusedComponent> fused_components_;
    //! Per (active track, sensor): the cloud's bounding sphere reaches that sensor's volume even
    //! though no particle is inside it. Only maintained while the fused proposal is on.
    std::vector<char> reach_flags_;

public:
    //! One posterior application (one track, one update), for diagnostics.
    struct PosteriorRecord {
        double time = 0.0;
        uint64_t birth_time = 0;
        uint32_t index = 0;
        double ess = 0.0;               //!< Effective sample size 1 / sum w^2 of the posterior weights
        size_t num_particles = 0;
        double detection_mass = 0.0;    //!< Posterior probability the track was detected this update
        bool regularized = false;
        int fused_components = 0;       //!< Detection components drawn from the fused proposal
        int fallback_components = 0;    //!< ... of which used the Gaussian fallback
        double fused_ess = 0.0;         //!< Smallest fused-component ESS (0 when none)
        int best_measurement = -1;      //!< Measurement with the largest association marginal, or -1
        double best_coefficient = 0.0;  //!< That marginal
    };

    /**
     * @brief Wall-clock totals and work counts, accumulated while profiling is on.
     *
     * Read-only instrumentation: nothing in the filter reads it, and collecting it draws no random
     * number and changes no track, so a profiled run is bit-for-bit the unprofiled one. Timers wrap
     * whole loops (not single tracks), so their own cost is a few clock reads per call.
     */
    struct Profile {
        // Seconds.
        double predict = 0.0;               //!< Whole predict()
        double predict_propagate = 0.0;     //!< ... of which cloud propagation
        double update = 0.0;                //!< Whole update()
        double refresh = 0.0;               //!< Lazy refresh: observability checks + propagation
        double refresh_propagate = 0.0;     //!< ... of which propagation
        double refresh_check = 0.0;         //!< ... of which observability tests
        double synchronize = 0.0;           //!< Explicit synchronize() calls
        double coverage = 0.0;              //!< compute_coverage
        double likelihood = 0.0;            //!< Per-pair particle sums, fused builds excluded
        double fused = 0.0;                 //!< Fused-proposal builds
        double assignment = 0.0;            //!< Cost matrix + ranked assignment + hypothesis weights
        double posterior = 0.0;             //!< apply_posterior (marginals, mixture, resampling, jitter)
        double birth = 0.0;                 //!< Birth of new tracks
        // Counts.
        uint64_t predict_calls = 0;
        uint64_t update_calls = 0;
        uint64_t updates_with_measurements = 0;
        uint64_t predict_particle_steps = 0;     //!< Particles x substeps propagated by predict()
        uint64_t refresh_particle_steps = 0;     //!< ... by the lazy refresh in update()
        uint64_t synchronize_particle_steps = 0; //!< ... by synchronize()
        uint64_t refresh_checks = 0;             //!< Observability tests of lagging tracks
        uint64_t refresh_slept = 0;              //!< Lagging tracks skipped because they were asleep
        uint64_t sleep_resets = 0;               //!< Times every sleep was cleared (sensor jump, ...)
        //! "Not observable" answers by the sleep they earned: none, < 1 s, 1-5 s, 5-20 s, >= 20 s.
        uint64_t sleep_histogram[5] = {0, 0, 0, 0, 0};
        uint64_t tracks_refreshed = 0;           //!< Tracks the refresh brought to the clock
        uint64_t refreshed_tracks_covered = 0;   //!< ... with some particle inside a sensor volume
        //! Refreshed tracks bucketed by the smallest (particle-to-sensor distance - max_range) after
        //! propagation: inside (<= 0), within 5 km, 5-50 km, beyond 50 km.
        uint64_t refresh_gap_inside = 0;
        uint64_t refresh_gap_under_5km = 0;
        uint64_t refresh_gap_5_to_50km = 0;
        uint64_t refresh_gap_over_50km = 0;
        uint64_t active_tracks = 0;              //!< Summed over updates
        uint64_t likelihood_evaluations = 0;     //!< Particle likelihoods in the per-pair sums
        uint64_t fused_builds = 0;
        uint64_t fused_kernel_evaluations = 0;   //!< Draws x kernels in the fused prior density
        uint64_t posterior_updates = 0;          //!< apply_posterior calls that resampled
        uint64_t regularizations = 0;
        uint64_t births = 0;
    };

    void set_profiling(bool enabled) { profiling_ = enabled; }
    bool profiling() const { return profiling_; }
    const Profile& profile() const { return profile_; }
    void reset_profile() { profile_ = Profile{}; }

private:
    bool record_diagnostics_ = false;
    std::vector<PosteriorRecord> diagnostics_;

    bool profiling_ = false;
    Profile profile_;
    //! Per filter track: brought to the clock by this update's lazy refresh (profiling only).
    std::vector<char> refreshed_scratch_;

    //! Profiling only: bucket a freshly refreshed track by how close its particles came to a sensor
    //! (0 inside, 1 within 5 km, 2 within 50 km, 3 beyond, -1 nothing to measure).
    int refresh_gap_bucket(const Track& track, const sensor::SensorArray& sensors) const;

    //! Liu-West kernel jitter of a freshly resampled cloud, about the weighted posterior mean and
    //! covariance of the cloud it was drawn from.
    void regularize(std::vector<Particle>& resampled, const std::vector<Particle>& predicted,
                    const std::vector<double>& posterior_weights);

    //! Systematic resampling of n particles from `source` with normalised `weights`. One draw from
    //! resample_rng_. Output weights are 1/n.
    std::vector<Particle> systematic_resample(const std::vector<Particle>& source,
                                              const std::vector<double>& weights, size_t n);

    /**
     * @brief Build the detection component of (track, measurement) from the fused proposal.
     *
     * Used when the ordinary particle update has collapsed. The proposal is the Gaussian product of
     * (i) a kernel density of the track's particles nearest the measured state and (ii) the
     * measurement expressed as a Gaussian in state space; draws are importance-weighted with the
     * exact likelihood. Returns false (leaving the ordinary result in place) when the geometry is
     * degenerate. On success writes the association likelihood L (measurement-space units, the same
     * quantity the particle sum estimates) and the weighted component.
     */
    bool build_fused_component(const Track& track, const Measurement& measurement,
                               const MeasurementLikelihoodCache& cache, int sensor_index,
                               const sensor::SensorArray* sensors, double& likelihood,
                               FusedComponent& out);

    void ensure_models_configured() const;

    //! Shared body of both update() overloads. sensors == nullptr selects the historical
    //! omniscient-sensor behaviour, in which every track is fully observable.
    void update_impl(const std::vector<Measurement>& measurements, const sensor::SensorArray* sensors);

    /**
     * @brief Map every measurement to the index of the sensor that produced it (meas_sensor_).
     *
     * Throws when a measurement's sensor_id_ is not in the array: guessing would silently score
     * the association with the wrong sensor's volume. Runs before anything mutates the filter, so
     * a rejected update leaves the state exactly as it was.
     */
    void resolve_measurement_sensors(const std::vector<Measurement>& measurements,
                                     const sensor::SensorArray* sensors);

    /**
     * @brief Decide which tracks take part in this update and measure their coverage.
     *
     * Without a sensor array every track is active and fully observable. With one, a track is
     * active when some particle of it is inside some sensor's volume (coverage_union > 0) and its
     * cloud is current. Every other track has exactly one hypothesis open to it -- not detected,
     * with P_D = 0 everywhere on its cloud -- whose posterior is its prior, so it is left out of the
     * assignment altogether. That is exact, and it is what keeps the assignment the size of the
     * handful of tracks near a sensor rather than the whole catalogue.
     */
    void compute_coverage(const sensor::SensorArray* sensors);

    //! Posterior existence, mixture and resampling for active track `a`, from its hypothesis
    //! marginals: det_coefficients[j] = total weight of hypotheses assigning it measurement j,
    //! miss_coefficient = total weight of hypotheses leaving it undetected.
    void apply_posterior(size_t a, size_t num_meas, const std::vector<double>& det_coefficients,
                         const std::vector<char>& det_used, double miss_coefficient, bool miss_used,
                         size_t contributing_hypotheses);

    //! Drop tracks whose existence probability fell below prune_threshold_, in place.
    void prune_tracks(std::vector<Track>& tracks);

    //! Annealed process-noise scale for a track at time `time` (1 when annealing is off).
    double noise_scale_at(const Track& track, double time) const;

    //! Propagate one track from its propagated_time to `target_time` in equal substeps no longer
    //! than max_pending_, then refresh its cached cloud bound. Returns particles x substeps.
    uint64_t propagate_track_to(Track& track, double target_time);

    //! Whether any particle of a lagging track could be inside a sensor volume at `now`. Conservative:
    //! a false positive only costs an early propagation.
    bool may_be_observable(Track& track, const sensor::SensorArray& sensors, double now,
                           std::vector<size_t>& candidates, double& sleep_for) const;

    //! Particle gate: whether any particle of a lagging track could be within reach of one of
    //! `candidates` after tau seconds. Conservative in the same sense as may_be_observable.
    bool particles_may_be_observable(const Track& track, const sensor::SensorArray& sensors,
                                     double tau, double noise, const std::vector<size_t>& candidates,
                                     double& sleep_for, const Eigen::Vector3d& cloud_predicted,
                                     double nearest_other) const;

    //! Particle gate: before a refresh, check that every sensor moved no faster than the speed bound
    //! the current sleeps assume (and that the clock, the sensor count and the range are unchanged);
    //! otherwise wake every track. Sets sleep_speed_bound_ and sleep_noise_cap_ for this update.
    void validate_sleeps(const sensor::SensorArray& sensors, double now);


    //! Lazy mode: bring every lagging track that may be observable up to the clock.
    void refresh_observable_tracks(const sensor::SensorArray& sensors);

    //! Stamp tracks that no tracker has timed yet (fresh from set_tracks) with the clock.
    void stamp_unstamped_tracks();

    /**
     * @brief Whether active track `a` could have produced measurement j at all.
     *
     * False exactly when a sensor array is in play and none of the track's particles are inside the
     * volume of the sensor that produced j. Such a pair is impossible, so update_impl skips its
     * likelihood pass and writes INF_COST into the cost matrix. Always true without a sensor array.
     */
    bool pair_is_observable(size_t a, size_t meas_index) const {
        const int sensor_index = meas_sensor_[meas_index];
        if (sensor_index < 0) {
            return true;
        }
        const size_t slot = a * num_sensors_ + static_cast<size_t>(sensor_index);
        if (coverage_per_sensor_[slot] > 0.0) {
            return true;
        }
        // With the fused proposal on, a cloud that reaches the volume without any particle inside
        // it can still have produced the measurement; its density there decides, not a particle
        // count.
        return fused_proposal_ && reach_flags_[slot] != 0;
    }

    //! Whether particle p of active track `a` is inside the volume of sensor `sensor_index`.
    //! sensor_index < 0 is the legacy omniscient sensor, which sees everything.
    bool particle_visible_to(size_t a, size_t p, int sensor_index) const {
        return sensor_index < 0 || particle_sensor_[track_particle_offsets_[a] + p] == sensor_index;
    }

    //! Whether particle p of active track `a` is inside any sensor's volume.
    bool particle_visible_anywhere(size_t a, size_t p) const {
        return !sensor_array_ || particle_sensor_[track_particle_offsets_[a] + p] >= 0;
    }

    /**
     * @brief P_D-weighted coverage of active track `a`: <p, P_D> = P_D * q_union.
     *
     * The probability that the track, if it exists, is detected by somebody. A track outside every
     * field of view has 0, so its existence probability is left exactly where it was.
     */
    double detection_mass(size_t a) const {
        return sensor_array_ ? p_detection_ * coverage_union_[a] : p_detection_;
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
     * @brief Turn lazy propagation on or off.
     *
     * With it on, predict() only advances the clock (and the survival probability); a track's
     * cloud is propagated when it has lagged the clock by max_pending seconds, when update() finds
     * that some particle of it could be inside a sensor volume, or on synchronize(). Everything a
     * sensor can see is therefore propagated at the caller's fine step, and everything else in
     * coarse substeps of at most max_pending.
     *
     * Requires a propagator whose process noise is time-consistent (TwoBodyPropagator with
     * noise_reference_dt), because the same track is stepped with different step lengths and
     * because the observability test needs a bound on how far noise can move a particle. Throws
     * std::invalid_argument otherwise.
     *
     * Tracks returned by get_tracks() may lag the clock; call synchronize() first when their state
     * at the current time is wanted, and read Track::propagated_time() to tell.
     */
    void set_lazy_propagation(bool enabled, double max_pending = 60.0);

    /**
     * @brief Turn the particle gate of lazy propagation on or off.
     *
     * Lazy propagation decides whether a lagging cloud might be visible from one bounding sphere.
     * For a long curved cloud the sphere is far larger than the cloud, so the whole cloud gets
     * stepped every second while no particle is anywhere near a sensor (measured: 97% of such
     * refreshes on the 10-orbit ring run had no particle within 5 km of a sensor's reach). With the
     * gate on, a sphere "maybe" is re-tested particle by particle, with the same Taylor bound and no
     * cloud spread, against only the sensors the sphere flagged; the cloud is propagated only if some
     * particle could be in reach. Sensors that reported a measurement keep the sphere test, so the
     * fused proposal still sees every cloud near a detection.
     *
     * It changes when clouds are propagated, never what an update learns: a cloud left lagging has
     * no particle inside any sensor volume, exactly as with the sphere test. The substep partition,
     * and with it the noise draws, differ, so results are statistically, not bitwise, the same. Off
     * (the default), the filter is bit-for-bit unchanged.
     */
    void set_particle_gate(bool enabled) { particle_gate_ = enabled; }
    bool particle_gate() const { return particle_gate_; }

    /**
     * @brief Audit the particle gate (validation only; slow).
     *
     * Every time the gate leaves a cloud lagging, a noise-free copy of the cloud is propagated to the
     * clock (drawing no random number, so the run itself is unchanged) and tested against every
     * sensor. Any particle inside a volume is a violation: the gate would have dropped evidence.
     * Returns (decisions audited, violations).
     */
    void set_gate_audit(bool enabled) { gate_audit_ = enabled; }

    /**
     * @brief Sleep between gate tests (on by default with the gate; for testing).
     *
     * After the gate answers "not observable", the cloud is not tested again until the same test is
     * guaranteed to answer "not observable" no longer: the margin is taken at its largest over the
     * remaining lag, the cloud and every sensor move at bounded speed (sensors are held to the bound
     * every update, and a violation wakes every track), and steps with a reported measurement test
     * everything. Skipping only answers already known therefore leaves results bit-for-bit unchanged.
     */
    void set_gate_sleep(bool enabled) { gate_sleep_ = enabled; }
    std::pair<uint64_t, uint64_t> gate_audit_counts() const {
        return {gate_audit_checks_, gate_audit_violations_};
    }
    //! (sleeping skips audited, skips whose test would have said "maybe", first such case).
    std::tuple<uint64_t, uint64_t, std::string> sleep_audit() const {
        return {sleep_audit_checks_, sleep_audit_violations_, sleep_audit_first_};
    }
    bool lazy_propagation() const { return lazy_propagation_; }
    double max_pending() const { return max_pending_; }

    /**
     * @brief Turn the regularization (kernel jitter) step on or off.
     *
     * After a track's cloud is resampled, if the posterior's effective sample size was below
     * ess_threshold * N, every resampled particle is moved as
     *
     *     x <- m + a (x - m) + h L eps,    a = sqrt(1 - h^2),   L L^T = S,   eps ~ N(0, I6)
     *
     * with m and S the weighted mean and covariance of the posterior before resampling (Liu & West
     * 2001 kernel shrinkage), and h = bandwidth_scale * h_opt, h_opt = (4 / (N (d + 2)))^(1/(d+4))
     * the Gaussian-kernel bandwidth of the regularized particle filter (Musso, Oudjane & Le Gland
     * 2001), d = 6. The shrinkage keeps the cloud's mean and covariance, so repeated resampling
     * does not inflate it; what it restores is diversity, which resampling destroys by duplicating
     * particles. Off, the filter is bit-for-bit unchanged.
     */
    void set_regularization(bool enabled, double bandwidth_scale = 1.0, double ess_threshold = 0.5);
    bool regularization() const { return regularization_; }

    /**
     * @brief Turn the fused proposal on or off.
     *
     * For each (track, measurement) pair the ordinary particle update runs first. If its effective
     * sample size is below ess_min (it has collapsed: the measurement is far sharper than the gaps
     * between the track's particles), that pair's detection component is rebuilt from a proposal
     * centred where the track's cloud and the measurement overlap: the Gaussian product of a kernel
     * density of the `neighbours` particles nearest the measured state (0 = max(30, 5% of the
     * cloud)) and the measurement as a Gaussian in state space, importance-weighted with the exact
     * likelihood and the kernel density. Its association likelihood replaces the particle sum, which
     * in that regime underflows to zero. A cloud whose bounding sphere reaches the sensor but has no
     * particle inside it is scored the same way rather than ruled out. If the fused weights
     * themselves collapse (ESS below fallback_ess_min), the component falls back to uniform draws from
     * the Gaussian product with a closed-form likelihood -- an approximation (it fits a Gaussian to a
     * cut-out of the cloud), kept as a last resort. Off, the filter is bit-for-bit unchanged.
     */
    void set_fused_proposal(bool enabled, double ess_min = 20.0, size_t neighbours = 0,
                            double fallback_ess_min = 20.0);
    bool fused_proposal() const { return fused_proposal_; }

    /**
     * @brief Turn fast mode on or off.
     *
     * Fast mode draws process noise from keyed streams (fast_random.h) through a ziggurat sampler,
     * and propagates a cloud in one batched call instead of one virtual call per particle. The
     * noise a particle gets is a pure function of (seed, track, substep, particle), so it does not
     * depend on how often, or in which order, anything else was propagated. The deterministic RK4
     * step is bit-for-bit the legacy one.
     *
     * The random streams differ from the legacy ones, so a fast-mode run is statistically, not
     * bitwise, equivalent to a legacy run. Off (the default) the filter is bit-for-bit unchanged.
     * Requires a propagator that supports keyed propagation (TwoBodyPropagator does); throws
     * std::invalid_argument otherwise.
     */
    void set_fast_mode(bool enabled);
    bool fast_mode() const { return fast_mode_; }

    //! Record one PosteriorRecord per track per update (off by default).
    void set_record_diagnostics(bool enabled) { record_diagnostics_ = enabled; }
    //! Return the records collected since the last call, and clear them.
    std::vector<PosteriorRecord> take_diagnostics() {
        std::vector<PosteriorRecord> out;
        out.swap(diagnostics_);
        return out;
    }

    //! Propagate every lagging track to the filter clock. A no-op when nothing lags.
    void synchronize();

    //! The filter clock [s].
    double timestamp() const { return current_state_.timestamp(); }

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