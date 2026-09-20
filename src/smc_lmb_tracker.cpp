#include "smc_lmb_tracker.h"
#include "assignment.h"
#include "in_orbit_sensor_model.h"
#include "validation.h"
#include <iostream>
#include <random>
#include <stdexcept>

void SMC_LMB_Tracker::ensure_models_configured() const {
    validation::require_models(propagator_, sensor_model_, birth_model_);
}

SMC_LMB_Tracker::SMC_LMB_Tracker(std::shared_ptr<IOrbitPropagator> propagator,
                                                                 std::shared_ptr<ISensorModel> sensor_model,
                                                                 std::shared_ptr<IBirthModel> birth_model,
                                                                 double survival_probability,
                                                                 int k_best,
                                                                 double prune_threshold,
                                                                 double clutter_intensity,
                                                                 double p_detection,
                                                                 double noise_decay_rate,
                                                                 double noise_min_scale,
                                                                 std::optional<uint64_t> seed)
        : current_state_(0.0, std::vector<Track>{}),
            propagator_(std::move(propagator)),
            sensor_model_(std::move(sensor_model)),
            birth_model_(std::move(birth_model)),
            survival_probability_(survival_probability),
            k_best_(k_best),
            prune_threshold_(prune_threshold),
            clutter_intensity_(clutter_intensity),
            p_detection_(p_detection),
            noise_decay_rate_(noise_decay_rate),
            noise_min_scale_(noise_min_scale),
            resample_rng_(seed.has_value() ? *seed : std::mt19937_64::result_type(std::random_device{}())) {
    validation::require_models(propagator_, sensor_model_, birth_model_);
    validation::require_k_best(k_best_);
    validation::require_clutter_intensity(clutter_intensity_);
}

void SMC_LMB_Tracker::predict(double dt) {
    ensure_models_configured();
    // Projects the filter state forward in time by operating in-place.
    const double previous_time = current_state_.timestamp();
    const double new_time = current_state_.timestamp() + dt;
    current_state_.set_timestamp(new_time);
    // Get a direct, MODIFIABLE reference to the internal track vector.
    std::vector<Track>& tracks = current_state_.tracks();
    for (Track& track : tracks) {
        // Update existence probability in-place.
        track.set_existence_probability(track.existence_probability() * survival_probability_);
        
        // --- Process Noise Annealing ---
        // Calculate track age (time since birth)
        double track_birth_time = static_cast<double>(track.label().birth_time);
        double age = new_time - track_birth_time;
        
        // Calculate noise scale using exponential decay
        // alpha(age) = alpha_min + (1 - alpha_min) * exp(-lambda * age)
        double noise_scale = 1.0;
        if (noise_decay_rate_ > 0.0 && age > 0.0) {
            noise_scale = noise_min_scale_ + (1.0 - noise_min_scale_) * std::exp(-noise_decay_rate_ * age);
        }
        
        // Propagate particles with adaptive noise scaling.
        std::vector<Particle> propagated_particles;
        propagated_particles.reserve(track.particles().size());
        for (const Particle& particle : track.particles()) {
            Particle new_particle = propagator_->propagate(particle, dt, previous_time, noise_scale);
            propagated_particles.push_back(new_particle);
        }
        // Replace the track's old particle cloud with the new one.
        track.set_particles(propagated_particles);
    }
    // No need to call set_tracks(), as we have modified the state directly.
}

void SMC_LMB_Tracker::update(const std::vector<Measurement>& measurements) {
    ensure_models_configured();

    for (const auto& measurement : measurements) {
        validation::require_measurement(measurement);
    }

    std::vector<Track>& tracks = current_state_.tracks();
    size_t num_tracks = tracks.size();
    size_t num_meas = measurements.size();

    // Handle the case of no existing tracks - just create new ones from all measurements
    if (num_tracks == 0) {
        std::vector<Track> born_tracks = birth_model_->generate_new_tracks(measurements, current_state_.timestamp());
        current_state_.tracks().swap(born_tracks);
        return;
    }

    // Handle the case of no measurements - nothing to update
    if (num_meas == 0) {
        return;
    }

    std::vector<MeasurementLikelihoodCache> meas_caches;
    meas_caches.reserve(num_meas);
    for (const auto& measurement : measurements) {
        meas_caches.push_back(InOrbitSensorModel::buildCache(measurement));
    }

    // Step 2: Compute normalized association weights for each track-measurement pair.
    //
    // Storage is one flat buffer rather than a nested vector-of-vector-of-vector, which removes
    // num_tracks * num_meas allocations per call. Particle counts are NOT assumed uniform across
    // tracks -- adaptive, confidence-driven cloud sizing is expected -- so a track's block is
    // located through a prefix-sum offset table instead of a fixed stride.
    track_particle_offsets_.assign(num_tracks + 1, 0);
    for (size_t i = 0; i < num_tracks; ++i) {
        track_particle_offsets_[i + 1] = track_particle_offsets_[i] + tracks[i].particles().size();
    }
    association_weights_.assign(track_particle_offsets_[num_tracks] * num_meas, 0.0);

    Eigen::MatrixXd likelihood_matrix(num_tracks, num_meas);

    for (size_t i = 0; i < num_tracks; ++i) {
        const auto& current_particles = tracks[i].particles();
        const size_t num_particles = current_particles.size();

        for (size_t j = 0; j < num_meas; ++j) {
            const auto& measurement = measurements[j];
            double* assoc = association_weights_.data() + association_block_offset(i, j, num_meas);

            double total_likelihood = 0.0;

            for (size_t p = 0; p < num_particles; ++p) {
                const auto& current_particle = current_particles[p];
                const double particle_likelihood = sensor_model_->calculate_likelihood(
                    current_particle, measurement, meas_caches[j]);
                const double updated_weight = current_particle.weight * (particle_likelihood / clutter_intensity_);
                assoc[p] = updated_weight;
                total_likelihood += updated_weight;
            }

            likelihood_matrix(i, j) = total_likelihood;

            if (total_likelihood > 1e-12) {
                const double inv_total = 1.0 / total_likelihood;
                for (size_t p = 0; p < num_particles; ++p) {
                    assoc[p] *= inv_total;
                }
            } else if (num_particles > 0) {
                const double uniform_weight = 1.0 / static_cast<double>(num_particles);
                for (size_t p = 0; p < num_particles; ++p) {
                    assoc[p] = uniform_weight;
                }
            }
        }
    }
    
    // Step 3: Build augmented cost matrix (N × (M+N)) per Reuter LMB formulation
    // Left block [0, M): detection costs -ln(P_D * L / κ)
    // Right block [M, M+N): missed detection costs, diagonal = -ln(1 - P_D), off-diag = 1e9
    size_t augmented_cols = num_meas + num_tracks;
    Eigen::MatrixXd cost_matrix(num_tracks, augmented_cols);
    
    double miss_cost = -std::log(std::max(1.0 - p_detection_, 1e-12));
    const double INF_COST = 1e9;  // Large cost for impossible assignments
    
    for (size_t i = 0; i < num_tracks; ++i) {
        // Detection costs (left block: columns 0 to num_meas-1)
        for (size_t j = 0; j < num_meas; ++j) {
            double likelihood = likelihood_matrix(i, j);
            // Cost = -ln(P_D * L / κ), where L already includes the weight averaging
            cost_matrix(i, j) = -std::log(std::max(p_detection_ * likelihood / clutter_intensity_, 1e-12));
        }
        
        // Missed detection costs (right block: columns num_meas to num_meas+num_tracks-1)
        for (size_t j = 0; j < num_tracks; ++j) {
            if (i == j) {
                // Diagonal: this track missed detection
                cost_matrix(i, num_meas + j) = miss_cost;
            } else {
                // Off-diagonal: impossible (each track has its own miss column)
                cost_matrix(i, num_meas + j) = INF_COST;
            }
        }
    }
    
    // Step 4: Solve assignment (K-best)
    std::vector<Hypothesis> hypotheses = solve_assignment(cost_matrix, k_best_);

    if (hypotheses.empty()) {
        return;
    }
    
    // Normalize hypothesis weights (log-sum-exp)
    std::vector<double> log_weights;
    for (const auto& h : hypotheses) log_weights.push_back(-h.weight);
    double max_logw = *std::max_element(log_weights.begin(), log_weights.end());
    std::vector<double> norm_weights;
    double sum_exp = 0.0;
    for (double lw : log_weights) sum_exp += std::exp(lw - max_logw);
    if (sum_exp <= 0.0) {
        return;
    }
    for (double lw : log_weights) norm_weights.push_back(std::exp(lw - max_logw) / sum_exp);
    
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

    // Step 5: Combine, update and resample.
    //
    // The hypothesis mixture is grouped by association rather than enumerated. Every hypothesis
    // that assigns track i to measurement j contributes the same per-particle vector
    // association_weights_[i][j] scaled by a different scalar, and every miss column carries an
    // identical formula, so at most num_meas + 1 distinct per-particle passes are needed no
    // matter how large k_best_ is. Exchanging the order of the double sum
    //     W[p] = sum_h e(h, p) = sum_j c[j] * nw[j][p] + c_miss * w[p]
    // is exact; only the floating-point summation order and the granularity of the resampling
    // list change, and the latter changes favorably (see the S <= 1e-12 note below and the
    // systematic-resampling argument: the stratification guarantee now applies to each particle
    // rather than to each of the k_best_ scattered fragments of that particle's mass).
    std::vector<Track> updated_tracks;
    updated_tracks.reserve(num_tracks);

    for (size_t i = 0; i < num_tracks; ++i) {
        const auto& predicted_particles = tracks[i].particles();
        const size_t num_particles = predicted_particles.size();

        // Step 5a: collapse the hypotheses onto their distinct associations. O(k_best_); no
        // particle is touched here.
        assoc_coefficients_.assign(num_meas, 0.0);
        assoc_used_.assign(num_meas, 0);
        double miss_coefficient = 0.0;
        bool miss_used = false;
        size_t contributing_hypotheses = 0;

        for (size_t h = 0; h < hypotheses.size(); ++h) {
            const int assoc_idx = hypotheses[h].associations[i];
            const double hyp_weight = norm_weights[h];

            if (assoc_idx >= 0 && static_cast<size_t>(assoc_idx) < num_meas) {
                assoc_coefficients_[static_cast<size_t>(assoc_idx)] += hyp_weight;
                assoc_used_[static_cast<size_t>(assoc_idx)] = 1;
                ++contributing_hypotheses;
            } else if (assoc_idx == -1 ||
                       (static_cast<size_t>(assoc_idx) >= num_meas &&
                        static_cast<size_t>(assoc_idx) < num_meas + num_tracks)) {
                // All miss columns in [num_meas, num_meas + num_tracks) share one formula, so
                // they collapse into a single bucket. One bucket per miss column would break the
                // num_meas + 1 bound for no benefit.
                miss_coefficient += hyp_weight;
                miss_used = true;
                ++contributing_hypotheses;
            }
            // Any other assoc_idx falls through without contributing, as it did when this loop
            // appended entries directly.
        }

        for (size_t j = 0; j < num_meas; ++j) {
            assoc_coefficients_[j] *= p_detection_ * likelihood_matrix(i, j);
        }
        miss_coefficient *= (1.0 - p_detection_);

        // Step 5b: one pass per used association. O(D * num_particles) with D <= num_meas + 1.
        // Measurements ascending, then the miss term, so the summation order is fixed run to run.
        mixture_weights_.assign(num_particles, 0.0);

        for (size_t j = 0; j < num_meas; ++j) {
            if (!assoc_used_[j]) {
                continue;
            }
            const double coefficient = assoc_coefficients_[j];
            const double* assoc = association_weights_.data() + association_block_offset(i, j, num_meas);
            for (size_t p = 0; p < num_particles; ++p) {
                mixture_weights_[p] += coefficient * assoc[p];
            }
        }

        if (miss_used) {
            for (size_t p = 0; p < num_particles; ++p) {
                mixture_weights_[p] += miss_coefficient * predicted_particles[p].weight;
            }
        }

        // Step 5c: totals, existence update and resampling, now over num_particles entries
        // instead of hypotheses.size() * num_particles.
        double sum_weights = 0.0;
        for (size_t p = 0; p < num_particles; ++p) {
            sum_weights += mixture_weights_[p];
        }

        const double r_legacy = tracks[i].existence_probability();
        const double r_new = (r_legacy * sum_weights) / (1.0 - r_legacy + r_legacy * sum_weights);

        // Counts hypotheses that took a branch, not buckets that ended up non-zero: a hypothesis
        // with weight exactly 0.0 still contributed, and previously produced a non-empty mixture.
        if (contributing_hypotheses == 0 || num_particles == 0) {
            Track final_track = tracks[i];
            final_track.set_existence_probability(r_new);
            updated_tracks.push_back(final_track);
            continue;
        }

        if (sum_weights > 1e-12) {
            const double inv_sum = 1.0 / sum_weights;
            for (size_t p = 0; p < num_particles; ++p) {
                mixture_weights_[p] *= inv_sum;
            }
        } else {
            // No information to go on. Equal slots make the systematic walk keep every particle
            // exactly once, which is the sensible reading of that. This is a deliberate behavior
            // change: spreading the same uniform weight across the old scattered list instead
            // produced a strided permutation with duplicates.
            const double uniform_weight = 1.0 / static_cast<double>(num_particles);
            for (size_t p = 0; p < num_particles; ++p) {
                mixture_weights_[p] = uniform_weight;
            }
        }

        std::vector<Particle> resampled_particles;
        resampled_particles.reserve(num_particles);

        const double u = unit_dist(resample_rng_) / static_cast<double>(num_particles);
        double cumsum = 0.0;
        size_t idx = 0;

        for (size_t p = 0; p < num_particles; ++p) {
            const double threshold = u + static_cast<double>(p) / static_cast<double>(num_particles);

            while (cumsum < threshold && idx < num_particles) {
                cumsum += mixture_weights_[idx];
                ++idx;
            }

            // The position in mixture_weights_ is the particle index, so no side table is needed.
            const size_t chosen_index = (idx > 0) ? idx - 1 : 0;

            Particle resampled;
            resampled.state_vector = predicted_particles[chosen_index].state_vector;
            resampled.weight = 1.0 / static_cast<double>(num_particles);
            resampled_particles.push_back(resampled);
        }

        Track final_track = tracks[i];
        final_track.set_existence_probability(r_new);
        final_track.set_particles(resampled_particles);
        updated_tracks.push_back(final_track);
    }
    
    // Track pruning: remove tracks with low existence probability
    std::vector<Track> pruned_tracks;
    for (const auto& track : updated_tracks) {
        if (track.existence_probability() >= prune_threshold_) {
            pruned_tracks.push_back(track);
        }
    }
    
    // Step 6: Adaptive Birth - Create new tracks from unused measurements
    std::vector<Track> born_tracks;
    
    if (!hypotheses.empty() && birth_model_) {
        std::vector<bool> measurement_used(num_meas, false);
        const Hypothesis& best_hypothesis = hypotheses[0];
        
        for (size_t track_idx = 0; track_idx < best_hypothesis.associations.size(); ++track_idx) {
            int meas_idx = best_hypothesis.associations[track_idx];
            if (meas_idx >= 0 && static_cast<size_t>(meas_idx) < num_meas) {
                measurement_used[meas_idx] = true;
            }
        }
        
        std::vector<Measurement> unused_measurements;
        unused_measurements.reserve(num_meas);
        for (size_t i = 0; i < num_meas; ++i) {
            if (!measurement_used[i]) {
                unused_measurements.push_back(measurements[i]);
            }
        }
        
        if (!unused_measurements.empty()) {
            born_tracks = birth_model_->generate_new_tracks(
                unused_measurements, 
                current_state_.timestamp());
        }
    }
    
    for (auto& new_track : born_tracks) {
        pruned_tracks.push_back(std::move(new_track));
    }
    
    current_state_.tracks().swap(pruned_tracks);
}

double SMC_LMB_Tracker::compute_association_likelihood(const Track& track, const Measurement& measurement) const {
    ensure_models_configured();
    validation::require_measurement(measurement);

    const MeasurementLikelihoodCache cache = InOrbitSensorModel::buildCache(measurement);
    double total_likelihood = 0.0;
    const auto& particles = track.particles();
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        double particle_likelihood = sensor_model_->calculate_likelihood(p, measurement, cache);
        double weighted_likelihood = particle_likelihood * p.weight;
        total_likelihood += weighted_likelihood;
    }
    return total_likelihood;
}

const std::vector<Track>& SMC_LMB_Tracker::get_tracks() const {
    return current_state_.tracks();
}

void SMC_LMB_Tracker::set_tracks(const std::vector<Track>& tracks) {
    current_state_.set_tracks(tracks);
}
