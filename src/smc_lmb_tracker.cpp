#include "smc_lmb_tracker.h"
#include "assignment.h"
#include <iostream>

SMC_LMB_Tracker::SMC_LMB_Tracker(std::shared_ptr<IOrbitPropagator> propagator,
                                                                 std::shared_ptr<ISensorModel> sensor_model,
                                                                 std::shared_ptr<IBirthModel> birth_model,
                                                                 double survival_probability,
                                                                 int k_best,
                                                                 double prune_threshold,
                                                                 double clutter_intensity)
        : current_state_(0.0, std::vector<Track>{}),
            propagator_(std::move(propagator)),
            sensor_model_(std::move(sensor_model)),
            birth_model_(std::move(birth_model)),
            survival_probability_(survival_probability),
            k_best_(k_best),
            prune_threshold_(prune_threshold),
            clutter_intensity_(clutter_intensity) {
}

void SMC_LMB_Tracker::predict(double dt) {
    // Projects the filter state forward in time by operating in-place.
    const double previous_time = current_state_.timestamp();
    current_state_.set_timestamp(current_state_.timestamp() + dt);
    // Get a direct, MODIFIABLE reference to the internal track vector.
    std::vector<Track>& tracks = current_state_.tracks();
    for (Track& track : tracks) {
        // Update existence probability in-place.
        track.set_existence_probability(track.existence_probability() * survival_probability_);
        // Propagate particles.
        std::vector<Particle> propagated_particles;
        propagated_particles.reserve(track.particles().size());
        for (const Particle& particle : track.particles()) {
            Particle new_particle = propagator_->propagate(particle, dt, previous_time);
            propagated_particles.push_back(new_particle);
        }
        // Replace the track's old particle cloud with the new one.
        track.set_particles(propagated_particles);
    }
    // No need to call set_tracks(), as we have modified the state directly.
}

void SMC_LMB_Tracker::update(const std::vector<Measurement>& measurements) {
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

    // Step 2: Create all hypothetical updated particle sets for each track-measurement pair
    // This 3D vector stores all hypothetical particle sets: [track_idx][meas_idx][particle_idx]
    std::vector<std::vector<std::vector<Particle>>> hypothetical_particle_sets(num_tracks);
    Eigen::MatrixXd likelihood_matrix(num_tracks, num_meas);

    // For each track
    for (size_t i = 0; i < num_tracks; ++i) {
        const auto& current_particles = tracks[i].particles();
        size_t num_particles = current_particles.size();

        // Initialize storage for this track's hypothetical sets
        hypothetical_particle_sets[i].resize(num_meas);
        
        // For each measurement, create a hypothetically updated set
        for (size_t j = 0; j < num_meas; ++j) {
            const auto& measurement = measurements[j];
            hypothetical_particle_sets[i][j].reserve(num_particles);
            
            double total_likelihood = 0.0;

            // For each particle
            for (size_t p = 0; p < num_particles; ++p) {
                const auto& current_particle = current_particles[p];
                
                // Create an updated particle (Bootstrap/SIR update)
                Particle updated_particle;
                
                updated_particle.state_vector = current_particle.state_vector;
                
                // Calculate likelihood ratio (likelihood / clutter intensity) and update weight
                double particle_likelihood = sensor_model_->calculate_likelihood(current_particle, measurement);
                double likelihood_ratio = particle_likelihood / clutter_intensity_;
                updated_particle.weight = current_particle.weight * likelihood_ratio;
                total_likelihood += updated_particle.weight;
                
                // Add to hypothetical set
                hypothetical_particle_sets[i][j].push_back(updated_particle);
            }
            
            // Store the total likelihood for this track-measurement association
            likelihood_matrix(i, j) = total_likelihood;
            
            // Normalize weights within this hypothetical set
            if (total_likelihood > 1e-12) {
                for (auto& particle : hypothetical_particle_sets[i][j]) {
                    particle.weight /= total_likelihood;
                }
            } else {
                // Handle the case of zero likelihood
                for (auto& particle : hypothetical_particle_sets[i][j]) {
                    particle.weight = 1.0 / num_particles;
                }
            }
        }
    }
    
    // Step 3: Build simplified cost matrix using the likelihoods from hypothetical updates
    Eigen::MatrixXd cost_matrix(num_tracks, num_meas);
    for (size_t i = 0; i < num_tracks; ++i) {
        double track_r = tracks[i].existence_probability();
        
        // Association costs - simplified without detection probability
        for (size_t j = 0; j < num_meas; ++j) {
            double likelihood = likelihood_matrix(i, j);
            cost_matrix(i, j) = -std::log(std::max(track_r * likelihood, 1e-12));
        }
    }
    
    // Step 4: Solve assignment (K-best)
    std::vector<Hypothesis> hypotheses = solve_assignment(cost_matrix, k_best_);
    
    // Normalize hypothesis weights (log-sum-exp)
    std::vector<double> log_weights;
    for (const auto& h : hypotheses) log_weights.push_back(-h.weight);
    double max_logw = *std::max_element(log_weights.begin(), log_weights.end());
    std::vector<double> norm_weights;
    double sum_exp = 0.0;
    for (double lw : log_weights) sum_exp += std::exp(lw - max_logw);
    for (double lw : log_weights) norm_weights.push_back(std::exp(lw - max_logw) / sum_exp);
    
    // Step 5: Combine, update and resample
    std::vector<Track> updated_tracks;
    for (size_t i = 0; i < num_tracks; ++i) {
        // Create a large mixture of particles from all hypothetical sets
        std::vector<Particle> mixed_particles;
            // For each hypothesis
            for (size_t h = 0; h < hypotheses.size(); ++h) {
                int assoc_idx = hypotheses[h].associations[i];
                double hyp_weight = norm_weights[h];
                // Only consider valid measurement associations
                if (assoc_idx >= 0 && assoc_idx < num_meas) {
                    // Add these particles to the mixture with appropriate weights
                    for (const auto& particle : hypothetical_particle_sets[i][assoc_idx]) {
                        Particle mixed_particle;
                        mixed_particle.state_vector = particle.state_vector;
                        mixed_particle.weight = particle.weight * hyp_weight;
                        mixed_particles.push_back(mixed_particle);
                    }
                }
            }

            // --- Corrected existence probability calculation ---
            // c. Calculate the new existence probability. It is the sum of all
            // unnormalized weights in the mixed particle cloud.
            double new_existence_probability = 0.0;
            for (const auto& p : mixed_particles) {
                new_existence_probability += p.weight;
            }

            // d. Normalize the particle weights in the mixed cloud.
            if (new_existence_probability > 1e-12) {
                for (auto& p : mixed_particles) {
                    p.weight /= new_existence_probability;
                }
            } else {
                // Handle case of zero total weight to prevent division by zero
                for (auto& p : mixed_particles) {
                    if (!mixed_particles.empty()) {
                        p.weight = 1.0 / mixed_particles.size();
                    }
                }
            }

            // Perform systematic resampling to get back to the standard number of particles
            size_t num_particles = tracks[i].particles().size();
            std::vector<Particle> resampled_particles;
            resampled_particles.reserve(num_particles);

            double u = ((double)rand() / RAND_MAX) / num_particles;
            double cumsum = 0.0;
            size_t idx = 0;

            for (size_t p = 0; p < num_particles; ++p) {
                double threshold = u + (double)p / num_particles;

                while (cumsum < threshold && idx < mixed_particles.size()) {
                    cumsum += mixed_particles[idx].weight;
                    ++idx;
                }

                // Get the chosen particle
                const Particle& chosen_particle = (idx > 0) ? mixed_particles[idx-1] : mixed_particles[0];

                // Create the resampled particle with equal weight
                Particle resampled;
                resampled.state_vector = chosen_particle.state_vector;
                resampled.weight = 1.0 / num_particles;
                resampled_particles.push_back(resampled);
            }

            // Create final track
            Track final_track = tracks[i];
            final_track.set_existence_probability(new_existence_probability);
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
    // 
    // Algorithm:
    // 1. Identify which measurements were used by the best hypothesis
    // 2. Collect measurements that were NOT used (unassociated)
    // 3. Pass unused measurements to birth model to create new track hypotheses
    
    std::vector<Track> born_tracks;
    
    if (!hypotheses.empty() && birth_model_) {
        // Step 6a: Create a flag vector to track which measurements were used
        std::vector<bool> measurement_used(num_meas, false);
        
        // Step 6b: The best hypothesis is at index 0 (solve_assignment returns sorted by cost)
        const Hypothesis& best_hypothesis = hypotheses[0];
        
        // Step 6c: Mark measurements that are associated with tracks in the best hypothesis
        for (size_t track_idx = 0; track_idx < best_hypothesis.associations.size(); ++track_idx) {
            int meas_idx = best_hypothesis.associations[track_idx];
            // Valid measurement index means this measurement was used
            if (meas_idx >= 0 && static_cast<size_t>(meas_idx) < num_meas) {
                measurement_used[meas_idx] = true;
            }
        }
        
        // Step 6d: Collect unused measurements
        std::vector<Measurement> unused_measurements;
        unused_measurements.reserve(num_meas);
        for (size_t i = 0; i < num_meas; ++i) {
            if (!measurement_used[i]) {
                unused_measurements.push_back(measurements[i]);
            }
        }
        
        // Step 6e: Generate new tracks from unused measurements
        if (!unused_measurements.empty()) {
            born_tracks = birth_model_->generate_new_tracks(
                unused_measurements, 
                current_state_.timestamp());
        }
    }
    
    // Combine pruned existing tracks with newly born tracks
    for (auto& new_track : born_tracks) {
        pruned_tracks.push_back(std::move(new_track));
    }
    
    // Update the current state with the combined tracks
    current_state_.tracks().swap(pruned_tracks);
}

// Helper: average likelihood over all particles

double SMC_LMB_Tracker::compute_association_likelihood(const Track& track, const Measurement& measurement) const {
    double total_likelihood = 0.0;
    const auto& particles = track.particles();
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        double particle_likelihood = sensor_model_->calculate_likelihood(p, measurement);
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