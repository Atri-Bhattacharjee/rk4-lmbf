#include "adaptive_birth_model.h"
#include "los_geometry.h"
#include "validation.h"
#include <random>
#include <vector>

namespace {

constexpr int kMaxRangeRedraws = 100;

}  // namespace

AdaptiveBirthModel::AdaptiveBirthModel(int particles_per_track,
                                       double initial_existence_probability,
                                       const Eigen::MatrixXd& birth_covariance_local,
                                       std::optional<uint64_t> seed)
    : particles_per_track_(particles_per_track),
      initial_existence_probability_(initial_existence_probability),
      birth_covariance_local_(MeasCovariance::Zero()),
      birth_noise_L_(MeasCovariance::Zero()),
      rng_(seed.has_value() ? *seed : std::mt19937_64::result_type(std::random_device{}())) {
    validation::require_particles_per_track(particles_per_track_);
    validation::require_covariance_6x6_positive_definite(birth_covariance_local,
                                                         "birth_covariance (local tangent frame)");
    birth_covariance_local_ = birth_covariance_local;
    Eigen::LLT<MeasCovariance> llt(birth_covariance_local_);
    birth_noise_L_ = llt.matrixL();
}

std::vector<Track> AdaptiveBirthModel::generate_new_tracks(const std::vector<Measurement>& unused_measurements,
                                                           double current_time) const {
    std::vector<Track> new_tracks;
    new_tracks.reserve(unused_measurements.size());
    std::normal_distribution<double> std_normal(0.0, 1.0);
    const double weight = 1.0 / static_cast<double>(particles_per_track_);

    for (size_t measurement_idx = 0; measurement_idx < unused_measurements.size(); ++measurement_idx) {
        const Measurement& measurement = unused_measurements[measurement_idx];
        LMB_VALIDATION_ONLY(validation::require_measurement(measurement));

        const los::LosObservation observed = measurement.observation();

        TrackLabel label;
        label.birth_time = static_cast<uint64_t>(current_time);
        label.index = static_cast<uint32_t>(measurement_idx);

        std::vector<Particle> particles;
        particles.reserve(particles_per_track_);

        for (int particle_idx = 0; particle_idx < particles_per_track_; ++particle_idx) {
            los::LosObservation sample;
            for (int attempt = 0; attempt <= kMaxRangeRedraws; ++attempt) {
                LocalMeasVector xi;
                for (int i = 0; i < 6; ++i) {
                    xi(i) = std_normal(rng_);
                }
                const LocalMeasVector eps = birth_noise_L_ * xi;
                sample = los::perturbed(observed, eps);
                if (sample.range > validation::RANGE_EPSILON) {
                    break;
                }
            }
            if (sample.range <= validation::RANGE_EPSILON) {
                sample.range = validation::RANGE_EPSILON;
            }

            Particle particle;
            particle.state_vector = los::toCartesian(sample, measurement.sensor_state_);
            particle.weight = weight;
            particles.push_back(particle);
        }

        new_tracks.emplace_back(label, initial_existence_probability_, particles);
    }

    return new_tracks;
}
