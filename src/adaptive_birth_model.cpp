#include "adaptive_birth_model.h"
#include "validation.h"
#include <random>
#include <vector>
#include <stdexcept>

// Define _USE_MATH_DEFINES before cmath to get M_PI on MSVC
#define _USE_MATH_DEFINES
#include <cmath>

// Fallback definition if M_PI is still not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

AdaptiveBirthModel::AdaptiveBirthModel(int particles_per_track, 
                                     double initial_existence_probability, 
                                     const Eigen::MatrixXd& initial_covariance)
    : particles_per_track_(particles_per_track),
      initial_existence_probability_(initial_existence_probability),
      initial_covariance_(initial_covariance) {
    validation::require_particles_per_track(particles_per_track_);
    validation::require_positive_definite(initial_covariance_, "initial_covariance");

    Eigen::LLT<ProcessNoiseCov> llt(initial_covariance_);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("AdaptiveBirthModel: initial_covariance is not positive definite");
    }
    birth_noise_L_ = llt.matrixL();
}

double AdaptiveBirthModel::computeCircularVelocity(double radius) const {
    // Clamp radius to minimum safe altitude (100km above Earth surface)
    // to avoid numerical issues for invalid measurements
    double safe_radius = std::max(radius, R_EARTH + 100.0e3);
    return std::sqrt(MU_EARTH / safe_radius);
}

std::vector<Track> AdaptiveBirthModel::generate_new_tracks(const std::vector<Measurement>& unused_measurements, double current_time) const {
    std::vector<Track> new_tracks;
    
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> std_normal(0.0, 1.0);
    
    for (size_t measurement_idx = 0; measurement_idx < unused_measurements.size(); ++measurement_idx) {
        const auto& measurement = unused_measurements[measurement_idx];
        validation::require_measurement(measurement);
        
        double range = measurement.value_(0);
        double range_rate = measurement.value_(1);
        double azimuth = measurement.value_(2);
        double elevation = measurement.value_(3);
        
        Eigen::Vector3d sensor_pos = measurement.sensor_state_.head<3>();
        
        Eigen::Vector3d u_radial;
        u_radial << std::cos(elevation) * std::cos(azimuth),
                    std::cos(elevation) * std::sin(azimuth),
                    std::sin(elevation);
        
        Eigen::Vector3d base_position = sensor_pos + range * u_radial;
        double target_radius = base_position.norm();
        double v_circular = computeCircularVelocity(target_radius);
        Eigen::Vector3d v_radial = range_rate * u_radial;
        
        Eigen::Vector3d u_tangent1, u_tangent2;
        
        if (std::abs(std::cos(elevation)) < 1e-9) {
            u_tangent1 << 1.0, 0.0, 0.0;
            u_tangent2 << 0.0, 1.0, 0.0;
        } else {
            u_tangent1 << -std::sin(azimuth),
                           std::cos(azimuth),
                           0.0;
            u_tangent2 = u_radial.cross(u_tangent1);
            u_tangent2.normalize();
        }
        
        TrackLabel label;
        label.birth_time = static_cast<uint64_t>(current_time);
        label.index = static_cast<uint32_t>(measurement_idx);
        
        std::vector<Particle> particles;
        particles.reserve(particles_per_track_);
        
        for (int particle_idx = 0; particle_idx < particles_per_track_; ++particle_idx) {
            double theta = (2.0 * M_PI * particle_idx) / particles_per_track_;
            
            Eigen::Vector3d v_tangent = v_circular * 
                (std::cos(theta) * u_tangent1 + std::sin(theta) * u_tangent2);
            
            StateVector base_state;
            base_state.head<3>() = base_position;
            base_state.tail<3>() = v_radial + v_tangent;
            
            StateVector noise;
            for (int i = 0; i < 6; ++i) {
                noise(i) = std_normal(gen);
            }
            
            Particle particle;
            particle.state_vector = base_state + birth_noise_L_ * noise;
            particle.weight = 1.0 / static_cast<double>(particles_per_track_);
            particles.push_back(particle);
        }
        
        Track track(label, initial_existence_probability_, particles);
        new_tracks.push_back(track);
    }
    
    return new_tracks;
}
