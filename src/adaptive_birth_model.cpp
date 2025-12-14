#include "adaptive_birth_model.h"
#include <random>
#include <vector>
#include <cmath>

AdaptiveBirthModel::AdaptiveBirthModel(int particles_per_track, 
                                     double initial_existence_probability, 
                                     const Eigen::MatrixXd& initial_covariance)
    : particles_per_track_(particles_per_track),
      initial_existence_probability_(initial_existence_probability),
      initial_covariance_(initial_covariance) {
}

double AdaptiveBirthModel::computeCircularVelocity(double radius) const {
    // Clamp radius to minimum safe altitude (100km above Earth surface)
    // to avoid numerical issues for invalid measurements
    double safe_radius = std::max(radius, R_EARTH + 100.0e3);
    return std::sqrt(MU_EARTH / safe_radius);
}

std::vector<Track> AdaptiveBirthModel::generate_new_tracks(const std::vector<Measurement>& unused_measurements, double current_time) const {
    // Initialize empty vector for new tracks
    std::vector<Track> new_tracks;
    
    // Initialize random number generators
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::normal_distribution<double> std_normal(0.0, 1.0);
    
    // Pre-calculate Cholesky decomposition of the initial covariance matrix
    // This is used to add correlated Gaussian noise to capture eccentricity variation
    Eigen::LLT<Eigen::MatrixXd> llt(initial_covariance_);
    Eigen::MatrixXd L = llt.matrixL();
    
    // Loop through each unused measurement
    for (size_t measurement_idx = 0; measurement_idx < unused_measurements.size(); ++measurement_idx) {
        const auto& measurement = unused_measurements[measurement_idx];
        
        // =================================================================
        // Step 1: Extract measurement components
        // Measurement format: [Range, RangeRate, Azimuth, Elevation]
        // =================================================================
        double range = measurement.value_(0);
        double range_rate = measurement.value_(1);
        double azimuth = measurement.value_(2);
        double elevation = measurement.value_(3);
        
        // Sensor position (first 3 components of sensor_state)
        Eigen::Vector3d sensor_pos = measurement.sensor_state_.head<3>();
        
        // =================================================================
        // Step 2: Compute radial unit vector (Line-of-Sight direction)
        // This points from the sensor toward the target
        // =================================================================
        Eigen::Vector3d u_radial;
        u_radial << std::cos(elevation) * std::cos(azimuth),
                    std::cos(elevation) * std::sin(azimuth),
                    std::sin(elevation);
        
        // =================================================================
        // Step 3: Compute target position and altitude
        // =================================================================
        Eigen::Vector3d base_position = sensor_pos + range * u_radial;
        double target_radius = base_position.norm();
        
        // =================================================================
        // Step 4: Compute circular orbital velocity at this altitude
        // v_circular = sqrt(mu / r)
        // =================================================================
        double v_circular = computeCircularVelocity(target_radius);
        
        // =================================================================
        // Step 5: Compute radial velocity (fixed from range-rate measurement)
        // This is the component we KNOW from the sensor
        // =================================================================
        Eigen::Vector3d v_radial = range_rate * u_radial;
        
        // =================================================================
        // Step 6: Compute two orthogonal tangent vectors
        // These span the plane perpendicular to line-of-sight
        // u_tangent1: Azimuth direction ("East")
        // u_tangent2: Elevation direction ("North")
        // =================================================================
        Eigen::Vector3d u_tangent1, u_tangent2;
        
        // Handle edge case: target directly overhead (elevation ≈ ±90°)
        if (std::abs(std::cos(elevation)) < 1e-9) {
            // Degenerate case: pick arbitrary orthogonal basis
            u_tangent1 << 1.0, 0.0, 0.0;
            u_tangent2 << 0.0, 1.0, 0.0;
        } else {
            // Standard case:
            // u_tangent1: perpendicular to u_radial in horizontal plane
            u_tangent1 << -std::sin(azimuth),
                           std::cos(azimuth),
                           0.0;
            
            // u_tangent2: completes the right-hand orthonormal system
            u_tangent2 = u_radial.cross(u_tangent1);
            u_tangent2.normalize();
        }
        
        // =================================================================
        // Step 7: Create track label
        // =================================================================
        TrackLabel label;
        label.birth_time = static_cast<uint64_t>(current_time);
        label.index = static_cast<uint32_t>(measurement_idx);
        
        // =================================================================
        // Step 8: Generate particles with tangent fan velocity distribution
        // =================================================================
        std::vector<Particle> particles;
        particles.reserve(particles_per_track_);
        
        for (int particle_idx = 0; particle_idx < particles_per_track_; ++particle_idx) {
            // Sample random direction in tangent plane (circular fan)
            // theta is uniformly distributed in [0, 2π)
            double theta = angle_dist(gen);
            
            // Tangent velocity: fixed magnitude (circular velocity), random direction
            Eigen::Vector3d v_tangent = v_circular * 
                (std::cos(theta) * u_tangent1 + std::sin(theta) * u_tangent2);
            
            // Assemble base state: position + (radial velocity + tangent velocity)
            Eigen::VectorXd base_state(6);
            base_state.head<3>() = base_position;
            base_state.tail<3>() = v_radial + v_tangent;
            
            // Generate 6D standard normal vector for correlated noise
            Eigen::VectorXd noise(6);
            for (int i = 0; i < 6; ++i) {
                noise(i) = std_normal(gen);
            }
            
            // Apply Cholesky factor to get correlated noise from birth covariance
            // This captures eccentricity variation and measurement uncertainty
            Eigen::VectorXd state = base_state + L * noise;
            
            // Create particle with uniform weight
            Particle particle;
            particle.state_vector = state;
            particle.weight = 1.0 / static_cast<double>(particles_per_track_);
            particles.push_back(particle);
        }
        
        // =================================================================
        // Step 9: Create the Track with label, existence probability, and particles
        // =================================================================
        Track track(label, initial_existence_probability_, particles);
        new_tracks.push_back(track);
    }
    
    return new_tracks;
}