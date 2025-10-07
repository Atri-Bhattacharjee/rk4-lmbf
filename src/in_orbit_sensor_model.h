#pragma once

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>

class InOrbitSensorModel : public ISensorModel {
public:
    InOrbitSensorModel();
    InOrbitSensorModel(double range_var, double range_rate_var, double azimuth_var, double elevation_var);
    
    double calculate_likelihood(const Particle& particle, const Measurement& measurement) const override;
    
    /**
     * @brief Convert particle state to measurement space
     * 
     * @param particle The particle with state to convert
     * @param sensor_state The sensor state to use as reference
     * @return Eigen::VectorXd The measurement space representation [range, range_rate, azimuth, elevation]
     */
    Eigen::VectorXd convertParticleToMeasurement(const Particle& particle, const Eigen::VectorXd& sensor_state) const;
    
private:
    // Default measurement noise parameters
    double range_var_;         // Range variance (m^2)
    double range_rate_var_;    // Range rate variance ((m/s)^2)
    double azimuth_var_;       // Azimuth variance (rad^2)
    double elevation_var_;     // Elevation variance (rad^2)
};
