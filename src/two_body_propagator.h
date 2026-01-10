#pragma once

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>

class TwoBodyPropagator : public IOrbitPropagator {
public:
    explicit TwoBodyPropagator(const Eigen::MatrixXd& process_noise_covariance);
    ~TwoBodyPropagator() override = default;
    Particle propagate(const Particle& particle, double dt, double current_time, double noise_scale = 1.0) const override;
private:
    Eigen::MatrixXd process_noise_covariance_;
};
