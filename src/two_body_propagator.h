#pragma once

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <random>

/**
 * @brief RK4 two-body orbit propagator with optional additive Gaussian process noise.
 *
 * The noise stream is owned by the instance. Passing a seed makes a propagator's output a
 * deterministic function of its inputs, which is what the golden-digest harness relies on;
 * an unseeded propagator draws its seed from std::random_device as before.
 */
class TwoBodyPropagator : public IOrbitPropagator {
public:
    explicit TwoBodyPropagator(const Eigen::MatrixXd& process_noise_covariance,
                               std::optional<uint64_t> seed = std::nullopt);
    ~TwoBodyPropagator() override = default;
    Particle propagate(const Particle& particle, double dt, double current_time, double noise_scale = 1.0) const override;
private:
    ProcessNoiseCov process_noise_covariance_;
    ProcessNoiseCov noise_L_;
    bool has_process_noise_ = false;
    mutable std::mt19937_64 rng_;  //!< Seeded from the ctor argument or std::random_device
};
