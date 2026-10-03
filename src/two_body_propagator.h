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
 *
 * Two process-noise models, chosen at construction:
 *
 * - Per call (noise_reference_dt omitted, the historical behaviour). process_noise_covariance is
 *   added in full on every propagate() call, whatever dt is. Correct only for a fixed step: a
 *   driver that halves dt doubles the noise it injects per unit time.
 *
 * - Time-consistent (noise_reference_dt = T). process_noise_covariance is read as the covariance a
 *   state accumulates over T seconds of continuous white noise, i.e. a diffusion D = Q / T on the
 *   state [r, v] with kinematic coupling r' = v. A step of dt then draws from
 *
 *       Qd(dt) = [ Dpp dt + (Dpv + Dvp) dt^2/2 + Dvv dt^3/3 ,  Dpv dt + Dvv dt^2/2 ]
 *                [ Dvp dt + Dvv dt^2/2                     ,  Dvv dt              ]
 *
 *   which is the exact covariance of that SDE, so sixty 1 s steps and one 60 s step inject the same
 *   noise up to the gravity-gradient coupling the kinematic model leaves out. Note what that means
 *   for tuning: velocity noise now also diffuses into position within a step (the Dvv dt^3/3 term),
 *   which the per-call model never did, so a Q tuned for the per-call model at dt = T spreads a
 *   cloud faster under this one. Lazy propagation in SMC_LMB_Tracker requires this model, because
 *   it propagates the same track with different step lengths.
 */
/**
 * @brief One noise-free RK4 two-body step, in place: exactly the deterministic part of
 *        TwoBodyPropagator::propagate (same derivative, same arithmetic).
 */
void two_body_rk4_step(StateVector& state, double dt);

/**
 * @brief two_body_rk4_step applied to `count` states in place, bit-for-bit the same per state, but
 *        stepping several at once. State i is the six doubles at first + i * stride.
 */
void two_body_rk4_step_strided(double* first, size_t stride, size_t count, double dt);

class TwoBodyPropagator : public IOrbitPropagator {
public:
    explicit TwoBodyPropagator(const Eigen::MatrixXd& process_noise_covariance,
                               std::optional<uint64_t> seed = std::nullopt,
                               std::optional<double> noise_reference_dt = std::nullopt);
    ~TwoBodyPropagator() override = default;
    Particle propagate(const Particle& particle, double dt, double current_time, double noise_scale = 1.0) const override;

    std::optional<double> noise_displacement_bound(double interval) const override;

    bool supports_keyed_propagation() const override { return true; }
    //! Same RK4 step as propagate(), applied in place to every particle, with each particle's noise
    //! drawn from fast_random::Stream(combine(stream_key, index)) through a ziggurat sampler.
    void propagate_cloud_keyed(std::vector<Particle>& particles, double dt, double current_time,
                               double noise_scale, uint64_t stream_key) const override;

    //! Reference interval of the time-consistent model, or nullopt for the per-call model.
    std::optional<double> noise_reference_dt() const { return noise_reference_dt_; }

    //! Covariance of the noise one propagate(dt) call adds at noise_scale = 1. Per-call model: the
    //! configured matrix whatever dt is. Time-consistent model: Qd(|dt|) above.
    ProcessNoiseCov step_noise_covariance(double dt) const;

private:
    ProcessNoiseCov process_noise_covariance_;
    ProcessNoiseCov noise_L_;
    bool has_process_noise_ = false;
    mutable std::mt19937_64 rng_;  //!< Seeded from the ctor argument or std::random_device
    // Kept on the instance so the distribution object is not reconstructed every propagate().
    // libstdc++ Box-Muller caches a spare Normal; with exactly six draws per call the spare is
    // empty at return, so the stream matches constructing a fresh distribution each time on this
    // toolchain. Treat any future change to the draw count as a stream change and re-gate.
    mutable std::normal_distribution<> unit_normal_{0.0, 1.0};
    // Time-consistent model only; declared after the hot per-call members above so their layout
    // is the historical one.
    std::optional<double> noise_reference_dt_;
    ProcessNoiseCov diffusion_ = ProcessNoiseCov::Zero();  //!< Q / T
    // Cholesky factor of Qd(dt) for the most recent dt. A driver alternates between very few step
    // lengths (the fine step and the lazy substep), and consecutive particles of one cloud share
    // one, so a single-entry cache hits almost always and keeps the factorisation off the
    // per-particle path.
    mutable double cached_dt_ = -1.0;
    mutable ProcessNoiseCov cached_L_ = ProcessNoiseCov::Zero();

    const ProcessNoiseCov& step_noise_factor(double dt) const;
    void add_time_consistent_noise(StateVector& state, double dt, double noise_scale) const;
};
