#pragma once

/**
 * @file particle_statistics.h
 * @brief Weighted summary statistics of a track's particle cloud.
 *
 * These were previously duplicated: a private `mean_state` inside metrics.cpp for the metric, and a
 * NumPy reimplementation inside each Python driver's `compute_track_mean`. The Python copy cost
 * about 13 ms per step for three 10,000-particle tracks, almost all of it pybind11 object churn.
 *
 * Note that there are deliberately *two* means here, because the two callers have always had
 * different contracts for a cloud that carries no usable weight. Collapsing them would silently
 * change one caller or the other, so both are kept and documented.
 */

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "datatypes.h"

//! 6x6 covariance of a state distribution, in the same ordering as StateVector.
using StateCovariance = Eigen::Matrix<double, 6, 6>;

namespace particle_stats {

/**
 * @brief Total weight at or below which a cloud is treated as carrying no information.
 *
 * Matches the threshold the Python drivers have always used in `compute_track_mean`.
 */
inline constexpr double kWeightSumFloor = 1e-12;

namespace detail {

struct WeightedAccumulation {
    StateVector weighted_sum;
    double total_weight;
};

/**
 * @brief Accumulate sum(x_p * w_p) and sum(w_p) in particle order.
 *
 * The iteration order and the shape of the accumulation are load-bearing: calculate_gospa_distance
 * is gated on producing bitwise-identical results, so this must stay exactly the loop that used to
 * live in metrics.cpp.
 */
inline WeightedAccumulation accumulate(const std::vector<Particle>& particles) {
    WeightedAccumulation accumulation{StateVector::Zero(), 0.0};
    for (const auto& particle : particles) {
        accumulation.weighted_sum += particle.state_vector * particle.weight;
        accumulation.total_weight += particle.weight;
    }
    return accumulation;
}

inline StateVector unweighted_mean(const std::vector<Particle>& particles) {
    StateVector sum = StateVector::Zero();
    for (const auto& particle : particles) {
        sum += particle.state_vector;
    }
    return sum / static_cast<double>(particles.size());
}

}  // namespace detail

/**
 * @brief Sum of the particle weights.
 */
inline double weight_sum(const Track& track) {
    return detail::accumulate(track.particles()).total_weight;
}

/**
 * @brief Weighted mean using the metric contract: the zero vector when the cloud is empty or
 *        carries exactly zero total weight.
 *
 * calculate_gospa_distance depends on this, so the arithmetic must not change. It reads only the
 * first three components, but the full six-component contract stays gated by
 * tests/test_particle_statistics.py and must not be weakened on that basis.
 */
inline StateVector weighted_mean(const Track& track) {
    const std::vector<Particle>& particles = track.particles();
    if (particles.empty()) {
        return StateVector::Zero();
    }
    const detail::WeightedAccumulation accumulation = detail::accumulate(particles);
    if (accumulation.total_weight == 0.0) {
        return StateVector::Zero();
    }
    return accumulation.weighted_sum / accumulation.total_weight;
}

/**
 * @brief Weighted mean using the driver contract: falls back to the *unweighted* mean when the
 *        total weight is at or below kWeightSumFloor, rather than returning zeros.
 *
 * This is what run_once.compute_track_mean has always done. The two contracts only diverge for a
 * cloud with no usable weight, which resampling makes unreachable in practice (weights come out of
 * update() as exactly 1/N), but the difference is real and both callers depend on their own.
 */
inline StateVector mean_state(const Track& track) {
    const std::vector<Particle>& particles = track.particles();
    if (particles.empty()) {
        return StateVector::Zero();
    }
    const detail::WeightedAccumulation accumulation = detail::accumulate(particles);
    if (accumulation.total_weight > kWeightSumFloor) {
        return accumulation.weighted_sum / accumulation.total_weight;
    }
    return detail::unweighted_mean(particles);
}

/**
 * @brief Weighted covariance about mean_state, normalized by the total weight.
 *
 * Uses the same weight-floor fallback as mean_state, so an information-free cloud reports its
 * unweighted spread instead of a covariance divided by a near-zero number.
 */
inline StateCovariance covariance(const Track& track) {
    const std::vector<Particle>& particles = track.particles();
    StateCovariance result = StateCovariance::Zero();
    if (particles.empty()) {
        return result;
    }

    const detail::WeightedAccumulation accumulation = detail::accumulate(particles);
    const bool weights_are_usable = accumulation.total_weight > kWeightSumFloor;
    const StateVector mean = weights_are_usable
        ? StateVector(accumulation.weighted_sum / accumulation.total_weight)
        : detail::unweighted_mean(particles);
    const double normalizer = weights_are_usable
        ? accumulation.total_weight
        : static_cast<double>(particles.size());

    for (const auto& particle : particles) {
        const StateVector deviation = particle.state_vector - mean;
        const double weight = weights_are_usable ? particle.weight : 1.0;
        result += (weight / normalizer) * (deviation * deviation.transpose());
    }
    return result;
}

}  // namespace particle_stats
