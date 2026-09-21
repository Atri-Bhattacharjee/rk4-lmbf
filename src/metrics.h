#pragma once
#include <cstddef>
#include <vector>
#include "datatypes.h"
#include <Eigen/Dense>

/**
 * @file metrics.h
 * @brief GOSPA (Generalized Optimal Sub-Pattern Assignment) tracking accuracy metric.
 *
 * Rahmathullah, Garcia-Fernandez & Svensson, "Generalized optimal sub-pattern assignment
 * metric", FUSION 2017.
 *
 *   GOSPA^p = min_gamma [ sum_{(i,j) in gamma} d(x_i, y_j)^p
 *                         + (c^p / alpha) (|X| - |gamma|)      // false tracks
 *                         + (c^p / alpha) (|Y| - |gamma|) ]    // missed truths
 *
 * where gamma ranges over partial assignments (each track matched to at most one truth and
 * vice versa). This implementation is unnormalised: there is no division by max(|X|,|Y|),
 * so the value grows as sqrt(k) with cardinality and is NOT bounded by the cutoff c. The
 * tight bound is c * sqrt((|X| + |Y|) / 2).
 *
 * alpha is fixed at 2. That is the only value for which the minimum decomposes exactly into
 * separate localisation, missed and false-track costs, which is the entire reason this metric was
 * chosen over the sub-pattern-assignment family. Do not parameterise it.
 *
 * The base distance d is the Euclidean norm over POSITION ONLY (the first three components,
 * metres). Including velocity would sum metres and metres-per-second under one square root.
 */

inline constexpr double kGospaDefaultCutoff = 10000.0;  //!< c, metres
inline constexpr double kGospaOrder         = 2.0;      //!< p
inline constexpr double kGospaAlpha         = 2.0;      //!< fixed; the decomposition requires it
inline constexpr int    kGospaPositionDim   = 3;        //!< base distance uses [x, y, z] only

/**
 * @brief GOSPA value together with its alpha=2 error decomposition.
 *
 * The three cost terms live in the p-th-power domain (m^p) and are exactly additive:
 *
 *     localisation_cost + missed_cost + false_cost == total^p
 *
 * The p-th roots are NOT additive, which is why the components are not themselves in metres.
 * That additivity is what makes the components meaningful to average over a Monte Carlo run
 * or to stack in a plot.
 */
struct GospaComponents {
    double localisation_cost = 0.0;  //!< sum of d^p over accepted pairs
    double missed_cost       = 0.0;  //!< (c^p / 2) * num_missed
    double false_cost        = 0.0;  //!< (c^p / 2) * num_false
    double total             = 0.0;  //!< metres: (localisation + missed + false)^(1/p)
    int num_assigned = 0;
    int num_missed   = 0;            //!< ground truths with no accepted pair
    int num_false    = 0;            //!< tracks with no accepted pair
    //! Track index -> ground-truth index under the optimal partial assignment, -1 if unassigned.
    std::vector<int> associations;
};

/**
 * @brief GOSPA with its localisation / missed / false decomposition.
 * @param tracks        Estimated tracks; each is reduced to its weighted particle mean.
 * @param ground_truths True states, each of size 6.
 * @param cutoff        c, metres. Must be finite and positive.
 * @throws std::invalid_argument if cutoff is non-finite or non-positive, or a ground truth
 *         is not a 6-vector.
 *
 * Degenerate cardinalities are not special-cased and fall out of the penalty terms:
 * both empty gives 0; m tracks against no truths gives c*sqrt(m/2); no tracks against
 * n truths gives c*sqrt(n/2).
 */
GospaComponents calculate_gospa_components(const std::vector<Track>& tracks,
                                           const std::vector<Eigen::VectorXd>& ground_truths,
                                           double cutoff = kGospaDefaultCutoff);

/**
 * @brief Scalar GOSPA in metres. Exactly calculate_gospa_components(...).total.
 */
double calculate_gospa_distance(const std::vector<Track>& tracks,
                                const std::vector<Eigen::VectorXd>& ground_truths,
                                double cutoff = kGospaDefaultCutoff);
