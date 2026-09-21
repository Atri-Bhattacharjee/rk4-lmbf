#include "metrics.h"
#include "assignment.h"
#include "particle_statistics.h"
#include "validation.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

static_assert(kGospaOrder == 2.0, "the p = 2 fast path below must be generalised first");
static_assert(kGospaAlpha == 2.0, "the drop rule and the c^p/2 penalties assume alpha = 2");

GospaComponents calculate_gospa_components(const std::vector<Track>& tracks,
                                           const std::vector<Eigen::VectorXd>& ground_truths,
                                           double cutoff) {
    if (!(std::isfinite(cutoff) && cutoff > 0.0)) {
        throw std::invalid_argument(
            "calculate_gospa_components: cutoff must be finite and positive, got " +
            std::to_string(cutoff));
    }

    const std::size_t m = tracks.size();
    const std::size_t n = ground_truths.size();

    // Release sets EIGEN_NO_DEBUG (CMakeLists.txt), so .head<3>() on a short vector is undefined
    // behaviour rather than an assert. This check has to be explicit and unconditional.
    for (const auto& truth : ground_truths) {
        validation::require_state_vector(truth, "calculate_gospa_components: ground_truth");
    }

    std::vector<StateVector> track_means;
    track_means.reserve(m);
    for (const auto& track : tracks) {
        track_means.push_back(particle_stats::weighted_mean(track));
    }

    const double c2 = cutoff * cutoff;          // c^p
    const double half_c2 = 0.5 * c2;            // c^p / alpha

    GospaComponents out;
    out.associations.assign(m, -1);

    // One code path: m == 0 or n == 0 simply falls through to the penalty terms below.
    if (m > 0 && n > 0) {
        Eigen::MatrixXd cost(m, n);
        Eigen::MatrixXd sqdist(m, n);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                // Position only. min(d, c)^2 == min(d^2, c^2), so clip in the squared domain
                // and skip a sqrt per pair.
                const double d2 = (track_means[i].head<kGospaPositionDim>() -
                                   ground_truths[j].head<kGospaPositionDim>()).squaredNorm();
                sqdist(i, j) = d2;
                cost(i, j) = std::min(d2, c2);
            }
        }

        // At alpha = 2 the forced min(m,n) matching over clipped costs attains the true
        // partial-assignment optimum, and the optimal partial assignment is recovered by
        // dropping every pair at d >= c: such a pair costs c^p assigned and c^p/2 + c^p/2
        // unassigned, so dropping it is exactly cost-neutral. Solving the rectangular problem
        // and then dropping is therefore equivalent to minimising over partial assignments.
        const auto hyps = solve_assignment(cost, 1);
        if (!hyps.empty()) {
            const std::vector<int>& assoc = hyps[0].associations;
            const std::size_t rows = std::min(m, assoc.size());
            for (std::size_t i = 0; i < rows; ++i) {
                const int j = assoc[i];
                if (j < 0 || j >= static_cast<int>(n)) continue;
                if (sqdist(i, j) >= c2) continue;  // beyond the cutoff: leave it unassigned
                out.associations[i] = j;
                out.localisation_cost += sqdist(i, j);
                ++out.num_assigned;
            }
        }
    }

    out.num_missed = static_cast<int>(n) - out.num_assigned;
    out.num_false  = static_cast<int>(m) - out.num_assigned;
    out.missed_cost = half_c2 * out.num_missed;
    out.false_cost  = half_c2 * out.num_false;
    out.total = std::sqrt(out.localisation_cost + out.missed_cost + out.false_cost);
    return out;
}

double calculate_gospa_distance(const std::vector<Track>& tracks,
                                const std::vector<Eigen::VectorXd>& ground_truths,
                                double cutoff) {
    return calculate_gospa_components(tracks, ground_truths, cutoff).total;
}
