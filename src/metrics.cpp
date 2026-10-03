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

namespace {

//! Above this many track-truth pairs the assignment is solved per connected component.
constexpr std::size_t kGospaClusterThreshold = 4096;

int find_root(std::vector<int>& parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/**
 * Optimal partial assignment by connected component. Rows (tracks) and columns (truths) are
 * linked when their squared distance is below c^2; each component with both a track and a truth
 * is solved with the same clipped costs and the same solver as the whole problem would be.
 */
void solve_by_component(const Eigen::MatrixXd& cost, const Eigen::MatrixXd& sqdist, double c2,
                        std::vector<int>& chosen) {
    const int m = static_cast<int>(cost.rows());
    const int n = static_cast<int>(cost.cols());
    std::vector<int> parent(static_cast<std::size_t>(m + n));
    for (int k = 0; k < m + n; ++k) {
        parent[static_cast<std::size_t>(k)] = k;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (sqdist(i, j) < c2) {
                const int a = find_root(parent, i);
                const int b = find_root(parent, m + j);
                if (a != b) {
                    parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
                }
            }
        }
    }
    // Members of each component, tracks and truths in increasing index order.
    std::vector<int> component(static_cast<std::size_t>(m + n), -1);
    std::vector<std::vector<int>> rows;
    std::vector<std::vector<int>> cols;
    for (int k = 0; k < m + n; ++k) {
        const int root = find_root(parent, k);
        int& id = component[static_cast<std::size_t>(root)];
        if (id < 0) {
            id = static_cast<int>(rows.size());
            rows.emplace_back();
            cols.emplace_back();
        }
        if (k < m) {
            rows[static_cast<std::size_t>(id)].push_back(k);
        } else {
            cols[static_cast<std::size_t>(id)].push_back(k - m);
        }
    }
    for (std::size_t c = 0; c < rows.size(); ++c) {
        const std::vector<int>& r = rows[c];
        const std::vector<int>& q = cols[c];
        if (r.empty() || q.empty()) {
            continue;
        }
        if (r.size() == 1 && q.size() == 1) {
            chosen[static_cast<std::size_t>(r[0])] = q[0];   // linked, so d < c: assign
            continue;
        }
        Eigen::MatrixXd sub(static_cast<Eigen::Index>(r.size()), static_cast<Eigen::Index>(q.size()));
        for (std::size_t a = 0; a < r.size(); ++a) {
            for (std::size_t b = 0; b < q.size(); ++b) {
                sub(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = cost(r[a], q[b]);
            }
        }
        const auto hyps = solve_assignment(sub, 1);
        if (hyps.empty()) {
            continue;
        }
        const std::vector<int>& assoc = hyps[0].associations;
        for (std::size_t a = 0; a < r.size() && a < assoc.size(); ++a) {
            const int b = assoc[a];
            if (b >= 0 && b < static_cast<int>(q.size())) {
                chosen[static_cast<std::size_t>(r[a])] = q[static_cast<std::size_t>(b)];
            }
        }
    }
}

}  // namespace

GospaComponents calculate_gospa_components(const std::vector<Track>& tracks,
                                           const std::vector<Eigen::VectorXd>& ground_truths,
                                           double cutoff) {
    std::vector<StateVector> track_means;
    track_means.reserve(tracks.size());
    for (const auto& track : tracks) {
        track_means.push_back(particle_stats::weighted_mean(track));
    }
    return calculate_gospa_components_from_means(track_means, ground_truths, cutoff);
}

GospaComponents calculate_gospa_components_from_means(const std::vector<StateVector>& track_means,
                                                      const std::vector<Eigen::VectorXd>& ground_truths,
                                                      double cutoff) {
    if (!(std::isfinite(cutoff) && cutoff > 0.0)) {
        throw std::invalid_argument(
            "calculate_gospa_components: cutoff must be finite and positive, got " +
            std::to_string(cutoff));
    }

    const std::size_t m = track_means.size();
    const std::size_t n = ground_truths.size();

    // Release sets EIGEN_NO_DEBUG (CMakeLists.txt), so .head<3>() on a short vector is undefined
    // behaviour rather than an assert. This check has to be explicit and unconditional.
    for (const auto& truth : ground_truths) {
        validation::require_state_vector(truth, "calculate_gospa_components: ground_truth");
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
        //
        // The same argument splits the problem: only pairs with d < c can end up assigned, so the
        // optimum is the sum of the optima of the connected components of the "d < c" bipartite
        // graph, each solved on its own. On a large catalogue that turns one O((m+n)^3) solve into
        // many tiny ones. Small problems keep the single solve, so their arithmetic is unchanged.
        std::vector<int> chosen(m, -1);
        if (m * n <= kGospaClusterThreshold) {
            const auto hyps = solve_assignment(cost, 1);
            if (!hyps.empty()) {
                const std::vector<int>& assoc = hyps[0].associations;
                const std::size_t rows = std::min(m, assoc.size());
                for (std::size_t i = 0; i < rows; ++i) {
                    chosen[i] = assoc[i];
                }
            }
        } else {
            solve_by_component(cost, sqdist, c2, chosen);
        }
        // Accumulated in track order whichever path chose the pairs.
        for (std::size_t i = 0; i < m; ++i) {
            const int j = chosen[i];
            if (j < 0 || j >= static_cast<int>(n)) continue;
            if (sqdist(i, j) >= c2) continue;  // beyond the cutoff: leave it unassigned
            out.associations[i] = j;
            out.localisation_cost += sqdist(i, j);
            ++out.num_assigned;
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
