#include "metrics.h"
#include "assignment.h"
#include <cmath>
#include <algorithm>
#include <vector>

static StateVector mean_state(const Track& track) {
    const auto& particles = track.particles();
    if (particles.empty()) return StateVector::Zero();
    StateVector sum = StateVector::Zero();
    double total_weight = 0.0;
    for (const auto& p : particles) {
        sum += p.state_vector * p.weight;
        total_weight += p.weight;
    }
    if (total_weight == 0.0) return StateVector::Zero();
    return sum / total_weight;
}

double calculate_ospa_distance(const std::vector<Track>& tracks,
                               const std::vector<Eigen::VectorXd>& ground_truths,
                               double cutoff) {
    const size_t m = tracks.size();
    const size_t n = ground_truths.size();

    std::vector<StateVector> track_means;
    track_means.reserve(m);
    for (const auto& track : tracks) {
        StateVector mean = mean_state(track);
        if (!ground_truths.empty() && mean.size() != ground_truths[0].size()) {
            return cutoff;
        }
        track_means.push_back(mean);
    }

    if (m == 0 && n == 0) return 0.0;
    if (n == 0) return cutoff;
    if (m == 0) return cutoff;

    const bool tracks_are_smaller = m <= n;
    const size_t rows = tracks_are_smaller ? m : n;
    const size_t cols = tracks_are_smaller ? n : m;
    Eigen::MatrixXd dist_matrix(rows, cols);

    if (tracks_are_smaller) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dist_matrix(i, j) = std::min((track_means[i] - ground_truths[j]).norm(), cutoff);
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            const Eigen::VectorXd& truth_state = ground_truths[i];
            for (size_t j = 0; j < m; ++j) {
                dist_matrix(i, j) = std::min((track_means[j] - truth_state).norm(), cutoff);
            }
        }
    }

    auto hyps = solve_assignment(dist_matrix, 1);
    double assignment_sum = 0.0;
    if (!hyps.empty()) {
        const auto& assoc_vec = hyps[0].associations;
        for (size_t i = 0; i < rows; ++i) {
            int j = (i < assoc_vec.size()) ? assoc_vec[i] : -1;
            if (j != -1 && j >= 0 && j < (int)cols)
                assignment_sum += dist_matrix(i, j);
            else
                assignment_sum += cutoff;
        }
    } else {
        assignment_sum = rows * cutoff;
    }
    double cardinality_error = cutoff * std::abs((int)m - (int)n);
    double ospa = (assignment_sum + cardinality_error) / std::max(m, n);
    return ospa;
}
