#include "assignment.h"
#include "munkres.h"
#include "matrix.h"
#include <queue>
#include <limits>
#include <algorithm>
#include <vector>
#include <map>
#include <set>

// Candidate structure for Murty's algorithm priority queue
struct Candidate {
    double cost;
    std::vector<int> assignment;
    std::map<int, int> enforced_assignments;  // row -> col mappings that must be preserved
    std::set<std::pair<int, int>> forbidden_assignments;  // (row, col) pairs that cannot be selected
    
    // Constructor
    Candidate(double c, const std::vector<int>& assn) 
        : cost(c), assignment(assn) {}
    
    Candidate(double c, const std::vector<int>& assn, 
              const std::map<int, int>& enforced, 
              const std::set<std::pair<int, int>>& forbidden)
        : cost(c), assignment(assn), enforced_assignments(enforced), forbidden_assignments(forbidden) {}
    
    // For priority queue (min-heap based on cost)
    bool operator>(const Candidate& other) const {
        return cost > other.cost;
    }
};

// Helper function: Munkres wrapper for single best assignment
static std::pair<std::vector<int>, double> solve_lap_munkres(const Eigen::MatrixXd& cost_matrix) {
    const int rows = cost_matrix.rows();
    const int cols = cost_matrix.cols();
    
    // Handle empty matrix
    if (rows == 0 || cols == 0) {
        return {std::vector<int>(), 0.0};
    }
    
    // The Munkres implementation assumes non-negative costs: its row/column reduction only subtracts
    // a positive minimum, and its first step stars exact zeros. A row holding negative costs and an
    // exact zero (e.g. [-62, 0]) therefore keeps the zero as its "assignment". Shift every finite
    // entry by the matrix minimum first. Each row takes exactly one column, so a constant shift
    // cannot change which assignment is optimal; the total below is taken from the unshifted costs.
    double shift = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const double value = cost_matrix(i, j);
            if (std::isfinite(value)) {
                shift = std::min(shift, value);
            }
        }
    }

    // Create Munkres matrix and copy data
    Matrix<double> munkres_matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            munkres_matrix(i, j) = cost_matrix(i, j) - shift;
        }
    }
    
    // Solve using Munkres algorithm
    Munkres<double> solver;
    solver.solve(munkres_matrix);
    
    // Extract assignment and calculate total cost
    std::vector<int> assignment(rows, -1);
    double total_cost = 0.0;
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (munkres_matrix(i, j) == 0) {
                assignment[i] = j;
                total_cost += cost_matrix(i, j);
                break;  // Only one assignment per row
            }
        }
    }
    
    return {assignment, total_cost};
}

// Helper function: Create constrained cost matrix
static Eigen::MatrixXd create_constrained_matrix(const Eigen::MatrixXd& original_matrix,
                                                const std::map<int, int>& enforced_assignments,
                                                const std::set<std::pair<int, int>>& forbidden_assignments) {
    const int rows = original_matrix.rows();
    const int cols = original_matrix.cols();
    
    // Start with a copy of the original matrix
    Eigen::MatrixXd constrained_matrix = original_matrix;
    
    // Set forbidden assignments to infinity
    for (const auto& forbidden : forbidden_assignments) {
        if (forbidden.first < rows && forbidden.second < cols) {
            constrained_matrix(forbidden.first, forbidden.second) = std::numeric_limits<double>::infinity();
        }
    }
    
    // Handle enforced assignments by setting all other entries in those rows/columns to infinity
    for (const auto& enforced : enforced_assignments) {
        int row = enforced.first;
        int col = enforced.second;
        
        if (row < rows && col < cols) {
            // Set all other columns in this row to infinity
            for (int j = 0; j < cols; ++j) {
                if (j != col) {
                    constrained_matrix(row, j) = std::numeric_limits<double>::infinity();
                }
            }
            
            // Set all other rows in this column to infinity
            for (int i = 0; i < rows; ++i) {
                if (i != row) {
                    constrained_matrix(i, col) = std::numeric_limits<double>::infinity();
                }
            }
        }
    }
    
    return constrained_matrix;
}

// Helper function: Reconstruct full assignment from reduced assignment
static std::vector<int> reconstruct_assignment(const std::vector<int>& reduced_assignment,
                                              const std::vector<int>& row_mapping,
                                              const std::vector<int>& col_mapping,
                                              int original_rows) {
    std::vector<int> full_assignment(original_rows, -1);
    
    for (int i = 0; i < reduced_assignment.size(); ++i) {
        if (reduced_assignment[i] != -1) {
            int original_row = row_mapping[i];
            int original_col = col_mapping[reduced_assignment[i]];
            full_assignment[original_row] = original_col;
        }
    }
    
    return full_assignment;
}

// Main function: Solve K-best assignment problem using Murty's algorithm
std::vector<Hypothesis> solve_assignment(const Eigen::MatrixXd& cost_matrix, int k_best) {
    std::vector<Hypothesis> results;
    
    if (k_best <= 0 || cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        return results;
    }
    
    // Priority queue for candidates (min-heap based on cost)
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
    
    // Step 1: Find the best (root) solution
    auto root_solution = solve_lap_munkres(cost_matrix);
    results.emplace_back(root_solution.first, root_solution.second);
    
    if (k_best == 1) {
        return results;
    }
    
    // Step 2: Initialize candidates with partitions of the root solution
    const std::vector<int>& root_assignment = root_solution.first;
    
    // Create initial partitions based on the root solution
    for (int i = 0; i < root_assignment.size(); ++i) {
        if (root_assignment[i] != -1) {  // Only consider actual assignments
            // Create enforced assignments (all assignments before position i)
            std::map<int, int> enforced;
            for (int j = 0; j < i; ++j) {
                if (root_assignment[j] != -1) {
                    enforced[j] = root_assignment[j];
                }
            }
            
            // Create forbidden assignment (the assignment at position i)
            std::set<std::pair<int, int>> forbidden;
            forbidden.insert({i, root_assignment[i]});
            
            // Create constrained matrix and solve
            Eigen::MatrixXd constrained_matrix = create_constrained_matrix(cost_matrix, enforced, forbidden);
            auto solution = solve_lap_munkres(constrained_matrix);
            
            // Check if solution is valid (finite cost)
            if (std::isfinite(solution.second)) {
                candidates.emplace(solution.second, solution.first, enforced, forbidden);
            }
        }
    }
    
    // Step 3: Main loop - extract best candidates and partition them
    while (results.size() < k_best && !candidates.empty()) {
        // Get the next best candidate
        Candidate best_candidate = candidates.top();
        candidates.pop();
        
        // Add to results
        results.emplace_back(best_candidate.assignment, best_candidate.cost);
        
        // Partition this candidate to generate new candidates
        const std::vector<int>& current_assignment = best_candidate.assignment;
        
        for (int i = 0; i < current_assignment.size(); ++i) {
            if (current_assignment[i] != -1) {  // Only consider actual assignments
                // Create new enforced assignments (all current enforced + assignments before position i)
                std::map<int, int> new_enforced = best_candidate.enforced_assignments;
                for (int j = 0; j < i; ++j) {
                    if (current_assignment[j] != -1) {
                        new_enforced[j] = current_assignment[j];
                    }
                }
                
                // Create new forbidden assignments (all current forbidden + assignment at position i)
                std::set<std::pair<int, int>> new_forbidden = best_candidate.forbidden_assignments;
                new_forbidden.insert({i, current_assignment[i]});
                
                // Create constrained matrix and solve
                Eigen::MatrixXd constrained_matrix = create_constrained_matrix(cost_matrix, new_enforced, new_forbidden);
                auto solution = solve_lap_munkres(constrained_matrix);
                
                // Check if solution is valid and not already found
                if (std::isfinite(solution.second)) {
                    bool is_duplicate = false;
                    for (const auto& existing : results) {
                        if (existing.associations == solution.first) {
                            is_duplicate = true;
                            break;
                        }
                    }
                    
                    if (!is_duplicate) {
                        candidates.emplace(solution.second, solution.first, new_enforced, new_forbidden);
                    }
                }
            }
        }
    }
    
    return results;
}