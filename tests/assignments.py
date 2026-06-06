import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from lmb_engine_loader import import_lmb_engine

lmb_engine = import_lmb_engine()

# Global test counters
tests_passed = 0
tests_failed = 0

def assert_test(condition, message):
    """Assert helper with test counting"""
    global tests_passed, tests_failed
    if condition:
        tests_passed += 1
        print(f"✓ PASS: {message}")
    else:
        tests_failed += 1
        print(f"✗ FAIL: {message}")

def calculate_assignment_cost(cost_matrix, assignment):
    """Calculate the total cost of an assignment"""
    total_cost = 0.0
    for row, col in enumerate(assignment):
        if col != -1 and row < cost_matrix.shape[0] and col < cost_matrix.shape[1]:
            total_cost += cost_matrix[row, col]
    return total_cost

def validate_assignment(cost_matrix, assignment):
    """Validate that an assignment is feasible"""
    used_cols = set()
    for row, col in enumerate(assignment):
        if col != -1:
            if col in used_cols:
                return False, f"Column {col} assigned multiple times"
            if col < 0 or col >= cost_matrix.shape[1]:
                return False, f"Invalid column index {col}"
            used_cols.add(col)
    return True, "Valid assignment"

def print_assignment_detailed(cost_matrix, k_best=1, test_name=""):
    """Enhanced assignment printing with validation"""
    print(f"\n--- {test_name} ---")
    print("Cost matrix:")
    print(cost_matrix)
    
    hyps = lmb_engine.solve_assignment(cost_matrix, k_best)
    print(f"Requested k_best: {k_best}, Got: {len(hyps)} solutions")
    
    for idx, h in enumerate(hyps):
        calculated_cost = calculate_assignment_cost(cost_matrix, h.associations)
        # Handle infinity comparisons properly
        if np.isinf(h.weight) and np.isinf(calculated_cost):
            cost_match = True  # Both infinite counts as match
        else:
            cost_match = abs(h.weight - calculated_cost) < 1e-10
        valid, msg = validate_assignment(cost_matrix, h.associations)
        
        print(f"Hypothesis {idx}: associations={h.associations}")
        print(f"  Reported cost: {h.weight:.6f}")
        print(f"  Calculated cost: {calculated_cost:.6f}")
        print(f"  Cost match: {cost_match}, Valid: {valid}")
        if not valid:
            print(f"  Error: {msg}")
    
    return hyps

def test_basic_square_matrix():
    """Test basic square matrix assignment"""
    print("\n=== Basic Square Matrix Tests ===")
    
    # Simple 2x2 case
    cost = np.array([[1, 2], [2, 1]], dtype=float)
    hyps = print_assignment_detailed(cost, k_best=1, test_name="2x2 Simple")
    
    # Verify optimal assignment
    assert_test(len(hyps) == 1, "Should return exactly 1 solution")
    assert_test(hyps[0].associations == [0, 1], "Optimal assignment should be [0,1]")
    assert_test(abs(hyps[0].weight - 2.0) < 1e-10, "Optimal cost should be 2.0")

def test_rectangular_matrices():
    """Test rectangular matrix assignments"""
    print("\n=== Rectangular Matrix Tests ===")
    
    # 2x3 matrix (more columns than rows)
    cost = np.array([[1, 2, 3], [2, 1, 3]], dtype=float)
    hyps = print_assignment_detailed(cost, k_best=1, test_name="2x3 Rectangular")
    
    assert_test(len(hyps) == 1, "Should return exactly 1 solution")
    expected_cost = 2.0  # [0->0, 1->1] = 1 + 1 = 2
    assert_test(abs(hyps[0].weight - expected_cost) < 1e-10, f"Cost should be {expected_cost}")
    
    # 3x2 matrix (more rows than columns)
    cost = np.array([[1, 2], [2, 1], [3, 3]], dtype=float)
    hyps = print_assignment_detailed(cost, k_best=1, test_name="3x2 Rectangular")
    
    assert_test(len(hyps) == 1, "Should return exactly 1 solution")
    # One row will be unassigned (-1)
    unassigned_count = sum(1 for x in hyps[0].associations if x == -1)
    assert_test(unassigned_count == 1, "Exactly one row should be unassigned")

def test_k_best_functionality():
    """Test K-best assignment functionality"""
    print("\n=== K-Best Assignment Tests ===")
    
    # Create a matrix with known multiple solutions
    cost = np.array([[2, 1, 3], [1, 3, 2], [3, 2, 1]], dtype=float)
    
    # Test k=1
    hyps1 = print_assignment_detailed(cost, k_best=1, test_name="3x3 K=1")
    assert_test(len(hyps1) == 1, "K=1 should return exactly 1 solution")
    
    # Test k=3
    hyps3 = print_assignment_detailed(cost, k_best=3, test_name="3x3 K=3")
    assert_test(len(hyps3) <= 3, "K=3 should return at most 3 solutions")
    assert_test(len(hyps3) >= 1, "K=3 should return at least 1 solution")
    
    # Verify cost ordering (non-decreasing)
    for i in range(1, len(hyps3)):
        assert_test(hyps3[i].weight >= hyps3[i-1].weight, 
                   f"Solution {i} cost should be >= solution {i-1} cost")
    
    # Verify no duplicate solutions
    assignments = [tuple(h.associations) for h in hyps3]
    unique_assignments = set(assignments)
    assert_test(len(assignments) == len(unique_assignments), "All solutions should be unique")
    
    # Test k > possible solutions
    hyps_large = print_assignment_detailed(cost, k_best=10, test_name="3x3 K=10")
    assert_test(len(hyps_large) <= 6, "Should not return more solutions than possible")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n=== Edge Case Tests ===")
    
    # Single element matrix
    cost = np.array([[5.0]], dtype=float)
    hyps = print_assignment_detailed(cost, k_best=1, test_name="1x1 Single Element")
    assert_test(len(hyps) == 1, "Single element should return 1 solution")
    assert_test(hyps[0].associations == [0], "Single element assignment should be [0]")
    assert_test(abs(hyps[0].weight - 5.0) < 1e-10, "Cost should be 5.0")
    
    # Matrix with infinite costs
    cost = np.array([[1, np.inf], [np.inf, 2]], dtype=float)
    hyps = print_assignment_detailed(cost, k_best=1, test_name="Matrix with Infinities")
    assert_test(len(hyps) == 1, "Should find valid assignment avoiding infinities")
    assert_test(np.isfinite(hyps[0].weight), "Returned cost should be finite")
    
    # All infinite costs (unassignable)
    cost = np.array([[np.inf, np.inf], [np.inf, np.inf]], dtype=float)
    hyps = print_assignment_detailed(cost, k_best=1, test_name="All Infinite Costs")
    # This should either return no solutions or solutions with infinite cost
    if len(hyps) > 0:
        # Handle the case where both costs are infinite (NaN comparison issue)
        if np.isinf(hyps[0].weight):
            assert_test(True, "Cost should be infinite for unassignable matrix")
        else:
            assert_test(False, "Cost should be infinite for unassignable matrix")

def test_cost_verification():
    """Test that reported costs match calculated costs"""
    print("\n=== Cost Verification Tests ===")
    
    # Random matrix for comprehensive cost testing
    np.random.seed(42)  # For reproducible results
    cost = np.random.rand(4, 5) * 10
    
    hyps = print_assignment_detailed(cost, k_best=5, test_name="4x5 Random Cost Verification")
    
    for i, h in enumerate(hyps):
        calculated_cost = calculate_assignment_cost(cost, h.associations)
        cost_match = abs(h.weight - calculated_cost) < 1e-10
        assert_test(cost_match, f"Solution {i}: reported cost should match calculated cost")
        
        valid, msg = validate_assignment(cost, h.associations)
        assert_test(valid, f"Solution {i}: assignment should be valid - {msg}")

def test_missed_detection_scenarios():
    """Test scenarios with missed detections (high cost assignments)"""
    print("\n=== Missed Detection Tests ===")
    
    # Matrix where missed detection (last column) is sometimes optimal
    cost = np.array([[1, 2, 0.5], [2, 1, 0.5]], dtype=float)  # Missed detection cost = 0.5
    hyps = print_assignment_detailed(cost, k_best=3, test_name="Missed Detection Optimal")
    
    # Should prefer missed detections over expensive assignments
    best_assignment = hyps[0].associations
    missed_detections = sum(1 for col in best_assignment if col == 2)  # Column 2 is missed detection
    assert_test(missed_detections > 0, "Should choose missed detection when optimal")

def test_murty_algorithm_specific():
    """Test Murty's algorithm specific functionality"""
    print("\n=== Murty's Algorithm Specific Tests ===")
    
    # Create a matrix where we can verify specific partitioning behavior
    cost = np.array([
        [1, 2, 4],
        [3, 1, 2],
        [2, 3, 1]
    ], dtype=float)
    
    hyps = print_assignment_detailed(cost, k_best=6, test_name="3x3 Full Partitioning")
    
    # Verify that we get multiple valid solutions
    assert_test(len(hyps) >= 2, "Should find at least 2 solutions for 3x3 matrix")
    
    # Verify strict cost ordering
    for i in range(1, len(hyps)):
        assert_test(hyps[i].weight > hyps[i-1].weight or abs(hyps[i].weight - hyps[i-1].weight) < 1e-10,
                   f"Solution {i} cost should be >= solution {i-1} cost (strict ordering)")
    
    # Test that first solution matches single best
    hyps_single = lmb_engine.solve_assignment(cost, 1)
    assert_test(abs(hyps[0].weight - hyps_single[0].weight) < 1e-10,
               "First k-best solution should match single best solution")
    assert_test(hyps[0].associations == hyps_single[0].associations,
               "First k-best assignment should match single best assignment")

def test_large_matrix_performance():
    """Test performance and correctness on larger matrices"""
    print("\n=== Large Matrix Performance Tests ===")
    
    # 6x6 matrix
    np.random.seed(123)
    cost = np.random.rand(6, 6) * 100
    
    hyps = print_assignment_detailed(cost, k_best=3, test_name="6x6 Performance Test")
    
    assert_test(len(hyps) >= 1, "Should find at least one solution for 6x6 matrix")
    
    # Verify all solutions are valid
    for i, h in enumerate(hyps):
        valid, msg = validate_assignment(cost, h.associations)
        assert_test(valid, f"Large matrix solution {i} should be valid - {msg}")

def run_all_tests():
    """Run all test suites"""
    global tests_passed, tests_failed
    
    print("Starting comprehensive assignment algorithm tests...")
    
    test_basic_square_matrix()
    test_rectangular_matrices()
    test_k_best_functionality()
    test_edge_cases()
    test_cost_verification()
    test_missed_detection_scenarios()
    test_murty_algorithm_specific()
    test_large_matrix_performance()
    
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"Total tests: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {tests_failed} tests failed.")
        return False

def main():
    """Main test execution"""
    success = run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)