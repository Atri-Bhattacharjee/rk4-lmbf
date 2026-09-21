"""Unit gate on the GOSPA metric itself (src/metrics.cpp).

The predecessor metric had no unit test at all -- it was only ever covered end to end, as one
number in a digest. That is not enough to regenerate fixtures against, so this file has to pass
before any fixture is written from a new build.

Three layers, because they fail differently:

* Hand-computable cases with exact expected values. These pin the decisions a reader cannot
  recover from the output alone: position-only base distance, the alpha=2 drop rule at d >= c,
  the boundary convention at d == c exactly, and the degenerate cardinalities (which are NOT
  the cutoff, unlike the OSPA predecessor).

* A brute-force reference that minimises over every partial assignment, straight from the
  definition. src/metrics.cpp solves a forced rectangular assignment and then drops the pairs
  at d >= c; that shortcut is provably equal to the true partial-assignment optimum at alpha=2,
  and this is what actually holds the proof to account.

* Structural identities that must hold for every input: exact additivity of the decomposition,
  the counts closing against the cardinalities, the sqrt((m+n)/2) upper bound, and the
  associations forming a valid injection.

Usage:
    python tests/test_gospa_metric.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import run_once  # noqa: E402

lmb = run_once.lmb_engine

C = 10000.0          # the cutoff these cases are written against
HALF_CP = 0.5 * C * C  # c^p / alpha at p = 2, alpha = 2 -> 5e7


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)

    def exact(self, actual: float, expected: float, message: str) -> None:
        """Bitwise equality. Every expected value in Part A is exactly representable."""
        self.count += 1
        if float(actual) != float(expected):
            raise AssertionError(f"{message}: got {actual!r}, expected {expected!r}")

    def close(self, actual, expected, message: str, rtol: float = 1e-12) -> None:
        self.count += 1
        if not np.allclose(np.asarray(actual, dtype=np.float64),
                           np.asarray(expected, dtype=np.float64), rtol=rtol, atol=0.0):
            raise AssertionError(f"{message}: got {actual!r}, expected {expected!r}")


def expect_raises(chk: Checker, exc_type, fn, label: str, message_contains: str):
    """Run fn; require exc_type whose message contains message_contains."""
    chk.count += 1
    try:
        fn()
    except exc_type as error:
        if message_contains not in str(error):
            raise AssertionError(
                f"{label}: {exc_type.__name__} message {str(error)!r} lacks {message_contains!r}"
            ) from None
        return error
    raise AssertionError(f"{label}: expected {exc_type.__name__}")


def make_track(*components: float):
    """A one-particle track at unit weight, so weighted_mean returns the state exactly."""
    state = np.zeros(6, dtype=np.float64)
    state[: len(components)] = components
    particle = lmb.Particle()
    particle.state_vector = state
    particle.weight = 1.0
    return lmb.Track(lmb.TrackLabel(), 0.5, [particle])


def make_truth(*components: float) -> np.ndarray:
    state = np.zeros(6, dtype=np.float64)
    state[: len(components)] = components
    return state


def check_build_is_current(chk: Checker) -> None:
    """Assertion zero: the Release and Debug presets share build/, so a stale .so is easy to
    import by accident. Without this the failure mode is silent -- fixtures regenerated with new
    field names from the old metric. Keep this first in the file."""
    chk.ok(hasattr(lmb, "calculate_gospa_components"),
           f"engine at {lmb.__file__} predates the GOSPA change -- rebuild before running this")
    chk.ok(hasattr(lmb, "calculate_gospa_distance"),
           "calculate_gospa_distance missing from the engine")
    chk.ok(not hasattr(lmb, "calculate_ospa_distance"),
           f"stale .so at {lmb.__file__}: the removed OSPA symbol is still present")
    chk.exact(lmb.GOSPA_ORDER_P, 2.0, "GOSPA_ORDER_P")
    chk.exact(lmb.GOSPA_ALPHA, 2.0, "GOSPA_ALPHA (the decomposition is only valid at 2)")
    chk.ok(0.0 < lmb.GOSPA_DEFAULT_CUTOFF < math.inf,
           f"GOSPA_DEFAULT_CUTOFF must be finite and positive, got {lmb.GOSPA_DEFAULT_CUTOFF}")


def check_hand_cases(chk: Checker) -> None:
    """Part A. Expected values are written as math.sqrt(<exactly representable>) rather than as
    decimal literals: IEEE sqrt is correctly rounded, so math.sqrt(5e7) matches the C++
    std::sqrt(5e7) bitwise, whereas 10000/math.sqrt(2) lands one ulp away."""
    root_half = math.sqrt(HALF_CP)  # 7071.067811865475, one missed or false track

    cases = [
        # label, tracks, truths, expected total, (assigned, missed, false)
        ("both empty", [], [], 0.0, (0, 0, 0)),
        ("no tracks, one truth", [], [make_truth(0, 0, 0)], root_half, (0, 1, 0)),
        ("one track, no truths", [make_track(0, 0, 0)], [], root_half, (0, 0, 1)),
        ("exact match", [make_track(0, 0, 0)], [make_truth(0, 0, 0)], 0.0, (1, 0, 0)),
        ("3-4-5 triangle", [make_track(0, 0, 0)], [make_truth(3000, 4000, 0)], 5000.0, (1, 0, 0)),
        # Velocity differs by 10 km/s. Under the old full-6D norm this would have been
        # hypot(5000, 10000) = 11180.34; position-only it must stay 5000.
        ("position-only", [make_track(0, 0, 0, 5000, 0, 0)],
         [make_truth(3000, 4000, 0, -5000, 0, 0)], 5000.0, (1, 0, 0)),
        # d > c: the pair is dropped, so this is one missed AND one false, not one localisation
        # error clipped at c. That is the alpha=2 signature.
        ("beyond cutoff", [make_track(0, 0, 0)], [make_truth(20000, 0, 0)], C, (0, 1, 1)),
        # The boundary convention: d == c exactly is dropped (the test is d^2 >= c^2).
        ("d == c exactly", [make_track(0, 0, 0)], [make_truth(C, 0, 0)], C, (0, 1, 1)),
        ("surplus track is false",
         [make_track(100, 0, 0), make_track(50000, 0, 0)], [make_truth(0, 0, 0)],
         math.sqrt(100.0 * 100.0 + HALF_CP), (1, 0, 1)),
        ("surplus truth is missed",
         [make_track(0, 0, 0)], [make_truth(0, 0, 0), make_truth(30000, 0, 0)],
         root_half, (1, 1, 0)),
        # Two tracks at the origin, one truth at the origin and one 100 km away. The far pair is
        # dropped, leaving one exact match plus one missed and one false.
        ("drop rule reproduces the forced total",
         [make_track(0, 0, 0), make_track(0, 0, 0)],
         [make_truth(0, 0, 0), make_truth(100000, 0, 0)], C, (1, 1, 1)),
        # Unnormalised: k objects each 1000 m off gives sqrt(k) * 1000, not 1000. OSPA would have
        # reported 1000 here regardless of k. This is the behaviour change most visible in plots.
        ("unnormalised sqrt(k) growth",
         [make_track(1000, 0, 0), make_track(0, 1000, 0), make_track(0, 0, 1000)],
         [make_truth(0, 0, 0), make_truth(0, 0, 0), make_truth(0, 0, 0)],
         math.sqrt(3.0e6), (3, 0, 0)),
    ]

    for label, tracks, truths, expected, (n_assigned, n_missed, n_false) in cases:
        result = lmb.calculate_gospa_components(tracks, truths, C)
        chk.exact(result.total, expected, f"{label}: total")
        chk.ok(result.num_assigned == n_assigned,
               f"{label}: num_assigned {result.num_assigned} != {n_assigned}")
        chk.ok(result.num_missed == n_missed,
               f"{label}: num_missed {result.num_missed} != {n_missed}")
        chk.ok(result.num_false == n_false,
               f"{label}: num_false {result.num_false} != {n_false}")
        chk.exact(HALF_CP * n_missed, result.missed, f"{label}: missed cost")
        chk.exact(HALF_CP * n_false, result.false_positive, f"{label}: false cost")
        chk.exact(lmb.calculate_gospa_distance(tracks, truths, C), result.total,
                  f"{label}: scalar entry point must equal .total bitwise")


def reference_gospa(track_positions, truth_positions, cutoff, order=2.0, alpha=2.0):
    """GOSPA straight from the definition: minimise over ALL partial assignments.

    Exponential, so only usable on tiny inputs -- which is the point. It shares no code with
    src/metrics.cpp and in particular does not know about the forced-assignment shortcut.
    Returns (total, num_assigned) for the best assignment found.
    """
    m, n = len(track_positions), len(truth_positions)
    penalty = (cutoff ** order) / alpha
    best_cost, best_k = math.inf, 0
    for k in range(min(m, n) + 1):
        for rows in itertools.combinations(range(m), k):
            for cols in itertools.permutations(range(n), k):
                cost = sum(
                    min(float(np.linalg.norm(track_positions[i] - truth_positions[j])), cutoff) ** order
                    for i, j in zip(rows, cols)
                )
                cost += penalty * (m - k) + penalty * (n - k)
                if cost < best_cost:
                    best_cost, best_k = cost, k
    return best_cost ** (1.0 / order), best_k


def random_instances(rng, count: int, cutoff: float):
    """Instances sized so the brute force stays tractable and roughly half the pairwise
    distances straddle the cutoff -- the regime where the drop rule actually matters."""
    for _ in range(count):
        m = int(rng.integers(0, 5))
        n = int(rng.integers(0, 5))
        box = 2.0 * cutoff
        track_pos = rng.uniform(-box, box, size=(m, 3))
        truth_pos = rng.uniform(-box, box, size=(n, 3))
        yield track_pos, truth_pos


def check_against_brute_force(chk: Checker, instances) -> int:
    """Part B. The forced rectangular assignment plus the d >= c drop must equal the true
    minimum over partial assignments."""
    compared = 0
    for track_pos, truth_pos in instances:
        tracks = [make_track(*p) for p in track_pos]
        truths = [make_truth(*p) for p in truth_pos]
        result = lmb.calculate_gospa_components(tracks, truths, C)
        expected, expected_k = reference_gospa(track_pos, truth_pos, C)
        chk.close(result.total, expected,
                  f"brute force disagrees for m={len(track_pos)} n={len(truth_pos)}")
        # The total is unique; the argmin assignment need not be. Only compare the split when
        # the number of assigned pairs is forced, otherwise a symmetric layout flakes.
        if result.num_assigned == expected_k:
            compared += 1
    return compared


def check_identities(chk: Checker, instances) -> None:
    """Part C. Must hold for every input, whatever the assignment chose."""
    for track_pos, truth_pos in instances:
        m, n = len(track_pos), len(truth_pos)
        tracks = [make_track(*p) for p in track_pos]
        truths = [make_truth(*p) for p in truth_pos]
        label = f"m={m} n={n}"
        g = lmb.calculate_gospa_components(tracks, truths, C)

        chk.exact(lmb.calculate_gospa_distance(tracks, truths, C), g.total,
                  f"{label}: scalar vs components")
        total_sq = g.total * g.total
        component_sum = g.localisation + g.missed + g.false_positive
        chk.ok(abs(component_sum - total_sq) <= 1e-9 * max(total_sq, 1.0),
               f"{label}: components {component_sum!r} do not sum to total^2 {total_sq!r}")
        chk.ok(g.num_assigned + g.num_missed == n,
               f"{label}: assigned+missed {g.num_assigned + g.num_missed} != {n} truths")
        chk.ok(g.num_assigned + g.num_false == m,
               f"{label}: assigned+false {g.num_assigned + g.num_false} != {m} tracks")
        chk.ok(min(g.localisation, g.missed, g.false_positive, g.total) >= 0.0,
               f"{label}: negative component")
        chk.ok(math.isfinite(g.total), f"{label}: non-finite total")
        # Unnormalised GOSPA is NOT bounded by the cutoff. This is the real bound, and it is what
        # the harness invariants were restated against.
        bound = C * math.sqrt((m + n) / 2.0)
        chk.ok(g.total <= bound * (1.0 + 1e-12),
               f"{label}: total {g.total} exceeds the bound c*sqrt((m+n)/2) = {bound}")

        # associations must be a valid injection, and every listed pair must be inside the cutoff.
        chk.ok(len(g.associations) == m, f"{label}: associations length {len(g.associations)} != {m}")
        assigned = [j for j in g.associations if j != -1]
        chk.ok(len(set(assigned)) == len(assigned), f"{label}: a truth is claimed by two tracks")
        chk.ok(len(assigned) == g.num_assigned, f"{label}: associations disagree with num_assigned")
        for i, j in enumerate(g.associations):
            if j == -1:
                continue
            chk.ok(0 <= j < n, f"{label}: association {j} out of range")
            distance = float(np.linalg.norm(track_pos[i] - truth_pos[j]))
            chk.ok(distance < C, f"{label}: pair at d={distance} >= c should have been dropped")


def check_parameters(chk: Checker) -> None:
    """The binding default, cutoff scaling, and the input guards."""
    tracks = [make_track(1000, 0, 0)]
    truths = [make_truth(0, 0, 0)]

    chk.exact(lmb.calculate_gospa_distance(tracks, truths),
              lmb.calculate_gospa_distance(tracks, truths, lmb.GOSPA_DEFAULT_CUTOFF),
              "omitting cutoff must equal passing GOSPA_DEFAULT_CUTOFF")
    chk.exact(lmb.calculate_gospa_components(tracks, truths).total,
              lmb.calculate_gospa_components(tracks, truths, lmb.GOSPA_DEFAULT_CUTOFF).total,
              "components: omitting cutoff must equal passing GOSPA_DEFAULT_CUTOFF")

    # Fully saturated (every pair dropped): the total is c*sqrt((m+n)/2), so doubling c doubles it.
    far_tracks = [make_track(0, 0, 0)]
    far_truths = [make_truth(1e9, 0, 0)]
    single = lmb.calculate_gospa_distance(far_tracks, far_truths, C)
    double = lmb.calculate_gospa_distance(far_tracks, far_truths, 2.0 * C)
    chk.close(double, 2.0 * single, "doubling the cutoff must double a saturated total")

    expect_raises(chk, ValueError, lambda: lmb.calculate_gospa_distance(tracks, truths, 0.0),
                  "zero cutoff", "cutoff must be finite and positive")
    expect_raises(chk, ValueError, lambda: lmb.calculate_gospa_distance(tracks, truths, -1.0),
                  "negative cutoff", "cutoff must be finite and positive")
    expect_raises(chk, ValueError,
                  lambda: lmb.calculate_gospa_distance(tracks, truths, float("nan")),
                  "nan cutoff", "cutoff must be finite and positive")
    # A short ground truth would be undefined behaviour under .head<3>() with EIGEN_NO_DEBUG,
    # so the guard has to be explicit rather than an Eigen assert.
    expect_raises(chk, ValueError,
                  lambda: lmb.calculate_gospa_distance(tracks, [np.zeros(5)], C),
                  "5-element ground truth", "state vector must have size 6")


def main() -> int:
    chk = Checker()
    check_build_is_current(chk)
    check_hand_cases(chk)

    rng = np.random.default_rng(20260920)
    instances = list(random_instances(rng, 200, C))
    compared = check_against_brute_force(chk, instances)
    check_identities(chk, instances)
    check_parameters(chk)

    print(f"  brute-force agreement on {len(instances)} random instances "
          f"({compared} with a forced assignment count)")
    print(f"PASS: test_gospa_metric ({chk.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
