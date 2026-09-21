"""Structural invariants asserted at every step of a seeded run.

Where the golden digest catches *changes*, this catches *illegal states* — the errors an
optimization is most likely to introduce (a dropped normalization, an off-by-one in a resampling
walk, an aliased buffer written through twice) show up here as a broken invariant rather than as a
numeric drift, and they show up on the first step that breaks rather than at the end.

Every property asserted here was verified against the pre-optimization engine, so a failure means
a phase changed behavior, not that the bar was set aspirationally high.

Usage:
    python tests/test_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_scenario as hs  # noqa: E402
import run_once  # noqa: E402

lmb = hs.lmb

CASES: tuple[tuple[hs.ScenarioConfig, tuple[int, ...]], ...] = (
    (hs.ScenarioConfig(num_steps=40, num_particles=200, k_best=2), (20260908, 11, 12)),
    (hs.ScenarioConfig(num_steps=60, num_particles=500, k_best=16), (20260908, 13)),
    (hs.ScenarioConfig(num_steps=60, num_particles=300, k_best=100), (20260908,)),
)

WEIGHT_SUM_TOL = 1e-12
SOLVER_TRIALS = 200
SOLVER_SEED = 20260908


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def check_run(checker: Checker, config: hs.ScenarioConfig, seed: int) -> None:
    label = f"[steps={config.num_steps} particles={config.num_particles} k_best={config.k_best} seed={seed}]"
    previous_cardinality = 0

    def observer(step, tracker, tracks, measurements, measurements_before, **_):
        nonlocal previous_cardinality
        where = f"{label} step {step}"

        # A new track is only ever created from an unused measurement, so cardinality cannot grow
        # by more than the number of measurements presented this step.
        checker.ok(
            len(tracks) <= previous_cardinality + len(measurements),
            f"{where}: cardinality {len(tracks)} exceeds previous {previous_cardinality} "
            f"+ measurements {len(measurements)}",
        )
        previous_cardinality = len(tracks)

        labels = [(int(t.label().birth_time), int(t.label().index)) for t in tracks]
        checker.ok(len(set(labels)) == len(labels), f"{where}: duplicate track labels {labels}")

        for position, track in enumerate(tracks):
            track_where = f"{where} track {position} label {labels[position]}"
            r = track.existence_probability()
            checker.ok(np.isfinite(r), f"{track_where}: existence probability {r} is not finite")
            checker.ok(0.0 <= r <= 1.0, f"{track_where}: existence probability {r} outside [0, 1]")
            checker.ok(
                r >= run_once.PRUNE_THRESHOLD,
                f"{track_where}: existence probability {r} survived pruning at "
                f"threshold {run_once.PRUNE_THRESHOLD}",
            )

            states, weights = hs.particle_arrays(track)
            checker.ok(
                weights.size == config.num_particles,
                f"{track_where}: {weights.size} particles, expected {config.num_particles}",
            )
            checker.ok(bool(np.isfinite(states).all()), f"{track_where}: non-finite state vector")
            checker.ok(bool(np.isfinite(weights).all()), f"{track_where}: non-finite particle weight")

            if weights.size:
                expected = 1.0 / weights.size
                # Post-resampling weights are set to exactly 1/N, not merely close to it.
                checker.ok(
                    bool(np.all(weights == expected)),
                    f"{track_where}: post-update weights are not all exactly 1/N "
                    f"(min {weights.min()!r}, max {weights.max()!r}, expected {expected!r})",
                )
                checker.ok(
                    abs(float(weights.sum()) - 1.0) <= WEIGHT_SUM_TOL,
                    f"{track_where}: weights sum to {weights.sum()!r}, off by more than {WEIGHT_SUM_TOL:g}",
                )

        for index, (measurement, snapshot) in enumerate(zip(measurements, measurements_before)):
            (
                range_, range_rate_, los_, los_rate_, sensor_state_, covariance_, timestamp_, sensor_id_
            ) = snapshot
            unchanged = (
                measurement.range_ == range_
                and measurement.range_rate_ == range_rate_
                and np.array_equal(np.asarray(measurement.los_), los_)
                and np.array_equal(np.asarray(measurement.los_rate_), los_rate_)
                and np.array_equal(np.asarray(measurement.sensor_state_), sensor_state_)
                and np.array_equal(np.asarray(measurement.covariance_), covariance_)
                and measurement.timestamp_ == timestamp_
                and measurement.sensor_id_ == sensor_id_
            )
            checker.ok(unchanged, f"{where}: update() mutated input measurement {index}")

    digest = hs.run_scenario(seed, config, observer=observer)

    checker.ok(bool(np.isfinite(digest.gospa).all()), f"{label}: non-finite GOSPA")
    checker.ok(bool((digest.gospa >= 0.0).all()), f"{label}: negative GOSPA")
    # Unnormalised GOSPA is NOT bounded by the cutoff -- it grows as sqrt(cardinality). The tight
    # bound is c * sqrt((m + n) / 2), attained when every pair is beyond the cutoff.
    bound = hs.gospa_upper_bound(digest.cardinality, digest.num_truths)
    checker.ok(
        bool((digest.gospa <= bound * (1.0 + 1e-12)).all()),
        f"{label}: GOSPA exceeds c*sqrt((m+n)/2) (worst ratio "
        f"{np.max(digest.gospa / np.where(bound > 0, bound, 1.0)):.6f})",
    )
    checker.ok(
        bool(np.isfinite(digest.track_cov_trace).all() & (digest.track_cov_trace >= 0.0).all()),
        f"{label}: weighted covariance trace is negative or non-finite",
    )
    print(f"  {label} {config.num_steps} steps clean, max cardinality {digest.cardinality.max()}")


def check_solver(checker: Checker) -> None:
    """Association-level invariants of solve_assignment on randomized augmented cost matrices.

    Note: hypothesis ordering by cost is deliberately *not* asserted. The current Murty
    implementation does not always return hypotheses in non-decreasing cost order, and Phase 0
    must not change engine behavior; Phase 6's differential test pins the existing behavior.
    """
    rng = np.random.default_rng(SOLVER_SEED)
    for _ in range(SOLVER_TRIALS):
        num_tracks = int(rng.integers(1, 6))
        num_meas = int(rng.integers(1, 6))
        k_best = int(rng.integers(1, 40))

        # Same shape as the tracker's augmented matrix: measurement columns then per-track miss columns.
        cost = np.full((num_tracks, num_meas + num_tracks), 1e9, dtype=np.float64)
        cost[:, :num_meas] = rng.normal(size=(num_tracks, num_meas)) * 3.0
        for i in range(num_tracks):
            cost[i, num_meas + i] = rng.normal() * 3.0

        hypotheses = lmb.solve_assignment(cost, k_best)
        where = f"[solver N={num_tracks} M={num_meas} K={k_best}]"

        checker.ok(len(hypotheses) <= k_best, f"{where}: returned {len(hypotheses)} > k_best hypotheses")
        checker.ok(len(hypotheses) >= 1, f"{where}: returned no hypotheses")

        seen: set[tuple[int, ...]] = set()
        for h in hypotheses:
            associations = tuple(int(a) for a in h.associations)
            checker.ok(
                len(associations) == num_tracks,
                f"{where}: hypothesis has {len(associations)} associations, expected {num_tracks}",
            )
            assigned = [a for a in associations if a >= 0]
            checker.ok(
                len(set(assigned)) == len(assigned),
                f"{where}: hypothesis {associations} assigns a column twice",
            )
            checker.ok(
                all(0 <= a < cost.shape[1] for a in assigned),
                f"{where}: hypothesis {associations} references a column outside the matrix",
            )
            checker.ok(np.isfinite(h.weight), f"{where}: hypothesis {associations} has cost {h.weight}")
            checker.ok(associations not in seen, f"{where}: duplicate hypothesis {associations}")
            seen.add(associations)

    print(f"  solver invariants clean over {SOLVER_TRIALS} randomized augmented matrices")


def main() -> int:
    checker = Checker()
    print("filter invariants")
    for config, seeds in CASES:
        for seed in seeds:
            check_run(checker, config, seed)
    check_solver(checker)
    print(f"PASS: test_invariants ({checker.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
