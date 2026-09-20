"""Phase 3 gate: localResidual with a precomputed tangent basis is bit-identical to the old overload.

The three-argument form exists so MeasurementLikelihoodCache can hoist tangentBasis(measured.los)
once per measurement instead of once per particle. That only preserves filter behavior if the new
overload returns exactly the same residual as the two-argument form whenever its basis argument is
tangent_basis(measured.los).

Coverage deliberately hits the sensitive branches in los_geometry.h:
  * ordinary random pairs of directions;
  * near-identical directions (series path in logMap);
  * near-antipodal directions (conventional pi * e1 branch);
  * near-pole directions (helper-axis choice in tangentBasis);
  * exactly equal directions (zero residual for the angular block when rates also match).

Usage:
    python tests/test_local_residual_cached_basis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

RNG_SEED = 20260920
N_RANDOM = 400
N_NEAR_IDENTICAL = 80
N_NEAR_ANTIPODAL = 80
N_NEAR_POLE = 80


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def make_observation(los: np.ndarray, los_rate: np.ndarray, range_: float, range_rate: float):
    obs = lmb.LosObservation()
    obs.los = los / np.linalg.norm(los)
    # Keep the rate tangent to the direction, matching the physical model.
    rate = los_rate - np.dot(los_rate, obs.los) * obs.los
    obs.los_rate = rate
    obs.range = float(range_)
    obs.range_rate = float(range_rate)
    return obs


def random_unit(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def random_tangent(rng: np.random.Generator, direction: np.ndarray, scale: float) -> np.ndarray:
    vector = rng.normal(size=3)
    vector = vector - np.dot(vector, direction) * direction
    norm = np.linalg.norm(vector)
    if norm < 1e-15:
        basis = np.asarray(lmb.tangent_basis(direction))
        return scale * basis[:, 0]
    return scale * vector / norm


def assert_bit_identical(checker: Checker, measured, predicted, where: str) -> None:
    basis = lmb.tangent_basis(measured.los)
    old = np.asarray(lmb.local_residual(measured, predicted), dtype=np.float64)
    new = np.asarray(lmb.local_residual(measured, predicted, basis), dtype=np.float64)
    checker.ok(
        old.shape == (6,) and new.shape == (6,),
        f"{where}: unexpected residual shape old={old.shape} new={new.shape}",
    )
    # Bitwise: no tolerance. equal_nan is unnecessary; residuals must be finite.
    checker.ok(bool(np.isfinite(old).all() and np.isfinite(new).all()), f"{where}: non-finite residual")
    mismatches = np.flatnonzero(old != new)
    checker.ok(
        mismatches.size == 0,
        f"{where}: {mismatches.size} bitwise mismatches, first at {int(mismatches[0]) if mismatches.size else -1} "
        f"(old {old!r}, new {new!r})",
    )


def check_random_pairs(checker: Checker, rng: np.random.Generator) -> None:
    for index in range(N_RANDOM):
        measured_los = random_unit(rng)
        predicted_los = random_unit(rng)
        measured = make_observation(
            measured_los,
            random_tangent(rng, measured_los, rng.uniform(1e-6, 1e-2)),
            rng.uniform(1e5, 1e7),
            rng.uniform(-1e3, 1e3),
        )
        predicted = make_observation(
            predicted_los,
            random_tangent(rng, predicted_los, rng.uniform(1e-6, 1e-2)),
            rng.uniform(1e5, 1e7),
            rng.uniform(-1e3, 1e3),
        )
        assert_bit_identical(checker, measured, predicted, f"random[{index}]")
    print(f"  random pairs: {N_RANDOM} bit-identical")


def check_near_identical(checker: Checker, rng: np.random.Generator) -> None:
    for index in range(N_NEAR_IDENTICAL):
        measured_los = random_unit(rng)
        angle = 10.0 ** rng.uniform(-16, -6)
        step = random_tangent(rng, measured_los, angle)
        predicted_los = np.asarray(lmb.exp_map(measured_los, step))
        measured = make_observation(
            measured_los,
            random_tangent(rng, measured_los, 1e-4),
            7e6,
            10.0,
        )
        predicted = make_observation(
            predicted_los,
            random_tangent(rng, predicted_los, 1e-4),
            7e6 + rng.normal() * 1e-3,
            10.0 + rng.normal() * 1e-4,
        )
        assert_bit_identical(checker, measured, predicted, f"near_identical[{index}]")
    print(f"  near-identical pairs: {N_NEAR_IDENTICAL} bit-identical")


def check_near_antipodal(checker: Checker, rng: np.random.Generator) -> None:
    for index in range(N_NEAR_ANTIPODAL):
        measured_los = random_unit(rng)
        # Exactly antipodal and nearly antipodal both exercise the logMap antipode branch.
        if index % 2 == 0:
            predicted_los = -measured_los
        else:
            wobble = random_tangent(rng, -measured_los, 10.0 ** rng.uniform(-16, -10))
            predicted_los = (-measured_los + wobble)
            predicted_los = predicted_los / np.linalg.norm(predicted_los)
        measured = make_observation(measured_los, random_tangent(rng, measured_los, 1e-5), 7e6, 0.0)
        predicted = make_observation(predicted_los, random_tangent(rng, predicted_los, 1e-5), 7e6, 0.0)
        assert_bit_identical(checker, measured, predicted, f"near_antipodal[{index}]")
    print(f"  near-antipodal pairs: {N_NEAR_ANTIPODAL} bit-identical")


def check_near_pole(checker: Checker, rng: np.random.Generator) -> None:
    poles = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for index in range(N_NEAR_POLE):
        pole = poles[index % len(poles)]
        measured_los = pole + rng.normal(size=3) * 10.0 ** rng.uniform(-10, -4)
        measured_los = measured_los / np.linalg.norm(measured_los)
        predicted_los = random_unit(rng)
        measured = make_observation(measured_los, random_tangent(rng, measured_los, 1e-4), 7e6, 1.0)
        predicted = make_observation(predicted_los, random_tangent(rng, predicted_los, 1e-4), 7e6, 1.0)
        assert_bit_identical(checker, measured, predicted, f"near_pole[{index}]")
    print(f"  near-pole pairs: {N_NEAR_POLE} bit-identical")


def check_exactly_equal(checker: Checker) -> None:
    measured = make_observation(
        np.array([0.1, 0.2, np.sqrt(1.0 - 0.01 - 0.04)]),
        np.array([1e-5, -2e-5, 0.0]),
        6.8e6,
        12.5,
    )
    predicted = make_observation(
        np.asarray(measured.los).copy(),
        np.asarray(measured.los_rate).copy(),
        measured.range,
        measured.range_rate,
    )
    assert_bit_identical(checker, measured, predicted, "exactly_equal")
    residual = np.asarray(lmb.local_residual(measured, predicted), dtype=np.float64)
    # Range and direction residuals must be exact zeros when the observations are identical.
    # Rate residuals can pick up ~1e-22 parallel-transport roundoff even at equal directions.
    checker.ok(bool(np.all(residual[:4] == 0.0)), f"exactly_equal range/angle residual not zero: {residual!r}")
    checker.ok(bool(np.max(np.abs(residual[4:])) < 1e-18), f"exactly_equal rate residual too large: {residual!r}")
    print("  exactly-equal pair: bit-identical; range/angle residual exactly zero")


def check_wrong_basis_differs(checker: Checker, rng: np.random.Generator) -> None:
    """Sanity: a deliberately wrong basis must not silently match, or the test is vacuous."""
    measured_los = random_unit(rng)
    predicted_los = random_unit(rng)
    measured = make_observation(measured_los, random_tangent(rng, measured_los, 1e-3), 7e6, 0.0)
    predicted = make_observation(predicted_los, random_tangent(rng, predicted_los, 1e-3), 7.1e6, 1.0)
    wrong_basis = lmb.tangent_basis(predicted_los)
    old = np.asarray(lmb.local_residual(measured, predicted), dtype=np.float64)
    spoofed = np.asarray(lmb.local_residual(measured, predicted, wrong_basis), dtype=np.float64)
    # Angular components (indices 2..5) are the ones that depend on the basis.
    checker.ok(
        not np.array_equal(old[2:], spoofed[2:]),
        "spoofed-basis residual matched the true one on the angular block; the differential test "
        "would not catch a wiring bug that ignores the basis argument",
    )
    print("  spoofed-basis check: angular block differs as expected")


def main() -> int:
    checker = Checker()
    rng = np.random.default_rng(RNG_SEED)
    print("localResidual cached-basis differential")
    check_random_pairs(checker, rng)
    check_near_identical(checker, rng)
    check_near_antipodal(checker, rng)
    check_near_pole(checker, rng)
    check_exactly_equal(checker)
    check_wrong_basis_differs(checker, rng)
    print(f"PASS: test_local_residual_cached_basis ({checker.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
