"""Phase 1 gate: the C++ cloud statistics and the zero-copy particle views.

Two things are checked, because they fail in different ways:

* `Track.mean_state()`, `Track.covariance()` and `Track.weight_sum()` must agree with an independent
  NumPy reference to rtol 1e-12, including the degenerate clouds where the contract is a fallback
  rather than a formula (empty, single particle, all-zero weights, weight sum below the 1e-12 floor).
  The pre-existing NumPy `compute_track_mean` is used as one of the references, since preserving its
  behavior is the actual requirement.

* The views must genuinely alias the track's memory rather than quietly copy it. A copy would pass
  every value comparison while silently keeping the cost the phase set out to remove, so the checks
  are on strides, ownership, base object, shared addresses and survival of the owner being deleted.

Usage:
    python tests/test_particle_statistics.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import run_once  # noqa: E402

lmb = run_once.lmb_engine

RTOL = 1e-12
WEIGHT_SUM_FLOOR = 1e-12


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)

    def close(self, actual, expected, message: str, rtol: float = RTOL) -> None:
        self.count += 1
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if not np.allclose(actual, expected, rtol=rtol, atol=0.0):
            deviation = np.max(np.abs(actual - expected) / np.where(np.abs(expected) > 0, np.abs(expected), 1.0))
            raise AssertionError(f"{message}: worst relative deviation {deviation:.3e} exceeds rtol {rtol:g}")


def make_track(states: np.ndarray, weights: np.ndarray):
    particles = []
    for state, weight in zip(states, weights):
        particle = lmb.Particle()
        particle.state_vector = np.asarray(state, dtype=np.float64)
        particle.weight = float(weight)
        particles.append(particle)
    return lmb.Track(lmb.TrackLabel(), 0.5, particles)


def reference_mean(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The contract Track.mean_state() must honor, written out independently."""
    if states.shape[0] == 0:
        return np.zeros(6)
    weight_sum = float(np.sum(weights))
    normalized = (
        weights / weight_sum
        if weight_sum > WEIGHT_SUM_FLOOR
        else np.full(states.shape[0], 1.0 / states.shape[0])
    )
    return np.average(states, weights=normalized, axis=0)


def reference_covariance(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if states.shape[0] == 0:
        return np.zeros((6, 6))
    weight_sum = float(np.sum(weights))
    normalized = (
        weights / weight_sum
        if weight_sum > WEIGHT_SUM_FLOOR
        else np.full(states.shape[0], 1.0 / states.shape[0])
    )
    deviations = states - reference_mean(states, weights)
    return np.einsum("p,pi,pj->ij", normalized, deviations, deviations)


def check_statistics(checker: Checker) -> None:
    rng = np.random.default_rng(20260908)

    cases: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("empty cloud", np.zeros((0, 6)), np.zeros(0)),
        ("single particle", rng.normal(size=(1, 6)) * 1e6, np.array([1.0])),
        ("uniform 1/N weights", rng.normal(size=(500, 6)) * 1e6, np.full(500, 1.0 / 500)),
        ("all-zero weights", rng.normal(size=(64, 6)) * 1e6, np.zeros(64)),
        ("weight sum below the floor", rng.normal(size=(64, 6)) * 1e6, np.full(64, 1e-18)),
        ("one dominant weight", rng.normal(size=(64, 6)) * 1e6, np.eye(64)[0] * 3.0),
        ("LEO-scale states", rng.normal(size=(2000, 6)) * 7e6, rng.random(2000)),
        ("tiny spread on a huge mean", np.full((300, 6), 7e6) + rng.normal(size=(300, 6)), rng.random(300)),
    ]

    for name, states, weights in cases:
        track = make_track(states, weights)

        checker.close(track.mean_state(), reference_mean(states, weights), f"[{name}] mean_state")
        checker.close(track.covariance(), reference_covariance(states, weights), f"[{name}] covariance")
        checker.close(track.weight_sum(), float(np.sum(weights)), f"[{name}] weight_sum")

        # The requirement is not "matches a reference" but "matches what the drivers used to do".
        checker.close(
            run_once.compute_track_mean(track),
            reference_mean(states, weights),
            f"[{name}] compute_track_mean still honors its original contract",
        )

        mean = np.asarray(track.mean_state())
        checker.ok(mean.shape == (6,), f"[{name}] mean_state shape {mean.shape}, expected (6,)")
        covariance = np.asarray(track.covariance())
        checker.ok(covariance.shape == (6, 6), f"[{name}] covariance shape {covariance.shape}")
        checker.close(covariance, covariance.T, f"[{name}] covariance is not symmetric")
        if states.shape[0] > 0:
            checker.ok(
                bool((np.diag(covariance) >= 0.0).all()),
                f"[{name}] covariance has a negative variance on the diagonal",
            )

        print(f"  [{name}] N={states.shape[0]} statistics match the reference")


def check_views(checker: Checker) -> None:
    rng = np.random.default_rng(11)
    count = 257  # deliberately not a round number, to catch a stride or shape assumption
    states = rng.normal(size=(count, 6)) * 7e6
    weights = rng.random(count)
    track = make_track(states, weights)

    view_states = track.particle_states()
    view_weights = track.particle_weights()

    checker.ok(view_states.shape == (count, 6), f"particle_states shape {view_states.shape}")
    checker.ok(view_weights.shape == (count,), f"particle_weights shape {view_weights.shape}")
    checker.ok(view_states.dtype == np.float64, f"particle_states dtype {view_states.dtype}")
    checker.ok(view_weights.dtype == np.float64, f"particle_weights dtype {view_weights.dtype}")

    # A copy would satisfy every value comparison, so ownership and strides are the real evidence.
    checker.ok(not view_states.flags.owndata, "particle_states owns its data, so it is a copy")
    checker.ok(not view_weights.flags.owndata, "particle_weights owns its data, so it is a copy")
    checker.ok(view_states.strides == (64, 8), f"particle_states strides {view_states.strides}, expected (64, 8)")
    checker.ok(view_weights.strides == (64,), f"particle_weights strides {view_weights.strides}, expected (64,)")
    checker.ok(view_states.base is track, "particle_states base is not the owning track")
    checker.ok(view_weights.base is track, "particle_weights base is not the owning track")

    checker.ok(not view_states.flags.writeable, "particle_states is writeable; it aliases filter state")
    checker.ok(not view_weights.flags.writeable, "particle_weights is writeable; it aliases filter state")

    states_address = view_states.__array_interface__["data"][0]
    weights_address = view_weights.__array_interface__["data"][0]
    checker.ok(
        track.particle_states().__array_interface__["data"][0] == states_address,
        "two particle_states views do not share an address, so at least one is a copy",
    )
    checker.ok(
        weights_address - states_address == 48,
        f"weight is {weights_address - states_address} bytes into Particle, expected 48",
    )

    checker.close(view_states, states, "particle_states values")
    checker.close(view_weights, weights, "particle_weights values")

    particles = track.particles()
    checker.close(
        np.array([p.state_vector for p in particles]), view_states, "particle_states vs particles()"
    )
    checker.close(np.array([p.weight for p in particles]), view_weights, "particle_weights vs particles()")

    # keep_alive: the array holds the last reference to the track once the name is gone.
    del track, particles
    gc.collect()
    checker.close(view_states, states, "particle_states after the owning track was deleted")
    checker.close(view_weights, weights, "particle_weights after the owning track was deleted")

    empty = lmb.Track(lmb.TrackLabel(), 0.5, [])
    checker.ok(empty.particle_states().shape == (0, 6), "empty particle_states shape")
    checker.ok(empty.particle_weights().shape == (0,), "empty particle_weights shape")

    print(f"  views alias {count} particles, are read-only, and survive owner deletion")


def check_against_live_tracker(checker: Checker) -> None:
    """The statistics must also agree on clouds the filter actually produces."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import harness_scenario as hs

    inspected = 0

    def observer(step, tracks, **_):
        nonlocal inspected
        for track in tracks:
            states = np.asarray(track.particle_states())
            weights = np.asarray(track.particle_weights())
            checker.close(
                track.mean_state(), reference_mean(states, weights), f"step {step} live mean_state"
            )
            checker.close(
                track.covariance(),
                reference_covariance(states, weights),
                f"step {step} live covariance",
            )
            inspected += 1

    hs.run_scenario(20260908, hs.ScenarioConfig(num_steps=40, num_particles=300, k_best=8), observer=observer)
    checker.ok(inspected > 0, "no live tracks were inspected")
    print(f"  {inspected} live filter clouds match the reference")


def main() -> int:
    checker = Checker()
    print("cloud statistics")
    check_statistics(checker)
    print("zero-copy views")
    check_views(checker)
    print("live tracker clouds")
    check_against_live_tracker(checker)
    print(f"PASS: test_particle_statistics ({checker.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
