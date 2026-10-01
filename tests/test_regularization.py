"""The regularization (kernel jitter) step and the posterior diagnostics.

SMC_LMB_Tracker.set_regularization moves each resampled particle as

    x <- m + a (x - m) + h L eps,   a = sqrt(1 - h^2),  L L^T = S,  eps ~ N(0, I6)

with m, S the weighted mean and covariance of the posterior it was resampled from (Liu & West
kernel shrinkage). Checked here:

  R1  off by default, and ess_threshold = 0 never fires: resampled particles are exact copies;
  R2  when it fires, the cloud keeps the exact posterior's mean and covariance (the point of the
      shrinkage) and no two particles coincide (the point of the jitter);
  R3  the recorded ESS is 1 / sum w^2 of the exact posterior weights, and the detection mass is
      the posterior probability of a detection;
  R4  argument validation.

The exact posterior comes from tests/lmb_reference.py.

Usage:
    python tests/test_regularization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

from lmb_engine_loader import import_lmb_engine  # noqa: E402
import lmb_reference as ref  # noqa: E402
import test_existence_enumeration as scene  # noqa: E402

lmb = import_lmb_engine()

SEED = 20260930
N = 4000
WIDE_SIGMAS = np.array([2000.0, 40.0, 0.15, 0.15, 0.01, 0.01])


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)

    def raises(self, fn, exc, fragment: str, message: str) -> None:
        self.count += 1
        try:
            fn()
        except exc as error:
            if fragment not in str(error):
                raise AssertionError(f"{message}: message {str(error)!r} lacks {fragment!r}")
            return
        raise AssertionError(f"{message}: no {exc.__name__} raised")


def build_case(rng):
    """One track straddling a sensor volume and a measurement near its centre."""
    sensors = scene.build_sensors()
    sensor = np.asarray(sensors.state(0))
    # A slow (~100 m/s) co-moving target. At the ~10 km/s of a real ring crossing, both rate
    # channels move by metres-per-second or more per metre of position error, so the posterior
    # collapses onto one particle -- and moments of a one-particle posterior say nothing about
    # the kernel. A slow target keeps the rates insensitive to position.
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    centre = np.concatenate([sensor[:3] + 12e3 * direction, sensor[3:] + rng.normal(0, 60.0, 3)])
    states, weights = scene.random_cloud(rng, centre, N, spread=3e3)
    track = ref.TrackInput(existence=0.7, states=states, weights=weights)
    target = centre + np.concatenate([rng.normal(0, 1000.0, 3), rng.normal(0, 10.0, 3)])
    k = sensors.visible_sensor(target)
    assert k >= 0, "scaffolding: the measurement must be visible"
    measurement = scene.make_measurement(target, sensors, k)
    measurement.covariance_ = np.diag(WIDE_SIGMAS**2)
    return sensors, track, measurement


def run_update(sensors, track, measurement, regularization, threshold=1.0, kappa=1e-12, p_detection=0.9):
    tracker, sensor_model = scene.make_tracker(kappa, p_detection, SEED)
    if regularization is not None:
        tracker.set_regularization(regularization, 1.0, threshold)
    tracker.set_record_diagnostics(True)
    tracker.set_tracks([scene.to_track(track, 0)])
    tracker.update([measurement], sensors)
    return tracker, sensor_model


def weighted_moments(states, weights):
    weights = np.asarray(weights) / np.sum(weights)
    mean = weights @ states
    offset = states - mean
    return mean, (offset * weights[:, None]).T @ offset


def check_off_and_threshold(chk, sensors, track, measurement):
    default, _ = scene.make_tracker(1e-12, 0.9, SEED)
    chk.ok(not default.regularization, "regularization must be off by default")

    for label, setting, threshold in (("default", None, 1.0), ("threshold 0", True, 0.0)):
        tracker, _ = run_update(sensors, track, measurement, setting, threshold)
        states = np.asarray(tracker.get_tracks()[0].particle_states())
        originals = {row.tobytes() for row in track.states}
        chk.ok(all(row.tobytes() in originals for row in states),
               f"{label}: without jitter every resampled particle must be an exact original")
        diagnostics = tracker.take_diagnostics()
        chk.ok(len(diagnostics["time"]) == 1 and not diagnostics["regularized"][0],
               f"{label}: exactly one record, marked not regularized")


def check_moments_preserved(chk, sensors, track, measurement):
    tracker, sensor_model = run_update(sensors, track, measurement, True, threshold=1.0)
    expected = ref.posterior(lmb, sensor_model, [track], [measurement], sensors, 0.9, 1e-12)
    mixture = expected.mixture[0]
    ess = 1.0 / np.sum(mixture**2)
    chk.ok(0.02 * N < ess < 0.9 * N,
           f"scaffolding: the posterior must be non-trivial but not degenerate, ESS/N = {ess / N:.3f}")

    ref_mean, ref_cov = weighted_moments(track.states, mixture)
    states = np.asarray(tracker.get_tracks()[0].particle_states())
    got_mean, got_cov = weighted_moments(states, np.ones(len(states)))

    distinct = len({row.tobytes() for row in states})
    chk.ok(distinct == N, f"regularized cloud must have no duplicates, {distinct} distinct of {N}")

    sigma = np.sqrt(np.diag(ref_cov))
    # Resampling from ESS effective particles, then jitter: the mean's sampling error is ~sigma/sqrt(ESS).
    tolerance = 6.0 * sigma / np.sqrt(min(ess, N)) + 1e-9
    chk.ok(np.all(np.abs(got_mean - ref_mean) < tolerance),
           f"regularized mean must match the posterior mean: gap/sigma {np.abs(got_mean - ref_mean) / sigma}")
    ratio = np.diag(got_cov) / np.diag(ref_cov)
    chk.ok(np.all(np.abs(ratio - 1.0) < 0.2),
           f"shrinkage must keep the posterior variance, ratios {np.round(ratio, 3)}")

    diagnostics = tracker.take_diagnostics()
    chk.ok(bool(diagnostics["regularized"][0]), "the record must say the update was regularized")
    chk.ok(np.isclose(diagnostics["ess"][0], ess, rtol=1e-6),
           f"recorded ESS {diagnostics['ess'][0]:.3f} != 1/sum w^2 of the exact posterior {ess:.3f}")
    detected = sum(w for w, theta in zip(expected.hypothesis_weights,
                                         ref.enumerate_hypotheses(1, 1)) if theta[0] >= 0)
    chk.ok(np.isclose(diagnostics["detection_mass"][0], detected, rtol=1e-9, atol=1e-12),
           f"recorded detection mass {diagnostics['detection_mass'][0]} != {detected}")
    chk.ok(len(tracker.take_diagnostics()["time"]) == 0, "take_diagnostics must clear the records")
    print(f"    ESS/N {ess / N:.3f}; variance ratios {np.round(ratio, 3)}; {distinct} distinct particles")


def check_validation(chk):
    tracker, _ = scene.make_tracker(1e-12, 0.9, SEED)
    chk.raises(lambda: tracker.set_regularization(True, 0.0), ValueError, "bandwidth_scale",
               "a zero bandwidth must be rejected")
    chk.raises(lambda: tracker.set_regularization(True, 1.0, 1.5), ValueError, "ess_threshold",
               "a threshold above 1 must be rejected")
    tracker.set_regularization(True)
    chk.ok(tracker.regularization, "the setter must round-trip")


def main() -> int:
    chk = Checker()
    rng = np.random.default_rng(SEED)
    sensors, track, measurement = build_case(rng)
    print("regularization")
    for name, fn in (
        ("R1 off by default, threshold respected", lambda: check_off_and_threshold(chk, sensors, track, measurement)),
        ("R2/R3 moments preserved, diversity restored, diagnostics", lambda: check_moments_preserved(chk, sensors, track, measurement)),
        ("R4 validation", lambda: check_validation(chk)),
    ):
        before = chk.count
        fn()
        print(f"  {name}: {chk.count - before} assertions")
    print(f"PASS: test_regularization ({chk.count} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
