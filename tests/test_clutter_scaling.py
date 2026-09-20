"""The clutter intensity kappa must enter the detection cost exactly once.

Step 2 of ``update`` stores the raw weighted-average likelihood ``L = sum_p w_p g(z|x_p)`` in
``likelihood_matrix``; Step 3 forms the detection cost ``-ln(P_D * L / kappa)`` and Step 5a the
mixture coefficient ``P_D * L / kappa``. Each consumer applies its own single ``1/kappa``.

Regression guarded here: ``likelihood_matrix`` used to hold ``L/kappa`` (the division sat in the
per-particle loop), while Step 3 divided by kappa *again*, so the cost matrix encoded
``-ln(P_D * L / kappa^2)``. That is not a constant offset -- it lands on the detection columns but
not on the missed-detection column, so it moves the detection/miss crossover by a factor of kappa
and skews hypothesis weights by ``kappa^-d`` in the number of detections ``d``.

The crossover is the sharp, observable consequence. With one track and one measurement the
augmented cost matrix is ``[detect_cost, miss_cost]``, and detection wins exactly when

    P_D * L / kappa > 1 - P_D        i.e.   kappa < kappa* = P_D * L / (1 - P_D)

The doubled-kappa form instead flips at ``kappa < sqrt(kappa*)``. Those differ by orders of
magnitude for the small kappa this filter is configured with, so a sweep separates them cleanly.

The best hypothesis is read through adaptive birth rather than through private state: Step 6 spawns
a track from every measurement the best hypothesis left unassigned, so a miss yields two tracks and
a detection yields one.

Usage:
    python tests/test_clutter_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

from lmb_engine_loader import import_lmb_engine  # noqa: E402
import reference_geometry as ref  # noqa: E402

lmb = import_lmb_engine()

FIXED_SEED = 20260920
SIGMAS = np.array([100.0, 1.0, 1e-3, 1e-3, 1e-5, 1e-5])
P_DETECTION = 0.9
P_SURVIVAL = 0.99
PRUNE_THRESHOLD = 1e-6      # low enough that neither branch prunes the surviving track
K_BEST = 8
N_PARTICLES = 32
CLOUD_SPREAD = 10.0         # m, tight enough that L is dominated by the offset below
RANGE_OFFSET = 1000.0       # m along the line of sight => ~10 sigma_range, so L is tiny and
                            # kappa* lands far below 1 (a discriminating window needs kappa* < 1)


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def scene(rng: np.random.Generator):
    """LEO sensor with a target ~1e6 m away; returns (target_state, sensor_state, los)."""
    sensor_dir = ref.random_unit_vectors(rng, 1)[0]
    sensor_pos = 6.771e6 * sensor_dir
    sensor_vel = 7.67e3 * ref.unit(np.cross(sensor_dir, ref.random_unit_vectors(rng, 1)[0]))
    los = ref.unit(np.cross(sensor_vel, ref.random_unit_vectors(rng, 1)[0]))
    rho = 1.0e6
    target = np.concatenate([sensor_pos + rho * los, sensor_vel])
    return target, np.concatenate([sensor_pos, sensor_vel]), los


def sample_cloud(rng: np.random.Generator, target: np.ndarray) -> np.ndarray:
    """One fixed particle cloud, drawn once and reused verbatim.

    Every tracker in the sweep must see the identical cloud, and it must be the same one the
    reference L was measured on -- otherwise the measured crossover is compared against an L the
    filter never actually computed, and the comparison silently loses a few percent.
    """
    offsets = np.concatenate(
        [rng.normal(0.0, CLOUD_SPREAD, (N_PARTICLES, 3)),
         rng.normal(0.0, CLOUD_SPREAD * 1e-3, (N_PARTICLES, 3))],
        axis=1,
    )
    return target[None, :] + offsets


def make_track(cloud: np.ndarray) -> "lmb.Track":
    particles = []
    for state in cloud:
        particle = lmb.Particle()
        particle.state_vector = state
        particle.weight = 1.0 / len(cloud)
        particles.append(particle)
    return lmb.Track(lmb.TrackLabel(), 0.8, particles)


def build_tracker(clutter_intensity: float, seed: int = FIXED_SEED):
    sensor_model = lmb.InOrbitSensorModel(*SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(N_PARTICLES, 0.5, np.diag(SIGMAS**2), seed=seed)
    propagator = lmb.TwoBodyPropagator(np.zeros((6, 6)))
    return lmb.SMC_LMB_Tracker(
        propagator, sensor_model, birth_model,
        P_SURVIVAL, K_BEST, PRUNE_THRESHOLD, clutter_intensity, P_DETECTION, 0.0, 1.0,
        seed=seed,
    )


def build_case(rng: np.random.Generator):
    """A track and a deliberately offset measurement, plus the L the filter will compute for them."""
    target, sensor, los = scene(rng)
    measurement = lmb.Measurement.fromCartesian(
        target + np.concatenate([RANGE_OFFSET * los, np.zeros(3)]), sensor
    )
    measurement.covariance_ = np.diag(SIGMAS**2)
    cloud = sample_cloud(rng, target)
    # compute_association_likelihood returns the raw L, with no kappa and no P_D applied -- the
    # same quantity Step 2 now stores. It is unaffected by the kappa placement under test.
    likelihood = build_tracker(1.0).compute_association_likelihood(make_track(cloud), measurement)
    return cloud, measurement, likelihood


def track_count_after_update(cloud, measurement, clutter_intensity: float) -> int:
    tracker = build_tracker(clutter_intensity)
    tracker.set_tracks([make_track(cloud)])
    tracker.update([measurement])
    return len(tracker.get_tracks())


def find_crossover(cloud, measurement, lo: float, hi: float) -> float:
    """Bisect on kappa for the largest value at which the best hypothesis still takes the detection.

    Detection (1 track) holds for small kappa and miss (2 tracks) for large kappa, so the predicate
    is monotone and bisection is well posed.
    """
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if track_count_after_update(cloud, measurement, mid) == 1:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def main() -> int:
    chk = Checker()
    rng = np.random.default_rng(FIXED_SEED)
    cloud, measurement, likelihood = build_case(rng)

    kappa_star = P_DETECTION * likelihood / (1.0 - P_DETECTION)
    kappa_star_buggy = np.sqrt(kappa_star)

    print("clutter scaling")
    print(f"  L (raw association likelihood) = {likelihood:.6e}")
    print(f"  predicted crossover, single kappa  kappa* = P_D*L/(1-P_D) = {kappa_star:.6e}")
    print(f"  predicted crossover, doubled kappa sqrt(kappa*)          = {kappa_star_buggy:.6e}")

    chk.ok(np.isfinite(likelihood) and likelihood > 0.0,
           f"association likelihood must be finite and positive, got {likelihood!r}")
    chk.ok(kappa_star < 1.0,
           f"test scene is not discriminating: kappa* = {kappa_star:.3e} must be < 1 so that "
           f"sqrt(kappa*) is distinguishable from it")

    # Bracket both candidates by a wide margin so the bisection cannot miss either.
    lo, hi = kappa_star * 1e-4, kappa_star_buggy * 1e4

    # The predicate must actually switch across the bracket, or the bisection is meaningless.
    chk.ok(track_count_after_update(cloud, measurement, lo) == 1,
           f"expected a detection (1 track) at kappa = {lo:.3e}")
    chk.ok(track_count_after_update(cloud, measurement, hi) == 2,
           f"expected a miss (2 tracks) at kappa = {hi:.3e}")

    crossover = find_crossover(cloud, measurement, lo, hi)
    print(f"  measured crossover                                       = {crossover:.6e}")

    # 5% in log space: the crossover is a sharp threshold on a scalar comparison, so the only
    # slack needed is the bisection's own resolution.
    log_err_correct = abs(np.log(crossover / kappa_star))
    log_err_buggy = abs(np.log(crossover / kappa_star_buggy))
    print(f"  |log| deviation from kappa*      = {log_err_correct:.3e}")
    print(f"  |log| deviation from sqrt(kappa*) = {log_err_buggy:.3e}")

    chk.ok(log_err_correct < 0.05,
           f"crossover {crossover:.6e} does not match the single-kappa prediction "
           f"{kappa_star:.6e}; kappa is not applied exactly once on the detection cost path")
    chk.ok(log_err_correct < log_err_buggy,
           f"crossover {crossover:.6e} is closer to the doubled-kappa prediction "
           f"{kappa_star_buggy:.6e} than to {kappa_star:.6e}")

    print(f"PASS: test_clutter_scaling ({chk.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
