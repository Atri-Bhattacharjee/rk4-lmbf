"""Targeted tests for the association-grouped mixture update.

``update`` used to enumerate one mixture entry per (hypothesis, particle) pair. It now groups the
hypotheses by the association they assign to a track, because only ``D <= num_meas + 1`` distinct
per-particle vectors exist no matter how large ``k_best`` is. This file pins the two things the
golden and statistical harnesses do not pin on their own:

1. the algebraic identity the grouping relies on -- ``sum_h e(h, p)`` really does equal the grouped
   ``W[p]`` -- checked against an explicit ``K*P`` reference summation;
2. the engine's behaviour on ragged and degenerate particle clouds, since the flat association
   buffer indexes tracks through a prefix-sum table specifically so that per-track particle counts
   are free to differ.

Why (1) needs its own test: in every scenario the golden fixtures cover, the hypothesis weights
saturate -- ``max(norm_weights)`` is exactly 1.0 -- so the mixture collapses to a single hypothesis
and the grouped and enumerated forms coincide bitwise. Those fixtures therefore confirm the
rewrite is faithful but never exercise a genuinely diffuse mixture. The reference summation here
does.

Usage:
    python tests/test_mixture_grouping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

import reference_geometry as ref  # noqa: E402
from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

GROUPING_RTOL = 1e-12


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


# --------------------------------------------------------------------------------------------
# 1. The algebraic identity
# --------------------------------------------------------------------------------------------

def enumerated_reference(hyp_weight, assoc, norm_weights, likelihood, p_detection, particle_weight):
    """The pre-grouping form: one entry per (hypothesis, particle), summed per particle.

    Mirrors the old inner loop exactly, including the miss branch and the fall-through for an
    association index that belongs to neither range.
    """
    num_meas = norm_weights.shape[0]
    num_particles = norm_weights.shape[1]
    totals = np.zeros(num_particles)
    contributing = 0
    for h, j in enumerate(assoc):
        if 0 <= j < num_meas:
            totals += norm_weights[j] * hyp_weight[h] * p_detection * likelihood[j]
            contributing += 1
        elif j == -1 or j >= num_meas:  # every miss column shares one formula
            totals += particle_weight * hyp_weight[h] * (1.0 - p_detection)
            contributing += 1
        # anything else contributes nothing, exactly as the engine's fall-through does
    return totals, contributing


def grouped(hyp_weight, assoc, norm_weights, likelihood, p_detection, particle_weight):
    """The grouped form the engine now implements: collapse hypotheses, then one pass per bucket."""
    num_meas = norm_weights.shape[0]
    coefficients = np.zeros(num_meas)
    used = np.zeros(num_meas, dtype=bool)
    miss_coefficient = 0.0
    miss_used = False
    contributing = 0
    for h, j in enumerate(assoc):
        if 0 <= j < num_meas:
            coefficients[j] += hyp_weight[h]
            used[j] = True
            contributing += 1
        elif j == -1 or j >= num_meas:
            miss_coefficient += hyp_weight[h]
            miss_used = True
            contributing += 1
    coefficients *= p_detection * likelihood
    miss_coefficient *= 1.0 - p_detection

    totals = np.zeros(norm_weights.shape[1])
    for j in range(num_meas):          # ascending, matching the engine's fixed summation order
        if used[j]:
            totals += coefficients[j] * norm_weights[j]
    if miss_used:
        totals += miss_coefficient * particle_weight
    return totals, contributing


def check_worked_example(chk: Checker) -> None:
    """The hand-computed case from the design discussion.

    Three hypotheses share one association with scales 0.72/0.54/0.18 over [0.2, 0.5, 0.3], and a
    fourth carries 0.90 over [0.6, 0.1, 0.3]. Grouping must give [0.828, 0.810, 0.702].
    """
    first = np.array([0.2, 0.5, 0.3])
    second = np.array([0.6, 0.1, 0.3])
    expected = (0.72 + 0.54 + 0.18) * first + 0.90 * second

    chk.ok(np.allclose(expected, [0.828, 0.810, 0.702], rtol=0, atol=1e-12),
           f"worked example arithmetic drifted: {expected}")

    # Route it through both implementations with p_detection and likelihood folded into the scales.
    norm_weights = np.vstack([first, second])
    likelihood = np.array([1.0, 1.0])
    p_detection = 1.0
    hyp_weight = np.array([0.72, 0.54, 0.18, 0.90])
    assoc = np.array([0, 0, 0, 1])
    particle_weight = np.full(3, 1.0 / 3.0)

    ref_totals, ref_contributing = enumerated_reference(
        hyp_weight, assoc, norm_weights, likelihood, p_detection, particle_weight)
    grp_totals, grp_contributing = grouped(
        hyp_weight, assoc, norm_weights, likelihood, p_detection, particle_weight)

    chk.ok(np.allclose(grp_totals, expected, rtol=GROUPING_RTOL, atol=0.0),
           f"grouped totals {grp_totals} != hand-computed {expected}")
    chk.ok(np.allclose(grp_totals, ref_totals, rtol=GROUPING_RTOL, atol=0.0),
           f"grouped {grp_totals} != enumerated {ref_totals}")
    chk.ok(ref_contributing == grp_contributing == 4,
           f"contributing count {ref_contributing}/{grp_contributing}, expected 4")
    print(f"  worked example: W = {grp_totals} (matches the hand computation)")


def check_randomized_identity(chk: Checker, rng: np.random.Generator) -> None:
    """Grouping must reproduce the enumerated K*P sum for diffuse mixtures too.

    Deliberately uses non-saturated hypothesis weights, which the golden scenarios never produce.
    """
    worst = 0.0
    for case in range(200):
        num_meas = int(rng.integers(1, 5))
        num_particles = int(rng.integers(1, 40))
        num_hyp = int(rng.integers(1, 60))
        num_tracks = int(rng.integers(1, 4))

        hyp_weight = rng.random(num_hyp)
        hyp_weight /= hyp_weight.sum()
        if case % 7 == 0 and num_hyp > 1:
            hyp_weight[0] = 0.0  # a zero-weight hypothesis still counts as contributing

        # Associations span detections, -1, the miss columns, and out-of-range fall-through values.
        choices = list(range(num_meas)) + [-1] + list(range(num_meas, num_meas + num_tracks)) + [-9]
        assoc = np.array([choices[int(rng.integers(0, len(choices)))] for _ in range(num_hyp)])

        norm_weights = rng.random((num_meas, num_particles))
        norm_weights /= norm_weights.sum(axis=1, keepdims=True)
        likelihood = 10.0 ** rng.uniform(-6.0, 3.0, size=num_meas)
        p_detection = float(rng.uniform(0.5, 0.999))
        particle_weight = np.full(num_particles, 1.0 / num_particles)

        ref_totals, ref_contributing = enumerated_reference(
            hyp_weight, assoc, norm_weights, likelihood, p_detection, particle_weight)
        grp_totals, grp_contributing = grouped(
            hyp_weight, assoc, norm_weights, likelihood, p_detection, particle_weight)

        chk.ok(ref_contributing == grp_contributing,
               f"case {case}: contributing {grp_contributing} != reference {ref_contributing}")
        chk.ok(np.allclose(grp_totals, ref_totals, rtol=GROUPING_RTOL, atol=0.0),
               f"case {case}: grouped totals differ from the enumerated reference")

        scale = np.maximum(np.abs(ref_totals), 1e-300)
        worst = max(worst, float(np.max(np.abs(grp_totals - ref_totals) / scale)))

        # The out-of-range value must never contribute.
        chk.ok(ref_contributing == int(np.sum(assoc != -9)),
               f"case {case}: fall-through association was counted as contributing")

    print(f"  randomized identity: 200 cases, worst relative deviation {worst:.3e} (rtol {GROUPING_RTOL:g})")


# --------------------------------------------------------------------------------------------
# 2. Engine behaviour on ragged and degenerate clouds
# --------------------------------------------------------------------------------------------

def build_tracker(seed: int = 5):
    sigmas = np.array([100.0, 1.0, 1e-3, 1e-3, 1e-5, 1e-5])
    sensor_model = lmb.InOrbitSensorModel(*sigmas**2)
    birth_model = lmb.AdaptiveBirthModel(8, 0.5, np.diag(sigmas**2), seed=seed)
    propagator = lmb.TwoBodyPropagator(np.zeros((6, 6)))
    # k_best is large enough that several hypotheses survive for a multi-track, multi-measurement step.
    return lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model,
                               0.99, 32, 1e-6, 1e-9, 0.9, 0.0, 1.0), sigmas


def scene(rng: np.random.Generator):
    sensor_dir = ref.random_unit_vectors(rng, 1)[0]
    sensor_pos = 6.771e6 * sensor_dir
    sensor_vel = 7.67e3 * ref.unit(np.cross(sensor_dir, ref.random_unit_vectors(rng, 1)[0]))
    los = ref.random_unit_vectors(rng, 1)[0]
    rho = rng.uniform(2.0e5, 2.0e6)
    transverse = ref.unit(np.cross(los, ref.random_unit_vectors(rng, 1)[0])) * 1e-3 * rho
    target = np.concatenate([sensor_pos + rho * los, sensor_vel + transverse])
    return target, np.concatenate([sensor_pos, sensor_vel])


def make_track(rng, target, counts, spread=50.0):
    particles = []
    for _ in range(counts):
        particle = lmb.Particle()
        particle.state_vector = target + np.concatenate([rng.normal(0, spread, 3), rng.normal(0, spread * 1e-3, 3)])
        particle.weight = 1.0 / counts if counts else 0.0
        particles.append(particle)
    return lmb.Track(lmb.TrackLabel(), 0.8, particles)


def check_ragged_clouds(chk: Checker, rng: np.random.Generator) -> None:
    """Per-track particle counts are allowed to differ, and must stay independent.

    The association buffer is one flat allocation, so a stride-based index would silently read a
    neighbouring track's particles here. Adaptive, confidence-driven cloud sizing is expected, so
    this is a supported configuration rather than a hypothetical.
    """
    sigmas = np.array([100.0, 1.0, 1e-3, 1e-3, 1e-5, 1e-5])
    counts = [17, 3, 40, 1]
    tracker, _ = build_tracker()

    targets, measurements = [], []
    for _ in range(len(counts)):
        target, sensor = scene(rng)
        targets.append(target)
        measurement = lmb.Measurement.fromCartesian(target, sensor)
        measurement.covariance_ = np.diag(sigmas**2)
        measurements.append(measurement)

    tracker.set_tracks([make_track(rng, target, count) for target, count in zip(targets, counts)])
    tracker.update(measurements)

    updated = tracker.get_tracks()
    chk.ok(len(updated) == len(counts), f"expected {len(counts)} tracks back, got {len(updated)}")

    for track, expected_count in zip(updated, counts):
        states = np.asarray(track.particle_states())
        weights = np.asarray([particle.weight for particle in track.particles()])
        chk.ok(states.shape[0] == expected_count,
               f"ragged cloud resized: {states.shape[0]} particles, expected {expected_count}")
        chk.ok(np.all(weights == 1.0 / expected_count),
               f"post-update weights are not exactly 1/{expected_count}")
        chk.ok(np.isfinite(states).all(), "ragged cloud produced a non-finite state")
        chk.ok(0.0 <= track.existence_probability() <= 1.0,
               f"existence probability {track.existence_probability()} outside [0, 1]")

    print(f"  ragged clouds: counts {counts} preserved, weights exactly 1/P per track")


def check_degenerate_branches(chk: Checker, rng: np.random.Generator) -> None:
    """The three branches that bypass or short-circuit the resampler."""
    sigmas = np.array([100.0, 1.0, 1e-3, 1e-3, 1e-5, 1e-5])

    # (a) A zero-particle track: no resampling is possible, the track survives with its updated r.
    tracker, _ = build_tracker()
    target, sensor = scene(rng)
    measurement = lmb.Measurement.fromCartesian(target, sensor)
    measurement.covariance_ = np.diag(sigmas**2)
    empty = lmb.Track(lmb.TrackLabel(), 0.8, [])
    populated = make_track(rng, target, 12)
    tracker.set_tracks([empty, populated])
    tracker.update([measurement])
    updated = tracker.get_tracks()
    chk.ok(len(updated) >= 1, "the zero-particle case dropped every track")
    for track in updated:
        chk.ok(0.0 <= track.existence_probability() <= 1.0,
               f"existence probability {track.existence_probability()} outside [0, 1]")
        chk.ok(np.isfinite(np.asarray(track.particle_states())).all(),
               "zero-particle case produced a non-finite state")

    # (b) A vanishing total mixture mass: the measurement is so far from the cloud that every
    #     likelihood underflows, which drives sum_weights to the <= 1e-12 fallback.
    tracker, _ = build_tracker()
    near, sensor = scene(rng)
    far = near + np.concatenate([np.full(3, 5.0e7), np.zeros(3)])
    measurement = lmb.Measurement.fromCartesian(far, sensor)
    measurement.covariance_ = np.diag(sigmas**2)
    tracker.set_tracks([make_track(rng, near, 9)])
    tracker.update([measurement])
    for track in tracker.get_tracks():
        weights = np.asarray([particle.weight for particle in track.particles()])
        if weights.size:
            chk.ok(np.all(weights == 1.0 / weights.size),
                   "degenerate-mass fallback left weights other than exactly 1/P")
        chk.ok(0.0 <= track.existence_probability() <= 1.0,
               f"existence probability {track.existence_probability()} outside [0, 1]")

    # (c) A single-particle cloud: the systematic walk must still terminate and pick that particle.
    tracker, _ = build_tracker()
    target, sensor = scene(rng)
    measurement = lmb.Measurement.fromCartesian(target, sensor)
    measurement.covariance_ = np.diag(sigmas**2)
    single = make_track(rng, target, 1)
    before = np.asarray(single.particle_states())[0].copy()
    tracker.set_tracks([single])
    tracker.update([measurement])
    for track in tracker.get_tracks():
        states = np.asarray(track.particle_states())
        chk.ok(states.shape[0] == 1, f"single-particle cloud became {states.shape[0]} particles")
        chk.ok(np.array_equal(states[0], before),
               "single-particle cloud did not resample to its only particle")

    print("  degenerate branches: empty cloud, vanishing mixture mass and P=1 all handled")


def main() -> int:
    chk = Checker()
    rng = np.random.default_rng(20260920)
    print("mixture grouping")
    check_worked_example(chk)
    check_randomized_identity(chk, rng)
    check_ragged_clouds(chk, rng)
    check_degenerate_branches(chk, rng)
    print(f"PASS: test_mixture_grouping ({chk.count} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
