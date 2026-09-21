"""End-to-end acceptance: short tracker runs on the run_once.py scenario, unrotated and rotated to the pole.

The rotated scene is the unrotated scene (truths, sensor and every measurement) rotated rigidly by the
rotation that puts the mid-run sensor->object-1 line of sight exactly on +z. The measurement model is
rotation invariant, so:
  * deterministic part: the likelihood of the truth particle against each measurement agrees to 1e-9
    relative between the two scenes at every step;
  * statistical part: with the propagator/resampler RNGs random_device-seeded, 5 runs per scene must give
    mean-GOSPA samples that overlap within 3 sqrt(s1^2/5 + s2^2/5) + 5% of the unrotated mean, with zero
    NaNs and the correct final cardinality in every run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

import run_once  # noqa: E402
import harness_scenario as hs  # noqa: E402
import reference_geometry as ref  # noqa: E402

lmb = run_once.lmb_engine

NUM_STEPS = 40
NUM_PARTICLES = 200
NUM_RUNS = 5
GOSPA_CUTOFF = hs.GOSPA_CUTOFF
BASE_SEED = 20260908


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def rotate_state(q: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.concatenate([q @ state[:3], q @ state[3:]])


def rotate_measurement(q: np.ndarray, measurement) -> "lmb.Measurement":
    rotated = lmb.Measurement()
    rotated.range_ = measurement.range_
    rotated.range_rate_ = measurement.range_rate_
    rotated.los_ = q @ measurement.los_
    rotated.los_rate_ = q @ measurement.los_rate_
    rotated.covariance_ = measurement.covariance_
    rotated.sensor_state_ = rotate_state(q, measurement.sensor_state_)
    rotated.timestamp_ = measurement.timestamp_
    rotated.sensor_id_ = measurement.sensor_id_
    return rotated


def simulate_truth():
    """Deterministic-noise truth and sensor trajectories over NUM_STEPS, exactly as run_once.py does."""
    propagator = run_once.get_ground_truth_propagator()
    sensor_state = run_once.SENSOR_STATE.copy()
    active = []
    truths_per_step = []
    sensor_per_step = []
    for step in range(NUM_STEPS):
        if step > 0:
            active = [(obj_id, run_once.propagate_truth_state(propagator, state, run_once.DT)) for obj_id, state in active]
            sensor_state = run_once.propagate_truth_state(propagator, sensor_state, run_once.DT)
        for obj_id, birth_step, initial_state in run_once.SCENARIO:
            if step == birth_step:
                active.append((obj_id, initial_state.copy()))
        truths_per_step.append([(obj_id, state.copy()) for obj_id, state in active])
        sensor_per_step.append(sensor_state.copy())
    return truths_per_step, sensor_per_step


def generate_measurement_stream(truths_per_step, sensor_per_step, seed: int):
    np.random.seed(seed)
    return [run_once.generate_measurements(truths_per_step[step], sensor_per_step[step], step * run_once.DT)
            for step in range(NUM_STEPS)]


def make_tracker(birth_seed: int):
    sensor_model = lmb.InOrbitSensorModel(*run_once.FILTER_SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(NUM_PARTICLES, run_once.P_BIRTH, run_once.BIRTH_COVARIANCE_LOCAL, seed=birth_seed)
    propagator = lmb.TwoBodyPropagator(run_once.Q_FILTER)
    return lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model, run_once.P_SURVIVAL, run_once.K_BEST,
                               run_once.PRUNE_THRESHOLD, run_once.CLUTTER_INTENSITY, run_once.P_DETECTION,
                               run_once.NOISE_DECAY_RATE, run_once.NOISE_MIN_SCALE)


def run_tracker(measurement_stream, truths_per_step, birth_seed: int):
    tracker = make_tracker(birth_seed)
    gospa = np.empty(NUM_STEPS)
    bound = np.empty(NUM_STEPS)
    for step in range(NUM_STEPS):
        if step > 0:
            tracker.predict(run_once.DT)
        tracker.update(measurement_stream[step])
        tracks = tracker.get_tracks()
        truth_states = [state for _, state in truths_per_step[step]]
        # No truth guard: GOSPA scores tracks against zero truths as false-track cost, not zero.
        gospa[step] = lmb.calculate_gospa_distance(tracks, truth_states, GOSPA_CUTOFF)
        bound[step] = hs.gospa_upper_bound(len(tracks), len(truth_states), GOSPA_CUTOFF)
    tracks = tracker.get_tracks()
    means = np.array([run_once.compute_track_mean(t) for t in tracks]) if tracks else np.zeros((0, 6))
    confirmed = sum(1 for t in tracks if t.existence_probability() > 0.5)
    saturation = np.divide(gospa, bound, out=np.zeros_like(gospa), where=bound > 0.0)
    return gospa, means, confirmed, saturation


def check_deterministic_likelihood(chk: Checker, q, stream, rotated_stream, truths_per_step) -> None:
    sensor_model = lmb.InOrbitSensorModel(*run_once.FILTER_SIGMAS**2)
    checked = 0
    for step in range(NUM_STEPS):
        truths = truths_per_step[step]
        chk.ok(len(stream[step]) == len(rotated_stream[step]), "rotated stream length mismatch")
        for measurement, rotated in zip(stream[step], rotated_stream[step]):
            # Each measurement was generated from one truth; find it by nearest Cartesian state.
            states = [state for _, state in truths]
            back = measurement.toCartesian()
            truth = min(states, key=lambda s: np.linalg.norm(s[:3] - back[:3]))
            particle = lmb.Particle()
            particle.state_vector = truth
            particle.weight = 1.0
            rotated_particle = lmb.Particle()
            rotated_particle.state_vector = rotate_state(q, truth)
            rotated_particle.weight = 1.0
            l_base = sensor_model.calculate_likelihood(particle, measurement)
            l_rot = sensor_model.calculate_likelihood(rotated_particle, rotated)
            chk.ok(np.isfinite(l_base) and l_base > 0.0, f"step {step}: non-finite likelihood")
            chk.ok(abs(l_rot - l_base) <= 1e-9 * l_base,
                   f"step {step}: rotated likelihood differs by {abs(l_rot - l_base) / l_base:.3e} relative")
            checked += 1
    chk.ok(checked >= NUM_STEPS, f"only {checked} likelihoods compared")


def main() -> None:
    chk = Checker()
    truths_per_step, sensor_per_step = simulate_truth()

    mid = NUM_STEPS // 2
    object_1 = next(state for obj_id, state in truths_per_step[mid] if obj_id == 1)
    los_mid = ref.unit(object_1[:3] - sensor_per_step[mid][:3])
    q = ref.rotation_taking(los_mid, np.array([0.0, 0.0, 1.0]))
    chk.ok(abs((q @ los_mid)[2] - 1.0) <= 1e-15, "rotation must place the mid-run LOS exactly on +z")
    chk.ok(np.max(np.abs(q @ q.T - np.eye(3))) <= 1e-15, "rotation must be orthogonal")

    rotated_truths = [[(obj_id, rotate_state(q, state)) for obj_id, state in truths] for truths in truths_per_step]

    base_means = []
    rot_means = []
    base_tracking = []
    for run in range(NUM_RUNS):
        seed = BASE_SEED + run
        stream = generate_measurement_stream(truths_per_step, sensor_per_step, seed)
        rotated_stream = [[rotate_measurement(q, m) for m in step_measurements] for step_measurements in stream]
        if run == 0:
            check_deterministic_likelihood(chk, q, stream, rotated_stream, truths_per_step)
            los_z = [abs(m.los_[2]) for step_measurements in rotated_stream[mid - 3:mid + 4] for m in step_measurements]
            chk.ok(max(los_z) > 0.999999, f"rotated scene never reaches the pole (max |los_z| = {max(los_z):.6f})")

        expected_cardinality = len(truths_per_step[-1])
        for label, measurement_stream, truths, sink in (("unrotated", stream, truths_per_step, base_means),
                                                        ("rotated", rotated_stream, rotated_truths, rot_means)):
            gospa, means, confirmed, saturation = run_tracker(measurement_stream, truths, birth_seed=seed)
            chk.ok(np.all(np.isfinite(gospa)), f"[{label} run {run}] NaN/Inf in GOSPA")
            chk.ok(np.all(np.isfinite(means)), f"[{label} run {run}] NaN/Inf in a track mean")
            chk.ok(confirmed == expected_cardinality,
                   f"[{label} run {run}] {confirmed} confirmed tracks, expected {expected_cardinality}")
            sink.append(float(np.mean(gospa)))
            tracking = float(np.mean(saturation < 1.0 - 1e-12))
            if label == "unrotated":
                base_tracking.append(tracking)
            print(f"  [{label} run {run}] mean GOSPA {sink[-1]:.1f} m "
                  f"(tracking fraction {tracking:.3f}), final {gospa[-1]:.1f} m, confirmed {confirmed}")

    base = np.array(base_means)
    rot = np.array(rot_means)
    s1 = base.std(ddof=1)
    s2 = rot.std(ddof=1)
    bound = 3.0 * np.sqrt(s1**2 / NUM_RUNS + s2**2 / NUM_RUNS) + 0.05 * base.mean()
    delta = abs(base.mean() - rot.mean())
    print(f"  unrotated mean GOSPA {base.mean():.1f} +- {s1:.1f}; rotated {rot.mean():.1f} +- {s2:.1f}; "
          f"|delta| {delta:.1f} <= {bound:.1f}")
    chk.ok(delta <= bound, f"rotated/unrotated mean GOSPA differ by {delta:.1f} m > bound {bound:.1f} m")
    # Lost-track gate. Not a fraction of the cutoff (unnormalised GOSPA is not bounded by it) and
    # not a mean saturation (at 200 particles a healthy run already sits near 1.0, leaving no
    # headroom). The fraction of steps that produced an accepted track/truth pair is 0.0 by
    # construction for a filter that has stopped tracking. Matches
    # hs.TRACKING_FRACTION_FLOOR, shared with test_golden_invariance.py.
    mean_tracking = float(np.mean(base_tracking))
    chk.ok(mean_tracking > hs.TRACKING_FRACTION_FLOOR,
           f"only {mean_tracking:.3f} of steps produced an accepted track/truth pair; "
           "tracker is not tracking")

    if chk.count <= 0:
        raise AssertionError("no assertions executed")
    print(f"PASS: test_end_to_end ({chk.count} assertions)")


if __name__ == "__main__":
    main()
