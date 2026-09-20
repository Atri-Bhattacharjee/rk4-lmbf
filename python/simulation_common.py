"""Shared configuration and helpers for run.py / run_once.py.

python/2026_ieee_aerospace.py is intentionally independent of this module.
"""
from __future__ import annotations

import os

import numpy as np

from lmb_engine_loader import import_lmb_engine

lmb_engine = import_lmb_engine(verbose=os.environ.get("LMB_ENGINE_VERBOSE") == "1")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

NUM_STEPS = 100
DT = 60.0
NUM_PARTICLES = 10000

P_DETECTION = 0.999999999
P_SURVIVAL = 0.999999999
P_BIRTH = 0.9
CLUTTER_INTENSITY = 1e-15
PRUNE_THRESHOLD = 0.001
K_BEST = 2

NOISE_DECAY_RATE = 0.001
NOISE_MIN_SCALE = 0.001

TRUTH_SIGMA_RANGE = 10.0
TRUTH_SIGMA_RANGE_RATE = 1.0
TRUTH_SIGMA_ANGLE = 1e-6
TRUTH_SIGMA_ANGLE_RATE = 1e-7

FILTER_SIGMA_RANGE = 5000.0
FILTER_SIGMA_RANGE_RATE = 500.0
FILTER_SIGMA_ANGLE = 1e-2
FILTER_SIGMA_ANGLE_RATE = 1e-3

TRUTH_SIGMAS = np.array([
    TRUTH_SIGMA_RANGE,
    TRUTH_SIGMA_RANGE_RATE,
    TRUTH_SIGMA_ANGLE,
    TRUTH_SIGMA_ANGLE,
    TRUTH_SIGMA_ANGLE_RATE,
    TRUTH_SIGMA_ANGLE_RATE,
])
FILTER_SIGMAS = np.array([
    FILTER_SIGMA_RANGE,
    FILTER_SIGMA_RANGE_RATE,
    FILTER_SIGMA_ANGLE,
    FILTER_SIGMA_ANGLE,
    FILTER_SIGMA_ANGLE_RATE,
    FILTER_SIGMA_ANGLE_RATE,
])

# Hoisted once: measurement.covariance_ and the truth-propagator epsilon noise.
FILTER_COVARIANCE = np.diag(FILTER_SIGMAS**2)
TRUTH_PROPAGATOR_NOISE = np.eye(6) * 1e-18

Q_FILTER = np.diag([
    500.0**2,
    500.0**2,
    500.0**2,
    50.0**2,
    50.0**2,
    50.0**2,
])

BIRTH_COVARIANCE_LOCAL = np.diag([
    1000.0**2,
    500.0**2,
    1e-4**2,
    1e-4**2,
    5e-5**2,
    5e-5**2,
])

# =============================================================================
# SCENARIO DEFINITION
# =============================================================================

R_EARTH = 6.371e6
ALTITUDE = 400e3
MU_EARTH = 3.986004418e14

ORBIT_RADIUS = R_EARTH + ALTITUDE
V_CIRCULAR = np.sqrt(MU_EARTH / ORBIT_RADIUS)

SENSOR_STATE = np.array([ORBIT_RADIUS, 0.0, 0.0, 0.0, V_CIRCULAR, 0.0])

SCENARIO = [
    (1, 0, np.array([
        +2.6544665658e+06,
        -1.5306571649e+06,
        -6.5227229294e+06,
        -4.5588669893e+03,
        +5.0941118632e+03,
        -2.9133829034e+03,
    ])),
    (2, 30, np.array([
        +3.7638743804e+06,
        -6.2234498679e+05,
        +6.0037549884e+06,
        +5.0461811923e+03,
        +4.9856212029e+03,
        -2.5111625240e+03,
    ])),
    (3, 50, np.array([
        -7.1242552406e+06,
        -2.0344233795e+06,
        +1.3982281842e+06,
        -1.3769335820e+02,
        -3.1939023671e+03,
        -6.3886825206e+03,
    ])),
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_ground_truth_propagator():
    """Deterministic two-body propagator (near-zero process noise)."""
    return lmb_engine.TwoBodyPropagator(TRUTH_PROPAGATOR_NOISE)


def propagate_truth_state(propagator, state_vector, dt):
    """Propagate a raw 6-D state vector one step."""
    particle = lmb_engine.Particle()
    particle.state_vector = state_vector.copy()
    particle.weight = 1.0
    propagated = propagator.propagate(particle, dt, 0.0)
    return np.array(propagated.state_vector)


def generate_measurements(active_truths, sensor_state, current_time):
    """Simulate sensor measurements from active ground-truth objects."""
    measurements = []

    for _, truth_state in active_truths:
        if np.random.random() > P_DETECTION:
            continue

        measurement = lmb_engine.Measurement.fromCartesian(truth_state, sensor_state).perturbed(
            np.random.normal(size=6) * TRUTH_SIGMAS
        )
        measurement.timestamp_ = current_time
        measurement.sensor_id_ = "sensor_0"
        # Copy so each Measurement owns its matrix; values are still the hoisted constant.
        measurement.covariance_ = FILTER_COVARIANCE.copy()
        measurements.append(measurement)

    return measurements


def compute_track_mean(track):
    """Weighted mean state of a track's particles (delegates to C++)."""
    return np.asarray(track.mean_state(), dtype=np.float64).reshape(6)


def run_single_simulation(verbose=False, collect_track_errors=True, seed=None):
    """
    Run one SMC-LMB simulation.

    Args:
        verbose: Print per-step progress when True.
        collect_track_errors: When False, skip Object-1 component-error work (single-run path).
        seed: Optional integer. When set, seeds NumPy and the filter/birth/resampler RNGs so a
            run is reproducible. Default None keeps the historical random_device behaviour.

    Returns:
        (ospa_results, track_error_history)
    """
    if seed is not None:
        np.random.seed(seed)

    if verbose:
        print("=" * 60)
        print("SMC-LMB Filter Validation Simulation")
        print("=" * 60)
        print("Configuration:")
        print(f"  Steps: {NUM_STEPS}, DT: {DT}s, Particles: {NUM_PARTICLES}")
        print(f"  P_D: {P_DETECTION}, P_S: {P_SURVIVAL}, P_B: {P_BIRTH}")
        print(f"  Clutter: {CLUTTER_INTENSITY}, K-best: {K_BEST}")
        print("=" * 60)

    # When seed is set, pin the truth propagator too: TRUTH_PROPAGATOR_NOISE still has a tiny
    # positive trace, so an unseeded instance draws from random_device and breaks reproducibility.
    if seed is None:
        truth_propagator = get_ground_truth_propagator()
    else:
        truth_propagator = lmb_engine.TwoBodyPropagator(TRUTH_PROPAGATOR_NOISE, seed=seed)
    filter_propagator = lmb_engine.TwoBodyPropagator(Q_FILTER, seed=seed)
    sensor_model = lmb_engine.InOrbitSensorModel(*FILTER_SIGMAS**2)
    birth_model = lmb_engine.AdaptiveBirthModel(
        NUM_PARTICLES,
        P_BIRTH,
        BIRTH_COVARIANCE_LOCAL,
        seed=seed,
    )
    tracker = lmb_engine.SMC_LMB_Tracker(
        filter_propagator,
        sensor_model,
        birth_model,
        P_SURVIVAL,
        K_BEST,
        PRUNE_THRESHOLD,
        CLUTTER_INTENSITY,
        P_DETECTION,
        NOISE_DECAY_RATE,
        NOISE_MIN_SCALE,
        seed=seed,
    )

    active_ground_truths = []
    sensor_state = SENSOR_STATE.copy()
    ospa_results = []
    track_error_history = []

    if verbose:
        print("\nSimulation Progress:")
        print("-" * 60)

    for step in range(NUM_STEPS):
        current_time = step * DT

        if step > 0:
            for i in range(len(active_ground_truths)):
                obj_id, state = active_ground_truths[i]
                new_state = propagate_truth_state(truth_propagator, state, DT)
                active_ground_truths[i] = (obj_id, new_state)
            sensor_state = propagate_truth_state(truth_propagator, sensor_state, DT)

        for obj_id, birth_step, initial_state in SCENARIO:
            if step == birth_step:
                active_ground_truths.append((obj_id, initial_state.copy()))
                if verbose:
                    print(f"  [Step {step:3d}] Object {obj_id} BORN at t={current_time:.0f}s")

        measurements = generate_measurements(
            active_ground_truths,
            sensor_state,
            current_time,
        )

        if step > 0:
            tracker.predict(DT)

        tracker.update(measurements)
        tracks = tracker.get_tracks()
        truth_states = [state.copy() for (_, state) in active_ground_truths]

        if collect_track_errors:
            truth_obj1_state = None
            for obj_id, state in active_ground_truths:
                if obj_id == 1:
                    truth_obj1_state = state
                    break

            if truth_obj1_state is not None and len(tracks) > 0:
                min_dist = float("inf")
                best_track = None
                for track in tracks:
                    track_mean = compute_track_mean(track)
                    pos_dist = np.linalg.norm(track_mean[:3] - truth_obj1_state[:3])
                    if pos_dist < min_dist:
                        min_dist = pos_dist
                        best_track = track_mean
                track_error_history.append(best_track - truth_obj1_state)
            else:
                track_error_history.append(np.zeros(6))

        if len(truth_states) > 0:
            ospa = lmb_engine.calculate_ospa_distance(
                tracks,
                truth_states,
                100000.0,
            )
        else:
            ospa = 0.0

        ospa_results.append(ospa)

        if verbose and (step % 10 == 0 or step in [0, 30, 50]):
            track_probs = [t.existence_probability() for t in tracks]
            prob_str = ", ".join([f"{p:.2f}" for p in track_probs[:5]])
            if len(track_probs) > 5:
                prob_str += ", ..."
            print(
                f"  [Step {step:3d}] t={current_time:6.0f}s | "
                f"Tracks: {len(tracks):2d} | Truths: {len(active_ground_truths)} | "
                f"Meas: {len(measurements)} | OSPA: {ospa:8.1f}m | "
                f"r=[{prob_str}]"
            )

    if verbose:
        print("-" * 60)
        print("\nFinal Results:")
        print(f"  Final OSPA: {ospa_results[-1]:.1f} m")
        print(f"  Mean OSPA (last 20 steps): {np.mean(ospa_results[-20:]):.1f} m")
        tracks = tracker.get_tracks()
        print("\nTrack Summary:")
        for i, track in enumerate(tracks):
            mean_state = compute_track_mean(track)
            pos_mag = np.linalg.norm(mean_state[:3]) / 1000
            vel_mag = np.linalg.norm(mean_state[3:6]) / 1000
            print(
                f"  Track {i+1}: r={track.existence_probability():.3f}, "
                f"|pos|={pos_mag:.1f} km, |vel|={vel_mag:.2f} km/s"
            )

    return ospa_results, track_error_history


__all__ = [
    "lmb_engine",
    "NUM_STEPS",
    "DT",
    "NUM_PARTICLES",
    "P_DETECTION",
    "P_SURVIVAL",
    "P_BIRTH",
    "CLUTTER_INTENSITY",
    "PRUNE_THRESHOLD",
    "K_BEST",
    "NOISE_DECAY_RATE",
    "NOISE_MIN_SCALE",
    "TRUTH_SIGMAS",
    "FILTER_SIGMAS",
    "FILTER_COVARIANCE",
    "TRUTH_PROPAGATOR_NOISE",
    "Q_FILTER",
    "BIRTH_COVARIANCE_LOCAL",
    "SENSOR_STATE",
    "SCENARIO",
    "get_ground_truth_propagator",
    "propagate_truth_state",
    "generate_measurements",
    "compute_track_mean",
    "run_single_simulation",
]
