"""Shared configuration and helpers for run.py / run_once.py.

python/2026_ieee_aerospace.py is intentionally independent of this module.
"""
from __future__ import annotations

import os
from pathlib import Path

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

# GOSPA cutoff c, metres. Read from the engine (src/metrics.h) rather than re-declared, so the
# drivers and the test harness cannot drift apart on the metric's parameters.
GOSPA_CUTOFF = lmb_engine.GOSPA_DEFAULT_CUTOFF
# An unnormalised GOSPA value is meaningless without its parameters, so every axis label and
# printed summary carries them.
GOSPA_PARAMS = (
    f"unnormalised, c = {GOSPA_CUTOFF / 1000:.0f} km, "
    f"p = {lmb_engine.GOSPA_ORDER_P:.0f}, alpha = {lmb_engine.GOSPA_ALPHA:.0f}"
)

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

# Unbounded in range. Paired with add_unpointed below this is the omniscient sensor the filter had
# before fields of view existed, which is why this scenario's numbers (and the committed golden
# fixtures) are unchanged by the sensor rework. Give it a max_range and a half_width, and use
# SensorArray.add instead, to bound it.
SENSOR_FOV = lmb_engine.SensorFovConfig()

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


def build_sensor_array(fov_config=None):
    """The shipped one-sensor array: a single unpointed sensor at SENSOR_STATE.

    Unpointed and unbounded by default, so the scenario sees exactly what it always saw. Pass a
    bounded SensorFovConfig, or build your own array with SensorArray.add, for a pointed sensor.
    """
    sensors = lmb_engine.SensorArray(SENSOR_FOV if fov_config is None else fov_config)
    sensors.add_unpointed("sensor_0", SENSOR_STATE)
    return sensors


def _as_sensor_array(sensors):
    """Accept a SensorArray, or a bare 6-D sensor state for callers that predate them.

    A bare state is wrapped in a one-sensor unbounded, unpointed array, which is the same
    omniscient sensor it used to denote. tests/harness_scenario.py and tests/bench_engine.py pass
    a state this way.
    """
    if isinstance(sensors, lmb_engine.SensorArray):
        return sensors
    array = lmb_engine.SensorArray(lmb_engine.SensorFovConfig())
    array.add_unpointed("sensor_0", np.asarray(sensors, dtype=np.float64).reshape(6))
    return array


def generate_measurements(active_truths, sensors, current_time):
    """Simulate sensor measurements from active ground-truth objects.

    An object is only reported when some sensor can actually observe it, and it is reported as a
    measurement of that sensor -- stamped with its id and its state, which is what lets the filter
    score the association against the right field of view. The visibility test is
    SensorArray.sees, the same predicate the filter uses on the particle clouds.

    ``sensors`` may be a SensorArray or a bare 6-D sensor state (see _as_sensor_array).
    """
    sensors = _as_sensor_array(sensors)
    measurements = []

    for _, truth_state in active_truths:
        sensor_index = sensors.visible_sensor(truth_state)
        if sensor_index < 0:
            # Outside every sensor's volume: nothing to detect, so no detection roll either.
            continue

        if np.random.random() > P_DETECTION:
            continue

        measurement = lmb_engine.Measurement.fromCartesian(
            truth_state, sensors.state(sensor_index)
        ).perturbed(np.random.normal(size=6) * TRUTH_SIGMAS)
        measurement.timestamp_ = current_time
        measurement.sensor_id_ = sensors.id(sensor_index)
        # Copy so each Measurement owns its matrix; values are still the hoisted constant.
        measurement.covariance_ = FILTER_COVARIANCE.copy()
        measurements.append(measurement)

    return measurements


def compute_track_mean(track):
    """Weighted mean state of a track's particles (delegates to C++)."""
    return np.asarray(track.mean_state(), dtype=np.float64).reshape(6)


def run_single_simulation(verbose=False, collect_track_errors=True, seed=None, collect_components=False):
    """
    Run one SMC-LMB simulation.

    Args:
        verbose: Print per-step progress when True.
        collect_track_errors: When False, skip Object-1 component-error work (single-run path).
        seed: Optional integer. When set, seeds NumPy and the filter/birth/resampler RNGs so a
            run is reproducible. Default None keeps the historical random_device behaviour.
        collect_components: When True, also accumulate the per-step GOSPA decomposition. This adds
            a third return value; the default two-tuple shape is pinned by
            tests/test_run_once_api_surface.py and must not change.

    Returns:
        (gospa_results, track_error_history), or
        (gospa_results, track_error_history, components) when collect_components is True.
        ``components`` is a dict of float lists keyed "localisation", "missed", "false_positive",
        each a p-th-power cost (m^p) that sums exactly to gospa**p.
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
    sensors = build_sensor_array()
    sensor_state = SENSOR_STATE.copy()
    gospa_results = []
    gospa_components = {"localisation": [], "missed": [], "false_positive": []}
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
            sensors.set_state(0, sensor_state)

        for obj_id, birth_step, initial_state in SCENARIO:
            if step == birth_step:
                active_ground_truths.append((obj_id, initial_state.copy()))
                if verbose:
                    print(f"  [Step {step:3d}] Object {obj_id} BORN at t={current_time:.0f}s")

        measurements = generate_measurements(
            active_ground_truths,
            sensors,
            current_time,
        )

        if step > 0:
            tracker.predict(DT)

        tracker.update(measurements, sensors)
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

        # No "if truth_states" guard: with GOSPA, m tracks against zero truths is c*sqrt(m/2) of
        # false-track cost, not zero. The engine handles n == 0 by construction.
        if collect_components:
            breakdown = lmb_engine.calculate_gospa_components(tracks, truth_states, GOSPA_CUTOFF)
            gospa = breakdown.total
            gospa_components["localisation"].append(breakdown.localisation)
            gospa_components["missed"].append(breakdown.missed)
            gospa_components["false_positive"].append(breakdown.false_positive)
        else:
            gospa = lmb_engine.calculate_gospa_distance(
                tracks,
                truth_states,
                GOSPA_CUTOFF,
            )

        gospa_results.append(gospa)

        if verbose and (step % 10 == 0 or step in [0, 30, 50]):
            track_probs = [t.existence_probability() for t in tracks]
            prob_str = ", ".join([f"{p:.2f}" for p in track_probs[:5]])
            if len(track_probs) > 5:
                prob_str += ", ..."
            print(
                f"  [Step {step:3d}] t={current_time:6.0f}s | "
                f"Tracks: {len(tracks):2d} | Truths: {len(active_ground_truths)} | "
                f"Meas: {len(measurements)} | GOSPA: {gospa:8.1f}m | "
                f"r=[{prob_str}]"
            )

    if verbose:
        print("-" * 60)
        print("\nFinal Results:")
        print(f"  Final GOSPA: {gospa_results[-1]:.1f} m")
        print(f"  Mean GOSPA (last 20 steps): {np.mean(gospa_results[-20:]):.1f} m")
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

    if collect_components:
        return gospa_results, track_error_history, gospa_components
    return gospa_results, track_error_history


# =============================================================================
# MONTE CARLO BATCH (sequential or ProcessPoolExecutor)
# =============================================================================


def derive_run_seeds(master_seed: int, num_runs: int) -> list[int]:
    """Explicit per-run seeds derived from a master seed (order-stable, independent of workers)."""
    children = np.random.SeedSequence(int(master_seed)).spawn(int(num_runs))
    # uint32 so the same integer is valid for np.random.seed and the C++ mt19937_64 seed argument.
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def resolve_max_workers(max_workers: int | None = None) -> int:
    """Resolve worker count: argument, else LMB_NUM_WORKERS, else cpu_count. 1 means serial."""
    if max_workers is not None:
        return max(1, int(max_workers))
    env = os.environ.get("LMB_NUM_WORKERS")
    if env is not None and str(env).strip() != "":
        return max(1, int(env))
    return max(1, int(os.cpu_count() or 1))


def _monte_carlo_worker(payload: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray | None]:
    """Top-level worker for ProcessPoolExecutor (must be picklable under spawn)."""
    run_index, seed = payload
    gospa, errors = run_single_simulation(
        verbose=False,
        collect_track_errors=(run_index == 0),
        seed=int(seed),
    )
    gospa_arr = np.asarray(gospa, dtype=np.float64)
    err_arr = np.asarray(errors, dtype=np.float64) if run_index == 0 else None
    return int(run_index), gospa_arr, err_arr


def run_monte_carlo(
    num_runs: int,
    *,
    master_seed: int,
    max_workers: int | None = None,
    on_run_complete=None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Run ``num_runs`` independent simulations with explicit derived seeds.

    Aggregation is always by run index (not completion order). ``representative_errors``
    always comes from run 0. ``max_workers == 1`` (or LMB_NUM_WORKERS=1) is a serial fallback.

    Returns:
        all_gospa: shape (num_runs, NUM_STEPS)
        representative_errors: shape (NUM_STEPS, 6) from run 0
        run_seeds: per-run seeds used
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    num_runs = int(num_runs)
    if num_runs <= 0:
        raise ValueError(f"num_runs must be positive, got {num_runs}")

    # Spawned workers re-import this package; keep the python/ directory on PYTHONPATH.
    python_dir = str(Path(__file__).resolve().parent)
    existing = os.environ.get("PYTHONPATH", "")
    if python_dir not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = python_dir + (os.pathsep + existing if existing else "")

    run_seeds = derive_run_seeds(master_seed, num_runs)
    workers = resolve_max_workers(max_workers)
    payloads = [(i, run_seeds[i]) for i in range(num_runs)]

    results_by_index: list[np.ndarray | None] = [None] * num_runs
    representative_errors: np.ndarray | None = None

    def _store(run_index: int, gospa: np.ndarray, errors: np.ndarray | None) -> None:
        nonlocal representative_errors
        results_by_index[run_index] = gospa
        if run_index == 0:
            representative_errors = errors
        if on_run_complete is not None:
            on_run_complete(run_index, gospa)

    if workers == 1:
        for payload in payloads:
            run_index, gospa, errors = _monte_carlo_worker(payload)
            _store(run_index, gospa, errors)
    else:
        # spawn avoids forking a process that already loaded the native extension / OpenMP.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {executor.submit(_monte_carlo_worker, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                run_index, gospa, errors = future.result()
                _store(run_index, gospa, errors)

    if any(item is None for item in results_by_index) or representative_errors is None:
        raise RuntimeError("Monte Carlo batch did not produce a complete result set")

    return np.stack(results_by_index, axis=0), representative_errors, run_seeds


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
    "GOSPA_CUTOFF",
    "GOSPA_PARAMS",
    "NOISE_DECAY_RATE",
    "NOISE_MIN_SCALE",
    "TRUTH_SIGMAS",
    "FILTER_SIGMAS",
    "FILTER_COVARIANCE",
    "TRUTH_PROPAGATOR_NOISE",
    "Q_FILTER",
    "BIRTH_COVARIANCE_LOCAL",
    "SENSOR_STATE",
    "SENSOR_FOV",
    "SCENARIO",
    "get_ground_truth_propagator",
    "propagate_truth_state",
    "build_sensor_array",
    "generate_measurements",
    "compute_track_mean",
    "run_single_simulation",
    "derive_run_seeds",
    "resolve_max_workers",
    "run_monte_carlo",
]
