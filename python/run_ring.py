"""Equatorial sensor ring against a sampled debris population: the naive-first large scenario.

N range-only sensors (default 100) sit evenly spaced on a circular equatorial orbit at 800 km, each
seeing anything within max_range (default 20 km). Every run samples num_objects (default 1000) rows
from python/data/eci_800km-altitude_20km-range_randomized_phase.csv, whose objects all pass within
20 km of the 800 km equatorial ring -- which is not the same as passing a *sensor*: with 100
sensors, a crossing meets one about 9% of the time.

Time is stepped on a fine global clock (dt, default 1 s): a 20 km pass lasts ~4 s at the ~10 km/s
crossing speeds here, so a coarser clock misses detections and, just as badly, misses the
negative information of a sensor that looked and saw nothing. The filter runs with lazy
propagation, so only clouds some sensor could see are stepped at dt; everything else is carried in
<= max_pending substeps. That needs time-consistent process noise, so the filter propagator is
built with noise_reference_dt: the process noise is the covariance accumulated over that interval.

Filter tuning lives in the "Filter configuration" block of RingConfig and is set for this
geometry, not imported from simulation_common (which is tuned for a 400 km sensor looking at objects
~1000 km away). The measurement likelihood and the birth covariance are 3x the truth noise; process
noise is very small (see RingConfig), because the truth follows the same two-body model with no
perturbations, and only has to keep resampled particles from collapsing onto duplicates;
process-noise annealing is off. A regularization (kernel jitter) step restores particle diversity
after resampling.

Truth, sensors and detections are propagated in NumPy with the same RK4 two-body model the engine
uses. Metrics are sampled every metric_interval seconds against the objects detected at least once
(a never-seen object is not a filter failure), and python/evaluation_plots.py turns the log into
figures.

The sensors can also be given a field of view (RingConfig.fov_half_angle_deg) and aimed every step
by a tasker handed to run(); python/tasking.py has the taskers and python/run_tasked.py compares
them. The command line below always runs the range-only sensors.

    python python/run_ring.py                       # defaults: 100 sensors, 1000 objects, 2 orbits
    python python/run_ring.py --particles 1000 --orbits 2 --seed 7
    LMB_RING_PARTICLES=500 python python/run_ring.py

Writes python/results/ring_seed<seed>/: ring_log.npz, summary.json, and the figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from simulation_common import (
    CLUTTER_INTENSITY,
    GOSPA_CUTOFF,
    GOSPA_PARAMS,
    K_BEST,
    MU_EARTH,
    P_BIRTH,
    P_DETECTION,
    P_SURVIVAL,
    PRUNE_THRESHOLD,
    R_EARTH,
    TRUTH_SIGMAS,
    lmb_engine,
)

PYTHON_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PYTHON_DIR / "data" / "eci_800km-altitude_20km-range_randomized_phase.csv"
RESULTS_DIR = PYTHON_DIR / "results"

# Same floor as the engine's gravity model (src/two_body_propagator.cpp).
MIN_RADIUS = R_EARTH + 100.0e3

# Detections of one object separated by more than this belong to different passes.
PASS_GAP = 60.0


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RingConfig:
    num_sensors: int = 100
    sensor_altitude: float = 800.0e3
    sensor_range: float = 20.0e3
    # Field-of-view half-angle of every sensor [deg], half-width and half-height alike. None: the
    # sensors see everything within range. Set: they are pointed, and something has to aim them at
    # every step -- the ``tasker`` argument of run() (python/tasking.py).
    fov_half_angle_deg: float | None = None
    num_objects: int = 1000
    num_orbits: float = 2.0
    dt: float = 1.0
    metric_interval: float = 60.0
    num_particles: int = 1000
    max_pending: float = 60.0
    noise_reference_dt: float = 60.0
    existence_threshold: float = 0.5      # tracks at or above this are the state estimate
    progress_interval: float = 600.0      # simulated seconds between progress lines
    profile: bool = True                  # engine phase timers (read-only; results unchanged)
    seed: int = 20260930
    csv_path: str = str(DEFAULT_CSV)
    output_dir: str = ""

    # --- Filter configuration (retune here) ---
    # Probabilities and clutter come from simulation_common; the noise model is set for this
    # geometry. Order of the six sigmas: range [m], range rate [m/s], two LOS angles [rad], two LOS
    # angular rates [rad/s], in the measurement's local tangent frame.
    p_detection: float = P_DETECTION
    p_survival: float = P_SURVIVAL
    p_birth: float = P_BIRTH
    # Density of "new object" detections in measurement space -- what a measurement no existing
    # track explains is weighed against. Derived from the scenario: ~1.2e-4 first detections per
    # sensor-second (142 over 100 sensors x 12,088 s in the 2-orbit reference run), spread over one
    # sensor's measurement space: 20 km of range x 30 km/s of closing speed x 4 pi sr x ~4 (rad/s)^2
    # of angular rate ~ 3e10. 1.2e-4 / 3e10 ~ 4e-15.
    clutter_intensity: float = 4e-15
    prune_threshold: float = PRUNE_THRESHOLD
    k_best: int = K_BEST
    # True sensor noise: range [m], range rate [m/s], two LOS angles [rad], two LOS rates [rad/s].
    # The angular terms are matched to range and range rate at ~15 km (10 m and 1 m/s cross-range);
    # --tight-angles restores the 1 urad / 0.1 urad/s placeholder from simulation_common.
    truth_sigmas: list = field(default_factory=lambda: [10.0, 1.0, 6.7e-4, 6.7e-4, 6.7e-5, 6.7e-5])
    filter_sigma_scale: float = 1.0       # likelihood and birth sigmas = scale * truth sigmas
    # Extra per-component multipliers on top of filter_sigma_scale, same order as the sigmas.
    filter_sigma_extra: list = field(default_factory=lambda: [1.0] * 6)
    # Position (m) and velocity (m/s) standard deviations accumulated over noise_reference_dt.
    # Tuned 2026-10-02 (sweep of 1x..0.01x on two seeds, confirmed at 30 orbits): the old 2 m /
    # 0.02 m/s made the filter forget ~0.2 m/s of velocity per orbit against a noise-free truth, so
    # well-observed tracks drifted ~0.9 km/h and stale NEES sat at ~0.7. At 0.03x drift is ~0.13 km/h,
    # stale NEES ~2.1 and the 30-orbit median error 2 km instead of 12. Gains flatten below 0.1x; the
    # cost is a few more "re-acquired, extra birth" passes (6-10 vs 2-3 of ~1700). If the truth gets
    # dynamics the filter lacks (J2, drag), raise these to cover that mismatch.
    q_position_sigma: float = 0.06
    q_velocity_sigma: float = 0.0006
    noise_decay_rate: float = 0.0         # process-noise annealing off
    noise_min_scale: float = 1.0
    regularization: bool = True
    regularization_bandwidth_scale: float = 1.0
    regularization_ess_threshold: float = 0.5
    fused_proposal: bool = True
    fused_ess_min: float = 20.0
    # --- Engine speed (results statistically, not bitwise, equal to the legacy engine) ---
    particle_gate: bool = True            # re-test a lazy cloud's "maybe visible" per particle
    fast_mode: bool = True                # keyed random streams + ziggurat + batched propagation
    gate_audit: bool = False              # validation only (slow): check every gate decision
    gate_sleep: bool = True               # skip gate tests whose answer is already known
    # Truth and sensors stepped in C++ (lmb_engine.two_body_rk4_steps, the engine's own RK4), a
    # chunk of truth_chunk steps per call, instead of NumPy every step. Off: the NumPy RK4 below.
    fast_truth: bool = True
    truth_chunk: int = 60

    @property
    def filter_sigmas(self) -> np.ndarray:
        return (self.filter_sigma_scale * np.asarray(self.filter_sigma_extra, dtype=np.float64)
                * np.asarray(self.truth_sigmas, dtype=np.float64))

    @property
    def q_filter_diag(self) -> np.ndarray:
        return np.array([self.q_position_sigma**2] * 3 + [self.q_velocity_sigma**2] * 3)

    @property
    def birth_covariance_local_diag(self) -> np.ndarray:
        return self.filter_sigmas**2

    @property
    def sensor_radius(self) -> float:
        return R_EARTH + self.sensor_altitude

    @property
    def orbit_period(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.sensor_radius**3 / MU_EARTH)

    @property
    def duration(self) -> float:
        return self.num_orbits * self.orbit_period


def config_from_args(argv=None) -> RingConfig:
    config = RingConfig()
    env = {
        "num_particles": ("LMB_RING_PARTICLES", int),
        "num_objects": ("LMB_RING_OBJECTS", int),
        "num_sensors": ("LMB_RING_SENSORS", int),
        "num_orbits": ("LMB_RING_ORBITS", float),
        "seed": ("LMB_RING_SEED", int),
    }
    for attr, (name, cast) in env.items():
        if os.environ.get(name, "").strip():
            setattr(config, attr, cast(os.environ[name]))

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sensors", type=int, dest="num_sensors")
    parser.add_argument("--objects", type=int, dest="num_objects")
    parser.add_argument("--orbits", type=float, dest="num_orbits")
    parser.add_argument("--particles", type=int, dest="num_particles")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--metric-interval", type=float, dest="metric_interval",
                        help="simulated seconds between metric samples")
    parser.add_argument("--progress-interval", type=float, dest="progress_interval",
                        help="simulated seconds between progress lines")
    parser.add_argument("--no-profile", action="store_false", dest="profile", default=None,
                        help="turn the engine's phase timers off")
    parser.add_argument("--range", type=float, dest="sensor_range", help="sensor max range [m]")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sigma-scale", type=float, dest="filter_sigma_scale",
                        help="filter likelihood/birth sigmas as a multiple of the truth sigmas")
    parser.add_argument("--truth-sigmas", type=float, nargs=6, dest="truth_sigmas",
                        metavar=("RANGE", "RATE", "ANG1", "ANG2", "ANGRATE1", "ANGRATE2"),
                        help="true sensor noise (the filter uses --sigma-scale times these)")
    parser.add_argument("--sigma-extra", type=float, nargs=6, dest="filter_sigma_extra",
                        metavar=("RANGE", "RATE", "ANG1", "ANG2", "ANGRATE1", "ANGRATE2"),
                        help="per-component multipliers on top of --sigma-scale")
    parser.add_argument("--no-regularization", action="store_false", dest="regularization",
                        default=None, help="turn the kernel-jitter step off")
    parser.add_argument("--no-fused", action="store_false", dest="fused_proposal", default=None,
                        help="turn the fused proposal off (ordinary particle update only)")
    parser.add_argument("--legacy-engine", action="store_true", default=False,
                        help="particle gate and fast mode off: the bit-for-bit legacy engine")
    parser.add_argument("--kappa", type=float, dest="clutter_intensity",
                        help="density of new-object detections in measurement space")
    parser.add_argument("--tight-angles", action="store_true", default=False,
                        help="use simulation_common's 1 urad / 0.1 urad/s angular noise")
    parser.add_argument("--csv", dest="csv_path")
    parser.add_argument("--output", dest="output_dir")
    args = parser.parse_args(argv)
    tight = vars(args).pop("tight_angles")
    if vars(args).pop("legacy_engine"):
        config.particle_gate = False
        config.fast_mode = False
        config.fast_truth = False
    if tight:
        config.truth_sigmas = TRUTH_SIGMAS.tolist()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    if not config.output_dir:
        config.output_dir = str(RESULTS_DIR / f"ring_seed{config.seed}")
    return config


# =============================================================================
# Truth: catalogue, ring, dynamics
# =============================================================================


def load_catalogue(path) -> tuple[np.ndarray, np.ndarray]:
    """(object ids, (N, 6) ECI states in m and m/s) from the MASTER-derived CSV."""
    ids, states = [], []
    with open(path, newline="") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in rows:
            ids.append(int(row["obj_id"]))
            states.append([float(row[key]) * 1e3 for key in
                           ("x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms")])
    return np.asarray(ids, dtype=np.int64), np.asarray(states, dtype=np.float64)


def ring_states(num_sensors: int, radius: float) -> np.ndarray:
    """Prograde circular equatorial orbit, sensors 360/N degrees apart, sensor 0 on +x."""
    speed = np.sqrt(MU_EARTH / radius)
    angles = 2.0 * np.pi * np.arange(num_sensors) / num_sensors
    return np.column_stack([
        radius * np.cos(angles), radius * np.sin(angles), np.zeros(num_sensors),
        -speed * np.sin(angles), speed * np.cos(angles), np.zeros(num_sensors),
    ])


def _derivative(states: np.ndarray) -> np.ndarray:
    position = states[:, :3]
    radius = np.linalg.norm(position, axis=1)
    radius_safe = np.maximum(radius, MIN_RADIUS)
    unit = np.divide(position, radius[:, None], out=np.zeros_like(position),
                     where=radius[:, None] > 1e-6)
    unit[radius <= 1e-6] = (1.0, 0.0, 0.0)
    out = np.empty_like(states)
    out[:, :3] = states[:, 3:]
    out[:, 3:] = -MU_EARTH * unit / (radius_safe**2)[:, None]
    return out


def rk4(states: np.ndarray, dt: float) -> np.ndarray:
    """The engine's RK4 two-body step, vectorised over rows."""
    k1 = _derivative(states)
    k2 = _derivative(states + 0.5 * dt * k1)
    k3 = _derivative(states + 0.5 * dt * k2)
    k4 = _derivative(states + dt * k3)
    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# =============================================================================
# Filter
# =============================================================================


def build_filter(config: RingConfig, seeds: dict):
    q_filter = np.diag(config.q_filter_diag)
    propagator = lmb_engine.TwoBodyPropagator(q_filter, seed=seeds["filter"],
                                              noise_reference_dt=config.noise_reference_dt)
    sensor_model = lmb_engine.InOrbitSensorModel(*np.asarray(config.filter_sigmas) ** 2)
    birth_model = lmb_engine.AdaptiveBirthModel(
        config.num_particles, config.p_birth, np.diag(config.birth_covariance_local_diag),
        seed=seeds["birth"])
    tracker = lmb_engine.SMC_LMB_Tracker(
        propagator, sensor_model, birth_model, config.p_survival, config.k_best,
        config.prune_threshold, config.clutter_intensity, config.p_detection,
        config.noise_decay_rate, config.noise_min_scale, seed=seeds["tracker"])
    tracker.set_lazy_propagation(True, config.max_pending)
    tracker.set_regularization(config.regularization, config.regularization_bandwidth_scale,
                               config.regularization_ess_threshold)
    tracker.set_fused_proposal(config.fused_proposal, config.fused_ess_min)
    tracker.set_particle_gate(config.particle_gate)
    tracker.set_gate_audit(config.gate_audit)
    tracker.set_gate_sleep(config.gate_sleep)
    tracker.set_fast_mode(config.fast_mode)
    tracker.set_record_diagnostics(True)
    tracker.set_profiling(config.profile)
    return tracker


def build_sensor_array(config: RingConfig, states: np.ndarray):
    if config.fov_half_angle_deg is None:
        sensors = lmb_engine.SensorArray(lmb_engine.SensorFovConfig(max_range=config.sensor_range))
        for k in range(len(states)):
            sensors.add_unpointed(f"ring_{k:03d}", states[k])
        return sensors
    half_angle = np.deg2rad(config.fov_half_angle_deg)
    sensors = lmb_engine.SensorArray(lmb_engine.SensorFovConfig(
        max_range=config.sensor_range, half_width=half_angle, half_height=half_angle))
    for k in range(len(states)):
        # Radially outward until the tasker's first call, which comes before the first detection.
        sensors.add(f"ring_{k:03d}", states[k], states[k, :3])
    return sensors


def derive_seeds(master_seed: int) -> dict:
    state = np.random.SeedSequence(master_seed).generate_state(5, dtype=np.uint64)
    return {"sample": int(state[0]), "measurement": int(state[1]), "filter": int(state[2]),
            "birth": int(state[3]), "tracker": int(state[4])}


# =============================================================================
# Bookkeeping
# =============================================================================


class Timers:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds


@dataclass
class Pass:
    object_index: int
    start_time: float
    first_pass: bool
    credited_alive_at_start: bool
    detections: int = 0
    birth: bool = False
    taken_by_own: bool = False      # some detection in the pass was taken by a track born from this object

    @property
    def outcome(self) -> str:
        if self.first_pass:
            return "first detection"
        if self.taken_by_own and not self.birth:
            return "re-acquired"
        if self.taken_by_own:
            return "re-acquired, extra birth"
        if not self.birth:
            return "taken by another track"
        return "duplicate birth" if self.credited_alive_at_start else "lost, re-born"


def label_keys(summary) -> np.ndarray:
    return summary["birth_time"].astype(np.int64) * 1_000_000 + summary["index"].astype(np.int64)


# =============================================================================
# The run
# =============================================================================


def run(config: RingConfig, verbose: bool = True, before_update=None, tasker=None,
        scored_objects=None) -> dict:
    """Run the scenario. ``before_update``, if given, is called after predict() and before
    update() at every step, with keyword arguments describing that step (diagnostics only).

    ``tasker`` aims pointed sensors (config.fov_half_angle_deg). ``tasker.point(...)`` is called at
    every step once truth and sensors are at that step and before anything is detected, so the
    detections and the filter's update both see the pointing it sets. ``tasker.after_update(...)``,
    if it has one, is called after the filter's update with that update's diagnostics.

    ``scored_objects``: indices of the objects the metrics are scored against. Default: the objects
    detected so far, which suits one run but depends on where the sensors looked; pass a fixed set
    to compare pointing policies."""
    if config.fov_half_angle_deg is not None and tasker is None:
        raise ValueError("pointed sensors (fov_half_angle_deg) need a tasker to aim them")
    if scored_objects is not None:
        scored_objects = np.asarray(scored_objects, dtype=np.int64)
    seeds = derive_seeds(config.seed)
    timers = Timers()
    wall_start = time.perf_counter()

    catalogue_ids, catalogue = load_catalogue(config.csv_path)
    if config.num_objects > len(catalogue):
        raise ValueError(f"asked for {config.num_objects} objects, catalogue has {len(catalogue)}")
    chosen = np.sort(np.random.default_rng(seeds["sample"]).choice(
        len(catalogue), size=config.num_objects, replace=False))
    object_ids = catalogue_ids[chosen]
    truth = catalogue[chosen].copy()

    sensor_states = ring_states(config.num_sensors, config.sensor_radius)
    sensors = build_sensor_array(config, sensor_states)
    tracker = build_filter(config, seeds)
    rng = np.random.default_rng(seeds["measurement"])
    truth_sigmas = np.asarray(config.truth_sigmas)
    measurement_covariance = np.diag(np.asarray(config.filter_sigmas) ** 2)

    num_steps = int(round(config.duration / config.dt))
    metric_every = max(1, int(round(config.metric_interval / config.dt)))
    progress_every = max(1, int(round(config.progress_interval / config.dt)))

    # Per-object detection history.
    first_detection = np.full(config.num_objects, np.nan)
    last_detection = np.full(config.num_objects, np.nan)
    detections_per_object = np.zeros(config.num_objects, dtype=np.int64)
    detections_per_sensor = np.zeros(config.num_sensors, dtype=np.int64)
    detection_events = []          # (time, object, sensor)
    overlap_events = 0
    passes: list[Pass] = []
    open_pass: dict[int, int] = {}  # object -> index into passes
    credited: dict[int, int] = {}   # object -> label key of the track last born from it
    births = []                     # (time, label key, attributed object or -1)
    known_labels: set[int] = set()
    posterior_records = []          # (time, ESS / N, detection mass, regularized, fused components, fallbacks)
    track_origin: dict[int, int] = {}   # label key -> object whose detection the track was born from
    association_events = []        # (time, object detected, origin object of the track that took it, marginal, fused)

    # Sampled metrics.
    samples = {name: [] for name in (
        "time", "gospa", "localisation", "missed", "false_positive", "num_assigned", "num_truths",
        "num_estimates", "num_tracks", "existence_sum", "num_lagging")}
    object_rows = []   # (time, object, credited error [m], gospa-matched error [m] or nan, since last det [s], nees or nan)
    track_rows = []    # (time, label key, existence, matched object or -1)

    sensor_lo = sensor_states[:, :3].min(axis=0)
    sensor_hi = sensor_states[:, :3].max(axis=0)

    if verbose:
        print("=" * 78)
        print("Equatorial sensor ring vs sampled debris")
        print("=" * 78)
        print(f"  sensors {config.num_sensors} at {config.sensor_altitude / 1e3:.0f} km, "
              f"range {config.sensor_range / 1e3:.0f} km | objects {config.num_objects} "
              f"of {len(catalogue)} | particles {config.num_particles}")
        print(f"  {config.num_orbits:g} orbits = {config.duration:.0f} s = {num_steps} steps of "
              f"{config.dt:g} s | lazy propagation, max_pending {config.max_pending:g} s | "
              f"seed {config.seed}")
        print("-" * 78)

    num_objects = len(truth)
    truth_block = None
    near_block = None        # per step of the block: indices of objects inside the sensors' box
    for step in range(num_steps + 1):
        now = step * config.dt

        # --- truth and sensors ---
        tick = time.perf_counter()
        if step > 0:
            if config.fast_truth:
                k = (step - 1) % config.truth_chunk
                if k == 0:
                    count = min(config.truth_chunk, num_steps - step + 1)
                    truth_block = lmb_engine.two_body_rk4_steps(
                        np.vstack([truth, sensor_states]), config.dt, count)
                    # The detection box test below, for every step of the block at once.
                    reach = config.sensor_range
                    block_lo = truth_block[:, num_objects:, :3].min(axis=1) - reach
                    block_hi = truth_block[:, num_objects:, :3].max(axis=1) + reach
                    positions = truth_block[:, :num_objects, :3]
                    inside = np.all((positions >= block_lo[:, None, :])
                                    & (positions <= block_hi[:, None, :]), axis=2)
                    rows, cols = np.nonzero(inside)
                    near_block = np.split(cols, np.searchsorted(rows, np.arange(1, count)))
                truth = truth_block[k, :num_objects]
                sensor_states = truth_block[k, num_objects:]
            else:
                truth = rk4(truth, config.dt)
                sensor_states = rk4(sensor_states, config.dt)
            sensors.set_states(sensor_states)
            if not config.fast_truth:
                sensor_lo = sensor_states[:, :3].min(axis=0)
                sensor_hi = sensor_states[:, :3].max(axis=0)
        timers.add("truth", time.perf_counter() - tick)

        if tasker is not None:
            tick = time.perf_counter()
            tasker.point(now=now, step=step, config=config, tracker=tracker, sensors=sensors,
                         truth=truth, sensor_states=sensor_states)
            timers.add("tasking", time.perf_counter() - tick)

        # --- detection: range prefilter in NumPy, confirmed by SensorArray.sees ---
        tick = time.perf_counter()
        reach = config.sensor_range
        if config.fast_truth and step > 0:
            near = near_block[(step - 1) % config.truth_chunk]
        else:
            near = np.flatnonzero(np.all((truth[:, :3] >= sensor_lo - reach)
                                         & (truth[:, :3] <= sensor_hi + reach), axis=1))
        measurements, measured_objects = [], []
        # Range prefilter for every near object at once; objects stay in index order, so the
        # detection draws below are taken in the same order as an object-by-object loop.
        in_reach = (np.linalg.norm(sensor_states[None, :, :3] - truth[near, None, :3], axis=2) <= reach
                    if len(near) else np.zeros((0, len(sensor_states)), dtype=bool))
        for row in np.flatnonzero(in_reach.any(axis=1)):
            o = near[row]
            candidates = np.flatnonzero(in_reach[row])
            seen_by = [int(k) for k in candidates if sensors.sees(int(k), truth[o])]
            if not seen_by:
                continue
            if len(seen_by) > 1:
                overlap_events += 1
            k = seen_by[0]
            if rng.random() > config.p_detection:
                continue
            measurement = lmb_engine.Measurement.fromCartesian(truth[o], sensor_states[k]).perturbed(
                rng.normal(size=6) * truth_sigmas)
            measurement.timestamp_ = now
            measurement.sensor_id_ = sensors.id(k)
            measurement.covariance_ = measurement_covariance.copy()
            measurements.append(measurement)
            measured_objects.append(int(o))
            detections_per_sensor[k] += 1
            detection_events.append((now, int(o), k))
        timers.add("detection", time.perf_counter() - tick)

        # --- pass bookkeeping (before the update, so "alive at start" means before this step) ---
        if measured_objects:
            live = set(label_keys(tracker.track_summary(with_means=False)).tolist())
            for o in measured_objects:
                gap = now - last_detection[o]
                if np.isnan(last_detection[o]) or gap > PASS_GAP:
                    passes.append(Pass(object_index=o, start_time=now,
                                       first_pass=bool(np.isnan(first_detection[o])),
                                       credited_alive_at_start=credited.get(o, -1) in live))
                    open_pass[o] = len(passes) - 1
                passes[open_pass[o]].detections += 1
                if np.isnan(first_detection[o]):
                    first_detection[o] = now
                last_detection[o] = now
                detections_per_object[o] += 1

        # --- filter ---
        tick = time.perf_counter()
        if step > 0:
            tracker.predict(config.dt)
        timers.add("predict", time.perf_counter() - tick)
        if before_update is not None:
            before_update(now=now, config=config, tracker=tracker, sensors=sensors, truth=truth,
                          sensor_states=sensor_states, measurements=measurements,
                          measured_objects=measured_objects, passes=passes, open_pass=open_pass,
                          credited=credited)
        tick = time.perf_counter()
        tracker.update(measurements, sensors)
        timers.add("update", time.perf_counter() - tick)
        records = tracker.take_diagnostics()
        if tasker is not None and hasattr(tasker, "after_update"):
            tick = time.perf_counter()
            tasker.after_update(now=now, step=step, config=config, tracker=tracker, records=records,
                                measurements=measurements, measured_objects=measured_objects)
            timers.add("tasking", time.perf_counter() - tick)
        if len(records["time"]):
            posterior_records.append(np.column_stack([
                records["time"], records["ess"] / np.maximum(records["num_particles"], 1),
                records["detection_mass"], records["regularized"].astype(np.float64),
                records["fused_components"].astype(np.float64),
                records["fallback_components"].astype(np.float64)]))
            # Which existing track took each detection (marginal >= 0.5), and was it the right one?
            keys_taken = (records["birth_time"].astype(np.int64) * 1_000_000
                          + records["index"].astype(np.int64))
            for key, j, coefficient, fused in zip(keys_taken.tolist(), records["best_measurement"].tolist(),
                                                  records["best_coefficient"].tolist(),
                                                  records["fused_components"].tolist()):
                if j >= 0 and coefficient >= 0.5 and j < len(measured_objects):
                    o = measured_objects[j]
                    association_events.append((now, o, track_origin.get(int(key), -1), coefficient,
                                               float(fused > 0)))
                    if o in open_pass and track_origin.get(int(key), -1) == o:
                        passes[open_pass[o]].taken_by_own = True

        # --- births, attributed to the nearest object measured this step ---
        if measured_objects:
            tick = time.perf_counter()
            summary = tracker.track_summary(with_means=False)
            keys = label_keys(summary)
            fresh = [i for i, key in enumerate(keys.tolist()) if key not in known_labels]
            if fresh:
                means = tracker.track_summary(with_means=True)["mean"]
                measured_positions = truth[measured_objects, :3]
                for i in fresh:
                    gaps = np.linalg.norm(measured_positions - means[i, :3], axis=1)
                    o = measured_objects[int(np.argmin(gaps))]
                    births.append((now, int(keys[i]), o))
                    credited[o] = int(keys[i])
                    track_origin[int(keys[i])] = o
                    if o in open_pass:
                        passes[open_pass[o]].birth = True
            known_labels.update(keys.tolist())
            timers.add("bookkeeping", time.perf_counter() - tick)

        # --- metrics ---
        if step % metric_every == 0 or step == num_steps:
            tick = time.perf_counter()
            lagging = int(np.sum(tracker.track_summary(with_means=False)["propagated_time"] < now))
            tracker.synchronize()
            # Means, covariances and GOSPA straight from the tracker: no particle cloud is copied.
            summary = tracker.track_summary(with_means=True, with_covariances=True)
            keys = label_keys(summary)
            existence = summary["existence"]
            num_tracks = len(existence)
            estimate_idx = np.flatnonzero(existence >= config.existence_threshold)
            detected = (scored_objects if scored_objects is not None
                        else np.flatnonzero(~np.isnan(first_detection)))
            breakdown = tracker.gospa_components(estimate_idx.tolist(), truth[detected], GOSPA_CUTOFF)

            samples["time"].append(now)
            samples["gospa"].append(breakdown.total)
            samples["localisation"].append(breakdown.localisation)
            samples["missed"].append(breakdown.missed)
            samples["false_positive"].append(breakdown.false_positive)
            samples["num_assigned"].append(breakdown.num_assigned)
            samples["num_truths"].append(len(detected))
            samples["num_estimates"].append(len(estimate_idx))
            samples["num_tracks"].append(num_tracks)
            samples["existence_sum"].append(float(np.sum(existence)))
            samples["num_lagging"].append(lagging)

            matched_object_of_track = np.full(num_tracks, -1, dtype=np.int64)
            matched_error = {}
            matched_nees = {}
            for e, j in enumerate(breakdown.associations):
                if j < 0:
                    continue
                track_index = int(estimate_idx[e])
                o = int(detected[j])
                matched_object_of_track[track_index] = o
                error = summary["mean"][track_index, :3] - truth[o, :3]
                matched_error[o] = float(np.linalg.norm(error))
                covariance = summary["covariance"][track_index, :3, :3]
                try:
                    matched_nees[o] = float(error @ np.linalg.solve(covariance, error))
                except np.linalg.LinAlgError:
                    matched_nees[o] = np.nan

            index_of_key = {int(k): i for i, k in enumerate(keys.tolist())}
            for o in detected:
                o = int(o)
                track_index = index_of_key.get(credited.get(o, -1))
                credited_error = (np.nan if track_index is None else
                                  float(np.linalg.norm(summary["mean"][track_index, :3] - truth[o, :3])))
                object_rows.append((now, o, credited_error, matched_error.get(o, np.nan),
                                    now - last_detection[o], matched_nees.get(o, np.nan)))
            for i, key in enumerate(keys.tolist()):
                track_rows.append((now, int(key), float(existence[i]), int(matched_object_of_track[i])))
            timers.add("metrics", time.perf_counter() - tick)

        if verbose and (step % progress_every == 0 or step == num_steps):
            elapsed = time.perf_counter() - wall_start
            eta = elapsed / max(step, 1) * (num_steps - step)
            print(f"  t={now:7.0f}s  detections={len(detection_events):5d}  "
                  f"objects seen={int(np.sum(~np.isnan(first_detection))):4d}  "
                  f"tracks={len(tracker.track_summary(with_means=False)['existence']):4d}  "
                  f"births={len(births):4d}  wall={elapsed:6.1f}s  eta={eta:6.1f}s", flush=True)

    wall = time.perf_counter() - wall_start
    outcomes = {}
    for p in passes:
        outcomes[p.outcome] = outcomes.get(p.outcome, 0) + 1

    config_record = asdict(config)
    config_record.update(filter_sigmas=config.filter_sigmas.tolist(),
                         q_filter_diag=config.q_filter_diag.tolist())
    log = {
        "config": config_record,
        "object_ids": object_ids,
        "detections_per_sensor": detections_per_sensor,
        "detections_per_object": detections_per_object,
        "detection_events": np.asarray(detection_events, dtype=np.float64).reshape(-1, 3),
        "births": np.asarray(births, dtype=np.float64).reshape(-1, 3),
        "passes": np.asarray([(p.object_index, p.start_time, p.first_pass, p.detections, p.birth,
                               p.credited_alive_at_start, p.taken_by_own) for p in passes],
                             dtype=np.float64).reshape(-1, 7),
        "pass_outcomes": outcomes,
        "object_rows": np.asarray(object_rows, dtype=np.float64).reshape(-1, 6),
        "track_rows": np.asarray(track_rows, dtype=np.float64).reshape(-1, 4),
        "posterior_records": (np.vstack(posterior_records) if posterior_records
                              else np.zeros((0, 6))),
        "association_events": np.asarray(association_events, dtype=np.float64).reshape(-1, 5),
        "timers": timers.totals,
        "engine_profile": tracker.profile() if config.profile else {},
        "gate_audit": (list(tracker.gate_audit_counts()) + list(tracker.sleep_audit()[:2])
                       if config.gate_audit else []),
        "wall_seconds": wall,
        "overlap_events": overlap_events,
        "gospa_params": GOSPA_PARAMS,
        "gospa_cutoff": GOSPA_CUTOFF,
        # What the metrics were scored against: the objects detected so far, or a given set.
        "scoring": "detected" if scored_objects is None else "given",
    }
    log.update({name: np.asarray(values, dtype=np.float64) for name, values in samples.items()})
    return log


# =============================================================================
# Output
# =============================================================================


ARRAY_KEYS = ("object_ids", "detections_per_sensor", "detections_per_object", "detection_events",
              "births", "passes", "object_rows", "track_rows", "posterior_records",
              "association_events", "time", "gospa", "localisation",
              "missed", "false_positive", "num_assigned", "num_truths", "num_estimates",
              "num_tracks", "existence_sum", "num_lagging")


def profile_summary(profile: dict) -> dict:
    """Engine phase seconds, work counts, and the derived rates worth reading (None when a
    denominator is zero)."""
    if not profile:
        return {}
    seconds, counts = profile["seconds"], profile["counts"]

    def ratio(num, den, scale=1.0):
        return round(scale * num / den, 4) if den else None

    refreshed = counts["tracks_refreshed"]
    derived = {
        "ns_per_particle_step_predict": ratio(seconds["predict_propagate"],
                                              counts["predict_particle_steps"], 1e9),
        "ns_per_particle_step_refresh": ratio(seconds["refresh_propagate"],
                                              counts["refresh_particle_steps"], 1e9),
        "refresh_share_of_update": ratio(seconds["refresh"], seconds["update"]),
        "refresh_propagate_share_of_update": ratio(seconds["refresh_propagate"], seconds["update"]),
        "fused_share_of_update": ratio(seconds["fused"], seconds["update"]),
        "tracks_refreshed_per_update": ratio(refreshed, counts["update_calls"]),
        "refreshed_fraction_covered": ratio(counts["refreshed_tracks_covered"], refreshed),
        "refreshed_fraction_gap_over_5km": ratio(counts["refresh_gap_5_to_50km"]
                                                 + counts["refresh_gap_over_50km"], refreshed),
    }
    return {"seconds": {k: round(v, 3) for k, v in seconds.items()}, "counts": dict(counts),
            "derived": derived}


def run_summary(log: dict) -> dict:
    tracking = np.divide(log["num_assigned"], log["num_truths"],
                         out=np.zeros_like(log["num_assigned"]), where=log["num_truths"] > 0)
    records = np.asarray(log["posterior_records"])
    records = records.reshape(-1, records.shape[1] if records.ndim == 2 and records.shape[1] else 6)
    events = np.asarray(log.get("association_events", np.zeros((0, 5)))).reshape(-1, 5)
    own = events[events[:, 1] == events[:, 2]]
    steals = events[(events[:, 2] >= 0) & (events[:, 1] != events[:, 2])]
    detection = records[records[:, 2] >= 0.5]
    miss = records[records[:, 2] < 0.5]
    nees = np.asarray(log["object_rows"]).reshape(-1, 6)[:, 5]
    nees = nees[np.isfinite(nees)]
    return {
        "median_ess_fraction_detection_updates": float(np.median(detection[:, 1])) if len(detection) else None,
        "median_ess_fraction_miss_updates": float(np.median(miss[:, 1])) if len(miss) else None,
        "fraction_of_updates_regularized": float(records[:, 3].mean()) if len(records) else 0.0,
        "updates_with_fused_component": int(np.sum(records[:, 4] > 0)) if records.shape[1] > 4 else 0,
        "fused_fallbacks": int(np.sum(records[:, 5])) if records.shape[1] > 5 else 0,
        "detections_taken_by_own_track": int(len(own)),
        "detections_taken_by_other_object_track": int(len(steals)),
        "detections_taken_with_fused_component": int(np.sum(events[:, 4] > 0)) if len(events) else 0,
        "median_nees": float(np.median(nees)) if len(nees) else None,
        "nees_inside_95pct_band": float(np.mean((nees >= 0.2158) & (nees <= 9.3484))) if len(nees) else None,
        "wall_seconds": round(log["wall_seconds"], 2),
        "wall_seconds_by_phase": {k: round(v, 2) for k, v in log["timers"].items()},
        "engine_profile": profile_summary(log.get("engine_profile") or {}),
        # (gate decisions audited, with a particle in a volume, sleeping skips audited, whose test
        # would have said "maybe")
        "gate_audit": list(log.get("gate_audit") or []),
        "detections": int(log["detections_per_sensor"].sum()),
        "objects_detected": int(np.sum(log["detections_per_object"] > 0)),
        "sensors_with_detections": int(np.sum(log["detections_per_sensor"] > 0)),
        "births": int(len(log["births"])),
        "pass_outcomes": log["pass_outcomes"],
        "overlap_events": int(log["overlap_events"]),
        "final_num_tracks": int(log["num_tracks"][-1]),
        "final_tracking_fraction": float(tracking[-1]),
        "mean_tracking_fraction": float(tracking[log["num_truths"] > 0].mean())
        if np.any(log["num_truths"] > 0) else 0.0,
    }


def save(log: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "ring_log.npz",
                        meta=json.dumps({k: log[k] for k in ("config", "pass_outcomes", "timers",
                                                             "engine_profile", "gate_audit",
                                                             "wall_seconds",
                                                             "overlap_events", "gospa_params",
                                                             "gospa_cutoff", "scoring") if k in log}),
                        **{k: log[k] for k in ARRAY_KEYS})
    summary = run_summary(log)
    (output_dir / "summary.json").write_text(json.dumps({"config": log["config"], "summary": summary},
                                                        indent=2))
    return output_dir


def load(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        log = {k: data[k] for k in ARRAY_KEYS}
        log.update(json.loads(str(data["meta"])))
    return log


def main(argv=None) -> None:
    config = config_from_args(argv)
    log = run(config)
    output_dir = save(log, Path(config.output_dir))
    summary = run_summary(log)
    print("-" * 78)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    try:
        from evaluation_plots import make_all_plots
    except ImportError as error:  # matplotlib missing: keep the log, skip the figures
        print(f"  figures skipped ({error})")
    else:
        for path in make_all_plots(log, output_dir):
            print(f"  wrote {path}")
    print(f"  log: {output_dir / 'ring_log.npz'}")


if __name__ == "__main__":
    main()
