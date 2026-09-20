"""Fully-seeded scenario runner shared by the Phase 0 verification harnesses.

The engine draws randomness from five independent places, and every one of them has to be pinned
for a run to be reproducible:

1. NumPy's legacy global stream  -- detection rolls and measurement noise in ``generate_measurements``
2. the truth propagator          -- ``get_ground_truth_propagator`` uses ``eye(6) * 1e-18``, whose trace
                                    clears the ``> 1e-24`` gate in ``TwoBodyPropagator``, so truth
                                    propagation genuinely consumes random numbers
3. the filter propagator         -- ``Q_FILTER`` process noise per particle per step
4. the birth model               -- particle sampling for new tracks
5. the tracker's resampler       -- the single systematic-resampling offset per track per step

``derive_seeds`` turns one master integer into all five via ``SeedSequence``, so a scenario is a
pure function of that integer.

The digest computes its own weighted mean and covariance directly from particle data rather than
calling into the drivers. That is deliberate: the digest is a measuring instrument for *filter
state*, so it must not change when a later phase changes how Python or C++ summarizes a cloud.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
for _search_path in (str(REPO_ROOT / "python"), str(TESTS_DIR)):
    if _search_path not in sys.path:
        sys.path.insert(0, _search_path)

import run_once  # noqa: E402

lmb = run_once.lmb_engine

FIXTURE_DIR = TESTS_DIR / "fixtures"
OSPA_CUTOFF = 100000.0

# Integer digest fields must match exactly even in --rtol mode: they encode track identity,
# cardinality and pruning decisions, none of which may drift for a floating-point reason.
INT_FIELDS = ("cardinality", "num_measurements", "track_step", "track_birth", "track_index")
FLOAT_FIELDS = ("ospa", "track_r", "track_mean", "track_cov_trace", "track_weight_sum")


@dataclass(frozen=True)
class ScenarioConfig:
    """Knobs for a harness run. Defaults are the short configuration the harnesses gate on."""

    num_steps: int = 40
    num_particles: int = 200
    k_best: int = 2


DEFAULT_CONFIG = ScenarioConfig()


def derive_seeds(master_seed: int) -> dict[str, int]:
    """Expand one master seed into the five independent streams the scenario needs."""
    state = np.random.SeedSequence(master_seed).generate_state(5, dtype=np.uint64)
    return {
        # np.random.seed only accepts a 32-bit value.
        "measurement": int(state[0] % (2**32)),
        "truth": int(state[1]),
        "filter": int(state[2]),
        "birth": int(state[3]),
        "tracker": int(state[4]),
    }


@dataclass
class Digest:
    """Canonical per-step, per-track summary of a run.

    Track rows are stored flat with a ``track_step`` column instead of a ragged per-step list,
    because the number of live tracks varies from step to step.
    """

    master_seed: int
    config: ScenarioConfig
    ospa: np.ndarray
    cardinality: np.ndarray
    num_measurements: np.ndarray
    track_step: np.ndarray
    track_birth: np.ndarray
    track_index: np.ndarray
    track_r: np.ndarray
    track_mean: np.ndarray
    track_cov_trace: np.ndarray
    track_weight_sum: np.ndarray

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        payload = {name: getattr(self, name) for name in INT_FIELDS + FLOAT_FIELDS}
        payload["master_seed"] = np.array([self.master_seed], dtype=np.uint64)
        payload["config"] = np.array(
            [self.config.num_steps, self.config.num_particles, self.config.k_best], dtype=np.int64
        )
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.to_npz_dict())

    @staticmethod
    def load(path: Path) -> "Digest":
        with np.load(path) as data:
            num_steps, num_particles, k_best = (int(value) for value in data["config"])
            fields = {name: data[name] for name in INT_FIELDS + FLOAT_FIELDS}
            return Digest(
                master_seed=int(data["master_seed"][0]),
                config=ScenarioConfig(num_steps=num_steps, num_particles=num_particles, k_best=k_best),
                **fields,
            )


def particle_arrays(track) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(states, weights)`` as float64 arrays for one track."""
    particles = track.particles()
    if len(particles) == 0:
        return np.zeros((0, 6), dtype=np.float64), np.zeros(0, dtype=np.float64)
    states = np.asarray([p.state_vector for p in particles], dtype=np.float64)
    weights = np.asarray([p.weight for p in particles], dtype=np.float64)
    return states, weights


def summarize_track(track) -> tuple[np.ndarray, float, float]:
    """Weighted mean, weighted covariance trace and raw weight sum for one particle cloud."""
    states, weights = particle_arrays(track)
    if states.shape[0] == 0:
        return np.zeros(6, dtype=np.float64), 0.0, 0.0

    weight_sum = float(np.sum(weights))
    if weight_sum > 1e-12:
        normalized = weights / weight_sum
    else:
        normalized = np.full(states.shape[0], 1.0 / states.shape[0], dtype=np.float64)

    mean = np.average(states, weights=normalized, axis=0)
    deviations = states - mean
    cov_trace = float(np.sum(normalized * np.sum(deviations * deviations, axis=1)))
    return mean, cov_trace, weight_sum


def build_models(seeds: dict[str, int], config: ScenarioConfig):
    """Construct the four seeded engine objects a run needs."""
    truth_propagator = lmb.TwoBodyPropagator(np.eye(6) * 1e-18, seed=seeds["truth"])
    filter_propagator = lmb.TwoBodyPropagator(run_once.Q_FILTER, seed=seeds["filter"])
    sensor_model = lmb.InOrbitSensorModel(*run_once.FILTER_SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(
        config.num_particles,
        run_once.P_BIRTH,
        run_once.BIRTH_COVARIANCE_LOCAL,
        seed=seeds["birth"],
    )
    tracker = lmb.SMC_LMB_Tracker(
        filter_propagator,
        sensor_model,
        birth_model,
        run_once.P_SURVIVAL,
        config.k_best,
        run_once.PRUNE_THRESHOLD,
        run_once.CLUTTER_INTENSITY,
        run_once.P_DETECTION,
        run_once.NOISE_DECAY_RATE,
        run_once.NOISE_MIN_SCALE,
        seed=seeds["tracker"],
    )
    return truth_propagator, tracker


def run_scenario(
    master_seed: int,
    config: ScenarioConfig = DEFAULT_CONFIG,
    observer: Optional[Callable[..., None]] = None,
) -> Digest:
    """Run a fully-seeded simulation and return its digest.

    ``observer``, when given, is called after every update as
    ``observer(step=..., tracker=..., tracks=..., measurements=..., measurements_before=...)``.
    ``test_invariants.py`` uses it so it does not have to duplicate this loop.
    """
    seeds = derive_seeds(master_seed)
    np.random.seed(seeds["measurement"])
    truth_propagator, tracker = build_models(seeds, config)

    sensor_state = run_once.SENSOR_STATE.copy()
    active_truths: list[tuple[int, np.ndarray]] = []

    ospa = np.zeros(config.num_steps, dtype=np.float64)
    cardinality = np.zeros(config.num_steps, dtype=np.int64)
    num_measurements = np.zeros(config.num_steps, dtype=np.int64)
    track_step: list[int] = []
    track_birth: list[int] = []
    track_index: list[int] = []
    track_r: list[float] = []
    track_mean: list[np.ndarray] = []
    track_cov_trace: list[float] = []
    track_weight_sum: list[float] = []

    for step in range(config.num_steps):
        current_time = step * run_once.DT

        if step > 0:
            for i, (obj_id, state) in enumerate(active_truths):
                active_truths[i] = (obj_id, run_once.propagate_truth_state(truth_propagator, state, run_once.DT))
            sensor_state = run_once.propagate_truth_state(truth_propagator, sensor_state, run_once.DT)

        for obj_id, birth_step, initial_state in run_once.SCENARIO:
            if step == birth_step:
                active_truths.append((obj_id, initial_state.copy()))

        measurements = run_once.generate_measurements(active_truths, sensor_state, current_time)
        # Snapshot so test_invariants.py can prove update() does not mutate its input.
        measurements_before = [
            (
                m.range_,
                m.range_rate_,
                np.array(m.los_, dtype=np.float64),
                np.array(m.los_rate_, dtype=np.float64),
                np.array(m.sensor_state_, dtype=np.float64),
                np.array(m.covariance_, dtype=np.float64),
                m.timestamp_,
                m.sensor_id_,
            )
            for m in measurements
        ]

        if step > 0:
            tracker.predict(run_once.DT)
        tracker.update(measurements)

        tracks = tracker.get_tracks()
        truth_states = [state.copy() for (_, state) in active_truths]

        cardinality[step] = len(tracks)
        num_measurements[step] = len(measurements)
        ospa[step] = (
            lmb.calculate_ospa_distance(tracks, truth_states, OSPA_CUTOFF) if truth_states else 0.0
        )

        for track in tracks:
            mean, cov_trace, weight_sum = summarize_track(track)
            label = track.label()
            track_step.append(step)
            track_birth.append(int(label.birth_time))
            track_index.append(int(label.index))
            track_r.append(float(track.existence_probability()))
            track_mean.append(mean)
            track_cov_trace.append(cov_trace)
            track_weight_sum.append(weight_sum)

        if observer is not None:
            observer(
                step=step,
                tracker=tracker,
                tracks=tracks,
                measurements=measurements,
                measurements_before=measurements_before,
            )

    return Digest(
        master_seed=master_seed,
        config=config,
        ospa=ospa,
        cardinality=cardinality,
        num_measurements=num_measurements,
        track_step=np.asarray(track_step, dtype=np.int64),
        track_birth=np.asarray(track_birth, dtype=np.int64),
        track_index=np.asarray(track_index, dtype=np.int64),
        track_r=np.asarray(track_r, dtype=np.float64),
        track_mean=(
            np.asarray(track_mean, dtype=np.float64) if track_mean else np.zeros((0, 6), dtype=np.float64)
        ),
        track_cov_trace=np.asarray(track_cov_trace, dtype=np.float64),
        track_weight_sum=np.asarray(track_weight_sum, dtype=np.float64),
    )


def compare_digests(
    reference: Digest,
    candidate: Digest,
    rtol: float,
    atol: float = 0.0,
    int_fields: tuple[str, ...] = INT_FIELDS,
    float_fields: tuple[str, ...] = FLOAT_FIELDS,
) -> list[str]:
    """Return a list of human-readable failures; empty means the digests agree.

    Integer fields are always compared exactly regardless of ``rtol``. ``int_fields`` and
    ``float_fields`` narrow the comparison to a subset, which
    ``test_golden_invariance.py`` uses off the reference platform to compare only the fields that
    are reproducible there.
    """
    failures: list[str] = []

    if reference.config != candidate.config:
        failures.append(f"config differs: reference {reference.config} vs candidate {candidate.config}")
        return failures
    if reference.master_seed != candidate.master_seed:
        failures.append(
            f"master_seed differs: reference {reference.master_seed} vs candidate {candidate.master_seed}"
        )
        return failures

    for name in int_fields:
        ref_values = getattr(reference, name)
        cand_values = getattr(candidate, name)
        if ref_values.shape != cand_values.shape:
            failures.append(f"{name}: shape {cand_values.shape} != reference {ref_values.shape}")
            continue
        mismatches = np.flatnonzero(ref_values != cand_values)
        if mismatches.size:
            first = int(mismatches[0])
            failures.append(
                f"{name}: {mismatches.size} exact mismatches, first at flat index {first} "
                f"(reference {ref_values.ravel()[first]}, candidate {cand_values.ravel()[first]})"
            )

    for name in float_fields:
        ref_values = getattr(reference, name)
        cand_values = getattr(candidate, name)
        if ref_values.shape != cand_values.shape:
            failures.append(f"{name}: shape {cand_values.shape} != reference {ref_values.shape}")
            continue
        if ref_values.size == 0:
            continue
        if rtol == 0.0 and atol == 0.0:
            mismatches = np.flatnonzero(
                (ref_values != cand_values).ravel() & ~(np.isnan(ref_values.ravel()) & np.isnan(cand_values.ravel()))
            )
            if mismatches.size:
                first = int(mismatches[0])
                failures.append(
                    f"{name}: {mismatches.size} bitwise mismatches, first at flat index {first} "
                    f"(reference {ref_values.ravel()[first]!r}, candidate {cand_values.ravel()[first]!r})"
                )
            continue
        close = np.isclose(cand_values, ref_values, rtol=rtol, atol=atol, equal_nan=True)
        if not close.all():
            bad = np.flatnonzero(~close.ravel())
            first = int(bad[0])
            denominator = np.where(np.abs(ref_values.ravel()) > 0, np.abs(ref_values.ravel()), 1.0)
            worst = float(np.max(np.abs(cand_values.ravel() - ref_values.ravel()) / denominator))
            failures.append(
                f"{name}: {bad.size}/{ref_values.size} values outside rtol={rtol:g}, "
                f"worst relative deviation {worst:.3e}, first at flat index {first} "
                f"(reference {ref_values.ravel()[first]!r}, candidate {cand_values.ravel()[first]!r})"
            )

    return failures
