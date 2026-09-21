"""Benchmark the engine's hot paths and peak memory, for the before/after numbers every phase records.

Rule 3 of the plan requires each phase to justify itself with measurements, and a phase whose gain
is under about 3% of the relevant benchmark is reverted even when it is correct. This script is the
instrument for that decision.

Rather than simulating 50 steps at production particle counts just to reach three live tracks, the
steady state is built directly: the truths are propagated cheaply to a chosen step, three
measurements are formed, and the birth model is asked for three full clouds. A few warm-up filter
cycles then settle the state before timing starts. That reproduces the N=3 / M=3 / P=10,000 /
K_BEST=100 regime the plan's baseline numbers came from, in a fraction of the time.

Usage:
    python tests/bench_engine.py                          # production sizing
    python tests/bench_engine.py --quick                  # small sizing, for a smoke run
    python tests/bench_engine.py --json before.json       # record a result set
    python tests/bench_engine.py --compare before.json    # diff against a recorded set
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_scenario as hs  # noqa: E402
import run_once  # noqa: E402

lmb = hs.lmb

# Phase 3's target is the per-particle likelihood work, so the benchmark must be run at a step
# where all three objects are alive and all three produce measurements.
WARMUP_STEP = 55
WARMUP_CYCLES = 3
SEED = 20260908


@dataclass(frozen=True)
class BenchConfig:
    num_particles: int
    k_best: int
    repeats: int
    warmups: int = 2

    @property
    def label(self) -> str:
        return f"P={self.num_particles} K_BEST={self.k_best} repeats={self.repeats}"


PRODUCTION = BenchConfig(num_particles=10000, k_best=100, repeats=21)
QUICK = BenchConfig(num_particles=1000, k_best=16, repeats=5)

# A fixed, engine-independent workload, timed alongside everything else and reported as a trust
# indicator. Wall-clock benchmarks on a developer machine drift with CPU frequency scaling, thermal
# state and background load, so without a reference a phase can be credited or blamed for the
# machine's mood. No phase in the plan can affect this number.
#
# It is deliberately *not* used as a divisor. The engine timings reproduce to about 2% run to run
# and this reference to about 4%, so scaling by it would add more noise than it removed. It is used
# only to flag a comparison as untrustworthy, which is the question it can actually answer.
#
# np.sin over a large preallocated buffer is chosen over a matmul because BLAS matmul is threaded
# and measured at 97% run-to-run spread here, against 4% for this.
CALIBRATION_ELEMENTS = 2_000_000
CALIBRATION_DRIFT_WARN = 0.05


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB. ru_maxrss is in KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def time_operation(operation, config: BenchConfig) -> tuple[float, float]:
    """Time ``operation``. Returns ``(median_ms, min_ms)``.

    The first calls touch fresh pages and grow the allocator, so ``config.warmups`` iterations are
    discarded. ``min`` is the headline statistic because it is the sample least contaminated by
    preemption and frequency transitions; the median is kept alongside it as a spread check.
    """
    for _ in range(config.warmups):
        operation()
    samples = []
    for _ in range(config.repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples)), float(np.min(samples))


def build_steady_state(config: BenchConfig):
    """Return ``(tracker, measurements, truth_states, snapshot_tracks)`` at the warm-up step."""
    seeds = hs.derive_seeds(SEED)
    np.random.seed(seeds["measurement"])

    truth_propagator = lmb.TwoBodyPropagator(np.eye(6) * 1e-18, seed=seeds["truth"])
    filter_propagator = lmb.TwoBodyPropagator(run_once.Q_FILTER, seed=seeds["filter"])
    sensor_model = lmb.InOrbitSensorModel(*run_once.FILTER_SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(
        config.num_particles, run_once.P_BIRTH, run_once.BIRTH_COVARIANCE_LOCAL, seed=seeds["birth"]
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

    # Cheap truth-only propagation up to the warm-up step: no particle clouds exist yet.
    sensor_state = run_once.SENSOR_STATE.copy()
    active: list[tuple[int, np.ndarray]] = []
    for step in range(WARMUP_STEP + 1):
        if step > 0:
            for i, (obj_id, state) in enumerate(active):
                active[i] = (obj_id, run_once.propagate_truth_state(truth_propagator, state, run_once.DT))
            sensor_state = run_once.propagate_truth_state(truth_propagator, sensor_state, run_once.DT)
        for obj_id, birth_step, initial_state in run_once.SCENARIO:
            if step == birth_step:
                active.append((obj_id, initial_state.copy()))

    current_time = WARMUP_STEP * run_once.DT
    measurements = run_once.generate_measurements(active, sensor_state, current_time)
    if len(measurements) != len(run_once.SCENARIO):
        raise RuntimeError(f"expected {len(run_once.SCENARIO)} measurements, got {len(measurements)}")

    tracker.set_tracks(birth_model.generate_new_tracks(measurements, current_time))

    # Each warm-up cycle advances the truths too. Re-using one stale measurement set would let the
    # predicted clouds drift away from it and prune a track before the benchmark even starts.
    for cycle in range(WARMUP_CYCLES):
        for i, (obj_id, state) in enumerate(active):
            active[i] = (obj_id, run_once.propagate_truth_state(truth_propagator, state, run_once.DT))
        sensor_state = run_once.propagate_truth_state(truth_propagator, sensor_state, run_once.DT)
        current_time = (WARMUP_STEP + cycle + 1) * run_once.DT
        measurements = run_once.generate_measurements(active, sensor_state, current_time)
        tracker.predict(run_once.DT)
        tracker.update(measurements)

    tracks = tracker.get_tracks()
    if len(tracks) != len(run_once.SCENARIO):
        raise RuntimeError(f"expected {len(run_once.SCENARIO)} tracks after warm-up, got {len(tracks)}")

    # The benchmark's measurements must belong to the step *after* the snapshot, so that the timed
    # predict and the timed update are correctly paired. Feeding stale measurements to update()
    # leaves them unassociated, which spawns birth tracks and silently changes the timed regime.
    for i, (obj_id, state) in enumerate(active):
        active[i] = (obj_id, run_once.propagate_truth_state(truth_propagator, state, run_once.DT))
    sensor_state = run_once.propagate_truth_state(truth_propagator, sensor_state, run_once.DT)
    bench_time = (WARMUP_STEP + WARMUP_CYCLES + 1) * run_once.DT
    bench_measurements = run_once.generate_measurements(active, sensor_state, bench_time)
    if len(bench_measurements) != len(run_once.SCENARIO):
        raise RuntimeError(
            f"expected {len(run_once.SCENARIO)} benchmark measurements, got {len(bench_measurements)}"
        )

    truth_states = [state.copy() for (_, state) in active]
    return tracker, bench_measurements, truth_states, tracks


def build_cost_matrix(tracker, tracks, measurements) -> np.ndarray:
    """Reproduce the tracker's augmented N x (M+N) cost matrix so the solver is timed on real data."""
    num_tracks, num_meas = len(tracks), len(measurements)
    cost = np.full((num_tracks, num_meas + num_tracks), 1e9, dtype=np.float64)
    miss_cost = -np.log(max(1.0 - run_once.P_DETECTION, 1e-12))
    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            likelihood = tracker.compute_association_likelihood(track, measurement)
            cost[i, j] = -np.log(
                max(run_once.P_DETECTION * likelihood / run_once.CLUTTER_INTENSITY, 1e-12)
            )
        cost[i, num_meas + i] = miss_cost
    return cost


def run_benchmarks(config: BenchConfig) -> dict:
    tracker, measurements, truth_states, snapshot = build_steady_state(config)
    num_tracks = len(snapshot)
    results: dict[str, dict[str, float]] = {}

    def record(name: str, median_ms: float, min_ms: float, note: str = "") -> None:
        results[name] = {"median_ms": median_ms, "min_ms": min_ms, "note": note}

    rng = np.random.default_rng(SEED)
    calibration_input = rng.normal(size=CALIBRATION_ELEMENTS)
    calibration_output = np.empty_like(calibration_input)
    calibration_median, calibration_min = time_operation(
        lambda: np.sin(calibration_input, out=calibration_output), config
    )
    record("calibration (numpy sin)", calibration_median, calibration_min, "engine-independent reference")

    # predict and update mutate the filter, so each repeat is restored from the same snapshot.
    def predict_once() -> None:
        tracker.set_tracks(snapshot)
        tracker.predict(run_once.DT)

    median_ms, min_ms = time_operation(predict_once, config)
    # set_tracks copies the clouds, so subtract it to leave the propagation cost alone.
    restore_median, restore_min = time_operation(lambda: tracker.set_tracks(snapshot), config)
    record("predict", median_ms - restore_median, min_ms - restore_min, f"{num_tracks} tracks, net of restore")
    record("set_tracks (restore)", restore_median, restore_min, f"{num_tracks} cloud copies")

    tracker.set_tracks(snapshot)
    tracker.predict(run_once.DT)
    predicted = tracker.get_tracks()

    def update_once() -> None:
        tracker.set_tracks(predicted)
        tracker.update(measurements)

    median_ms, min_ms = time_operation(update_once, config)
    record(
        "update",
        median_ms - restore_median,
        min_ms - restore_min,
        f"{num_tracks}x{len(measurements)}, net of restore",
    )

    tracker.set_tracks(predicted)
    tracker.update(measurements)
    updated = tracker.get_tracks()
    if len(updated) != num_tracks:
        raise RuntimeError(
            f"the timed update changed cardinality from {num_tracks} to {len(updated)}; the benchmark "
            "is no longer measuring the intended N x M regime"
        )

    median_ms, min_ms = time_operation(lambda: [t.particles() for t in updated], config)
    record("track.particles() copy", median_ms, min_ms, f"all {num_tracks} tracks")

    median_ms, min_ms = time_operation(lambda: [t.particle_states() for t in updated], config)
    record("track.particle_states() view", median_ms, min_ms, f"all {num_tracks} tracks")

    median_ms, min_ms = time_operation(lambda: [run_once.compute_track_mean(t) for t in updated], config)
    record("compute_track_mean (Python)", median_ms, min_ms, f"all {num_tracks} tracks")

    median_ms, min_ms = time_operation(lambda: tracker.get_tracks(), config)
    record("get_tracks()", median_ms, min_ms, f"{num_tracks} tracks across the binding")

    cost_matrix = build_cost_matrix(tracker, updated, measurements)
    median_ms, min_ms = time_operation(lambda: lmb.solve_assignment(cost_matrix, config.k_best), config)
    hypotheses = lmb.solve_assignment(cost_matrix, config.k_best)
    record(
        "solve_assignment",
        median_ms,
        min_ms,
        f"{cost_matrix.shape[0]}x{cost_matrix.shape[1]}, {len(hypotheses)} hypotheses returned",
    )

    median_ms, min_ms = time_operation(
        lambda: lmb.calculate_gospa_distance(updated, truth_states, hs.GOSPA_CUTOFF), config
    )
    record("calculate_gospa_distance", median_ms, min_ms, "")

    median_ms, min_ms = time_operation(
        lambda: lmb.calculate_gospa_components(updated, truth_states, hs.GOSPA_CUTOFF), config
    )
    record("calculate_gospa_components", median_ms, min_ms, "same work plus the decomposition")

    return {
        "config": {
            "num_particles": config.num_particles,
            "k_best": config.k_best,
            "repeats": config.repeats,
            "warmups": config.warmups,
            "num_tracks": num_tracks,
            "num_measurements": len(measurements),
            "hypotheses_returned": len(hypotheses),
        },
        "calibration_min_ms": calibration_min,
        "timings_ms": results,
        "peak_rss_mb": peak_rss_mb(),
    }


def print_report(report: dict, baseline: dict | None) -> None:
    config = report["config"]
    print(
        f"engine benchmark  particles={config['num_particles']} k_best={config['k_best']} "
        f"tracks={config['num_tracks']} measurements={config['num_measurements']} "
        f"hypotheses={config['hypotheses_returned']} repeats={config['repeats']}"
    )

    baseline_timings = (baseline or {}).get("timings_ms", {})
    drift = 0.0
    if baseline and baseline.get("calibration_min_ms", 0.0) > 0.0:
        drift = report["calibration_min_ms"] / baseline["calibration_min_ms"] - 1.0

    name_width = max(len(name) for name in report["timings_ms"])
    header = f"  {'operation':<{name_width}}  {'min':>10}  {'median':>10}"
    if baseline:
        header += f"  {'baseline':>10}  {'change':>8}"
    print(header)

    for name, entry in report["timings_ms"].items():
        line = f"  {name:<{name_width}}  {entry['min_ms']:9.3f}m  {entry['median_ms']:9.3f}m"
        if baseline:
            previous = baseline_timings.get(name)
            if previous is None or previous["min_ms"] <= 0.0:
                line += f"  {'-':>10}  {'new':>8}"
            else:
                before = previous["min_ms"]
                line += f"  {before:9.3f}m  {(entry['min_ms'] - before) / before * 100.0:+7.1f}%"
        if entry["note"]:
            line += f"   [{entry['note']}]"
        print(line)

    if baseline:
        print(f"  reference workload moved {drift * 100.0:+.1f}% since the baseline was recorded")
        if abs(drift) > CALIBRATION_DRIFT_WARN:
            print(
                f"  WARNING: the machine is running {drift * 100.0:+.1f}% differently from when the "
                f"baseline was recorded, which is more than the {CALIBRATION_DRIFT_WARN * 100.0:.0f}% "
                "trust threshold. Re-run on an idle machine before drawing a conclusion."
            )

    rss_line = f"  peak RSS {report['peak_rss_mb']:.1f} MiB"
    if baseline and "peak_rss_mb" in baseline:
        rss_line += (
            f"  (baseline {baseline['peak_rss_mb']:.1f} MiB, "
            f"{report['peak_rss_mb'] - baseline['peak_rss_mb']:+.1f} MiB)"
        )
    print(rss_line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="use the small sizing instead of production")
    parser.add_argument("--json", type=Path, default=None, help="write the results to a JSON file")
    parser.add_argument("--compare", type=Path, default=None, help="diff against a previously written JSON file")
    args = parser.parse_args()

    config = QUICK if args.quick else PRODUCTION
    report = run_benchmarks(config)

    baseline = None
    if args.compare is not None:
        if not args.compare.exists():
            raise SystemExit(f"no such baseline file: {args.compare}")
        baseline = json.loads(args.compare.read_text())

    print_report(report, baseline)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
