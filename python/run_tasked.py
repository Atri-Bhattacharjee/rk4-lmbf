"""Pointing policies compared in the full filter: python/run_ring.py with pointed sensors.

Every run is the ring scenario (same objects, same filter) with a field of view of the given
half-angle and one policy aiming the sensors (python/tasking.py):

  all-seeing       the sensors see everything in range: the reference, no pointing
  oracle           every sensor looks straight at whatever is in its bubble
  random           a random direction bin per sensor per slot, nothing else
  random+custody   random search, plus custody of known tracks
  sgd              the search schedule tuned by python/search_sgd.py, nothing else
  sgd+custody      the tuned schedule plus custody: the proposed tasker

All policies are scored against the same objects: the ones that enter a sensor's bubble at some
point in the run, whether or not the policy found them.

    python python/search_sgd.py --detection sample          # once: writes the tuned schedules
    python python/run_tasked.py                             # every policy, 45/20/10/5 deg, 3 seeds
    python python/run_tasked.py --policy oracle --fov 45 --seeds 1
    python python/run_tasked.py --policy sgd+custody --fov 20 --orbits 5 --workers 1

Writes python/results/tasked/: one <policy>_fov<deg>_seed<seed>.npz per run, summary.json, and the
two comparison figures of python/evaluation_plots.py with their CSV twins.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

POLICIES = ("all-seeing", "oracle", "random", "random+custody", "sgd", "sgd+custody")


def make_tasker(policy: str, config, seed: int, schedules: Path):
    from search_sgd import SearchPolicy, schedule_path
    from tasking import CustodyTasker, OracleTasker, ScheduleTasker

    if policy == "all-seeing":
        return None
    if policy == "oracle":
        return OracleTasker(config)
    search, _, custody = policy.partition("+")
    draw = int(np.random.SeedSequence([seed, 17]).generate_state(1)[0])
    if search == "random":
        tasker = ScheduleTasker.random(config, draw)
    elif search == "sgd":
        path = schedule_path(schedules, config.fov_half_angle_deg, "sample", config.dt)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run python/search_sgd.py --detection sample "
                                    f"--fov {config.fov_half_angle_deg:g} --dt {config.dt:g} first")
        tasker = ScheduleTasker.from_policy(config, SearchPolicy.load(path), draw)
    else:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
    return CustodyTasker(config, tasker) if custody else tasker


def pass_statistics(table, detection_events: np.ndarray, dt: float, pass_gap: float) -> dict:
    """What happened at each pass of an object through a bubble, from the truth's pass table and the
    run's detections: was the object already known, and how many of its samples were detected."""
    num_steps = table.config.num_steps
    order = np.lexsort((table.sample_step, table.sample_object))
    obj, sensor, step = table.sample_object[order], table.sample_sensor[order], table.sample_step[order]
    new = np.ones(len(obj), dtype=bool)
    new[1:] = (obj[1:] != obj[:-1]) | (sensor[1:] != sensor[:-1]) | ((step[1:] - step[:-1]) * dt > pass_gap)
    pass_of = np.cumsum(new) - 1
    num_passes = int(pass_of[-1]) + 1 if len(pass_of) else 0

    event_step = np.rint(detection_events[:, 0] / dt).astype(np.int64)
    event_object = detection_events[:, 1].astype(np.int64)
    detected = np.isin(obj * (num_steps + 1) + step, event_object * (num_steps + 1) + event_step)
    first = np.full(table.num_objects, np.iinfo(np.int64).max)
    np.minimum.at(first, event_object, event_step)

    samples = np.bincount(pass_of, minlength=num_passes)
    seen = np.bincount(pass_of, weights=detected, minlength=num_passes)
    start = step[new]
    end = step[np.append(np.flatnonzero(new)[1:] - 1, len(step) - 1)] if num_passes else start
    pass_object = obj[new]
    known = first[pass_object] < start
    found_here = (first[pass_object] >= start) & (first[pass_object] <= end)

    def mean(values) -> float:
        return float(np.mean(values)) if len(values) else float("nan")

    return {
        "passes": num_passes,
        "known_passes": int(known.sum()),
        "known_passes_detected": int(np.sum(known & (seen > 0))),
        "known_pass_detection_rate": mean(seen[known] > 0),
        "known_pass_samples_detected": mean(seen[known & (seen > 0)]),
        "known_pass_sample_share": float(seen[known].sum() / max(samples[known].sum(), 1)),
        "first_pass_samples_detected": mean(seen[found_here]),
        "unseen_passes_missed": int(np.sum(~known & ~found_here)),
    }


def run_one(payload: dict) -> dict:
    """One run. Top-level so a process pool can pickle it."""
    import run_ring
    from search_env import build_pass_table, make_scenario
    from tasking import search_config_for

    policy, half_angle, seed = payload["policy"], payload["fov"], payload["seed"]
    config = run_ring.RingConfig(
        num_orbits=payload["orbits"], seed=seed, dt=payload["dt"], num_particles=payload["particles"],
        fov_half_angle_deg=None if policy == "all-seeing" else half_angle,
        metric_interval=payload["metric_interval"], profile=False, gate_audit=payload["gate_audit"])
    search_config = search_config_for(config)
    table = build_pass_table(make_scenario(search_config).states, search_config)
    findable = np.flatnonzero(table.visible_objects("sample"))
    tasker = make_tasker(policy, config, seed, Path(payload["schedules"]))

    tick = time.perf_counter()
    log = run_ring.run(config, verbose=False, tasker=tasker, scored_objects=findable)
    wall = time.perf_counter() - tick

    events = log["detection_events"]
    first = np.full(config.num_objects, np.nan)
    if len(events):
        earliest = np.full(config.num_objects, np.inf)
        np.minimum.at(earliest, events[:, 1].astype(np.int64), events[:, 0])
        first = np.where(np.isfinite(earliest), earliest, np.nan)
    tracking = np.divide(log["num_assigned"], log["num_truths"], out=np.zeros_like(log["num_assigned"]),
                         where=log["num_truths"] > 0)
    rows = np.asarray(log["object_rows"]).reshape(-1, 6)
    final = rows[rows[:, 0] == rows[:, 0].max()] if len(rows) else rows
    matched = final[np.isfinite(final[:, 3]), 3]
    sensor_steps = (config.num_sensors * (search_config.num_steps + 1))
    summary = {
        "policy": policy, "fov_half_angle_deg": None if policy == "all-seeing" else float(half_angle),
        "seed": int(seed), "orbits": float(payload["orbits"]), "dt": float(payload["dt"]),
        "findable": int(len(findable)),
        "objects_found": int(np.sum(~np.isnan(first))),
        "found_share": float(np.sum(~np.isnan(first)) / max(len(findable), 1)),
        "detections": int(len(events)),
        "births": int(len(log["births"])),
        "final_tracks": int(log["num_tracks"][-1]),
        "final_tracked": int(log["num_assigned"][-1]),
        "final_tracking_fraction": float(tracking[-1]),
        "mean_tracking_fraction": float(tracking.mean()),
        "final_median_error_m": float(np.median(matched)) if len(matched) else float("nan"),
        "final_gospa": float(log["gospa"][-1]),
        "final_localisation": float(log["localisation"][-1]),
        "final_missed": float(log["missed"][-1]),
        "final_false": float(log["false_positive"][-1]),
        "custody_share": float(tasker.counts["custody"] / sensor_steps) if tasker is not None else float("nan"),
        "gate_audit": list(log.get("gate_audit") or []),
        "pass_outcomes": log["pass_outcomes"],
        "wall_seconds": wall, "timers": {k: round(v, 2) for k, v in log["timers"].items()},
        **pass_statistics(table, events, config.dt, run_ring.PASS_GAP),
    }
    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{run_name(policy, half_angle, seed)}.npz", summary=json.dumps(summary),
                        first_detection=first, findable=findable, time=log["time"], tracking=tracking,
                        gospa=log["gospa"], localisation=log["localisation"], missed=log["missed"],
                        false_positive=log["false_positive"], num_tracks=log["num_tracks"])
    return summary


def run_name(policy: str, half_angle, seed: int) -> str:
    return f"{policy}_seed{seed}" if policy == "all-seeing" else f"{policy}_fov{half_angle:g}_seed{seed}"


def summarize(records: list[dict]) -> list[dict]:
    """Means over seeds, one row per (field of view, policy)."""
    keys = ("findable", "objects_found", "found_share", "detections", "final_tracks", "final_tracked",
            "final_tracking_fraction", "mean_tracking_fraction", "final_median_error_m",
            "known_pass_detection_rate", "known_pass_samples_detected", "first_pass_samples_detected",
            "custody_share", "wall_seconds")
    rows = []
    groups = sorted({(r["fov_half_angle_deg"] or 0.0, POLICIES.index(r["policy"])) for r in records},
                    key=lambda g: (g[0] != 0.0, -g[0], g[1]))
    for half_angle, index in groups:
        group = [r for r in records
                 if (r["fov_half_angle_deg"] or 0.0) == half_angle and r["policy"] == POLICIES[index]]
        row = {"policy": POLICIES[index], "fov_half_angle_deg": half_angle or None, "seeds": len(group)}
        for key in keys:
            values = np.array([r[key] for r in group], dtype=np.float64)
            row[key] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"  {'fov':>5} {'policy':<15} {'found':>7} {'of findable':>12} {'tracked':>8} {'tracking':>9} "
          f"{'known passes seen':>18} {'samples/pass':>13} {'custody':>8} {'wall':>6}")
    for row in rows:
        angle = "  all" if row["fov_half_angle_deg"] is None else f"{row['fov_half_angle_deg']:5.1f}"
        print(f"  {angle} {row['policy']:<15} {row['objects_found']:7.1f} {row['found_share']:12.1%} "
              f"{row['final_tracked']:8.1f} {row['final_tracking_fraction']:9.3f} "
              f"{row['known_pass_detection_rate']:18.1%} {row['known_pass_samples_detected']:13.2f} "
              f"{row['custody_share']:8.2%} {row['wall_seconds']:5.0f}s")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--fov", type=float, nargs="+", default=[45.0, 20.0, 10.0, 5.0])
    parser.add_argument("--orbits", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, default=3, help="number of scenarios")
    parser.add_argument("--seed", type=int, default=20260930, help="first scenario seed")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--gate-audit", action="store_true", help="check every lazy-propagation gate decision (slow)")
    parser.add_argument("--schedules", default=None, help="folder holding search_sgd.py's schedules")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    from run_ring import RESULTS_DIR
    output = Path(args.output) if args.output else RESULTS_DIR / "tasked"
    schedules = Path(args.schedules) if args.schedules else RESULTS_DIR / "search_sgd"
    base = {"orbits": args.orbits, "dt": args.dt, "particles": args.particles, "gate_audit": args.gate_audit,
            "metric_interval": 600.0 if args.orbits >= 10 else 60.0, "schedules": str(schedules),
            "output": str(output)}
    payloads = []
    for k in range(args.seeds):
        for policy in args.policy:
            for half_angle in ([None] if policy == "all-seeing" else args.fov):
                payloads.append({**base, "policy": policy, "fov": half_angle, "seed": args.seed + k})

    print("=" * 110)
    print("Pointing policies in the full filter")
    print("=" * 110)
    print(f"  {args.orbits:g} orbits, dt {args.dt:g} s, {args.particles} particles | {len(payloads)} runs on "
          f"{args.workers} process(es) | seeds {args.seed}..{args.seed + args.seeds - 1}")
    print("-" * 110)
    tick = time.perf_counter()
    records = []

    def report(summary: dict) -> None:
        records.append(summary)
        print(f"  [{len(records):3d}/{len(payloads)}] {run_name(summary['policy'], summary['fov_half_angle_deg'], summary['seed']):<34} "
              f"found {summary['objects_found']:4d} of {summary['findable']}  tracking {summary['final_tracking_fraction']:.3f}  "
              f"{summary['wall_seconds']:5.0f} s", flush=True)

    if args.workers == 1:
        for payload in payloads:
            report(run_one(payload))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # One core per run: the engine is single-threaded and NumPy's own threads would only fight.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(run_one, payload) for payload in payloads]):
                report(future.result())

    rows = summarize(records)
    print("-" * 110)
    print_table(rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"args": vars(args), "summary": rows, "records": records},
                                                    indent=2, default=str))
    print(f"  {time.perf_counter() - tick:.0f} s in all | wrote {output / 'summary.json'}")
    if any(row["fov_half_angle_deg"] is not None for row in rows):
        try:
            from evaluation_plots import make_tasking_plots
        except ImportError as error:  # matplotlib missing: keep the results, skip the figures
            print(f"  figures skipped ({error})")
        else:
            for path in make_tasking_plots(output):
                print(f"  wrote {path}")


if __name__ == "__main__":
    main()
