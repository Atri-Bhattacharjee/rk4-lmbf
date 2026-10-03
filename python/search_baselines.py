"""Search baselines on the filter-free environment (python/search_env.py): the two ends of the scale.

  oracle   knows where every unseen object is and points each sensor at whatever new object is in
           its bubble. No policy without that knowledge can beat it; it falls short of sensors that
           see in every direction only when two new objects need one sensor in the same slot.
  random   a fresh random direction bin for every sensor in every slot.

Both are run for each field-of-view half-angle on the same scenarios, against the number of objects
that ever enter a bubble (what sensors seeing in every direction would find).

    python python/search_baselines.py                          # 30 orbits, 45/20/10/5 deg, 5 seeds
    python python/search_baselines.py --orbits 2 --fov 45 --seeds 1 --detection sample
    python python/search_baselines.py --custody                # also: what custody costs the oracle

Writes python/results/search_baselines/summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from run_ring import RESULTS_DIR, derive_seeds
from search_env import DETECTION_MODELS, SearchConfig, SearchEnv, build_pass_table, make_scenario


class OraclePolicy:
    """Perfect knowledge: each sensor takes the bin that catches the most new objects in its
    bubble this slot, and otherwise stays where it was."""

    def __init__(self, num_sensors: int) -> None:
        self.bins = np.zeros(num_sensors, dtype=np.int64)

    def __call__(self, env: SearchEnv) -> np.ndarray:
        objects, sensors, hits = env.bubble_contents()
        new = ~env.seen[objects]
        for sensor in np.unique(sensors[new]):
            rows = new & (sensors == sensor)
            # One row per new object: the bins that would catch it at any step of the slot.
            _, index = np.unique(objects[rows], return_inverse=True)
            caught = np.zeros((index.max() + 1, env.num_bins), dtype=bool)
            np.logical_or.at(caught, index, hits[rows])
            self.bins[sensor] = np.argmax(caught.sum(axis=0))
        return self.bins


def random_schedule(env: SearchEnv, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random bin for every sensor in every slot."""
    return rng.integers(0, env.num_bins, size=(env.num_slots, env.num_sensors))


def run_baselines(config: SearchConfig, fov_half_angles, seeds, detection: str = "streak",
                  slot: float = 10.0, random_draws: int = 5, custody: bool = False,
                  verbose: bool = True) -> list[dict]:
    """One record per (seed, field of view)."""
    orbit_ends = np.arange(1, int(np.ceil(config.num_orbits)) + 1) * config.orbit_period
    records = []
    for seed in seeds:
        tick = time.perf_counter()
        table = build_pass_table(make_scenario(config, seed).states, config)
        build_seconds = time.perf_counter() - tick
        for half_angle in fov_half_angles:
            tick = time.perf_counter()
            env = SearchEnv(table, fov_half_angle_deg=half_angle, detection=detection, slot=slot)
            oracle = env.run(OraclePolicy(env.num_sensors))
            rng = np.random.default_rng(derive_seeds(seed)["measurement"])
            draws = [env.run_schedule(random_schedule(env, rng)) for _ in range(random_draws)]
            record = {
                "seed": int(seed), "fov_half_angle_deg": float(half_angle), "bins": env.num_bins,
                "passes": table.num_passes, "bound": oracle.bound,
                "oracle_found": oracle.found,
                "random_found": float(np.mean([draw.found for draw in draws])),
                "oracle_mean_time_to_discovery": oracle.mean_time_to_discovery,
                "random_mean_time_to_discovery": float(np.mean(
                    [draw.mean_time_to_discovery for draw in draws])),
                "oracle_found_by_orbit": oracle.found_by(orbit_ends).tolist(),
                "random_found_by_orbit": np.mean(
                    [draw.found_by(orbit_ends) for draw in draws], axis=0).tolist(),
            }
            if custody:
                held = SearchEnv(table, fov_half_angle_deg=half_angle, detection=detection,
                                 slot=slot, custody=True)
                with_custody = held.run(OraclePolicy(held.num_sensors))
                record["oracle_found_with_custody"] = with_custody.found
                record["custody_fraction"] = with_custody.custody_fraction
            record["seconds"] = build_seconds + time.perf_counter() - tick
            records.append(record)
            if verbose:
                print(f"  seed {seed}  fov {half_angle:4.1f}  bound {record['bound']:4d}  "
                      f"oracle {record['oracle_found']:4d}  random {record['random_found']:6.1f}  "
                      f"({record['seconds']:.1f} s)", flush=True)
    return records


def summarize(records: list[dict]) -> list[dict]:
    """Means over seeds, one row per field of view."""
    rows = []
    for half_angle in sorted({r["fov_half_angle_deg"] for r in records}, reverse=True):
        group = [r for r in records if r["fov_half_angle_deg"] == half_angle]
        mean = {key: float(np.mean([r[key] for r in group]))
                for key in ("bound", "oracle_found", "random_found", "passes",
                            "oracle_mean_time_to_discovery", "random_mean_time_to_discovery")}
        row = {"fov_half_angle_deg": half_angle, "bins": group[0]["bins"], "seeds": len(group), **mean,
               "oracle_of_bound": mean["oracle_found"] / mean["bound"],
               "random_of_bound": mean["random_found"] / mean["bound"],
               "random_of_oracle": mean["random_found"] / mean["oracle_found"],
               "random_found_std": float(np.std([r["random_found"] for r in group]))}
        if "custody_fraction" in group[0]:
            row["oracle_found_with_custody"] = float(np.mean([r["oracle_found_with_custody"] for r in group]))
            row["custody_fraction"] = float(np.mean([r["custody_fraction"] for r in group]))
        rows.append(row)
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--orbits", type=float, default=30.0)
    parser.add_argument("--objects", type=int, default=SearchConfig.num_objects)
    parser.add_argument("--sensors", type=int, default=SearchConfig.num_sensors)
    parser.add_argument("--fov", type=float, nargs="+", default=[45.0, 20.0, 10.0, 5.0],
                        help="field-of-view half-angles [deg]")
    parser.add_argument("--seeds", type=int, default=5, help="number of scenarios")
    parser.add_argument("--seed", type=int, default=SearchConfig.seed, help="first scenario seed")
    parser.add_argument("--detection", choices=DETECTION_MODELS, default="streak")
    parser.add_argument("--slot", type=float, default=10.0, help="seconds a pointing is held")
    parser.add_argument("--random-draws", type=int, default=5,
                        help="random schedules averaged per scenario")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--rotate", action="store_true",
                        help="give every object a random node longitude")
    parser.add_argument("--custody", action="store_true",
                        help="also run the oracle with sensors reserved for already-seen objects")
    parser.add_argument("--output", default=str(RESULTS_DIR / "search_baselines"))
    args = parser.parse_args(argv)

    config = SearchConfig(num_orbits=args.orbits, num_objects=args.objects, num_sensors=args.sensors,
                          seed=args.seed, split=args.split, rotate=args.rotate)
    seeds = [args.seed + k for k in range(args.seeds)]
    print("=" * 78)
    print("Search baselines (no filter): oracle and random pointing")
    print("=" * 78)
    print(f"  sensors {config.num_sensors}, range {config.sensor_range / 1e3:.0f} km | objects "
          f"{config.num_objects} ({config.split}{', rotated' if config.rotate else ''}) | "
          f"{config.num_orbits:g} orbits")
    print(f"  detection {args.detection} | pointing held {args.slot:g} s | seeds {seeds[0]}..{seeds[-1]}")
    print("-" * 78)
    records = run_baselines(config, args.fov, seeds, detection=args.detection, slot=args.slot,
                            random_draws=args.random_draws, custody=args.custody)
    rows = summarize(records)

    print("-" * 78)
    print("  bound = objects that ever enter a bubble (sensors seeing in every direction)")
    print(f"  {'fov':>5} {'bins':>5} {'bound':>7} {'oracle':>7} {'of bound':>9} {'random':>7} "
          f"{'of bound':>9} {'of oracle':>10}")
    for row in rows:
        print(f"  {row['fov_half_angle_deg']:5.1f} {row['bins']:5d} {row['bound']:7.1f} "
              f"{row['oracle_found']:7.1f} {row['oracle_of_bound']:9.1%} {row['random_found']:7.1f} "
              f"{row['random_of_bound']:9.1%} {row['random_of_oracle']:10.1%}")
    if args.custody:
        print("  with custody (a sensor with a known object in its bubble does no search that slot):")
        for row in rows:
            print(f"  {row['fov_half_angle_deg']:5.1f}  oracle {row['oracle_found_with_custody']:7.1f}  "
                  f"sensor-slots on custody {row['custody_fraction']:.3%}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(
        {"config": asdict(config), "detection": args.detection, "slot": args.slot,
         "random_draws": args.random_draws, "summary": rows, "records": records}, indent=2))
    print(f"  wrote {output / 'summary.json'}")


if __name__ == "__main__":
    main()
