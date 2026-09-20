"""Phase 8 gate: parallel Monte Carlo matches serial and is stable across worker counts."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
PYTHON_DIR = TESTS_DIR.parent / "python"
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(TESTS_DIR))

# Workers spawned by ProcessPoolExecutor need the same import path.
_existing = os.environ.get("PYTHONPATH", "")
if str(PYTHON_DIR) not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = str(PYTHON_DIR) + (os.pathsep + _existing if _existing else "")

import simulation_common as sc  # noqa: E402


MASTER_SEED = 20260908
NUM_RUNS = 4
# Speedup measurement uses a slightly larger batch so process startup is amortized.
SPEEDUP_RUNS = 8
SPEEDUP_WORKERS = 4


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def main() -> int:
    checker = Checker()

    seeds = sc.derive_run_seeds(MASTER_SEED, NUM_RUNS)
    checker.ok(len(seeds) == NUM_RUNS, "derive_run_seeds length")
    checker.ok(len(set(seeds)) == NUM_RUNS, "derived seeds must be distinct")
    checker.ok(
        sc.derive_run_seeds(MASTER_SEED, NUM_RUNS) == seeds,
        "derive_run_seeds must be deterministic",
    )

    print(f"serial Monte Carlo  runs={NUM_RUNS} master_seed={MASTER_SEED}")
    t0 = time.perf_counter()
    serial_ospa, serial_errors, serial_seeds = sc.run_monte_carlo(
        NUM_RUNS, master_seed=MASTER_SEED, max_workers=1
    )
    serial_s = time.perf_counter() - t0
    checker.ok(serial_seeds == seeds, "serial run used the expected derived seeds")
    checker.ok(serial_ospa.shape == (NUM_RUNS, sc.NUM_STEPS), f"serial OSPA shape {serial_ospa.shape}")
    checker.ok(serial_errors.shape == (sc.NUM_STEPS, 6), f"serial errors shape {serial_errors.shape}")

    print(f"parallel Monte Carlo  runs={NUM_RUNS} workers=2")
    t0 = time.perf_counter()
    parallel_ospa, parallel_errors, parallel_seeds = sc.run_monte_carlo(
        NUM_RUNS, master_seed=MASTER_SEED, max_workers=2
    )
    parallel_2_s = time.perf_counter() - t0
    checker.ok(parallel_seeds == seeds, "parallel(2) run used the expected derived seeds")
    checker.ok(
        np.array_equal(serial_ospa, parallel_ospa),
        "parallel(2) OSPA is not bit-identical to serial",
    )
    checker.ok(
        np.array_equal(serial_errors, parallel_errors),
        "parallel(2) representative_errors is not bit-identical to serial run 0",
    )

    print(f"parallel Monte Carlo  runs={NUM_RUNS} workers=4")
    parallel4_ospa, parallel4_errors, parallel4_seeds = sc.run_monte_carlo(
        NUM_RUNS, master_seed=MASTER_SEED, max_workers=4
    )
    checker.ok(parallel4_seeds == seeds, "parallel(4) run used the expected derived seeds")
    checker.ok(
        np.array_equal(serial_ospa, parallel4_ospa),
        "parallel(4) OSPA is not bit-identical to serial",
    )
    checker.ok(
        np.array_equal(serial_errors, parallel4_errors),
        "parallel(4) representative_errors is not bit-identical to serial run 0",
    )
    checker.ok(
        np.array_equal(parallel_ospa, parallel4_ospa),
        "parallel output differs across worker counts",
    )

    # Confirm representative_errors matches a direct run-0 simulation under the same seed.
    direct_ospa, direct_errors = sc.run_single_simulation(
        verbose=False, collect_track_errors=True, seed=seeds[0]
    )
    checker.ok(
        np.array_equal(serial_ospa[0], np.asarray(direct_ospa, dtype=np.float64)),
        "run 0 OSPA does not match a direct seeded simulation",
    )
    checker.ok(
        np.array_equal(serial_errors, np.asarray(direct_errors, dtype=np.float64)),
        "representative_errors does not come from run 0",
    )

    print(
        f"speedup Monte Carlo  runs={SPEEDUP_RUNS} serial vs workers={SPEEDUP_WORKERS} "
        f"(master_seed={MASTER_SEED + 1})"
    )
    t0 = time.perf_counter()
    sc.run_monte_carlo(SPEEDUP_RUNS, master_seed=MASTER_SEED + 1, max_workers=1)
    speed_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    sc.run_monte_carlo(SPEEDUP_RUNS, master_seed=MASTER_SEED + 1, max_workers=SPEEDUP_WORKERS)
    speed_parallel = time.perf_counter() - t0
    speedup = speed_serial / speed_parallel if speed_parallel > 0 else 0.0
    # Account for spawn/startup overhead: require a clear win, not ideal linear scaling.
    checker.ok(
        speedup >= 1.5,
        f"expected clear parallel speedup (>=1.5x), got {speedup:.2f}x "
        f"(serial {speed_serial:.1f}s, parallel/{SPEEDUP_WORKERS} {speed_parallel:.1f}s)",
    )

    print(
        f"  gate timing: serial({NUM_RUNS})={serial_s:.1f}s  "
        f"parallel/2={parallel_2_s:.1f}s  "
        f"speedup batch {SPEEDUP_RUNS}: {speed_serial:.1f}s -> {speed_parallel:.1f}s "
        f"({speedup:.2f}x with {SPEEDUP_WORKERS} workers)"
    )
    print(f"PASS: test_monte_carlo_parallel ({checker.count} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
