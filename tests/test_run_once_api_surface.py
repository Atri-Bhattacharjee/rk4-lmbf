"""API-surface and bitwise-OSPA gate for the run_once / simulation_common consolidation.

Keeps the names that tests/test_end_to_end.py (and other harnesses) import from run_once stable,
and checks that a fixed-seed single run is bit-identical after the code motion.
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

REQUIRED_NAMES = (
    "SENSOR_STATE",
    "SCENARIO",
    "DT",
    "FILTER_SIGMAS",
    "Q_FILTER",
    "P_BIRTH",
    "BIRTH_COVARIANCE_LOCAL",
    "K_BEST",
    "PRUNE_THRESHOLD",
    "CLUTTER_INTENSITY",
    "P_DETECTION",
    "NOISE_DECAY_RATE",
    "NOISE_MIN_SCALE",
    "P_SURVIVAL",
    "get_ground_truth_propagator",
    "propagate_truth_state",
    "generate_measurements",
    "compute_track_mean",
    "lmb_engine",
    "run_single_simulation",
)

# Seeded end-to-end capture (seed pins NumPy + truth/filter/birth/resampler RNGs). Confirmed
# bit-identical to the pre-consolidation run_once.py under the same seed wiring.
EXPECTED_OSPA_SEED = 20260908
EXPECTED_OSPA_MEAN = 1679.1767480015253
EXPECTED_OSPA_FINAL = 1701.06238267292
EXPECTED_OSPA_HEAD16 = bytes.fromhex("ee1a14dc7eb53040e66bd240b3db8640")


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def main() -> int:
    checker = Checker()

    for name in REQUIRED_NAMES:
        checker.ok(hasattr(run_once, name), f"run_once is missing required name {name!r}")

    checker.ok(isinstance(run_once.DT, float), "DT must remain a float")
    checker.ok(run_once.DT == 60.0, f"DT changed: {run_once.DT}")
    checker.ok(run_once.K_BEST == 2, f"K_BEST changed: {run_once.K_BEST}")
    checker.ok(run_once.P_BIRTH == 0.9, f"P_BIRTH changed: {run_once.P_BIRTH}")
    checker.ok(run_once.PRUNE_THRESHOLD == 0.001, f"PRUNE_THRESHOLD changed: {run_once.PRUNE_THRESHOLD}")
    checker.ok(run_once.CLUTTER_INTENSITY == 1e-15, f"CLUTTER_INTENSITY changed: {run_once.CLUTTER_INTENSITY}")
    checker.ok(run_once.P_DETECTION == 0.999999999, f"P_DETECTION changed: {run_once.P_DETECTION}")
    checker.ok(run_once.P_SURVIVAL == 0.999999999, f"P_SURVIVAL changed: {run_once.P_SURVIVAL}")
    checker.ok(run_once.NOISE_DECAY_RATE == 0.001, f"NOISE_DECAY_RATE changed: {run_once.NOISE_DECAY_RATE}")
    checker.ok(run_once.NOISE_MIN_SCALE == 0.001, f"NOISE_MIN_SCALE changed: {run_once.NOISE_MIN_SCALE}")
    checker.ok(
        run_once.FILTER_SIGMAS.shape == (6,),
        f"FILTER_SIGMAS shape {run_once.FILTER_SIGMAS.shape}",
    )
    checker.ok(run_once.SENSOR_STATE.shape == (6,), "SENSOR_STATE shape")
    checker.ok(run_once.Q_FILTER.shape == (6, 6), "Q_FILTER shape")
    checker.ok(run_once.BIRTH_COVARIANCE_LOCAL.shape == (6, 6), "BIRTH_COVARIANCE_LOCAL shape")
    checker.ok(len(run_once.SCENARIO) == 3, "SCENARIO length")
    checker.ok(callable(run_once.get_ground_truth_propagator), "get_ground_truth_propagator")
    checker.ok(callable(run_once.propagate_truth_state), "propagate_truth_state")
    checker.ok(callable(run_once.generate_measurements), "generate_measurements")
    checker.ok(callable(run_once.compute_track_mean), "compute_track_mean")
    checker.ok(run_once.lmb_engine is not None, "lmb_engine")

    ospa, _ = run_once.run_single_simulation(
        verbose=False, collect_track_errors=False, seed=EXPECTED_OSPA_SEED
    )
    ospa = np.asarray(ospa, dtype=np.float64)
    checker.ok(ospa.shape == (run_once.NUM_STEPS,), f"OSPA length {ospa.shape}")
    checker.ok(
        ospa.tobytes()[:16] == EXPECTED_OSPA_HEAD16,
        "OSPA head bytes differ from the consolidation capture (not bit-identical)",
    )
    checker.ok(
        float(ospa.mean()) == EXPECTED_OSPA_MEAN and float(ospa[-1]) == EXPECTED_OSPA_FINAL,
        f"OSPA mean/final drifted: mean={ospa.mean()!r} final={ospa[-1]!r}",
    )
    ospa2, _ = run_once.run_single_simulation(
        verbose=False, collect_track_errors=False, seed=EXPECTED_OSPA_SEED
    )
    checker.ok(
        np.array_equal(ospa, np.asarray(ospa2, dtype=np.float64)),
        "two fixed-seed runs are not bit-identical to each other",
    )
    ospa3, _ = run_once.run_single_simulation(
        verbose=False, collect_track_errors=True, seed=EXPECTED_OSPA_SEED
    )
    checker.ok(
        np.array_equal(ospa, np.asarray(ospa3, dtype=np.float64)),
        "collect_track_errors must not change the OSPA array",
    )

    print(f"PASS: test_run_once_api_surface ({checker.count} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
