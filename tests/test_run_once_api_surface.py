"""API-surface and GOSPA gate for the run_once / simulation_common consolidation.

Keeps the names that tests/test_end_to_end.py (and other harnesses) import from run_once stable.
On the reference platform (x86-64 Linux / libstdc++) a fixed-seed GOSPA array is checked bit-identical
against a committed capture. Off that platform the C++ RNG stream differs (libc++ on macOS), so the
gate is shape + self-consistency only -- the same split as test_golden_invariance.py.
"""

from __future__ import annotations

import os
import platform
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
    "GOSPA_CUTOFF",
    "get_ground_truth_propagator",
    "propagate_truth_state",
    "generate_measurements",
    "compute_track_mean",
    "lmb_engine",
    "run_single_simulation",
)

# Seeded capture on the reference platform only (see PLATFORM GATING in test_golden_invariance.py).
EXPECTED_GOSPA_SEED = 20260908
EXPECTED_GOSPA_MEAN = 2581.2268826891814
EXPECTED_GOSPA_FINAL = 3459.342975801599
EXPECTED_GOSPA_HEAD16 = bytes.fromhex("b4c0fad989563040b64bd246ccda8640")
# A fully lost run -- every track beyond the cutoff from every truth -- reports
# c * sqrt((m + n) / 2), which for the scenario's objects tracked one-for-one is c * sqrt(k).
# Gate on a fraction of that rather than a fraction of the cutoff: unnormalised GOSPA is not
# bounded by c, so "below 0.9 * c" is not a statement about it. See the same restatement in
# test_golden_invariance.py and tests/harness_scenario.py::gospa_saturation.
PORTABLE_SATURATION_CEILING = 0.9
LOST_RUN_GOSPA = run_once.GOSPA_CUTOFF * np.sqrt(len(run_once.SCENARIO))

REFERENCE_PLATFORM = "linux"
REFERENCE_MACHINES = ("x86_64", "amd64")


def is_reference_platform() -> bool:
    return sys.platform.startswith(REFERENCE_PLATFORM) and platform.machine().lower() in REFERENCE_MACHINES


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def main() -> int:
    checker = Checker()
    reference = is_reference_platform()

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

    gospa, _ = run_once.run_single_simulation(
        verbose=False, collect_track_errors=False, seed=EXPECTED_GOSPA_SEED
    )
    gospa = np.asarray(gospa, dtype=np.float64)
    checker.ok(gospa.shape == (run_once.NUM_STEPS,), f"GOSPA length {gospa.shape}")
    checker.ok(bool(np.isfinite(gospa).all()), "GOSPA contains non-finite values")
    checker.ok(bool((gospa >= 0.0).all()), "GOSPA contains negative values")

    if reference:
        checker.ok(
            gospa.tobytes()[:16] == EXPECTED_GOSPA_HEAD16,
            "GOSPA head bytes differ from the consolidation capture (not bit-identical)",
        )
        checker.ok(
            float(gospa.mean()) == EXPECTED_GOSPA_MEAN and float(gospa[-1]) == EXPECTED_GOSPA_FINAL,
            f"GOSPA mean/final drifted: mean={gospa.mean()!r} final={gospa[-1]!r}",
        )
    else:
        saturation = float(gospa.mean()) / LOST_RUN_GOSPA
        checker.ok(
            saturation < PORTABLE_SATURATION_CEILING,
            f"mean GOSPA {gospa.mean():.1f} m is {saturation:.3f} of the fully-lost value "
            f"{LOST_RUN_GOSPA:.1f} m; looks like a lost-track run on this platform",
        )
        print(
            f"  portable GOSPA gate on {sys.platform}/{platform.machine()}: "
            f"mean={gospa.mean():.1f} m final={gospa[-1]:.1f} m "
            f"saturation={saturation:.3f} (absolute bytes are Linux-only)"
        )

    gospa2, _ = run_once.run_single_simulation(
        verbose=False, collect_track_errors=False, seed=EXPECTED_GOSPA_SEED
    )
    gospa2 = np.asarray(gospa2, dtype=np.float64)
    if reference:
        checker.ok(np.array_equal(gospa, gospa2), "two fixed-seed runs are not bit-identical to each other")
    else:
        checker.ok(
            np.allclose(gospa, gospa2, rtol=1e-12, atol=0.0),
            "two fixed-seed runs disagree beyond rtol 1e-12 on this platform",
        )

    gospa3, _ = run_once.run_single_simulation(
        verbose=False, collect_track_errors=True, seed=EXPECTED_GOSPA_SEED
    )
    gospa3 = np.asarray(gospa3, dtype=np.float64)
    if reference:
        checker.ok(np.array_equal(gospa, gospa3), "collect_track_errors must not change the GOSPA array")
    else:
        checker.ok(
            np.allclose(gospa, gospa3, rtol=1e-12, atol=0.0),
            "collect_track_errors changed GOSPA beyond rtol 1e-12",
        )

    print(f"PASS: test_run_once_api_surface ({checker.count} assertions)")
    return 0


def capture() -> int:
    """Print the three pinned constants for this build, to paste over the block above.

    Every other fixture in this repo regenerates through a --write flag; this one used to need a
    hand-written heredoc, so it is here for the next person.
    """
    gospa, _ = run_once.run_single_simulation(
        verbose=False, collect_track_errors=False, seed=EXPECTED_GOSPA_SEED
    )
    gospa = np.asarray(gospa, dtype=np.float64)
    print(f"EXPECTED_GOSPA_MEAN = {float(gospa.mean())!r}")
    print(f"EXPECTED_GOSPA_FINAL = {float(gospa[-1])!r}")
    print(f'EXPECTED_GOSPA_HEAD16 = bytes.fromhex("{gospa.tobytes()[:16].hex()}")')
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true",
                        help="print the pinned GOSPA constants for this build instead of testing")
    args = parser.parse_args()
    raise SystemExit(capture() if args.capture else main())
