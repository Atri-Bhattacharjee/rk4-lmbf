"""Golden-digest invariance: the primary gate for every optimization phase.

Runs a fixed set of fully-seeded scenarios and compares the resulting digest against a committed
fixture. Because all five random streams are pinned (see ``harness_scenario``), a correct
refactor reproduces the fixture *bitwise*; only a phase that deliberately changes floating-point
summation order needs the relaxed ``--rtol`` mode.

Usage:
    python tests/test_golden_invariance.py                 # bitwise comparison (default)
    python tests/test_golden_invariance.py --rtol           # tolerant, rtol 1e-12
    python tests/test_golden_invariance.py --rtol 1e-9      # tolerant, explicit rtol
    python tests/test_golden_invariance.py --write          # regenerate the fixtures
    python tests/test_golden_invariance.py --no-selfcheck   # skip the run-to-run repeat

Integer digest fields (cardinality, measurement counts, track labels) are compared exactly in
every mode, so a phase that perturbs track identity or a pruning decision fails even under
``--rtol``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_scenario as hs  # noqa: E402

# Cases are chosen to span the regimes the phases touch: `short` mirrors the end-to-end test,
# `multihyp` guarantees several hypotheses share an association (what Phase 2 restructures), and
# `dense` approaches the production K_BEST=100 / 3-track / 3-measurement configuration.
CASES: dict[str, tuple[hs.ScenarioConfig, tuple[int, ...]]] = {
    "short": (hs.ScenarioConfig(num_steps=40, num_particles=200, k_best=2), (20260908, 20260909)),
    "multihyp": (hs.ScenarioConfig(num_steps=60, num_particles=500, k_best=16), (20260908, 20260909)),
    "dense": (hs.ScenarioConfig(num_steps=60, num_particles=1000, k_best=100), (20260908,)),
}

DEFAULT_RTOL = 1e-12


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def fixture_path(case_name: str, seed: int) -> Path:
    return hs.FIXTURE_DIR / f"golden_{case_name}_{seed}.npz"


def iter_cases():
    for case_name, (config, seeds) in CASES.items():
        for seed in seeds:
            yield case_name, config, seed


def write_fixtures() -> None:
    for case_name, config, seed in iter_cases():
        digest = hs.run_scenario(seed, config)
        path = fixture_path(case_name, seed)
        digest.save(path)
        print(
            f"  wrote {path.relative_to(hs.REPO_ROOT)}  "
            f"steps={config.num_steps} particles={config.num_particles} k_best={config.k_best} "
            f"track_rows={digest.track_mean.shape[0]} mean_ospa={digest.ospa.mean():.1f}"
        )
    print(f"Wrote {sum(len(seeds) for _, seeds in CASES.values())} golden fixtures.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="regenerate the committed fixtures")
    parser.add_argument(
        "--rtol",
        nargs="?",
        type=float,
        const=DEFAULT_RTOL,
        default=None,
        help=f"compare with a relative tolerance instead of bitwise (default {DEFAULT_RTOL:g})",
    )
    parser.add_argument("--exact", action="store_true", help="force bitwise comparison (the default)")
    parser.add_argument("--no-selfcheck", action="store_true", help="skip the run-to-run repeatability check")
    args = parser.parse_args()

    if args.write:
        write_fixtures()
        return 0

    if args.exact and args.rtol is not None:
        parser.error("--exact and --rtol are mutually exclusive")
    rtol = 0.0 if args.rtol is None else args.rtol
    mode = "bitwise" if rtol == 0.0 else f"rtol={rtol:g}"

    checker = Checker()
    missing = [
        fixture_path(name, seed) for name, _, seed in iter_cases() if not fixture_path(name, seed).exists()
    ]
    if missing:
        names = "\n  ".join(str(path.relative_to(hs.REPO_ROOT)) for path in missing)
        raise AssertionError(
            f"missing golden fixtures:\n  {names}\nGenerate them with:\n"
            "  python tests/test_golden_invariance.py --write"
        )

    print(f"golden invariance ({mode})")
    for case_name, config, seed in iter_cases():
        reference = hs.Digest.load(fixture_path(case_name, seed))
        candidate = hs.run_scenario(seed, config)

        failures = hs.compare_digests(reference, candidate, rtol=rtol)
        checker.ok(
            not failures,
            f"[{case_name} seed {seed}] digest differs from the fixture ({mode}):\n  "
            + "\n  ".join(failures),
        )

        if not args.no_selfcheck:
            repeat = hs.run_scenario(seed, config)
            repeat_failures = hs.compare_digests(candidate, repeat, rtol=0.0)
            checker.ok(
                not repeat_failures,
                f"[{case_name} seed {seed}] harness is not self-reproducible bitwise:\n  "
                + "\n  ".join(repeat_failures),
            )

        # A digest full of zeros or NaNs would silently satisfy the comparison above.
        checker.ok(candidate.ospa.size == config.num_steps, f"[{case_name} seed {seed}] wrong OSPA length")
        checker.ok(bool((candidate.ospa >= 0.0).all()), f"[{case_name} seed {seed}] negative OSPA")
        checker.ok(candidate.track_mean.shape[0] > 0, f"[{case_name} seed {seed}] digest recorded no tracks")
        print(
            f"  [{case_name} seed {seed}] steps={config.num_steps} k_best={config.k_best} "
            f"track_rows={candidate.track_mean.shape[0]} mean_ospa={candidate.ospa.mean():.1f} matched"
        )

    print(f"PASS: test_golden_invariance ({checker.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
