"""Statistical equivalence against a committed baseline, for phases that legitimately change
floating-point ordering or the RNG stream.

Some optimizations cannot be bitwise: Phase 2 changes the summation order of the mixture and the
granularity of the resampling walk, and Phase 5 hoists a ``normal_distribution`` out of a loop,
which shifts the propagator's random stream. For those, "unchanged" has to mean "draws from the
same distribution", so this harness runs many independent scenarios per arm and compares the
resulting mean-OSPA distributions.

Three comparisons must all pass:

1. Welch's t-test on per-run mean OSPA (location), ``p > 0.01``.
2. Two-sample KS test on per-run mean OSPA (whole distribution, not just its mean), ``p > 0.01``.
3. A per-step Welch test on the mean-OSPA curve, Bonferroni-corrected across steps. A phase that
   shifts only, say, the convergence transient would pass 1 and 2 while failing this.

The baseline is generated once from the pre-change build and committed. Regenerating it discards
the reference point, so only do it deliberately and say why in the commit message.

Usage:
    python tests/test_statistical_equivalence.py            # compare against the fixture
    python tests/test_statistical_equivalence.py --write    # regenerate the baseline
    python tests/test_statistical_equivalence.py --seeds 96 # widen the sample (both arms must match)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_scenario as hs  # noqa: E402
import statistics_helpers as st  # noqa: E402

STAT_CONFIG = hs.ScenarioConfig(num_steps=60, num_particles=500, k_best=16)
NUM_SEEDS = 48
SEED_SOURCE = 20260908
ALPHA = 0.01
BASELINE_PATH = hs.FIXTURE_DIR / "statistical_baseline.npz"


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def scenario_seeds(num_seeds: int) -> list[int]:
    """Well-separated master seeds, so the runs are independent rather than adjacent integers."""
    state = np.random.SeedSequence(SEED_SOURCE).generate_state(num_seeds, dtype=np.uint32)
    return [int(value) for value in state]


def collect_arm(num_seeds: int, config: hs.ScenarioConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run one arm. Returns ``(mean_ospa_per_run, ospa_curves)``."""
    curves = np.zeros((num_seeds, config.num_steps), dtype=np.float64)
    for row, seed in enumerate(scenario_seeds(num_seeds)):
        curves[row, :] = hs.run_scenario(seed, config).ospa
    return curves.mean(axis=1), curves


def write_baseline(num_seeds: int) -> None:
    mean_ospa, curves = collect_arm(num_seeds, STAT_CONFIG)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        BASELINE_PATH,
        mean_ospa=mean_ospa,
        curves=curves,
        config=np.array(
            [STAT_CONFIG.num_steps, STAT_CONFIG.num_particles, STAT_CONFIG.k_best], dtype=np.int64
        ),
        seed_source=np.array([SEED_SOURCE], dtype=np.int64),
    )
    print(
        f"  wrote {BASELINE_PATH.relative_to(hs.REPO_ROOT)}  runs={num_seeds} "
        f"mean={mean_ospa.mean():.1f} m  sd={mean_ospa.std(ddof=1):.1f} m"
    )
    print("Baseline written. Record the regeneration and its reason in the commit message.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="regenerate the committed baseline")
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS, help=f"runs per arm (default {NUM_SEEDS})")
    args = parser.parse_args()

    if args.seeds < 40:
        parser.error("the gate requires at least 40 runs per arm")

    if args.write:
        write_baseline(args.seeds)
        return 0

    checker = Checker()
    # A bug in the test machinery would otherwise look like a passing gate.
    self_test_assertions = st.self_test()
    checker.count += self_test_assertions
    print(f"statistics helpers self-test clean ({self_test_assertions} assertions)")

    if not BASELINE_PATH.exists():
        raise AssertionError(
            f"missing baseline fixture {BASELINE_PATH.relative_to(hs.REPO_ROOT)}\nGenerate it with:\n"
            "  python tests/test_statistical_equivalence.py --write"
        )

    with np.load(BASELINE_PATH) as data:
        baseline_mean = data["mean_ospa"]
        baseline_curves = data["curves"]
        baseline_config = hs.ScenarioConfig(*(int(value) for value in data["config"]))
        baseline_seed_source = int(data["seed_source"][0])

    checker.ok(
        baseline_config == STAT_CONFIG,
        f"baseline was generated with {baseline_config} but this harness uses {STAT_CONFIG}",
    )
    checker.ok(
        baseline_seed_source == SEED_SOURCE,
        f"baseline seed source {baseline_seed_source} != {SEED_SOURCE}",
    )
    checker.ok(
        baseline_mean.size == args.seeds,
        f"baseline has {baseline_mean.size} runs but {args.seeds} were requested; "
        "regenerate the baseline with --write --seeds N to change the sample size",
    )

    candidate_mean, candidate_curves = collect_arm(args.seeds, STAT_CONFIG)
    checker.ok(bool(np.isfinite(candidate_mean).all()), "candidate arm produced a non-finite mean OSPA")

    t_statistic, degrees_of_freedom, t_p = st.welch_t_test(baseline_mean, candidate_mean)
    ks_statistic, ks_p = st.ks_2samp(baseline_mean, candidate_mean)

    pooled_sd = float(np.sqrt(0.5 * (baseline_mean.var(ddof=1) + candidate_mean.var(ddof=1))))
    delta = float(candidate_mean.mean() - baseline_mean.mean())
    effect_size = delta / pooled_sd if pooled_sd > 0.0 else 0.0

    print(f"mean-OSPA distributions over {args.seeds} runs per arm")
    print(f"  baseline   {baseline_mean.mean():9.1f} m  sd {baseline_mean.std(ddof=1):8.1f} m")
    print(f"  candidate  {candidate_mean.mean():9.1f} m  sd {candidate_mean.std(ddof=1):8.1f} m")
    print(f"  delta      {delta:9.1f} m  (Cohen's d {effect_size:+.3f})")
    print(f"  Welch t = {t_statistic:+.4f}, df = {degrees_of_freedom:.1f}, p = {t_p:.4f} (need p > {ALPHA})")
    print(f"  KS    D = {ks_statistic:.4f}, p = {ks_p:.4f} (need p > {ALPHA})")

    checker.ok(
        t_p > ALPHA,
        f"Welch t-test rejects equality of mean OSPA: t={t_statistic:+.4f}, df={degrees_of_freedom:.1f}, "
        f"p={t_p:.6f} <= {ALPHA} (delta {delta:.1f} m, Cohen's d {effect_size:+.3f})",
    )
    checker.ok(
        ks_p > ALPHA,
        f"KS test rejects equality of the mean-OSPA distribution: D={ks_statistic:.4f}, "
        f"p={ks_p:.6f} <= {ALPHA}",
    )

    # Per-step curve comparison, Bonferroni-corrected: with 60 independent tests at alpha=0.01 you
    # would expect ~0.6 spurious rejections, so the uncorrected threshold would flag healthy runs.
    step_threshold = ALPHA / STAT_CONFIG.num_steps
    step_p_values = np.ones(STAT_CONFIG.num_steps, dtype=np.float64)
    for step in range(STAT_CONFIG.num_steps):
        step_p_values[step] = st.welch_t_test(baseline_curves[:, step], candidate_curves[:, step])[2]

    worst_step = int(np.argmin(step_p_values))
    worst_p = float(step_p_values[worst_step])
    print(
        f"  per-step curve: worst p = {worst_p:.2e} at step {worst_step} "
        f"(Bonferroni threshold {step_threshold:.2e})"
    )
    checker.ok(
        worst_p > step_threshold,
        f"per-step mean-OSPA curve differs at step {worst_step}: p={worst_p:.3e} <= {step_threshold:.3e} "
        f"(baseline {baseline_curves[:, worst_step].mean():.1f} m, "
        f"candidate {candidate_curves[:, worst_step].mean():.1f} m)",
    )

    identical = bool(np.array_equal(baseline_curves, candidate_curves))
    print(f"  arms are bitwise identical: {identical}")

    print(f"PASS: test_statistical_equivalence ({checker.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
