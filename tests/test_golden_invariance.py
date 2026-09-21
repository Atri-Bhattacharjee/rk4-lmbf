"""Golden-digest invariance: the primary gate for every optimization phase.

Runs a fixed set of fully-seeded scenarios and compares the resulting digest against a committed
fixture. Because all five random streams are pinned (see ``harness_scenario``), a correct
refactor reproduces the fixture *bitwise*; only a phase that deliberately changes floating-point
summation order needs the relaxed ``--rtol`` mode.

Bitwise reproduction holds only on the toolchain that wrote the fixtures, so it is gated to the
reference platform (x86-64 Linux / libstdc++) and every other platform runs ``--portable`` instead.
See the PLATFORM GATING block below for why, and for what replaces it elsewhere.

Usage:
    python tests/test_golden_invariance.py                 # bitwise on the reference platform,
                                                           # portable everywhere else
    python tests/test_golden_invariance.py --exact          # force bitwise
    python tests/test_golden_invariance.py --rtol           # tolerant, rtol 1e-12
    python tests/test_golden_invariance.py --rtol 1e-9      # tolerant, explicit rtol
    python tests/test_golden_invariance.py --portable       # force portable (platform-independent)
    python tests/test_golden_invariance.py --write          # regenerate the fixtures
    python tests/test_golden_invariance.py --no-selfcheck   # skip the run-to-run repeat

Integer digest fields (cardinality, measurement counts, track labels) are compared exactly in the
bitwise and ``--rtol`` modes, so a phase that perturbs track identity or a pruning decision fails
even under ``--rtol``.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np

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

# --- PLATFORM GATING ---------------------------------------------------------------------------
# The fixtures pin a bit pattern, and two things in the engine make that bit pattern a property of
# the toolchain rather than of the algorithm:
#
#   1. The RNG stream. ``std::mt19937_64`` is specified bit-for-bit by the standard, but the
#      distributions layered on it are not: ``std::normal_distribution`` (birth particles in
#      adaptive_birth_model.cpp, process noise in two_body_propagator.cpp) and
#      ``std::uniform_real_distribution`` (the resampler offset in smc_lmb_tracker.cpp) are
#      implementation-defined in both algorithm and engine consumption. libstdc++ and the MSVC STL
#      happen to agree; libc++ (macOS) draws a different sequence from the same seed, and the
#      scenario diverges on the very first birth.
#   2. Floating-point rounding. Clang contracts ``a * b + c`` into a single-rounding ``fma``
#      wherever the ISA has one -- always, on arm64, and never on the x86-64 baseline GCC targets.
#      Apple's libm ``sin``/``cos``/``atan2``/``exp``/``log`` are not bit-identical to glibc's, and
#      Eigen reduces in NEON lane order rather than SSE lane order.
#
# Cause 1 is not a rounding difference and no tolerance absorbs it: a different stream is a
# different draw from the same distribution. Measured over these cases, changing only the C++ stream
# moves a scenario's mean GOSPA by a factor of 0.3 to 2 and individual per-step values by several
# hundred percent, while the birth cloud's mean moves by the sampling error of the birth covariance
# over ``num_particles`` from the very first step. So off the reference platform this test compares
# only what is reproducible anywhere (PORTABLE_INT_FIELDS exactly, mean GOSPA below
# PORTABLE_MEAN_GOSPA_CEILING, and the run-to-run self-check), and the numerical gate becomes the
# distributional one in test_statistical_equivalence.py, which ci-test.sh and ci-test.ps1 run off the
# reference platform for exactly this reason.
REFERENCE_PLATFORM = "linux"
REFERENCE_MACHINES = ("x86_64", "amd64")

# ``num_measurements`` is driven entirely by NumPy's legacy global stream -- the detection rolls and
# measurement noise in ``run_once.generate_measurements`` -- and MT19937 plus NumPy's own
# distributions are portable, so this one has to hold on every platform. Every other digest field is
# downstream of the C++ RNG or of floating-point ordering.
PORTABLE_INT_FIELDS = ("num_measurements", "num_truths")

# One scenario's mean GOSPA is a single draw, so comparing it against the fixture's value is not a
# useful gate. Under the OSPA predecessor the coefficient of variation was ~30% and the 48-run
# baseline spanned 0.39-1.60 of its own mean; under GOSPA at this cutoff it is 8.5% and 0.74-1.17,
# but that tightening is clipping, not precision -- most steps are pinned at the maximum attainable
# value, so the spread that remains is mostly the unclipped minority. Either way no band tight
# enough to mean something would be safe here. So portable mode
# asserts only that the filter has not stopped tracking -- a run that loses its tracks pins every
# step at the maximum attainable GOSPA -- and the distributional comparison is left to
# test_statistical_equivalence.py, which has 48 runs per arm to do it properly.
#
# The gate is the fraction of steps that produced an accepted track/truth pair, not a fraction of
# the cutoff: unnormalised GOSPA is unbounded by c and grows as sqrt(cardinality), so "mean below
# 0.9 * c" is not a meaningful statement about it. A mean-saturation threshold does not work either
# -- at 200 particles this harness errs by more than c at most steps, so a healthy run already sits
# at 0.66-0.88 mean saturation with nothing between it and 1.0. hs.gospa_tracking_fraction separates
# cleanly instead: 0.35-0.83 across these five cases, and 0.0 for a filter that has stopped
# tracking. The floor lives in harness_scenario.py so every lost-track gate shares one number.
PORTABLE_TRACKING_FRACTION_FLOOR = hs.TRACKING_FRACTION_FLOOR


def is_reference_platform() -> bool:
    """True on the platform the committed fixtures were written on."""
    return sys.platform.startswith(REFERENCE_PLATFORM) and platform.machine().lower() in REFERENCE_MACHINES


def platform_tag() -> str:
    return f"{sys.platform}/{platform.machine()}"


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


def drift_summary(reference: hs.Digest, candidate: hs.Digest) -> list[str]:
    """Per-field worst relative deviation, for the portable mode's informational report.

    Fields whose row count depends on how many tracks survived pruning can differ in shape between
    platforms; those are reported as a shape, since no elementwise deviation exists.
    """
    entries: list[str] = []
    for name in hs.FLOAT_FIELDS:
        ref_values = getattr(reference, name).ravel()
        cand_values = getattr(candidate, name).ravel()
        if ref_values.shape != cand_values.shape:
            entries.append(f"{name} shape {cand_values.shape} vs {ref_values.shape}")
            continue
        if ref_values.size == 0:
            continue
        denominator = np.where(np.abs(ref_values) > 0.0, np.abs(ref_values), 1.0)
        entries.append(f"{name} {np.max(np.abs(cand_values - ref_values) / denominator):.2e}")
    return entries


def identity_summary(reference: hs.Digest, candidate: hs.Digest) -> str:
    """One line on the digest fields that portable mode does not gate on."""
    failures = hs.compare_digests(
        reference,
        candidate,
        rtol=0.0,
        int_fields=tuple(name for name in hs.INT_FIELDS if name not in PORTABLE_INT_FIELDS),
        float_fields=(),
    )
    if not failures:
        return "cardinality and track identity match the fixture"
    return "cardinality/track identity differ from the fixture: " + "; ".join(failures)


def write_fixtures() -> None:
    if not is_reference_platform():
        print(
            f"WARNING: writing fixtures on {platform_tag()}, which is not the reference platform "
            f"({REFERENCE_PLATFORM}/{'|'.join(REFERENCE_MACHINES)}). The bitwise gate runs there, so "
            "fixtures written here will fail it."
        )
    for case_name, config, seed in iter_cases():
        digest = hs.run_scenario(seed, config)
        path = fixture_path(case_name, seed)
        digest.save(path)
        print(
            f"  wrote {path.relative_to(hs.REPO_ROOT)}  "
            f"steps={config.num_steps} particles={config.num_particles} k_best={config.k_best} "
            f"track_rows={digest.track_mean.shape[0]} mean_gospa={digest.gospa.mean():.1f}"
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
    parser.add_argument(
        "--exact",
        action="store_true",
        help="force bitwise comparison (the default on the reference platform)",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="compare only the platform-independent digest fields (the default off the reference platform)",
    )
    parser.add_argument("--no-selfcheck", action="store_true", help="skip the run-to-run repeatability check")
    args = parser.parse_args()

    if args.write:
        write_fixtures()
        return 0

    requested = (
        ("--exact", args.exact),
        ("--portable", args.portable),
        ("--rtol", args.rtol is not None),
    )
    explicit = [name for name, given in requested if given]
    if len(explicit) > 1:
        parser.error(f"{', '.join(explicit)} are mutually exclusive")

    # Nothing was requested explicitly, so the platform decides: bitwise where the fixtures were
    # written, portable everywhere else.
    portable = args.portable or (not explicit and not is_reference_platform())
    rtol = 0.0 if args.rtol is None else args.rtol
    if portable:
        mode = "portable"
    elif rtol == 0.0:
        mode = "bitwise"
    else:
        mode = f"rtol={rtol:g}"

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

    print(f"golden invariance ({mode}) on {platform_tag()}")
    if portable:
        print(
            f"  the fixtures are bitwise only on {REFERENCE_PLATFORM}/{'|'.join(REFERENCE_MACHINES)}; "
            f"gating on {', '.join(PORTABLE_INT_FIELDS)}, GOSPA tracking fraction above "
            f"{PORTABLE_TRACKING_FRACTION_FLOOR:.2f} and run-to-run repeatability "
            "(see the PLATFORM GATING block in this file)"
        )
    for case_name, config, seed in iter_cases():
        reference = hs.Digest.load(fixture_path(case_name, seed))
        candidate = hs.run_scenario(seed, config)

        if portable:
            failures = hs.compare_digests(
                reference, candidate, rtol=0.0, int_fields=PORTABLE_INT_FIELDS, float_fields=()
            )
        else:
            failures = hs.compare_digests(reference, candidate, rtol=rtol)
        checker.ok(
            not failures,
            f"[{case_name} seed {seed}] digest differs from the fixture ({mode}):\n  "
            + "\n  ".join(failures),
        )

        if portable:
            candidate_tracking = hs.gospa_tracking_fraction(candidate)
            reference_tracking = hs.gospa_tracking_fraction(reference)
            checker.ok(
                candidate_tracking > PORTABLE_TRACKING_FRACTION_FLOOR,
                f"[{case_name} seed {seed}] only {candidate_tracking:.3f} of steps produced an "
                f"accepted track/truth pair (fixture: {reference_tracking:.3f}); the filter is not "
                "tracking on this platform",
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
        checker.ok(candidate.gospa.size == config.num_steps, f"[{case_name} seed {seed}] wrong GOSPA length")
        checker.ok(bool((candidate.gospa >= 0.0).all()), f"[{case_name} seed {seed}] negative GOSPA")
        checker.ok(bool(np.isfinite(candidate.gospa).all()), f"[{case_name} seed {seed}] non-finite GOSPA")
        checker.ok(candidate.track_mean.shape[0] > 0, f"[{case_name} seed {seed}] digest recorded no tracks")
        print(
            f"  [{case_name} seed {seed}] steps={config.num_steps} k_best={config.k_best} "
            f"track_rows={candidate.track_mean.shape[0]} mean_gospa={candidate.gospa.mean():.1f} matched"
        )
        if portable:
            # Not gated on, but printed so a platform's drift is visible in the CI log rather than
            # being something you have to reproduce locally to see.
            print(f"      drift vs fixture: {', '.join(drift_summary(reference, candidate))}")
            print(f"      {identity_summary(reference, candidate)}")

    print(f"PASS: test_golden_invariance ({mode}, {checker.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
