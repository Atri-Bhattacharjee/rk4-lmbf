"""Tests for the gradient-tuned search schedules (python/search_sgd.py).

The trainer rests on one formula: with independent draws, an object is found unless every one of
its visits is missed. If that formula or its hand-written gradient were wrong, training would still
run and still print numbers. So:

  M   the formula against the environment: its expected share of objects found equals the average
      of sampled schedules played through SearchEnv, for uniform odds (the random baseline), for
      arbitrary odds, for both rungs and both detection models;
  G   the gradient against central finite differences;
  T   training on toy problems whose optimum is known: commit to the better bin, split evenly
      between two bins, and give two orbits different bins;
  P   the policy object: slots map to the right orbit, sampled schedules follow the odds, save and
      load round-trip, and the best fixed bin is the one the environment scores highest.

Usage:
    python tests/test_search_sgd.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

import search_env as se  # noqa: E402
import search_sgd as sgd  # noqa: E402

SEED = 20260930


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def small_table(seed: int = SEED) -> se.PassTable:
    config = se.SearchConfig(num_orbits=6.0, seed=seed)
    return se.build_pass_table(se.make_scenario(config).states, config)


def policy_for(table: se.PassTable, half_angle: float, detection: str, rung: str, logits=None) -> sgd.SearchPolicy:
    config = table.config
    contexts = 1 if rung == "mix" else int(np.ceil(config.num_orbits))
    bins = se.direction_bins(half_angle)
    return sgd.SearchPolicy(half_angle_deg=half_angle, detection=detection, slot=10.0, dt=config.dt,
                            orbit_period=config.orbit_period, rung=rung,
                            logits=np.zeros((contexts, len(bins))) if logits is None else logits)


# ---------------------------------------------------------------------------------------------
# M: the formula against the environment
# ---------------------------------------------------------------------------------------------


def test_formula_against_environment(chk: Checker) -> None:
    table = small_table()
    rng = np.random.default_rng(SEED)
    for detection in se.DETECTION_MODELS:
        for half_angle in (45.0, 20.0):
            env = se.SearchEnv(table, fov_half_angle_deg=half_angle, detection=detection)
            bank = sgd.bank_from_tables([table], env.bins, env.slot_steps, detection)
            chk.ok(bank.num_objects == int(env.visible.sum()),
                   f"bank holds {bank.num_objects} objects, the environment can find {int(env.visible.sum())}")
            for rung, scale in (("mix", 0.0), ("mix", 1.5), ("orbit", 1.5)):
                policy = policy_for(table, half_angle, detection, rung)
                policy.logits = scale * rng.normal(size=policy.logits.shape)
                expected = sgd.bank_value(policy, bank)
                played = np.array([env.run_schedule(policy.sample_schedule(env.num_slots, env.num_sensors, rng)).found
                                   for _ in range(300)]) / bank.num_objects
                error = played.std(ddof=1) / np.sqrt(len(played))
                chk.ok(abs(played.mean() - expected) < 4.0 * error + 1e-9,
                       f"{detection} {half_angle} {rung} scale {scale}: formula {expected:.4f}, "
                       f"environment {played.mean():.4f} +- {error:.4f}")
            # Whole-bank value in chunks equals the value in one piece.
            policy = policy_for(table, half_angle, detection, "orbit")
            policy.logits = rng.normal(size=policy.logits.shape)
            whole, _ = sgd.expected_found(policy.logits, bank.hits.astype(np.float64),
                                          policy.contexts(bank.slot), bank.start, gradient=False)
            chk.ok(abs(sgd.bank_value(policy, bank, chunk=37) - whole) < 1e-6, "chunked value differs")


# ---------------------------------------------------------------------------------------------
# G: gradient
# ---------------------------------------------------------------------------------------------


def test_gradient(chk: Checker) -> None:
    table = small_table()
    rng = np.random.default_rng(SEED + 1)
    for half_angle, rung in ((45.0, "mix"), (20.0, "mix"), (20.0, "orbit")):
        bins = se.direction_bins(half_angle)
        bank = sgd.bank_from_tables([table], bins, 10, "streak")
        policy = policy_for(table, half_angle, "streak", rung)
        logits = rng.normal(size=policy.logits.shape)
        hits, context = bank.hits.astype(np.float64), policy.contexts(bank.slot)
        value, slope = sgd.expected_found(logits, hits, context, bank.start)
        chk.ok(0.0 < value < 1.0, f"value {value} is not a share")
        chk.ok(np.allclose(slope.sum(axis=1), 0.0, atol=1e-12),
               "gradient is not orthogonal to adding a constant to a row of logits")
        worst = 0.0
        for _ in range(40):
            c, b = rng.integers(logits.shape[0]), rng.integers(logits.shape[1])
            bump = np.zeros_like(logits)
            bump[c, b] = 1e-5
            up, _ = sgd.expected_found(logits + bump, hits, context, bank.start, gradient=False)
            down, _ = sgd.expected_found(logits - bump, hits, context, bank.start, gradient=False)
            worst = max(worst, abs((up - down) / 2e-5 - slope[c, b]))
        chk.ok(worst < 1e-8 + 1e-5 * np.abs(slope).max(),
               f"{half_angle} {rung}: gradient differs from finite differences by {worst:.2e} "
               f"(largest entry {np.abs(slope).max():.2e})")


# ---------------------------------------------------------------------------------------------
# T: toy problems with known optima
# ---------------------------------------------------------------------------------------------


def toy_policy(contexts: int) -> sgd.SearchPolicy:
    # Slot length 10 s and a 100 s "orbit": slots 0-9 are orbit 0, slots 10-19 orbit 1.
    return sgd.SearchPolicy(half_angle_deg=45.0, detection="streak", slot=10.0, dt=1.0, orbit_period=100.0,
                            logits=np.zeros((contexts, 2)), rung="mix" if contexts == 1 else "orbit")


def test_toys(chk: Checker) -> None:
    only_first, only_second = [True, False], [False, True]

    # One visit each; 60% of objects are caught by bin 0 only, 40% by bin 1 only. Value is
    # 0.6 p + 0.4 (1 - p): commit to bin 0.
    hits = np.array([only_first] * 600 + [only_second] * 400)
    bank = sgd.Bank(hits=hits, slot=np.zeros(1000, dtype=np.int64), start=np.arange(1001))
    policy = toy_policy(1)
    sgd.train(policy, bank, steps=400, batch_objects=250, seed=1)
    chk.ok(policy.odds[0, 0] > 0.98 and sgd.bank_value(policy, bank) > 0.59,
           f"commit toy: odds {policy.odds[0].round(3)}, value {sgd.bank_value(policy, bank):.3f}")

    # Two visits each, in different orbits; half the objects need bin 0 at either visit, half need
    # bin 1. One mix: 1 - (p^2 + (1 - p)^2) / 2, best at p = 1/2 with value 0.75.
    hits = np.array([only_first, only_first] * 500 + [only_second, only_second] * 500)
    slot = np.tile([0, 10], 1000)
    bank = sgd.Bank(hits=hits, slot=slot, start=np.arange(0, 2001, 2))
    policy = toy_policy(1)
    sgd.train(policy, bank, steps=400, batch_objects=250, seed=2)
    chk.ok(abs(policy.odds[0, 0] - 0.5) < 0.03 and abs(sgd.bank_value(policy, bank) - 0.75) < 0.002,
           f"split toy: odds {policy.odds[0].round(3)}, value {sgd.bank_value(policy, bank):.3f}")

    # Same objects with odds per orbit: one orbit looks at bin 0 and the other at bin 1, and every
    # object is found. Uniform odds are a saddle, so this also checks that the rows come apart.
    policy = toy_policy(2)
    sgd.train(policy, bank, steps=600, batch_objects=250, seed=3)
    odds = policy.odds
    chk.ok(sgd.bank_value(policy, bank) > 0.98 and abs(odds[0, 0] - odds[1, 0]) > 0.9,
           f"per-orbit toy: odds {odds.round(3).tolist()}, value {sgd.bank_value(policy, bank):.3f}")


# ---------------------------------------------------------------------------------------------
# P: the policy object
# ---------------------------------------------------------------------------------------------


def test_policy(chk: Checker) -> None:
    table = small_table()
    config = table.config
    rng = np.random.default_rng(SEED + 2)
    policy = policy_for(table, 20.0, "streak", "orbit")
    policy.logits = rng.normal(size=policy.logits.shape)

    slots = np.arange(config.num_steps // 10 + 1)
    context = policy.contexts(slots)
    chk.ok(context[0] == 0 and context.max() == len(policy.logits) - 1 and np.all(np.diff(context) >= 0),
           "slots do not map onto orbits 0..C-1 in order")
    boundary = int(np.ceil(config.orbit_period / 10.0))
    chk.ok(context[boundary - 1] == 0 and context[boundary] == 1, "the first orbit does not end at one period")
    chk.ok(np.all(policy_for(table, 20.0, "streak", "mix").contexts(slots) == 0), "a mix has more than one context")
    chk.ok(np.allclose(policy.odds.sum(axis=1), 1.0) and np.all(policy.odds > 0.0), "odds are not distributions")

    schedule = policy.sample_schedule(len(slots), config.num_sensors, rng)
    chk.ok(schedule.shape == (len(slots), config.num_sensors) and schedule.min() >= 0
           and schedule.max() < policy.logits.shape[1], "sampled schedule has the wrong shape or range")
    for c in (0, len(policy.logits) - 1):
        drawn = schedule[context == c].ravel()
        frequency = np.bincount(drawn, minlength=policy.logits.shape[1]) / len(drawn)
        tolerance = 5.0 * np.sqrt(policy.odds[c] * (1.0 - policy.odds[c]) / len(drawn)) + 1e-4
        chk.ok(np.all(np.abs(frequency - policy.odds[c]) < tolerance), f"orbit {c}: draws do not follow the odds")

    with tempfile.TemporaryDirectory() as folder:
        path = sgd.schedule_path(folder, 20.0, "streak", config.dt, "orbit")
        policy.save(path)
        loaded = sgd.SearchPolicy.load(path)
    chk.ok(np.array_equal(loaded.logits, policy.logits) and loaded.rung == "orbit" and loaded.detection == "streak"
           and loaded.half_angle_deg == 20.0 and loaded.slot == 10.0 and loaded.orbit_period == policy.orbit_period,
           "save / load changed the policy")

    # The best fixed bin: its share is what the environment gives for holding it, and no bin beats it.
    for detection in se.DETECTION_MODELS:
        env = se.SearchEnv(table, fov_half_angle_deg=20.0, detection=detection)
        bank = sgd.bank_from_tables([table], env.bins, env.slot_steps, detection)
        best, share = sgd.best_fixed_bin(bank)
        held = [env.run_schedule(np.full((env.num_slots, env.num_sensors), b)).found for b in range(env.num_bins)]
        chk.ok(held[best] == max(held) and abs(share - held[best] / bank.num_objects) < 1e-12,
               f"{detection}: best fixed bin {best} finds {held[best]}, the best bin finds {max(held)}")


def main() -> None:
    chk = Checker()
    print("search schedules by gradient ascent")
    for name, test in (("M formula vs environment", test_formula_against_environment),
                       ("G gradient", test_gradient),
                       ("T toy optima", test_toys),
                       ("P policy", test_policy)):
        before = chk.count
        test(chk)
        print(f"  {name}: {chk.count - before} assertions")
    print(f"PASS: test_search_sgd ({chk.count} assertions)")


if __name__ == "__main__":
    main()
