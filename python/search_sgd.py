"""Search schedules tuned by gradient ascent on the expected number of objects found.

A search policy here is a set of odds over the direction bins: every slot, every sensor draws its
pointing from them. With independent draws, an object is found unless every one of its visits (a
stretch in one sensor's bubble during one slot) is missed, so for odds p

    P(found) = 1 - prod over its visits of (1 - p . hits_of_that_visit)

which is exact for the search environment and smooth in p. Its gradient is written out below, so
the odds are tuned directly on the quantity that is scored, with no reward sampling. Three rungs:

  fixed   the best single bin, by counting (no training);
  mix     one set of odds for the whole run;
  orbit   separate odds for each orbit of the run, shared by all sensors.

Training objects come from the training orbit families with a random node longitude each
(SearchConfig(split="train", rotate=True)); scores are sampled schedules played through SearchEnv on
scenarios drawn from the held-out families.

    python python/search_sgd.py                                 # 45/20/10/5 deg, streak and sample
    python python/search_sgd.py --fov 20 --detection sample --bank 20 --test 5
    python python/search_sgd.py --dt 0.25 --detection sample    # schedules for a 4 Hz filter run

Writes python/results/search_sgd/: summary_dt<dt>.json and, for each field of view and detection
model, schedule_fov<deg>_<detection>_dt<dt>.npz (the rung that did best on validation scenarios),
which python/run_tasked.py loads, with the two rungs beside it as ..._mix.npz and ..._orbit.npz.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from run_ring import RESULTS_DIR
from search_baselines import OraclePolicy
from search_env import (DETECTION_MODELS, SearchConfig, SearchEnv, build_pass_table, direction_bins,
                        make_scenario, visit_hits)

RUNGS = ("mix", "orbit")


# =============================================================================
# Training objects
# =============================================================================


@dataclass
class Bank:
    """Visits of many objects, ordered by object. Objects that never enter a bubble are left out:
    no pointing finds them, so every share below is of the objects that could be found."""
    hits: np.ndarray        # (V, B) bool: bins that catch the object during the visit
    slot: np.ndarray        # (V,) slot of the visit
    start: np.ndarray       # (N + 1,) first visit of each object

    @property
    def num_objects(self) -> int:
        return len(self.start) - 1


def bank_from_tables(tables, bins, slot_steps: int, detection: str) -> Bank:
    hits, slots, counts = [], [], []
    for table in tables:
        visits = visit_hits(table, bins, slot_steps, detection)
        hits.append(visits.hits)
        slots.append(visits.slot)
        counts.append(np.unique(visits.object, return_counts=True)[1])
    counts = np.concatenate(counts)
    return Bank(hits=np.vstack(hits), slot=np.concatenate(slots),
                start=np.concatenate([[0], np.cumsum(counts)]))


def scenario_tables(config: SearchConfig, seeds) -> list:
    return [build_pass_table(make_scenario(config, seed).states, config) for seed in seeds]


# =============================================================================
# Policy
# =============================================================================


@dataclass
class SearchPolicy:
    """Odds over direction bins, one row per context. One context is a single mix for the whole
    run; more than one is a row per orbit of the run."""
    half_angle_deg: float
    detection: str
    slot: float               # seconds a pointing is held
    dt: float
    orbit_period: float
    logits: np.ndarray        # (C, B)
    rung: str = "mix"

    @property
    def odds(self) -> np.ndarray:
        return softmax(self.logits)

    def contexts(self, slots) -> np.ndarray:
        """Row of the odds each slot uses."""
        slots = np.asarray(slots)
        if len(self.logits) == 1:
            return np.zeros(slots.shape, dtype=np.int64)
        orbit = np.floor(slots * self.slot / self.orbit_period).astype(np.int64)
        return np.minimum(orbit, len(self.logits) - 1)

    def sample_schedule(self, num_slots: int, num_sensors: int, rng: np.random.Generator) -> np.ndarray:
        """(num_slots, num_sensors) bins, each drawn independently from its slot's odds."""
        odds = self.odds
        context = self.contexts(np.arange(num_slots))
        schedule = np.empty((num_slots, num_sensors), dtype=np.int64)
        for c in np.unique(context):
            rows = np.flatnonzero(context == c)
            schedule[rows] = rng.choice(odds.shape[1], size=(len(rows), num_sensors), p=odds[c])
        return schedule

    def save(self, path) -> None:
        np.savez(path, half_angle_deg=self.half_angle_deg, detection=self.detection, slot=self.slot,
                 dt=self.dt, orbit_period=self.orbit_period, logits=self.logits, rung=self.rung)

    @classmethod
    def load(cls, path) -> "SearchPolicy":
        with np.load(path, allow_pickle=False) as data:
            return cls(half_angle_deg=float(data["half_angle_deg"]), detection=str(data["detection"]),
                       slot=float(data["slot"]), dt=float(data["dt"]),
                       orbit_period=float(data["orbit_period"]), logits=np.array(data["logits"]),
                       rung=str(data["rung"]))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    odds = np.exp(shifted)
    return odds / odds.sum(axis=1, keepdims=True)


def schedule_path(output, half_angle_deg: float, detection: str, dt: float, rung: str = "") -> Path:
    suffix = f"_{rung}" if rung else ""
    return Path(output) / f"schedule_fov{half_angle_deg:g}_{detection}_dt{dt:g}{suffix}.npz"


# =============================================================================
# Objective and gradient
# =============================================================================


def expected_found(logits: np.ndarray, hits: np.ndarray, context: np.ndarray, start: np.ndarray,
                   gradient: bool = True):
    """Expected share of objects found, and its gradient with respect to ``logits``.

    ``hits`` (V, B) and ``context`` (V,) describe the visits, ordered by object; ``start`` (N + 1,)
    is the first visit of each object. A visit catches its object with probability
    a = odds[context] . hits; the object is missed with probability prod(1 - a) over its visits.
    """
    odds = softmax(logits)
    catch = np.empty(len(hits))
    rows_of = [np.flatnonzero(context == c) for c in range(len(logits))]
    for c, rows in enumerate(rows_of):
        if len(rows):
            catch[rows] = hits[rows] @ odds[c].astype(hits.dtype)
    escape = np.clip(1.0 - catch, 1e-12, 1.0)
    missed = np.exp(np.add.reduceat(np.log(escape), start[:-1]))
    num_objects = len(start) - 1
    value = float(np.mean(1.0 - missed))
    if not gradient:
        return value, None
    # d value / d catch of a visit: the chance every other visit of its object misses, over N.
    per_visit = np.repeat(missed, np.diff(start)) / escape / num_objects
    by_odds = np.zeros_like(odds)
    for c, rows in enumerate(rows_of):
        if len(rows):
            by_odds[c] = (hits[rows].T @ per_visit[rows].astype(hits.dtype)).astype(np.float64)
    return value, odds * (by_odds - np.sum(odds * by_odds, axis=1, keepdims=True))


def bank_value(policy: SearchPolicy, bank: Bank, chunk: int = 4096) -> float:
    """Expected share of the bank's objects found, in chunks of objects."""
    context = policy.contexts(bank.slot)
    total = 0.0
    for low in range(0, bank.num_objects, chunk):
        high = min(low + chunk, bank.num_objects)
        rows = slice(bank.start[low], bank.start[high])
        value, _ = expected_found(policy.logits, bank.hits[rows].astype(np.float32), context[rows],
                                  bank.start[low:high + 1] - bank.start[low], gradient=False)
        total += value * (high - low)
    return total / bank.num_objects


def best_fixed_bin(bank: Bank) -> tuple[int, float]:
    """(bin, share of the bank found) for every sensor holding one bin for the whole run."""
    found = np.zeros(bank.hits.shape[1], dtype=np.int64)
    for low in range(0, bank.num_objects, 4096):
        high = min(low + 4096, bank.num_objects)
        rows = slice(bank.start[low], bank.start[high])
        found += np.logical_or.reduceat(bank.hits[rows], bank.start[low:high] - bank.start[low], axis=0).sum(axis=0)
    best = int(np.argmax(found))
    return best, float(found[best] / bank.num_objects)


def train(policy: SearchPolicy, bank: Bank, steps: int = 1500, batch_objects: int = 4096,
          learning_rate: float = 0.1, seed: int = 0, init_noise: float = 0.01) -> list[float]:
    """Adam ascent on minibatches of objects, in place. Starts from uniform odds, which is the
    random baseline, plus a little noise so that the rows of a per-orbit policy can come apart.
    Returns the minibatch values, one per step."""
    rng = np.random.default_rng(seed)
    policy.logits = init_noise * rng.normal(size=policy.logits.shape)
    context = policy.contexts(bank.slot)
    counts = np.diff(bank.start)
    first, second = np.zeros_like(policy.logits), np.zeros_like(policy.logits)
    history = []
    order, cursor = rng.permutation(bank.num_objects), 0
    for step in range(1, steps + 1):
        if cursor + batch_objects > bank.num_objects:
            order, cursor = rng.permutation(bank.num_objects), 0
        objects = order[cursor:cursor + batch_objects]
        cursor += batch_objects
        size = counts[objects]
        rows = np.repeat(bank.start[objects], size) + (np.arange(size.sum()) - np.repeat(np.cumsum(size) - size, size))
        value, slope = expected_found(policy.logits, bank.hits[rows].astype(np.float32), context[rows],
                                      np.concatenate([[0], np.cumsum(size)]))
        history.append(value)
        first = 0.9 * first + 0.1 * slope
        second = 0.999 * second + 0.001 * slope**2
        rate = learning_rate * (0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * step / steps)))
        policy.logits = policy.logits + rate * (first / (1.0 - 0.9**step)) / (
            np.sqrt(second / (1.0 - 0.999**step)) + 1e-8)
    return history


# =============================================================================
# Scoring in the environment
# =============================================================================


def score(tables, half_angle_deg: float, detection: str, slot: float, policies: dict, fixed_bin: int,
          draws: int, rng: np.random.Generator) -> dict:
    """Share of findable objects found on each scenario, per policy: {name: [share per scenario]}.
    ``policies`` are sampled; "fixed" holds one bin; "oracle" knows the truth."""
    shares = {name: [] for name in ("fixed", "oracle", *policies)}
    for table in tables:
        env = SearchEnv(table, fov_half_angle_deg=half_angle_deg, detection=detection, slot=slot)
        bound = max(int(env.visible.sum()), 1)
        shares["oracle"].append(env.run(OraclePolicy(env.num_sensors)).found / bound)
        shares["fixed"].append(env.run_schedule(
            np.full((env.num_slots, env.num_sensors), fixed_bin)).found / bound)
        for name, policy in policies.items():
            shares[name].append(float(np.mean([
                env.run_schedule(policy.sample_schedule(env.num_slots, env.num_sensors, rng)).found
                for _ in range(draws)])) / bound)
    return shares


def tune(config: SearchConfig, half_angle_deg: float, detection: str, train_tables, valid_tables,
         test_tables, slot: float = 10.0, steps: int = 1500, batch_objects: int = 4096,
         learning_rate: float = 0.1, draws: int = 3, seed: int = 0, verbose: bool = True):
    """Train both rungs for one field of view and detection model, score everything on the test
    scenarios. Returns (record, {rung: policy}, selected rung)."""
    tick = time.perf_counter()
    bins = direction_bins(half_angle_deg)
    slot_steps = max(1, int(round(slot / config.dt)))
    bank = bank_from_tables(train_tables, bins, slot_steps, detection)
    valid = bank_from_tables(valid_tables, bins, slot_steps, detection)
    fixed_bin, fixed_train = best_fixed_bin(bank)

    def blank(rung: str) -> SearchPolicy:
        contexts = 1 if rung == "mix" else int(np.ceil(config.num_orbits))
        return SearchPolicy(half_angle_deg=half_angle_deg, detection=detection, slot=slot_steps * config.dt,
                            dt=config.dt, orbit_period=config.orbit_period,
                            logits=np.zeros((contexts, len(bins))), rung=rung)

    uniform = blank("mix")
    policies, train_value, valid_value = {}, {}, {}
    for rung in RUNGS:
        policy = blank(rung)
        train(policy, bank, steps=steps, batch_objects=batch_objects, learning_rate=learning_rate, seed=seed)
        policies[rung] = policy
        train_value[rung], valid_value[rung] = bank_value(policy, bank), bank_value(policy, valid)
    selected = max(RUNGS, key=lambda rung: valid_value[rung])

    shares = score(test_tables, half_angle_deg, detection, slot, {"random": uniform, **policies},
                   fixed_bin, draws, np.random.default_rng(seed))
    record = {
        "fov_half_angle_deg": float(half_angle_deg), "detection": detection, "dt": config.dt,
        "bins": len(bins), "bank_objects": bank.num_objects, "bank_visits": int(len(bank.slot)),
        "fixed_bin": fixed_bin, "fixed_bin_boresight": bins.boresight[fixed_bin].round(3).tolist(),
        "selected": selected,
        "expected_uniform_bank": bank_value(uniform, bank),
        "train": {"fixed": fixed_train, **train_value}, "valid": valid_value,
        "test": {name: float(np.mean(values)) for name, values in shares.items()},
        "test_std": {name: float(np.std(values)) for name, values in shares.items()},
        "largest_odds": {rung: float(policies[rung].odds.max(axis=1).mean()) for rung in RUNGS},
        "seconds": time.perf_counter() - tick,
    }
    if verbose:
        test = record["test"]
        print(f"  {half_angle_deg:5.1f} {detection:>7} {test['random']:8.1%} {test['fixed']:8.1%} "
              f"{test['mix']:8.1%} {test['orbit']:8.1%} {test['oracle']:8.1%}   "
              f"{train_value['mix']:6.1%} / {valid_value['mix']:6.1%} / {test['mix']:6.1%}   "
              f"{selected:>5}  ({record['seconds']:.0f} s)", flush=True)
    return record, policies, selected


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fov", type=float, nargs="+", default=[45.0, 20.0, 10.0, 5.0])
    parser.add_argument("--detection", choices=DETECTION_MODELS, nargs="+", default=list(DETECTION_MODELS)[::-1])
    parser.add_argument("--orbits", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=SearchConfig.dt)
    parser.add_argument("--slot", type=float, default=10.0, help="seconds a pointing is held")
    parser.add_argument("--bank", type=int, default=50, help="training scenarios (1000 objects each)")
    parser.add_argument("--valid", type=int, default=5, help="validation scenarios, from the training families")
    parser.add_argument("--test", type=int, default=10, help="test scenarios, from the held-out families")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=4096, help="objects per step")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--draws", type=int, default=3, help="sampled schedules per test scenario")
    parser.add_argument("--seed", type=int, default=SearchConfig.seed)
    parser.add_argument("--output", default=str(RESULTS_DIR / "search_sgd"))
    args = parser.parse_args(argv)

    train_config = SearchConfig(num_orbits=args.orbits, dt=args.dt, split="train", rotate=True)
    test_config = SearchConfig(num_orbits=args.orbits, dt=args.dt, split="test", rotate=True)
    print("=" * 96)
    print("Search schedules by gradient ascent on expected detections")
    print("=" * 96)
    print(f"  {args.orbits:g} orbits, dt {args.dt:g} s, pointing held {args.slot:g} s | {args.bank} training, "
          f"{args.valid} validation, {args.test} test scenarios of {train_config.num_objects} objects")
    tick = time.perf_counter()
    train_tables = scenario_tables(train_config, [args.seed + 1000 + k for k in range(args.bank)])
    valid_tables = scenario_tables(train_config, [args.seed + 3000 + k for k in range(args.valid)])
    test_tables = scenario_tables(test_config, [args.seed + 5000 + k for k in range(args.test)])
    print(f"  pass tables built in {time.perf_counter() - tick:.0f} s")
    print("-" * 96)
    print("  share of findable objects found on test scenarios                    mix: train / valid / test")
    print(f"  {'fov':>5} {'detect':>7} {'random':>8} {'fixed':>8} {'mix':>8} {'orbit':>8} {'oracle':>8}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for detection in args.detection:
        for half_angle in args.fov:
            record, policies, selected = tune(
                train_config, half_angle, detection, train_tables, valid_tables, test_tables,
                slot=args.slot, steps=args.steps, batch_objects=args.batch,
                learning_rate=args.learning_rate, draws=args.draws, seed=args.seed)
            records.append(record)
            for rung, policy in policies.items():
                policy.save(schedule_path(output, half_angle, detection, args.dt, rung))
            policies[selected].save(schedule_path(output, half_angle, detection, args.dt))
    summary = output / f"summary_dt{args.dt:g}.json"
    summary.write_text(json.dumps({"orbits": args.orbits, "dt": args.dt, "slot": args.slot,
                                   "bank": args.bank, "valid": args.valid, "test": args.test,
                                   "steps": args.steps, "records": records}, indent=2))
    print("-" * 96)
    print(f"  wrote {summary} and the schedules beside it")


if __name__ == "__main__":
    main()
