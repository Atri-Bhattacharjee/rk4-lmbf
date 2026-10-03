"""Tests for the filter-free search environment (python/search_env.py).

The environment replaces "step every object through the whole run and test its range to every
sensor" with a table built from the closed-form two-body solution around each node crossing. So
the checks are against the slow way of doing the same thing:

  K   the closed-form motion against the engine's RK4, the ring against ring_states, and the
      Kepler solver's residual;
  T   the table's samples against a brute-force detection loop with run_ring's rule -- on the
      default scenario (where the counts are also the ring run's own) and on the orbits the node
      windows are most likely to get wrong (eccentric, nearly equatorial);
  S   the segments: every sample starts one, a ten-times finer clock finds nothing outside them,
      and the straight path they assume is the true one to within metres;
  F   the field of view: the NumPy predicate against SensorArray.sees, the bins leaving no
      direction uncovered, and the segment test against dense sampling along the path;
  E   the episode bookkeeping: an object is found once, a schedule played in one pass equals the
      same schedule stepped, the oracle finds every object at its first pass, and custody only
      ever takes detections away;
  C   the scenario: run_ring's objects for the same seed, the family split, the rotation.

Usage:
    python tests/test_search_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

import search_env as se  # noqa: E402
from run_ring import RingConfig, derive_seeds, load_catalogue, ring_states  # noqa: E402
from search_baselines import OraclePolicy, random_schedule  # noqa: E402

SEED = 20260930


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)

    def raises(self, fn, exc, fragment: str, message: str) -> None:
        self.count += 1
        try:
            fn()
        except exc as error:
            if fragment not in str(error):
                raise AssertionError(f"{message}: message {str(error)!r} lacks {fragment!r}")
            return
        raise AssertionError(f"{message}: no {exc.__name__} raised")


def brute_force_samples(states: np.ndarray, config: se.SearchConfig) -> set:
    """{(step, object, sensor)} for every step an object is within range of a sensor: truth and
    sensors stepped with the engine's RK4 and tested the way run_ring tests them."""
    num_objects = len(states)
    current = np.vstack([states, ring_states(config.num_sensors, config.sensor_radius)])
    found = set()

    def detect(block: np.ndarray, first_step: int) -> None:
        position = block[:, :num_objects, :3]
        k, o = np.nonzero(np.abs(position[:, :, 2]) <= config.sensor_range)
        distance = np.linalg.norm(block[k, num_objects:, :3] - position[k, o][:, None, :], axis=2)
        row, sensor = np.nonzero(distance <= config.sensor_range)
        found.update(zip((first_step + k[row]).tolist(), o[row].tolist(), sensor.tolist()))

    detect(current[None], 0)
    step = 0
    while step < config.num_steps:
        count = min(200, config.num_steps - step)
        block = lmb.two_body_rk4_steps(current, config.dt, count)
        detect(block, step + 1)
        current = np.array(block[-1])
        step += count
    return found


def table_samples(table: se.PassTable) -> set:
    return set(zip(table.sample_step.tolist(), table.sample_object.tolist(), table.sample_sensor.tolist()))


def awkward_states(count: int) -> np.ndarray:
    """Catalogue orbits that stress the node windows: the most eccentric and the least inclined."""
    _, catalogue = load_catalogue(se.DEFAULT_CSV)
    orbits = se.orbits_from_states(catalogue)
    normal = np.cross(orbits.periapsis, orbits.quadrature)
    inclination = np.arccos(np.clip(normal[:, 2], -1.0, 1.0))
    rows = np.unique(np.concatenate([np.argsort(orbits.eccentricity)[-count // 2:],
                                     np.argsort(inclination)[:count // 2]]))
    return catalogue[rows]


# ---------------------------------------------------------------------------------------------
# K: closed-form motion
# ---------------------------------------------------------------------------------------------


def test_kepler(chk: Checker) -> None:
    rng = np.random.default_rng(SEED)
    mean = rng.uniform(-10.0, 10.0, size=200_000)
    eccentricity = rng.uniform(0.0, 0.8, size=mean.size)
    eccentric = se.solve_kepler(mean, eccentricity)
    residual = np.remainder(eccentric - eccentricity * np.sin(eccentric) - mean + np.pi, 2.0 * np.pi) - np.pi
    chk.ok(np.max(np.abs(residual)) < 1e-12, f"Kepler residual {np.max(np.abs(residual)):.2e} rad")

    # Against the engine's RK4, on a spread of the catalogue plus its awkward orbits, 5 orbits.
    _, catalogue = load_catalogue(se.DEFAULT_CSV)
    states = np.vstack([catalogue[rng.choice(len(catalogue), size=40, replace=False)], awkward_states(20)])
    config = se.SearchConfig(num_orbits=5.0)
    orbits = se.orbits_from_states(states)
    chk.ok(np.allclose(se.kepler_states(orbits, np.arange(len(states)), np.zeros(len(states))), states,
                       rtol=1e-12, atol=1e-6), "closed form does not return the initial states at t = 0")
    index = np.tile(np.arange(len(states)), 500)
    current, step, worst_position, worst_velocity = states, 0, 0.0, 0.0
    while step < config.num_steps:
        count = min(500, config.num_steps - step)
        block = lmb.two_body_rk4_steps(current, config.dt, count)
        times = np.repeat((step + 1 + np.arange(count)) * config.dt, len(states))
        closed = se.kepler_states(orbits, index[:count * len(states)], times).reshape(count, len(states), 6)
        worst_position = max(worst_position, float(np.max(np.linalg.norm(closed[..., :3] - block[..., :3], axis=2))))
        worst_velocity = max(worst_velocity, float(np.max(np.linalg.norm(closed[..., 3:] - block[..., 3:], axis=2))))
        current, step = np.array(block[-1]), step + count
    chk.ok(worst_position < 0.05, f"closed form vs RK4: {worst_position:.4f} m after 5 orbits")
    chk.ok(worst_velocity < 1e-4, f"closed form vs RK4: {worst_velocity:.2e} m/s after 5 orbits")

    # The ring: sensor k at angle 2 pi k / N + w t, on the circle ring_states starts it on.
    ring = ring_states(config.num_sensors, config.sensor_radius)
    sensors = np.arange(config.num_sensors)
    for when in (0.0, 1234.0):
        angle = se.sensor_angles(config, sensors, when)
        expected = (ring if when == 0.0 else
                    lmb.two_body_rk4_steps(ring, 1.0, int(when))[-1])
        mine = config.sensor_radius * np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
        chk.ok(np.max(np.linalg.norm(mine - expected[:, :3], axis=1)) < 1e-3,
               f"analytic ring is off the propagated ring at t = {when}")
    chk.ok(config.num_steps == int(round(RingConfig(num_orbits=5.0).duration / config.dt)),
           "SearchConfig and RingConfig disagree on the number of steps")


# ---------------------------------------------------------------------------------------------
# T: the table's samples against brute force
# ---------------------------------------------------------------------------------------------


def test_samples_against_brute_force(chk: Checker) -> None:
    # The default scenario is run_ring's: its 2-orbit run made 434 detections of 142 objects.
    config = se.SearchConfig(num_orbits=2.0, seed=SEED)
    scenario = se.make_scenario(config)
    table = se.build_pass_table(scenario.states, config)
    mine, reference = table_samples(table), brute_force_samples(scenario.states, config)
    chk.ok(mine == reference, f"default scenario: {len(mine - reference)} samples not in the brute-force "
           f"set, {len(reference - mine)} missing")
    chk.ok(len(mine) == 434 and int(table.visible_objects("sample").sum()) == 142,
           f"default 2-orbit scenario should give 434 samples of 142 objects, got {len(mine)} of "
           f"{int(table.visible_objects('sample').sum())}")
    chk.ok(len(mine) == len(table.sample_step), "duplicate sample rows")
    chk.ok(np.all(np.linalg.norm(table.sample_position, axis=1) <= config.sensor_range),
           "a sample lies outside the bubble")
    chk.ok(np.all(np.diff(table.sample_step) >= 0) and np.all(np.diff(table.segment_step) >= 0),
           "table rows are not in step order")

    # Eccentric and nearly equatorial orbits: long node windows, no node-radius shortcut.
    states = awkward_states(400)
    config = se.SearchConfig(num_orbits=3.0, num_objects=len(states))
    table = se.build_pass_table(states, config)
    mine, reference = table_samples(table), brute_force_samples(states, config)
    chk.ok(mine == reference, f"awkward orbits: {len(mine - reference)} samples not in the brute-force "
           f"set, {len(reference - mine)} missing")
    chk.ok(len(reference) > 50, f"awkward-orbit check is vacuous: only {len(reference)} samples")

    # A different ring: fewer sensors, longer reach, coarser clock.
    config = se.SearchConfig(num_orbits=1.0, num_sensors=30, sensor_range=50.0e3, dt=2.0, seed=7)
    scenario = se.make_scenario(config)
    mine = table_samples(se.build_pass_table(scenario.states, config))
    reference = brute_force_samples(scenario.states, config)
    chk.ok(mine == reference and len(reference) > 50,
           f"30-sensor ring: {len(mine ^ reference)} samples differ of {len(reference)}")


# ---------------------------------------------------------------------------------------------
# S: segments
# ---------------------------------------------------------------------------------------------


def test_segments(chk: Checker) -> None:
    config = se.SearchConfig(num_orbits=2.0, seed=SEED)
    scenario = se.make_scenario(config)
    table = se.build_pass_table(scenario.states, config)
    segments = set(zip(table.segment_step.tolist(), table.segment_object.tolist(), table.segment_sensor.tolist()))
    chk.ok(len(segments) == len(table.segment_step), "duplicate segment rows")
    chk.ok(table_samples(table) <= segments, "a sample is not the start of a segment")
    inside = table.segment_inside
    chk.ok(np.all((inside[:, 0] >= 0.0) & (inside[:, 1] <= 1.0) & (inside[:, 0] <= inside[:, 1])),
           "a segment's inside interval is not within [0, 1]")
    starts_inside = np.linalg.norm(table.segment_start, axis=1) <= config.sensor_range
    chk.ok(np.all(inside[starts_inside, 0] == 0.0), "a segment starting inside does not start at 0")
    chk.ok(table.num_passes > 0 and len(np.unique(table.segment_pass)) == table.num_passes,
           "pass indices are not 0..num_passes-1")

    # The straight path against the true one, at the middle of each segment's time in the bubble.
    orbits = se.orbits_from_states(scenario.states)
    middle = inside.mean(axis=1)
    when = (table.segment_step + middle) * config.dt
    true = se.local_offset(config, se.kepler_states(orbits, table.segment_object, when)[:, :3],
                           table.segment_sensor, when)
    straight = table.segment_start + middle[:, None] * (table.segment_end - table.segment_start)
    gap = np.linalg.norm(true - straight, axis=1)
    chk.ok(np.max(gap) < 10.0, f"straight segment is {np.max(gap):.1f} m off the true path")
    chk.ok(np.all(np.linalg.norm(true, axis=1) <= config.sensor_range + 10.0),
           "a segment's inside interval is outside the bubble on the true path")

    # A clock ten times finer: everything it sees well inside a bubble lies on a segment.
    fine_config = se.SearchConfig(num_orbits=2.0, seed=SEED, dt=0.1)
    fine = se.build_pass_table(scenario.states, fine_config)
    when = fine.sample_step * fine_config.dt
    deep = ((np.linalg.norm(fine.sample_position, axis=1) <= config.sensor_range - 100.0)
            & (when < config.num_steps * config.dt))
    step = np.floor(when[deep] / config.dt + 1e-9).astype(np.int64)
    fraction = when[deep] / config.dt - step
    lookup = {key: row for row, key in enumerate(zip(table.segment_step.tolist(), table.segment_object.tolist(),
                                                     table.segment_sensor.tolist()))}
    rows = [lookup.get(key, -1) for key in zip(step.tolist(), fine.sample_object[deep].tolist(),
                                               fine.sample_sensor[deep].tolist())]
    rows = np.asarray(rows)
    chk.ok(np.all(rows >= 0), f"{int(np.sum(rows < 0))} fine samples lie on no segment")
    chk.ok(np.all((fraction >= inside[rows, 0] - 0.02) & (fraction <= inside[rows, 1] + 0.02)),
           "a fine sample lies outside its segment's inside interval")
    chk.ok(int(deep.sum()) > 3000, f"fine-clock check is vacuous: {int(deep.sum())} samples")


# ---------------------------------------------------------------------------------------------
# F: field of view
# ---------------------------------------------------------------------------------------------


def test_field_of_view(chk: Checker) -> None:
    rng = np.random.default_rng(SEED)
    config = se.SearchConfig()
    reach = config.sensor_range

    # No direction is left uncovered, and at 45 degrees the six faces partition the sky.
    directions = rng.normal(size=(200_000, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for half_angle, count in ((45.0, 6), (30.0, 24), (20.0, 54), (10.0, 216), (5.0, 726)):
        bins = se.direction_bins(half_angle)
        chk.ok(len(bins) == count, f"{half_angle} deg: {len(bins)} bins, expected {count}")
        chk.ok(np.allclose(np.linalg.norm(bins.boresight, axis=1), 1.0)
               and np.allclose(np.einsum("ij,ij->i", bins.boresight, bins.up), 0.0, atol=1e-12)
               and np.allclose(np.cross(bins.up, bins.boresight), bins.width),
               f"{half_angle} deg: bins are not orthonormal frames")
        covered = np.zeros(len(directions), dtype=np.int64)
        for low in range(0, len(directions), 20_000):
            covered[low:low + 20_000] = se.point_hits(directions[low:low + 20_000], bins).sum(axis=1)
        chk.ok(np.all(covered >= 1), f"{half_angle} deg: {int(np.sum(covered == 0))} directions uncovered")
        if half_angle == 45.0:
            chk.ok(np.all(covered == 1), "45 deg: the six faces overlap")
    chk.raises(lambda: se.direction_bins(50.0), ValueError, "half_angle_deg", "half-angle above 45 accepted")

    # The NumPy predicate against the engine's, sensor anywhere on the ring.
    for half_angle in (45.0, 20.0, 5.0):
        bins = se.direction_bins(half_angle)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        speed = config.sensor_radius * config.angular_rate
        state = np.array([config.sensor_radius * np.cos(angle), config.sensor_radius * np.sin(angle), 0.0,
                          -speed * np.sin(angle), speed * np.cos(angle), 0.0])
        if half_angle == 45.0:
            # SensorFovConfig wants a half-angle below 90 degrees; 45 is fine, and its tangent is 1.
            chk.ok(abs(bins.tan_half_angle - 1.0) < 1e-12, "tan(45 deg) is not 1")
        sensors = lmb.SensorArray(lmb.SensorFovConfig(min_range=0.0, max_range=reach,
                                                      half_width=bins.half_angle, half_height=bins.half_angle))
        sensors.add("probe", state, se.local_to_eci(bins.boresight[0], angle)[0])
        for b in rng.choice(len(bins), size=min(len(bins), 12), replace=False):
            sensors.set_pointing(0, se.local_to_eci(bins.boresight[b], angle)[0],
                                 se.local_to_eci(bins.up[b], angle)[0])
            chk.ok(np.allclose(sensors.width_axis(0), se.local_to_eci(bins.width[b], angle)[0], atol=1e-12),
                   "width axis differs from the engine's")
            # Random points through and just beyond the bubble, plus points either side of each edge.
            points = rng.normal(size=(300, 3))
            points *= (rng.uniform(0.0, 1.1 * reach, size=300) / np.linalg.norm(points, axis=1))[:, None]
            edge = []
            for axis in (bins.width[b], bins.up[b]):
                for sign in (1.0, -1.0):
                    for nudge in (1.0 - 1e-6, 1.0 + 1e-6):
                        edge.append(0.5 * reach * (bins.boresight[b] + sign * nudge * bins.tan_half_angle * axis)
                                    / np.sqrt(1.0 + bins.tan_half_angle**2))
            points = np.vstack([points, edge])
            mine = (np.linalg.norm(points, axis=1) <= reach) & se.point_hits(points, bins)[:, b]
            engine = np.array([sensors.sees(0, state[:3] + se.local_to_eci(point, angle)[0]) for point in points])
            chk.ok(np.array_equal(mine, engine),
                   f"{half_angle} deg, bin {b}: {int(np.sum(mine != engine))} points disagree with SensorArray.sees")
            chk.ok(mine[-8:].tolist() == [True, False] * 4, "edge points are not classified in/out")

    # The segment test against points sampled along the path.
    table_config = se.SearchConfig(num_orbits=2.0, seed=SEED)
    table = se.build_pass_table(se.make_scenario(table_config).states, table_config)
    step = table.segment_end - table.segment_start
    for half_angle in (20.0, 5.0):
        bins = se.direction_bins(half_angle)
        hits = se.segment_hits(table.segment_start, table.segment_end, table.segment_inside, bins)
        chk.ok(np.all(hits >= se.point_hits(table.segment_start, bins)
                      & (np.linalg.norm(table.segment_start, axis=1) <= reach)[:, None]),
               f"{half_angle} deg: a sample detection is not a streak detection")
        dense = np.zeros_like(hits)
        for fraction in np.linspace(0.0, 1.0, 201):
            along = table.segment_inside[:, :1] + fraction * np.diff(table.segment_inside, axis=1)
            dense |= se.point_hits(table.segment_start + along * step, bins)
        chk.ok(not np.any(dense & ~hits), f"{half_angle} deg: a path point is in view but the segment is not")
        # The other way round, a hit can be a sliver between two sampled points: resample finely.
        row, column = np.nonzero(hits & ~dense)
        confirmed = 0
        for r, c in zip(row.tolist(), column.tolist()):
            along = np.linspace(table.segment_inside[r, 0], table.segment_inside[r, 1], 20_001)[:, None]
            one = se.DirectionBins(bins.half_angle, bins.boresight[c:c + 1], bins.up[c:c + 1], bins.width[c:c + 1])
            confirmed += bool(se.point_hits(table.segment_start[r] + along * step[r], one).any())
        chk.ok(confirmed >= 0.98 * len(row),
               f"{half_angle} deg: {len(row) - confirmed} of {int(hits.sum())} segment hits have no path point in view")
        chk.ok(hits.any(axis=1).all(), f"{half_angle} deg: a segment is visible from no bin")


# ---------------------------------------------------------------------------------------------
# E: episode bookkeeping
# ---------------------------------------------------------------------------------------------


def test_episode(chk: Checker) -> None:
    config = se.SearchConfig(num_orbits=6.0, seed=SEED)
    table = se.build_pass_table(se.make_scenario(config).states, config)
    rng = np.random.default_rng(SEED)

    for detection in se.DETECTION_MODELS:
        for half_angle in (45.0, 10.0):
            env = se.SearchEnv(table, fov_half_angle_deg=half_angle, detection=detection)
            chk.ok(env.num_slots == config.num_steps // 10 + 1, "wrong number of slots")
            schedule = random_schedule(env, rng)

            # Stepping: each object is reported once, at its first detection.
            env.reset()
            reported, last_seen = [], 0
            while not env.done:
                found = env.step(schedule[env.slot])
                reported.append(found)
                if int(env.seen.sum()) < last_seen:
                    raise AssertionError("an object became unseen")
                last_seen = int(env.seen.sum())
            stepped = env.result()
            reported = np.concatenate(reported)
            chk.ok(len(reported) == len(np.unique(reported)) == stepped.found,
                   f"{detection}: an object was reported more than once")
            chk.ok(np.array_equal(np.sort(reported), np.flatnonzero(env.seen)), "reported objects != seen objects")
            chk.raises(lambda: env.step(schedule[0]), RuntimeError, "episode is over", "stepped past the end")

            # The same schedule in one pass, and through run().
            whole = env.run_schedule(schedule)
            chk.ok(np.array_equal(whole.first_detection, stepped.first_detection, equal_nan=True),
                   f"{detection} {half_angle}: run_schedule differs from stepping")
            replay = env.run(lambda e: schedule[e.slot])
            chk.ok(np.array_equal(replay.first_detection, stepped.first_detection, equal_nan=True),
                   "run() differs from stepping")
            chk.ok(stepped.found <= stepped.bound and np.all(stepped.visible[env.seen]),
                   "found an object that never entered a bubble")
            chk.ok(np.array_equal(stepped.found_by([0.0, config.duration]), [int(np.sum(stepped.first_detection <= 0.0)),
                                                                             stepped.found]),
                   "discovery curve does not end at the number found")

            # The oracle finds each object in the slot it first enters a bubble, and beats random.
            oracle = env.run(OraclePolicy(env.num_sensors))
            first_step = np.full(table.num_objects, np.iinfo(np.int64).max)
            if detection == "sample":
                np.minimum.at(first_step, table.sample_object, table.sample_step)
            else:
                np.minimum.at(first_step, table.segment_object, table.segment_step)
            found_slot = oracle.first_detection[oracle.visible] // (env.slot_steps * config.dt)
            at_first = np.sum(found_slot == first_step[oracle.visible] // env.slot_steps)
            chk.ok(oracle.found >= 0.99 * oracle.bound and at_first >= 0.98 * oracle.bound,
                   f"{detection} {half_angle}: oracle found {oracle.found} of {oracle.bound}, "
                   f"{at_first} in the slot of their first pass")
            chk.ok(oracle.found >= stepped.found, "random pointing beat the oracle")
            chk.ok(oracle.mean_time_to_discovery <= stepped.mean_time_to_discovery,
                   "random pointing found objects sooner than the oracle")

            # Custody can only remove detections, and only costs a small share of sensor time.
            held = se.SearchEnv(table, fov_half_angle_deg=half_angle, detection=detection, custody=True)
            with_custody = held.run(lambda e: schedule[e.slot])
            later = np.where(np.isnan(with_custody.first_detection), np.inf, with_custody.first_detection)
            sooner = np.where(np.isnan(stepped.first_detection), np.inf, stepped.first_detection)
            chk.ok(np.all(later >= sooner), "custody made a detection happen earlier")
            chk.ok(0.0 < with_custody.custody_fraction < 0.01,
                   f"custody share of sensor-slots is {with_custody.custody_fraction:.4f}")
            chk.raises(lambda: held.run_schedule(schedule), ValueError, "custody off", "run_schedule with custody")

    env = se.SearchEnv(table, fov_half_angle_deg=45.0)
    chk.raises(lambda: env.step(np.zeros(3, dtype=int)), ValueError, "bins must be", "wrong-length action accepted")
    chk.raises(lambda: env.step(np.full(env.num_sensors, env.num_bins)), ValueError, "bins must be",
               "out-of-range bin accepted")
    chk.raises(lambda: se.SearchEnv(table, detection="radar"), ValueError, "detection must be",
               "unknown detection model accepted")

    # Every direction at once: with the six 45-degree faces, each object in a bubble is in exactly
    # one face's view at a sample, so the sample model summed over the six fixed pointings is the bound.
    env = se.SearchEnv(table, fov_half_angle_deg=45.0, detection="sample")
    seen_by_some_face = np.zeros(table.num_objects, dtype=bool)
    for b in range(6):
        result = env.run_schedule(np.full((env.num_slots, env.num_sensors), b))
        seen_by_some_face |= ~np.isnan(result.first_detection)
    chk.ok(np.array_equal(seen_by_some_face, env.visible), "six fixed faces together do not see every object")

    # The public state.
    env.reset()
    chk.ok(env.time == 0.0 and env.slot == 0 and not env.seen.any(), "reset() did not clear the episode")
    chk.ok(np.allclose(env.sensor_angles(), 2.0 * np.pi * np.arange(env.num_sensors) / env.num_sensors),
           "sensor angles at t = 0")
    objects, sensors, hits = env.bubble_contents()
    chk.ok(len(objects) == len(sensors) == len(hits) and hits.shape[1:] == (env.num_bins,),
           "bubble_contents shapes")


# ---------------------------------------------------------------------------------------------
# C: scenario
# ---------------------------------------------------------------------------------------------


def test_scenario(chk: Checker) -> None:
    ids, catalogue = load_catalogue(se.DEFAULT_CSV)

    # split "all", rotate off: run_ring's objects for the seed.
    config = se.SearchConfig(seed=SEED)
    scenario = se.make_scenario(config)
    chosen = np.sort(np.random.default_rng(derive_seeds(SEED)["sample"]).choice(
        len(catalogue), size=config.num_objects, replace=False))
    chk.ok(np.array_equal(scenario.rows, chosen) and np.array_equal(scenario.states, catalogue[chosen])
           and np.array_equal(scenario.object_ids, ids[chosen]), "default scenario is not run_ring's")
    chk.ok(not np.array_equal(se.make_scenario(config, seed=SEED + 1).rows, scenario.rows),
           "a different seed gave the same objects")

    # Family split: disjoint in families, complete, repeatable, about the asked share.
    train, test = se.family_split(se.DEFAULT_CSV, 0.2, 0)
    family = se._family_ids(str(se.DEFAULT_CSV))
    chk.ok(len(np.intersect1d(family[train], family[test])) == 0, "a family is on both sides of the split")
    chk.ok(len(train) + len(test) == len(catalogue) and len(np.intersect1d(train, test)) == 0,
           "the split is not a partition of the catalogue")
    chk.ok(0.15 < len(test) / len(catalogue) < 0.25, f"test share is {len(test) / len(catalogue):.3f}")
    again = se.family_split(se.DEFAULT_CSV, 0.2, 0)
    chk.ok(np.array_equal(train, again[0]) and np.array_equal(test, again[1]), "the split is not repeatable")
    chk.ok(family.max() + 1 < len(catalogue), "the catalogue has no repeated orbits; the family split is moot")
    for split, rows in (("train", train), ("test", test)):
        drawn = se.make_scenario(se.SearchConfig(split=split, seed=SEED)).rows
        chk.ok(np.all(np.isin(drawn, rows)), f"a {split} scenario drew a row from outside its split")
    chk.raises(lambda: se.make_scenario(se.SearchConfig(split="validation")), ValueError, "split must be",
               "unknown split accepted")
    chk.raises(lambda: se.make_scenario(se.SearchConfig(split="test", num_objects=len(test) + 1)), ValueError,
               "asked for", "more objects than the split holds")

    # Rotation about the Earth's axis: same orbit shape and inclination, new node longitude.
    plain = se.make_scenario(se.SearchConfig(split="train", seed=SEED))
    turned = se.make_scenario(se.SearchConfig(split="train", seed=SEED, rotate=True))
    chk.ok(np.array_equal(plain.rows, turned.rows), "rotation changed which objects were drawn")
    for first in (0, 3):
        chk.ok(np.allclose(np.linalg.norm(plain.states[:, first:first + 3], axis=1),
                           np.linalg.norm(turned.states[:, first:first + 3], axis=1), rtol=1e-12)
               and np.array_equal(plain.states[:, first + 2], turned.states[:, first + 2]),
               "rotation changed a length or a z component")
    before, after = se.orbits_from_states(plain.states), se.orbits_from_states(turned.states)
    chk.ok(np.allclose(before.semi_major, after.semi_major, rtol=1e-12)
           and np.allclose(before.eccentricity, after.eccentricity, atol=1e-12)
           and np.allclose(np.cross(before.periapsis, before.quadrature)[:, 2],
                           np.cross(after.periapsis, after.quadrature)[:, 2], atol=1e-12),
           "rotation changed an orbit's shape or inclination")
    longitude = np.arctan2(turned.states[:, 1], turned.states[:, 0]) - np.arctan2(plain.states[:, 1], plain.states[:, 0])
    chk.ok(np.std(np.remainder(longitude, 2.0 * np.pi)) > 1.0, "rotation angles are not spread out")


def main() -> None:
    chk = Checker()
    print("search environment")
    for name, test in (("K closed-form motion", test_kepler),
                       ("T samples vs brute force", test_samples_against_brute_force),
                       ("S segments", test_segments),
                       ("F field of view", test_field_of_view),
                       ("E episode", test_episode),
                       ("C scenario", test_scenario)):
        before = chk.count
        test(chk)
        print(f"  {name}: {chk.count - before} assertions")
    print(f"PASS: test_search_env ({chk.count} assertions)")


if __name__ == "__main__":
    main()
