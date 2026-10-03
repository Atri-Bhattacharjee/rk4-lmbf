"""Tests for the taskers that aim the pointed ring (python/tasking.py), with the filter in the loop.

A tasker that aimed the sensors slightly wrong would still produce a run and a tracking fraction.
What pins it down is that the filter-free search environment already says, event for event, what
any pointing must detect:

  A   aiming: local-frame directions reach the engine as the ECI boresight and roll the search
      environment assumes;
  O   oracle pointing at 45 degrees detects exactly what sensors seeing in every direction detect;
  S   a search schedule run through the filter makes exactly the detections the environment gives
      for the same schedule;
  P   custody prediction: particles on a true orbit, handed over at a later time, are predicted
      into the bubble at the steps and places the pass table has that object;
  C   the custody rule: a known track's pass takes its sensor out of search and aims it where the
      most cloud is (never less than any direction bin would see), a track below the existence
      threshold or with no pass coming changes nothing;
  E   end to end on a short run: custody re-detects known objects that search alone misses and
      follows a new object through its first pass, costs a sliver of sensor time, and leaves the
      lazy-propagation gate sound.

Usage:
    python tests/test_tasking.py
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

import run_ring  # noqa: E402
import search_env as se  # noqa: E402
import tasking  # noqa: E402
from run_tasked import pass_statistics  # noqa: E402

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


def event_set(events: np.ndarray, dt: float) -> set:
    return set(zip(np.rint(events[:, 0] / dt).astype(int).tolist(), events[:, 1].astype(int).tolist(),
                   events[:, 2].astype(int).tolist()))


def scenario_table(config: run_ring.RingConfig) -> se.PassTable:
    search = tasking.search_config_for(config)
    return se.build_pass_table(se.make_scenario(search).states, search)


# ---------------------------------------------------------------------------------------------
# A: aiming
# ---------------------------------------------------------------------------------------------


def test_aiming(chk: Checker) -> None:
    config = run_ring.RingConfig(fov_half_angle_deg=20.0)
    search = tasking.search_config_for(config)
    chk.ok(search.num_steps == int(round(config.duration / config.dt)) and search.sensor_radius == config.sensor_radius,
           "search_config_for changed the ring or the clock")
    states = lmb.two_body_rk4_steps(run_ring.ring_states(config.num_sensors, config.sensor_radius), 1.0, 777)[-1]
    sensors = run_ring.build_sensor_array(config, states)
    bins = se.direction_bins(20.0)
    chosen = np.random.default_rng(SEED).integers(0, len(bins), size=config.num_sensors)
    tasking.apply_pointing(sensors, states, bins.boresight[chosen], bins.up[chosen])
    angle = se.sensor_angles(search, np.arange(config.num_sensors), 777.0)
    for k in (0, 17, 99):
        chk.ok(np.allclose(sensors.boresight(k), se.local_to_eci(bins.boresight[chosen[k]], angle[k])[0], atol=1e-9)
               and np.allclose(sensors.up(k), se.local_to_eci(bins.up[chosen[k]], angle[k])[0], atol=1e-9)
               and np.allclose(sensors.width_axis(k), se.local_to_eci(bins.width[chosen[k]], angle[k])[0], atol=1e-9),
               f"sensor {k}: the engine's frame is not the bin's frame turned into ECI")

    # aim(): orthonormal frames for any boresight, including straight along cross-track.
    directions = np.vstack([np.random.default_rng(SEED).normal(size=(50, 3)), [[0, 0, 1.0], [0, 0, -3.0], [2.0, 0, 0]]])
    frames = se.aim(directions)
    chk.ok(np.allclose(np.linalg.norm(frames.boresight, axis=1), 1.0) and np.allclose(np.linalg.norm(frames.up, axis=1), 1.0)
           and np.allclose(np.einsum("ij,ij->i", frames.boresight, frames.up), 0.0, atol=1e-12)
           and np.allclose(np.cross(frames.up, frames.boresight), frames.width),
           "aim() frames are not orthonormal")
    chk.ok(np.allclose(np.cross(frames.boresight, directions), 0.0, atol=1e-9)
           and np.all(np.einsum("ij,ij->i", frames.boresight, directions) > 0), "aim() changed a direction")
    # A point on the boresight is in view at any half-angle; one behind it never is.
    narrow = se.with_half_angle(frames, 1.0)
    chk.ok(np.all(np.diag(se.point_hits(directions, narrow))) and not np.any(np.diag(se.point_hits(-directions, narrow))),
           "a sensor aimed at a point does not see it")

    chk.raises(lambda: tasking.OracleTasker(run_ring.RingConfig()), ValueError, "fov_half_angle_deg",
               "a tasker accepted unpointed sensors")
    chk.raises(lambda: run_ring.run(config, verbose=False), ValueError, "need a tasker",
               "pointed sensors ran with nothing aiming them")
    chk.raises(lambda: tasking.ScheduleTasker(config, np.zeros((3, 100), dtype=int)), ValueError, "schedule must be",
               "a schedule of the wrong length was accepted")


# ---------------------------------------------------------------------------------------------
# O, S: the filter's detections against the environment's
# ---------------------------------------------------------------------------------------------


def test_detections_against_environment(chk: Checker) -> None:
    # Oracle pointing with the widest field of view: every object in a bubble is on a boresight.
    base = run_ring.RingConfig(num_orbits=1.0, seed=SEED, profile=False)
    reference = run_ring.run(base, verbose=False)
    pointed = run_ring.RingConfig(num_orbits=1.0, seed=SEED, profile=False, fov_half_angle_deg=45.0)
    oracle = tasking.OracleTasker(pointed)
    log = run_ring.run(pointed, verbose=False, tasker=oracle)
    chk.ok(np.array_equal(log["detection_events"], reference["detection_events"]) and len(log["detection_events"]) > 100,
           "oracle pointing does not reproduce the all-seeing detections")
    chk.ok(len(log["births"]) == len(reference["births"]), "oracle pointing changed the number of births")
    chk.ok(oracle.counts["custody"] == len(log["detection_events"]),
           "the oracle's occupied sensor-steps are not the detections")
    chk.ok(log["scoring"] == "detected" and "tasking" in log["timers"], "run log lacks the tasking fields")

    # A fixed-in-advance schedule: the same detections as the search environment, for narrow and wide.
    for half_angle, orbits in ((45.0, 1.0), (20.0, 2.0)):
        config = run_ring.RingConfig(num_orbits=orbits, seed=SEED, profile=False, fov_half_angle_deg=half_angle)
        table = scenario_table(config)
        tasker = tasking.ScheduleTasker.random(config, seed=5)
        log = run_ring.run(config, verbose=False, tasker=tasker,
                           scored_objects=np.flatnonzero(table.visible_objects("sample")))
        env = se.SearchEnv(table, fov_half_angle_deg=half_angle, detection="sample")
        objects, sensors, steps, hits = se.detection_rows(table, env.bins, "sample")
        seen = hits[np.arange(len(objects)), tasker.schedule[steps // tasker.slot_steps, sensors]]
        expected = set(zip(steps[seen].tolist(), objects[seen].tolist(), sensors[seen].tolist()))
        chk.ok(event_set(log["detection_events"], config.dt) == expected and len(expected) > 5,
               f"{half_angle} deg: the filter made {len(log['detection_events'])} detections, the environment {len(expected)}")
        result = env.run_schedule(tasker.schedule)
        first = np.full(config.num_objects, np.inf)
        np.minimum.at(first, log["detection_events"][:, 1].astype(int), log["detection_events"][:, 0])
        chk.ok(np.array_equal(np.where(np.isfinite(first), first, np.nan), result.first_detection, equal_nan=True),
               f"{half_angle} deg: first-detection times differ from the environment's")
        chk.ok(log["scoring"] == "given" and np.all(log["num_truths"] == int(table.visible_objects("sample").sum())),
               "metrics were not scored against the given objects")
        chk.ok(tasker.counts == {"search": config.num_sensors * (env.config.num_steps + 1), "custody": 0},
               "a schedule tasker counted custody steps")


# ---------------------------------------------------------------------------------------------
# P: custody prediction
# ---------------------------------------------------------------------------------------------


def test_prediction(chk: Checker) -> None:
    config = se.SearchConfig(num_orbits=4.0, seed=SEED)
    scenario = se.make_scenario(config)
    table = se.build_pass_table(scenario.states, config)
    orbits = se.orbits_from_states(scenario.states)
    visible = np.flatnonzero(table.visible_objects("sample"))[:60]
    # Hand each object over at its own, different, time; predict a window that starts later still.
    epochs = np.linspace(100.0, 5000.0, len(visible))
    handed = se.kepler_states(orbits, visible, epochs)
    first_step, last_step = 5200, 20000
    index, sensor, step, position = se.predict_samples(handed, epochs, config, first_step, last_step)
    window = (table.sample_step >= first_step) & (table.sample_step <= last_step) & np.isin(table.sample_object, visible)
    expected = {(int(o), int(s), int(k)): p for o, s, k, p in zip(
        table.sample_object[window], table.sample_sensor[window], table.sample_step[window], table.sample_position[window])}
    predicted = {(int(visible[i]), int(s), int(k)): p for i, s, k, p in zip(index, sensor, step, position)}
    chk.ok(set(predicted) == set(expected) and len(expected) > 50,
           f"predicted {len(predicted)} samples, the pass table has {len(expected)} in the window")
    worst = max(np.linalg.norm(predicted[key] - expected[key]) for key in expected)
    chk.ok(worst < 0.01, f"predicted positions are {worst:.4f} m off the pass table")
    chk.ok(np.all(np.diff(index) >= 0), "predictions are not ordered by index")
    empty = se.predict_samples(handed[:0], epochs[:0], config, 0, 100)
    chk.ok(len(empty[0]) == 0 and empty[3].shape == (0, 3), "no states should predict nothing")
    chk.ok(len(se.predict_samples(handed, epochs, config, 30, 20)[0]) == 0, "an empty window should predict nothing")


# ---------------------------------------------------------------------------------------------
# C: the custody rule
# ---------------------------------------------------------------------------------------------


def cloud_tracker(states: np.ndarray, existences) -> "lmb.SMC_LMB_Tracker":
    """A tracker holding one track per (K, 6) cloud in ``states``, at time 0."""
    tracks = []
    for k, (cloud, existence) in enumerate(zip(states, existences)):
        particles = []
        for state in cloud:
            particle = lmb.Particle()
            particle.state_vector = state
            particle.weight = 1.0 / len(cloud)
            particles.append(particle)
        label = lmb.TrackLabel()
        label.birth_time = 0
        label.index = k
        tracks.append(lmb.Track(label, existence, particles))
    tracker = lmb.SMC_LMB_Tracker(lmb.TwoBodyPropagator(np.eye(6) * 1e-18, seed=1),
                                  lmb.InOrbitSensorModel(100.0, 1.0, 1e-6, 1e-6, 1e-8, 1e-8),
                                  lmb.AdaptiveBirthModel(10, 0.5, np.eye(6), seed=2), 0.999, 1, 1e-3, 1e-15, 0.99, seed=3)
    tracker.set_tracks(tracks)
    return tracker


def test_custody_rule(chk: Checker) -> None:
    half_angle = 10.0
    config = run_ring.RingConfig(num_orbits=1.0, seed=SEED, fov_half_angle_deg=half_angle)
    search = tasking.search_config_for(config)
    truth = se.make_scenario(search).states
    table = se.build_pass_table(truth, search)
    # Two objects with a pass well into the orbit, and one that never passes at all.
    later = table.sample_step > 600
    passing = np.unique(table.sample_object[later])[:2]
    absent = int(np.flatnonzero(~table.visible_objects("sample"))[0])
    rng = np.random.default_rng(SEED)
    spread = np.array([400.0, 400.0, 400.0, 0.4, 0.4, 0.4])

    def cloud(index: int) -> np.ndarray:
        return truth[index] + rng.normal(size=(300, 6)) * spread

    # Track 0: confirmed and passing. Track 1: passing but below the existence threshold. Track 2: no pass.
    tracker = cloud_tracker([cloud(passing[0]), cloud(passing[1]), cloud(absent)], [0.9, 0.2, 0.9])
    schedule = tasking.ScheduleTasker.fixed(config, 0)
    custody = tasking.CustodyTasker(config, schedule, particles=128)
    sensor_states = run_ring.ring_states(config.num_sensors, config.sensor_radius)
    plain_boresight, _ = schedule.directions(step=0)

    rows = np.flatnonzero(later & (table.sample_object == passing[0]))
    step, sensor = int(table.sample_step[rows[0]]), int(table.sample_sensor[rows[0]])
    custody.directions(step=0, tracker=tracker, sensor_states=sensor_states, truth=truth)      # first call: predicts
    chk.ok(set(custody._predicted) == {0}, f"tracks with predictions: {sorted(custody._predicted)}, expected only track 0")
    boresight, up = custody.directions(step=step, tracker=tracker, sensor_states=sensor_states, truth=truth)
    moved = np.flatnonzero(np.any(boresight != plain_boresight, axis=1))
    chk.ok(moved.tolist() == [sensor], f"custody moved sensors {moved.tolist()}, the pass is at sensor {sensor}")
    chk.ok(abs(np.dot(boresight[sensor], up[sensor])) < 1e-12 and abs(np.linalg.norm(boresight[sensor]) - 1.0) < 1e-12,
           "custody pointing is not a unit boresight with a perpendicular up")

    # The direction it chose sees the true object, and at least as much of the cloud as any bin.
    at_step = custody._rows_at(step)
    position, weight = custody._position[at_step], custody._weight[at_step]
    chk.ok(len(position) > 10 and np.isclose(weight.sum(), 0.9 * len(position) / 128), "predicted cloud weights")
    chosen = se.DirectionBins(np.deg2rad(half_angle), boresight[sensor][None], up[sensor][None],
                              np.cross(up[sensor], boresight[sensor])[None])
    in_view = float(weight @ se.point_hits(position, chosen)[:, 0])
    per_bin = weight @ se.point_hits(position, se.direction_bins(half_angle))
    chk.ok(in_view >= per_bin.max() - 1e-12, f"custody sees {in_view:.3f} of the cloud, the best bin {per_bin.max():.3f}")
    chk.ok(se.point_hits(table.sample_position[rows[:1]], chosen)[0, 0], "custody is not looking at the true object")
    log = custody.custody_log[-1]
    chk.ok(log[0] == step and log[1] == sensor and np.isclose(log[3], in_view), "custody log entry")

    # A step with no predicted cloud anywhere: pure search.
    quiet = next(k for k in range(1, 600) if custody._rows_at(k).stop == custody._rows_at(k).start)
    boresight, _ = custody.directions(step=quiet, tracker=tracker, sensor_states=sensor_states, truth=truth)
    chk.ok(np.array_equal(boresight, plain_boresight), "custody moved a sensor with no cloud in any bubble")
    chk.ok(custody.counts["custody"] == 1 and custody.counts["search"] == 3 * config.num_sensors - 1,
           f"sensor-step counts {custody.counts}")

    # A threshold above the cloud's weight leaves the sensor searching.
    strict = tasking.CustodyTasker(config, tasking.ScheduleTasker.fixed(config, 0), particles=128, min_weight=5.0)
    strict.directions(step=0, tracker=tracker, sensor_states=sensor_states, truth=truth)
    boresight, _ = strict.directions(step=step, tracker=tracker, sensor_states=sensor_states, truth=truth)
    chk.ok(np.array_equal(boresight, plain_boresight), "custody ignored its threshold")


# ---------------------------------------------------------------------------------------------
# E: end to end
# ---------------------------------------------------------------------------------------------


def test_end_to_end(chk: Checker) -> None:
    config = run_ring.RingConfig(num_orbits=5.0, seed=SEED, profile=False, metric_interval=600.0, fov_half_angle_deg=45.0)
    table = scenario_table(config)
    findable = np.flatnonzero(table.visible_objects("sample"))
    sensor_steps = config.num_sensors * (tasking.search_config_for(config).num_steps + 1)

    search_only = tasking.ScheduleTasker.random(config, seed=5)
    plain = run_ring.run(config, verbose=False, tasker=search_only, scored_objects=findable)
    custody = tasking.CustodyTasker(config, tasking.ScheduleTasker.random(config, seed=5))
    held = run_ring.run(config, verbose=False, tasker=custody, scored_objects=findable)
    before = pass_statistics(table, plain["detection_events"], config.dt, run_ring.PASS_GAP)
    after = pass_statistics(table, held["detection_events"], config.dt, run_ring.PASS_GAP)

    chk.ok(before["known_passes"] >= 20, f"too few known passes to judge custody: {before['known_passes']}")
    chk.ok(after["known_pass_detection_rate"] > 0.85 > 0.5 > before["known_pass_detection_rate"],
           f"known passes detected: {before['known_pass_detection_rate']:.2f} without custody, "
           f"{after['known_pass_detection_rate']:.2f} with")
    chk.ok(after["first_pass_samples_detected"] > before["first_pass_samples_detected"] + 0.4,
           f"samples on the pass that finds an object: {before['first_pass_samples_detected']:.2f} without follow, "
           f"{after['first_pass_samples_detected']:.2f} with")
    chk.ok(0 < custody.counts["custody"] < 0.005 * sensor_steps and sum(custody.counts.values()) == sensor_steps,
           f"custody took {custody.counts['custody']} of {sensor_steps} sensor-steps")
    chk.ok(held["num_assigned"][-1] > plain["num_assigned"][-1],
           f"custody did not raise the number of tracked objects ({plain['num_assigned'][-1]} -> {held['num_assigned'][-1]})")
    # Search itself is untouched: the objects found are the schedule's, give or take the handful of
    # slots custody took over.
    found_plain = len(np.unique(plain["detection_events"][:, 1]))
    found_held = len(np.unique(held["detection_events"][:, 1]))
    chk.ok(abs(found_held - found_plain) <= 0.03 * found_plain, f"custody changed objects found: {found_plain} -> {found_held}")
    keys = held["births"][:, 1]
    chk.ok(len(np.unique(keys)) == len(keys), "a track was born twice")

    # The lazy-propagation gate tests range only, so pointing must not make it miss anything.
    audited = run_ring.RingConfig(num_orbits=0.8, seed=SEED, profile=False, fov_half_angle_deg=20.0, gate_audit=True)
    log = run_ring.run(audited, verbose=False, tasker=tasking.CustodyTasker(audited, tasking.ScheduleTasker.random(audited, seed=5)))
    checks, violations = log["gate_audit"][0], log["gate_audit"][1]
    chk.ok(checks > 0 and violations == 0, f"gate audit with pointed sensors: {violations} violations in {checks} checks")


def main() -> None:
    chk = Checker()
    print("tasking")
    for name, test in (("A aiming", test_aiming),
                       ("O/S detections vs environment", test_detections_against_environment),
                       ("P custody prediction", test_prediction),
                       ("C custody rule", test_custody_rule),
                       ("E end to end", test_end_to_end)):
        before = chk.count
        test(chk)
        print(f"  {name}: {chk.count - before} assertions")
    print(f"PASS: test_tasking ({chk.count} assertions)")


if __name__ == "__main__":
    main()
