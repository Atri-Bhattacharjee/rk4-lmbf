"""SMC_LMB_Tracker.update against an exhaustive enumeration of the joint hypotheses.

The golden and statistical fixtures cannot see errors in the association scaling (every recorded
existence probability saturates at 1.0, and kappa cancels in the normalised mixture), so this file
checks the update's algebra directly. tests/lmb_reference.py rebuilds the LMB posterior from its
definition -- every injective track-to-measurement map enumerated, weighted by
prod eta, marginalised by hand -- with the per-particle likelihoods and visibility taken from the
engine's public primitives. The engine must match it on:

- posterior existence, to rtol 1e-9, on every track;
- the posterior mixture, through the resampled cloud: systematic resampling keeps particle p in
  floor(N W_p) or ceil(N W_p) copies, so every multiplicity must be within one of N W_p;
- which measurements the best hypothesis left unassigned, through the births they cause.

Scenes are drawn so that hypotheses with *different* detection counts compete: P_D in [0.5, 0.95],
kappa placed near the detection/miss crossover of an observable pair, clouds straddling the edge of
a 20 km range-only sensor volume (partial coverage), and some tracks shared between two sensors.
A legacy (no sensor array) variant runs the same scenes with every particle visible.

Usage:
    python tests/test_existence_enumeration.py [--cases N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

from lmb_engine_loader import import_lmb_engine  # noqa: E402
import lmb_reference as ref  # noqa: E402

lmb = import_lmb_engine()

SEED = 20260930
RANGE = 20.0e3
RADIUS = 6.371e6 + 800.0e3
SPEED = np.sqrt(3.986004418e14 / RADIUS)
# Wide enough that a cloud of a few km gives non-vanishing likelihoods across most of it.
MEAS_SIGMAS = np.array([3000.0, 60.0, 0.15, 0.15, 0.02, 0.02])
K_BEST = 128           # > the 34 hypotheses of a 3 x 3 scene, so nothing is truncated
EXISTENCE_RTOL = 1e-9


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def sensor_state(angle):
    return np.array([RADIUS * np.cos(angle), RADIUS * np.sin(angle), 0.0,
                     -SPEED * np.sin(angle), SPEED * np.cos(angle), 0.0])


def build_sensors():
    sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
    for k, angle in enumerate((0.0, 0.3, 1.1)):
        sensors.add_unpointed(f"s{k}", sensor_state(angle))
    return sensors


def random_target(rng, sensor, inside=True):
    """A crossing target: 5-18 km from the sensor (inside), or 22-30 km (outside)."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    distance = rng.uniform(5e3, 18e3) if inside else rng.uniform(22e3, 30e3)
    relative_velocity = rng.normal(size=3)
    relative_velocity *= rng.uniform(1e3, 1.2e4) / np.linalg.norm(relative_velocity)
    return np.concatenate([sensor[:3] + distance * direction, sensor[3:] + relative_velocity])


def random_cloud(rng, centre, count, spread):
    states = centre + np.concatenate(
        [rng.normal(0.0, spread, size=(count, 3)), rng.normal(0.0, 20.0, size=(count, 3))], axis=1)
    weights = rng.uniform(0.5, 1.5, size=count)
    return states, weights / weights.sum()


def make_measurement(target, sensors, index):
    measurement = lmb.Measurement.fromCartesian(target, np.asarray(sensors.state(index)))
    measurement.timestamp_ = 0.0
    measurement.sensor_id_ = sensors.id(index)
    measurement.covariance_ = np.diag(MEAS_SIGMAS**2)
    return measurement


def make_tracker(kappa, p_detection, seed):
    propagator = lmb.TwoBodyPropagator(np.zeros((6, 6)), seed=seed)
    sensor_model = lmb.InOrbitSensorModel(*MEAS_SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(8, 0.5, np.diag(MEAS_SIGMAS**2), seed=seed)
    return lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 1.0, K_BEST, 0.0, kappa,
                               p_detection, 0.0, 1.0, seed=seed), sensor_model


def to_track(track_input, label_index):
    particles = []
    for state, weight in zip(track_input.states, track_input.weights):
        particle = lmb.Particle()
        particle.state_vector = state
        particle.weight = weight
        particles.append(particle)
    label = lmb.TrackLabel()
    label.birth_time = 0
    label.index = label_index
    return lmb.Track(label, track_input.existence, particles)


def draw_scene(rng, sensors):
    """1-3 tracks around sensor volumes, 0-3 measurements; returns (track inputs, measurements)."""
    num_tracks = int(rng.integers(1, 4))
    num_meas = int(rng.integers(0, 4))
    tracks = []
    for _ in range(num_tracks):
        k = int(rng.integers(0, sensors.size()))
        centre = random_target(rng, np.asarray(sensors.state(k)), inside=rng.random() < 0.7)
        count = int(rng.integers(12, 40))
        states, weights = random_cloud(rng, centre, count, spread=rng.uniform(2e3, 9e3))
        tracks.append(ref.TrackInput(existence=float(rng.uniform(0.2, 0.95)), states=states,
                                     weights=weights))
    measurements = []
    for _ in range(num_meas):
        # Mostly near an existing track's centre, so pairs are genuinely competitive.
        if tracks and rng.random() < 0.8:
            source = tracks[int(rng.integers(0, len(tracks)))]
            target = source.states.mean(axis=0) + np.concatenate(
                [rng.normal(0, 1500.0, 3), rng.normal(0, 30.0, 3)])
        else:
            target = random_target(rng, np.asarray(sensors.state(int(rng.integers(0, sensors.size())))))
        k = sensors.visible_sensor(target)
        if k < 0:
            continue
        measurements.append(make_measurement(target, sensors, k))
    return tracks, measurements


def pick_kappa(rng, sensor_model, tracks, measurements, sensors, p_detection):
    """Kappa near the detection/miss crossover of some observable pair, so hypotheses compete."""
    crossovers = []
    for track in tracks:
        vis = ref.particle_visibility(lmb, sensors, track.states)
        wn = track.weights / track.weights.sum()
        q = 1.0 if vis is None else float(wn[vis >= 0].sum())
        miss = 1.0 - track.existence * p_detection * q
        for measurement in measurements:
            g = ref.particle_likelihoods(lmb, sensor_model, track.states, measurement)
            if vis is not None:
                g = np.where(vis == sensors.index_of(measurement.sensor_id_), g, 0.0)
            likelihood = float(np.sum(wn * g))
            if likelihood > 0.0 and miss > 0.0:
                crossovers.append(track.existence * p_detection * likelihood / miss)
    if not crossovers:
        return 1e-12
    return float(np.exp(np.mean(np.log(crossovers)))) * 10.0 ** rng.uniform(-1.0, 1.0)


def check_case(chk, rng, case, use_sensors):
    sensors = build_sensors()
    tracks, measurements = draw_scene(rng, sensors)
    array = sensors if use_sensors else None
    p_detection = float(rng.uniform(0.5, 0.95))
    _, sensor_model = make_tracker(1.0, p_detection, SEED + case)
    kappa = pick_kappa(rng, sensor_model, tracks, measurements, array, p_detection)
    tracker, sensor_model = make_tracker(kappa, p_detection, SEED + case)

    expected = ref.posterior(lmb, sensor_model, tracks, measurements, array, p_detection, kappa)

    tracker.set_tracks([to_track(t, i) for i, t in enumerate(tracks)])
    if use_sensors:
        tracker.update(measurements, sensors)
    elif measurements:
        tracker.update(measurements)
    else:
        return 0  # the legacy empty step is a documented no-op; nothing to compare
    updated = tracker.get_tracks()

    # Prune threshold 0 keeps every input track, in order, ahead of any births.
    by_index = {track.label().index: track for track in updated[:len(tracks)]}

    where = f"case {case} ({'sensors' if use_sensors else 'legacy'}, {len(tracks)}x{len(measurements)})"
    chk.ok(len(by_index) == len(tracks), f"{where}: prune threshold 0 must keep every input track")

    for i, track_input in enumerate(tracks):
        track = by_index[i]
        actual = track.existence_probability()
        want = expected.existence[i]
        chk.ok(np.isclose(actual, want, rtol=EXISTENCE_RTOL, atol=1e-15),
               f"{where}: track {i} existence {actual!r} != reference {want!r}")
        resampled = np.asarray(track.particle_states())
        if expected.active[i] and (measurements or use_sensors):
            ok, worst = ref.expected_multiplicity_ok(track_input.states, resampled, expected.mixture[i])
            chk.ok(ok, f"{where}: track {i} resampled multiplicities off the reference mixture by "
                       f"{worst + 1.0:.3f} particles")
        else:
            chk.ok(np.array_equal(resampled, track_input.states),
                   f"{where}: track {i} is unobservable and its cloud must be untouched")

    # Births: one per measurement the best hypothesis left unassigned.
    births = len(updated) - len(tracks)
    unassigned = len(measurements) - sum(1 for j in expected.best_assignment if j >= 0)
    top = np.sort(expected.hypothesis_weights)[::-1]
    if len(top) < 2 or top[0] > top[1] * (1.0 + 1e-6):
        chk.ok(births == unassigned,
               f"{where}: {births} births, but the best hypothesis leaves {unassigned} unassigned")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=150)
    args = parser.parse_args()

    chk = Checker()
    rng = np.random.default_rng(SEED)
    compared = {True: 0, False: 0}
    for case in range(args.cases):
        for use_sensors in (True, False):
            compared[use_sensors] += check_case(chk, rng, case, use_sensors)
    if chk.count == 0:
        raise AssertionError("no assertions executed")
    print(f"PASS: test_existence_enumeration ({chk.count} assertions; {compared[True]} sensor-array "
          f"and {compared[False]} legacy scenes against the exhaustive reference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
