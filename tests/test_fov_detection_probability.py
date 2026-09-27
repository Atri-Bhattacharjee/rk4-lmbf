"""Tests for the field-of-view-scaled detection probability in src/smc_lmb_tracker.cpp.

The filter's effective detection probability is the configured P_D scaled by the fraction of a
track's particle cloud that a sensor can actually see:

    pd_det(i, j)  = P_D * q[i][sensor that produced measurement j]
    pd_miss(i)    = P_D * q_union[i]

None of the intermediate quantities are exposed, so these tests go after the one observable the
math fully determines: the posterior existence probability. ``expected_existence`` below is an
independent NumPy reimplementation of the per-track update -- cost matrix, log-sum-exp over the
hypotheses, mixture collapse, Bernoulli update -- driven by the *public* likelihood and coverage
accessors. It never calls the tracker, so agreement is evidence about the C++ formula rather than
a restatement of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

from lmb_engine_loader import import_lmb_engine  # noqa: E402
import harness_scenario as hs  # noqa: E402

lmb = import_lmb_engine()

FIXED_SEED = 20260921

P_DETECTION = 0.9
P_SURVIVAL = 0.99
CLUTTER_INTENSITY = 1.0e-6
K_BEST = 8
PRUNE_THRESHOLD = 0.0
FILTER_SIGMAS = np.array([5000.0, 500.0, 1e-2, 1e-2, 1e-3, 1e-3])
FILTER_COVARIANCE = np.diag(FILTER_SIGMAS**2)

RANGE = 1.0e6
HALF_ANGLE = np.deg2rad(5.0)


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)

    def close(self, actual, expected, message: str, rtol: float = 1e-12) -> None:
        self.count += 1
        if not np.isclose(actual, expected, rtol=rtol, atol=0.0):
            raise AssertionError(f"{message}: got {actual!r}, expected {expected!r}")

    def raises(self, fn, exc, fragment: str, message: str) -> None:
        self.count += 1
        try:
            fn()
        except exc as error:
            if fragment not in str(error):
                raise AssertionError(f"{message}: message {str(error)!r} lacks {fragment!r}")
            return
        raise AssertionError(f"{message}: no {exc.__name__} raised")


# ---------------------------------------------------------------------------------------------
# Independent reference for a one-track filter's existence update
# ---------------------------------------------------------------------------------------------


def expected_existence(r, likelihoods, pd_det, pd_miss, kappa=CLUTTER_INTENSITY, k_best=K_BEST):
    """Posterior existence probability of a single track, rebuilt from scratch in NumPy.

    Mirrors the documented formulation: an augmented 1 x (M+1) cost matrix, k-best assignment,
    log-sum-exp hypothesis weights, then the mixture collapsed onto its distinct associations. The
    per-particle vectors drop out of the total: each measurement's normalised association weights
    sum to 1, and so do the (normalised) particle weights, so sum_weights is just the sum of the
    association coefficients plus the miss coefficient.
    """
    likelihoods = np.asarray(likelihoods, dtype=np.float64)
    pd_det = np.asarray(pd_det, dtype=np.float64)
    num_meas = likelihoods.size

    cost = np.full((1, num_meas + 1), 1e9, dtype=np.float64)
    for j in range(num_meas):
        cost[0, j] = -np.log(max(pd_det[j] * likelihoods[j] / kappa, 1e-12))
    cost[0, num_meas] = -np.log(max(1.0 - pd_miss, 1e-12))

    hypotheses = lmb.solve_assignment(cost, k_best)
    log_weights = np.array([-h.weight for h in hypotheses], dtype=np.float64)
    normalised = np.exp(log_weights - log_weights.max())
    normalised /= normalised.sum()

    coefficients = np.zeros(num_meas, dtype=np.float64)
    miss_coefficient = 0.0
    for hypothesis, weight in zip(hypotheses, normalised):
        association = hypothesis.associations[0]
        if 0 <= association < num_meas:
            coefficients[association] += weight
        elif association == -1 or num_meas <= association < num_meas + 1:
            miss_coefficient += weight

    for j in range(num_meas):
        coefficients[j] *= pd_det[j] * likelihoods[j] / kappa
    miss_coefficient *= 1.0 - pd_miss

    sum_weights = float(coefficients.sum() + miss_coefficient)
    return (r * sum_weights) / (1.0 - r + r * sum_weights)


def expected_existence_miss_only(r, pd_miss):
    """Existence update for a step in which no sensor reported anything."""
    sum_weights = 1.0 - pd_miss
    return (r * sum_weights) / (1.0 - r + r * sum_weights)


# ---------------------------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------------------------


def state(position, velocity=(0.0, 0.0, 0.0)):
    return np.concatenate([np.asarray(position, dtype=np.float64),
                           np.asarray(velocity, dtype=np.float64)])


def make_track(positions, existence=0.6):
    """A track whose particle weights are normalised, as a resampled cloud's always are."""
    positions = np.asarray(positions, dtype=np.float64)
    weight = 1.0 / len(positions)
    particles = []
    for position in positions:
        particle = lmb.Particle()
        particle.state_vector = state(position)
        particle.weight = weight
        particles.append(particle)
    return lmb.Track(lmb.TrackLabel(), existence, particles)


def make_tracker(seed=FIXED_SEED, p_detection=P_DETECTION):
    propagator = lmb.TwoBodyPropagator(np.eye(6) * 1e-18, seed=seed)
    sensor_model = lmb.InOrbitSensorModel(*FILTER_SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(16, 0.5, np.diag([1000.0**2, 500.0**2, 1e-4**2,
                                                          1e-4**2, 5e-5**2, 5e-5**2]), seed=seed)
    return lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model, P_SURVIVAL, K_BEST,
                               PRUNE_THRESHOLD, CLUTTER_INTENSITY, p_detection, 0.0, 1.0, seed=seed)


def make_measurement(target_position, sensor_state, sensor_id):
    measurement = lmb.Measurement.fromCartesian(state(target_position), sensor_state)
    measurement.timestamp_ = 0.0
    measurement.sensor_id_ = sensor_id
    measurement.covariance_ = FILTER_COVARIANCE.copy()
    return measurement


def two_lobe_sensors(half_angle=HALF_ANGLE, max_range=np.inf):
    """Two co-located sensors 90 degrees apart, so their narrow volumes are disjoint.

    Co-locating them means a measurement's geometry -- and therefore its likelihood -- is identical
    whichever sensor it is attributed to, which is what isolates the attribution itself.
    """
    fov = lmb.SensorFovConfig(max_range=max_range, half_width=half_angle, half_height=half_angle)
    sensors = lmb.SensorArray(fov)
    sensors.add("east", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0]))
    sensors.add("north", state((0.0, 0.0, 0.0)), np.array([0.0, 1.0, 0.0]))
    return sensors


def split_cloud(num_east, num_north):
    """A cloud with exactly num_east particles down +x and num_north down +y."""
    return np.array([[RANGE, 0.0, 0.0]] * num_east + [[0.0, RANGE, 0.0]] * num_north)


# ---------------------------------------------------------------------------------------------
# B1: the effective P_D of a detection is the fraction inside THAT sensor's volume
# ---------------------------------------------------------------------------------------------


def check_detection_uses_the_right_sensor(chk: Checker) -> None:
    sensors = two_lobe_sensors()
    positions = split_cloud(70, 30)

    fractions = np.asarray(sensors.coverage_fractions(make_track(positions)))
    chk.ok(np.allclose(fractions, [0.7, 0.3], atol=1e-12),
           f"scaffolding: expected a 70/30 split, got {fractions}")

    # Same measurement, same geometry, same likelihood -- only the attribution differs.
    for sensor_id, expected_q in (("east", 0.7), ("north", 0.3)):
        tracker = make_tracker()
        track = make_track(positions)
        tracker.set_tracks([track])

        measurement = make_measurement((RANGE + 2000.0, 1500.0, 0.0),
                                       state((0.0, 0.0, 0.0)), sensor_id)
        likelihood = tracker.compute_association_likelihood(track, measurement)
        chk.ok(likelihood > 0.0, "scaffolding: the measurement must have a non-zero likelihood")

        reference = expected_existence(
            r=track.existence_probability(),
            likelihoods=[likelihood],
            pd_det=[P_DETECTION * expected_q],
            pd_miss=P_DETECTION * 1.0,   # the cloud is wholly inside the union of the two volumes
        )

        tracker.update([measurement], sensors)
        chk.close(tracker.get_tracks()[0].existence_probability(), reference,
                  f"a measurement from '{sensor_id}' must be scored with q={expected_q}")

    # The two attributions must actually differ, or the test above proves nothing.
    east = make_tracker()
    east.set_tracks([make_track(positions)])
    east.update([make_measurement((RANGE + 2000.0, 1500.0, 0.0), state((0.0, 0.0, 0.0)), "east")],
                sensors)
    north = make_tracker()
    north.set_tracks([make_track(positions)])
    north.update([make_measurement((RANGE + 2000.0, 1500.0, 0.0), state((0.0, 0.0, 0.0)), "north")],
                 sensors)
    chk.ok(east.get_tracks()[0].existence_probability()
           > north.get_tracks()[0].existence_probability() + 1e-9,
           "attributing the same measurement to a better-covered sensor must raise existence more")


# ---------------------------------------------------------------------------------------------
# B2: partial coverage scales P_D on the missed-detection branch too
# ---------------------------------------------------------------------------------------------


def check_partial_coverage(chk: Checker) -> None:
    sensors = two_lobe_sensors()
    # 40 particles inside the east lobe, 60 far off-axis where neither sensor looks.
    positions = np.array([[RANGE, 0.0, 0.0]] * 40 + [[0.0, 0.0, RANGE]] * 60)
    track_probe = make_track(positions)
    fractions = np.asarray(sensors.coverage_fractions(track_probe))
    union = sensors.coverage_fraction(track_probe)
    chk.ok(np.allclose(fractions, [0.4, 0.0], atol=1e-12), f"scaffolding: got {fractions}")
    chk.close(union, 0.4, "scaffolding: the union must be 0.4")

    tracker = make_tracker()
    track = make_track(positions)
    tracker.set_tracks([track])
    measurement = make_measurement((RANGE + 2000.0, 1500.0, 0.0), state((0.0, 0.0, 0.0)), "east")
    likelihood = tracker.compute_association_likelihood(track, measurement)

    reference = expected_existence(
        r=track.existence_probability(),
        likelihoods=[likelihood],
        pd_det=[P_DETECTION * 0.4],
        pd_miss=P_DETECTION * 0.4,
    )
    tracker.update([measurement], sensors)
    chk.close(tracker.get_tracks()[0].existence_probability(), reference,
              "a half-covered track must use P_D * q on both the detection and the miss branch")


# ---------------------------------------------------------------------------------------------
# B3: steps in which no sensor reported anything
# ---------------------------------------------------------------------------------------------


def check_empty_step(chk: Checker) -> None:
    sensors = two_lobe_sensors()

    # Wholly inside a field of view: a sensor was looking and saw nothing, so the full penalty.
    inside = make_tracker()
    track = make_track(np.array([[RANGE, 0.0, 0.0]] * 32))
    r0 = track.existence_probability()
    inside.set_tracks([track])
    inside.update([], sensors)
    chk.close(inside.get_tracks()[0].existence_probability(),
              expected_existence_miss_only(r0, P_DETECTION * 1.0),
              "a fully covered track must take the full missed-detection hit on an empty step")

    # Wholly outside every field of view: nobody was looking, so nothing is learned.
    outside = make_tracker()
    hidden = make_track(np.array([[0.0, 0.0, RANGE]] * 32))
    chk.close(sensors.coverage_fraction(hidden), 0.0, "scaffolding: the cloud must be invisible")
    outside.set_tracks([hidden])
    outside.update([], sensors)
    chk.ok(outside.get_tracks()[0].existence_probability() == r0,
           "an unobservable track must keep its existence probability EXACTLY on an empty step")

    # Partially covered: scaled by the union fraction.
    partial = make_tracker()
    split = make_track(np.array([[RANGE, 0.0, 0.0]] * 25 + [[0.0, 0.0, RANGE]] * 75))
    partial.set_tracks([split])
    partial.update([], sensors)
    chk.close(partial.get_tracks()[0].existence_probability(),
              expected_existence_miss_only(r0, P_DETECTION * 0.25),
              "a quarter-covered track must take a quarter of the missed-detection hit")

    # An unobservable track is also untouched by a step that carries somebody else's measurements.
    with_traffic = make_tracker()
    with_traffic.set_tracks([make_track(np.array([[0.0, 0.0, RANGE]] * 32))])
    with_traffic.update(
        [make_measurement((RANGE + 2000.0, 1500.0, 0.0), state((0.0, 0.0, 0.0)), "east")], sensors)
    chk.ok(with_traffic.get_tracks()[0].existence_probability() == r0,
           "an unobservable track must keep its existence probability EXACTLY through other traffic")

    # Without a sensor array an empty step is still a no-op, as it always was.
    legacy = make_tracker()
    legacy.set_tracks([make_track(np.array([[RANGE, 0.0, 0.0]] * 32))])
    legacy.update([])
    chk.ok(legacy.get_tracks()[0].existence_probability() == r0,
           "update(measurements) with no measurements must remain a no-op")


# ---------------------------------------------------------------------------------------------
# B4: an unbounded, unpointed array must reproduce the legacy path exactly
# ---------------------------------------------------------------------------------------------


def check_legacy_equivalence(chk: Checker) -> None:
    omniscient = lmb.SensorArray(lmb.SensorFovConfig())
    omniscient.add_unpointed("sensor_0", state((0.0, 0.0, 0.0)))

    positions = np.array([[RANGE + 300.0 * k, 120.0 * k, -80.0 * k] for k in range(24)])
    measurements = [
        make_measurement((RANGE + 2000.0, 1500.0, 0.0), state((0.0, 0.0, 0.0)), "sensor_0"),
        make_measurement((RANGE - 4000.0, -2500.0, 900.0), state((0.0, 0.0, 0.0)), "sensor_0"),
    ]

    legacy = make_tracker()
    legacy.set_tracks([make_track(positions)])
    legacy.update(measurements)

    with_array = make_tracker()
    with_array.set_tracks([make_track(positions)])
    with_array.update(measurements, omniscient)

    legacy_tracks, array_tracks = legacy.get_tracks(), with_array.get_tracks()
    chk.ok(len(legacy_tracks) == len(array_tracks),
           f"track counts must match: {len(legacy_tracks)} vs {len(array_tracks)}")
    for index, (a, b) in enumerate(zip(legacy_tracks, array_tracks)):
        chk.ok(a.existence_probability() == b.existence_probability(),
               f"track {index} existence must be bitwise identical to the legacy path")
        chk.ok(np.array_equal(np.asarray(a.particle_states()), np.asarray(b.particle_states())),
               f"track {index} particle states must be bitwise identical to the legacy path")


# ---------------------------------------------------------------------------------------------
# B5: a measurement the array cannot explain is an error, not a guess
# ---------------------------------------------------------------------------------------------


def check_sensor_resolution_errors(chk: Checker) -> None:
    sensors = two_lobe_sensors()
    tracker = make_tracker()
    tracker.set_tracks([make_track(split_cloud(8, 8))])

    stray = make_measurement((RANGE, 0.0, 0.0), state((0.0, 0.0, 0.0)), "west")
    chk.raises(lambda: tracker.update([stray], sensors), ValueError, "not in the SensorArray",
               "an unknown sensor_id_ must be rejected")

    # Also on a birth-only step, where nothing downstream would otherwise consult the array.
    empty_filter = make_tracker()
    chk.raises(lambda: empty_filter.update([stray], sensors), ValueError, "not in the SensorArray",
               "an unknown sensor_id_ must be rejected before births too")
    chk.ok(len(empty_filter.get_tracks()) == 0,
           "a rejected birth-only update must not have created tracks")

    unnamed = make_measurement((RANGE, 0.0, 0.0), state((0.0, 0.0, 0.0)), "")
    chk.raises(lambda: tracker.update([unnamed], sensors), ValueError, "not in the SensorArray",
               "an empty sensor_id_ must be rejected")

    empty_array = lmb.SensorArray(lmb.SensorFovConfig())
    good = make_measurement((RANGE, 0.0, 0.0), state((0.0, 0.0, 0.0)), "east")
    chk.raises(lambda: tracker.update([good], empty_array), ValueError, "SensorArray is empty",
               "measurements against an empty SensorArray must be rejected")

    # A rejected update must not have half-applied itself.
    chk.close(tracker.get_tracks()[0].existence_probability(), 0.6,
              "a rejected update must leave the filter state alone")

    # An empty array with no measurements is legitimate: nothing was observable, nothing changes.
    empty_tracker = make_tracker()
    empty_tracker.set_tracks([make_track(split_cloud(4, 4))])
    empty_tracker.update([], empty_array)
    chk.close(empty_tracker.get_tracks()[0].existence_probability(), 0.6,
              "no sensors and no measurements must leave existence untouched")


# ---------------------------------------------------------------------------------------------
# B6: the whole path, on a seeded scenario
# ---------------------------------------------------------------------------------------------


def check_seeded_scenario(chk: Checker) -> None:
    # 60 steps so object 2 (born at step 30) is active too: a sensor tasked on object 1 should
    # then miss it most of the time, which is what makes the measurement-count comparison mean
    # something. Over 30 steps only one object exists and a tasked sensor sees everything.
    config = hs.ScenarioConfig(num_steps=60, num_particles=200, k_best=4)

    baseline = hs.run_scenario(FIXED_SEED, config)

    bounded = lmb.SensorArray(lmb.SensorFovConfig(max_range=2.0e7,
                                                  half_width=np.deg2rad(20.0),
                                                  half_height=np.deg2rad(20.0)))
    bounded.add("sensor_0", hs.run_once.SENSOR_STATE.copy(), np.array([-1.0, 0.0, 0.0]))

    def task_at_first_truth(step, sensors, truths):
        """Per-timestep pointing: keep the boresight on the oldest active object."""
        del step
        if truths:
            sensors.point_at(0, truths[0][1][:3])

    pointed = hs.run_scenario(FIXED_SEED, config, sensors=bounded,
                              point_sensors=task_at_first_truth)

    total_baseline = int(np.sum(baseline.num_measurements))
    total_pointed = int(np.sum(pointed.num_measurements))
    chk.ok(total_baseline > 0, "scaffolding: the unbounded run must produce measurements")
    chk.ok(total_pointed < total_baseline,
           f"a 20-degree FOV must cut the measurement count ({total_pointed} vs {total_baseline})")
    chk.ok(total_pointed > 0,
           f"a 20-degree FOV must still see something ({total_pointed} measurements)")
    chk.ok(np.all(np.isfinite(pointed.gospa)), "the bounded run must produce finite GOSPA")

    # An unbounded, unpointed array through the same path must reproduce the baseline digest.
    omniscient = lmb.SensorArray(lmb.SensorFovConfig())
    omniscient.add_unpointed("sensor_0", hs.run_once.SENSOR_STATE.copy())
    mirrored = hs.run_scenario(FIXED_SEED, config, sensors=omniscient)
    chk.ok(np.array_equal(mirrored.num_measurements, baseline.num_measurements),
           "an unbounded unpointed array must produce the same measurements as the legacy path")
    chk.ok(np.array_equal(mirrored.gospa, baseline.gospa),
           "an unbounded unpointed array must reproduce the legacy GOSPA bitwise")
    chk.ok(np.array_equal(mirrored.cardinality, baseline.cardinality),
           "an unbounded unpointed array must reproduce the legacy cardinality")


# ---------------------------------------------------------------------------------------------
# B7: pairs no sensor can explain are skipped, and contribute exactly nothing
# ---------------------------------------------------------------------------------------------


def check_impossible_pairs_are_inert(chk: Checker) -> None:
    """Track A lives only in the east lobe, track B only in the north lobe.

    The cross pairs (A, north measurement) and (B, east measurement) are impossible, so update()
    skips their likelihood pass and prices them at INF_COST. With the cross pairs out of play the
    two tracks decouple: the joint hypotheses factorise, so each track's posterior existence must
    equal a single-track reference that never saw the other track's measurement. Anything a
    skipped pair leaked into the mixture -- a stale L, an unzeroed association block, a NaN --
    would move that number.

    Existence sits within ~1e-7 of 1 here (L / kappa is huge), so the comparison is made on the
    complement 1 - r, which is where a leak would actually show up.
    """
    sensors = two_lobe_sensors()
    track_a = make_track(split_cloud(32, 0), existence=0.6)
    track_b = make_track(split_cloud(0, 32), existence=0.7)
    east = make_measurement((RANGE + 2000.0, 1500.0, 0.0), state((0.0, 0.0, 0.0)), "east")
    north = make_measurement((1500.0, RANGE + 2000.0, 0.0), state((0.0, 0.0, 0.0)), "north")

    chk.close(sensors.coverage_fractions(track_a)[0], 1.0, "scaffolding: A wholly in the east lobe")
    chk.close(sensors.coverage_fractions(track_a)[1], 0.0, "scaffolding: A invisible to north")
    chk.close(sensors.coverage_fractions(track_b)[1], 1.0, "scaffolding: B wholly in the north lobe")
    chk.close(sensors.coverage_fractions(track_b)[0], 0.0, "scaffolding: B invisible to east")

    tracker = make_tracker()
    tracker.set_tracks([track_a, track_b])
    likelihood_a = tracker.compute_association_likelihood(track_a, east)
    likelihood_b = tracker.compute_association_likelihood(track_b, north)
    chk.ok(likelihood_a > 0.0 and likelihood_b > 0.0,
           "scaffolding: each track must have a non-zero likelihood for its own measurement")

    expected_a = expected_existence(0.6, [likelihood_a], [P_DETECTION], P_DETECTION)
    expected_b = expected_existence(0.7, [likelihood_b], [P_DETECTION], P_DETECTION)

    tracker.update([east, north], sensors)
    updated = tracker.get_tracks()
    chk.ok(len(updated) == 2, f"both tracks must survive and nothing may be born, got {len(updated)}")

    for track, expected, name in ((updated[0], expected_a, "A"), (updated[1], expected_b, "B")):
        actual = track.existence_probability()
        chk.ok(np.isfinite(actual) and 0.0 <= actual <= 1.0,
               f"track {name} existence must be a probability, got {actual!r}")
        chk.close(1.0 - actual, 1.0 - expected,
                  f"track {name} must match a single-track reference that never saw the other "
                  "track's measurement", rtol=1e-6)
        weights = np.asarray(track.particle_weights())
        chk.ok(np.all(np.isfinite(weights)) and abs(weights.sum() - 1.0) < 1e-12,
               f"track {name} must keep a normalised, finite cloud")


# ---------------------------------------------------------------------------------------------


def main() -> None:
    chk = Checker()
    print("fov-scaled detection probability")
    for name, fn in (
        ("B1 detection uses the producing sensor's coverage", check_detection_uses_the_right_sensor),
        ("B2 partial coverage", check_partial_coverage),
        ("B3 empty steps", check_empty_step),
        ("B4 legacy equivalence", check_legacy_equivalence),
        ("B5 sensor resolution errors", check_sensor_resolution_errors),
        ("B6 seeded scenario", check_seeded_scenario),
        ("B7 impossible pairs are inert", check_impossible_pairs_are_inert),
    ):
        before = chk.count
        fn(chk)
        print(f"  {name}: {chk.count - before} assertions")
    if chk.count <= 0:
        raise AssertionError("no assertions executed")
    print(f"PASS: test_fov_detection_probability ({chk.count} assertions)")


if __name__ == "__main__":
    main()
