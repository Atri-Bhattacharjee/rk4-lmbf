"""Tests for the sensor pointing / field-of-view geometry in src/sensor_fov.h.

The visibility predicate is the single source of truth shared by the measurement generator and
the filter's coverage fractions, so it is checked against an independent NumPy formulation
(explicit atan2 on tangent-plane angles) rather than a restatement of the C++ arithmetic. Points
are also placed on each of the four field-of-view edges and nudged either side of them, because
the boundary is where a cached tan() and an atan2 can disagree.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

FIXED_SEED = 20260921


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


def state(position, velocity=(0.0, 0.0, 0.0)):
    return np.concatenate([np.asarray(position, dtype=np.float64),
                           np.asarray(velocity, dtype=np.float64)])


def sees_reference(sensor_position, boresight, up, target_position, fov, pointed=True):
    """Independent visibility reference: explicit angles, no cached tangents.

    Mirrors the documented convention -- (width, height, boresight) right-handed with
    width = up x boresight -- but reaches the answer through atan2 rather than through the
    multiply-by-tan form the implementation uses.
    """
    offset = np.asarray(target_position, dtype=np.float64) - np.asarray(sensor_position, dtype=np.float64)
    rho = float(np.linalg.norm(offset))
    if rho < fov.min_range or rho > fov.max_range:
        return False
    if not pointed:
        return True
    w = np.asarray(boresight, dtype=np.float64) / np.linalg.norm(boresight)
    v = np.asarray(up, dtype=np.float64)
    v = v - np.dot(v, w) * w
    v = v / np.linalg.norm(v)
    u = np.cross(v, w)
    z = float(np.dot(offset, w))
    if z <= 0.0:
        return False
    return (abs(np.arctan2(float(np.dot(offset, u)), z)) <= fov.half_width
            and abs(np.arctan2(float(np.dot(offset, v)), z)) <= fov.half_height)


# ---------------------------------------------------------------------------------------------
# A1: defaults and configuration validation
# ---------------------------------------------------------------------------------------------


def check_config(chk: Checker, rng) -> None:
    default = lmb.SensorFovConfig()
    chk.ok(default.min_range == 0.0, "default min_range must be 0")
    chk.ok(np.isinf(default.max_range) and default.max_range > 0, "default max_range must be +inf")
    chk.ok(0.0 < default.half_width < lmb.SENSOR_MAX_HALF_ANGLE, "default half_width must be in (0, pi/2)")
    chk.ok(default.half_height == default.half_width, "default half-angles must match")

    configured = lmb.SensorFovConfig(min_range=1.0e5, max_range=2.0e6,
                                     half_width=0.1, half_height=0.05)
    chk.ok(configured.min_range == 1.0e5 and configured.max_range == 2.0e6,
           "SensorFovConfig must store its range bounds")
    chk.ok(configured.half_width == 0.1 and configured.half_height == 0.05,
           "SensorFovConfig must store its half-angles")

    chk.raises(lambda: lmb.SensorFovConfig(min_range=-1.0), ValueError, "min_range",
               "negative min_range must be rejected")
    chk.raises(lambda: lmb.SensorFovConfig(min_range=10.0, max_range=1.0), ValueError, "max_range",
               "max_range below min_range must be rejected")
    chk.raises(lambda: lmb.SensorFovConfig(half_width=0.0), ValueError, "half_width",
               "zero half_width must be rejected")
    chk.raises(lambda: lmb.SensorFovConfig(half_width=np.pi / 2), ValueError, "half_width",
               "half_width at pi/2 must be rejected")
    chk.raises(lambda: lmb.SensorFovConfig(half_height=np.pi), ValueError, "half_height",
               "half_height beyond pi/2 must be rejected")
    chk.raises(lambda: lmb.SensorFovConfig(half_height=np.nan), ValueError, "half_height",
               "non-finite half_height must be rejected")

    # The getter hands back a copy, and assigning one back re-validates and re-caches.
    sensors = lmb.SensorArray(lmb.SensorFovConfig(half_width=0.2, half_height=0.2))
    fetched = sensors.fov_config
    fetched.half_width = 0.01
    chk.ok(sensors.fov_config.half_width == 0.2, "fov_config getter must return a copy")
    sensors.fov_config = fetched
    chk.ok(sensors.fov_config.half_width == 0.01, "assigning fov_config must take effect")

    # A config mutated into an invalid state must be rejected on assignment, not silently cached.
    invalid = lmb.SensorFovConfig()
    invalid.half_width = 10.0
    chk.raises(lambda: setattr(sensors, "fov_config", invalid), ValueError, "half_width",
               "assigning an invalid fov_config must be rejected")
    chk.ok(sensors.fov_config.half_width == 0.01,
           "a rejected fov_config assignment must leave the array unchanged")

    del rng


# ---------------------------------------------------------------------------------------------
# A2: the rectangular predicate against the atan2 reference
# ---------------------------------------------------------------------------------------------


def check_predicate_vs_reference(chk: Checker, rng) -> None:
    fov = lmb.SensorFovConfig(min_range=1.0e4, max_range=3.0e6,
                              half_width=np.deg2rad(8.0), half_height=np.deg2rad(3.0))
    sensors = lmb.SensorArray(fov)

    sensor_position = np.array([6.771e6, 0.0, 0.0])
    boresight = np.array([0.3, 1.0, -0.2])
    up = np.array([0.1, 0.05, 1.0])
    sensors.add("s0", state(sensor_position), boresight)
    sensors.set_pointing(0, boresight, up)

    stored_boresight = np.asarray(sensors.boresight(0))
    stored_up = np.asarray(sensors.up(0))

    inside = 0
    for _ in range(4000):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        # Bias half the draws into the cone so both branches get exercised, and span the range
        # bounds either side so the range test is hit too.
        if rng.random() < 0.5:
            direction = stored_boresight + 0.12 * rng.normal(size=3)
            direction /= np.linalg.norm(direction)
        rho = float(np.exp(rng.uniform(np.log(1.0e3), np.log(6.0e6))))
        target = sensor_position + rho * direction

        expected = sees_reference(sensor_position, stored_boresight, stored_up, target, fov)
        chk.ok(sensors.sees(0, target) == expected,
               f"sees must match the atan2 reference at rho={rho:.3e}, dir={direction}")
        inside += int(expected)

    chk.ok(inside > 200, f"test is degenerate: only {inside} of 4000 draws were visible")

    # A 6-D state and its position must be read identically.
    target = sensor_position + 1.0e6 * stored_boresight
    chk.ok(sensors.sees(0, state(target, (1.0, 2.0, 3.0))) == sensors.sees(0, target),
           "sees must ignore the velocity half of a 6-D state")


# ---------------------------------------------------------------------------------------------
# A3: the four field-of-view edges, and the back hemisphere
# ---------------------------------------------------------------------------------------------


def check_edges(chk: Checker, rng) -> None:
    half_width, half_height = np.deg2rad(6.0), np.deg2rad(2.5)
    fov = lmb.SensorFovConfig(half_width=half_width, half_height=half_height)
    sensors = lmb.SensorArray(fov)
    sensor_position = np.array([0.0, 0.0, 0.0])
    sensors.add("s0", state(sensor_position), np.array([1.0, 0.0, 0.0]))
    sensors.set_pointing(0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

    w = np.asarray(sensors.boresight(0))
    v = np.asarray(sensors.up(0))
    u = np.asarray(sensors.width_axis(0))
    chk.ok(np.allclose(np.cross(u, v), w, atol=1e-12), "width x height must be the boresight")

    z = 1.0e6
    nudge = 1.0e-9
    for half, axis, name in ((half_width, u, "width"), (half_height, v, "height")):
        for sign in (+1.0, -1.0):
            edge = z * w + sign * z * np.tan(half) * axis
            chk.ok(sensors.sees(0, edge), f"a point exactly on the {name} edge must be visible")
            chk.ok(sensors.sees(0, z * w + sign * z * np.tan(half - nudge) * axis),
                   f"a point just inside the {name} edge must be visible")
            chk.ok(not sensors.sees(0, z * w + sign * z * np.tan(half + nudge) * axis),
                   f"a point just outside the {name} edge must not be visible")

    # A corner is inside only because both angles are; the far corner of the bounding cone is not.
    corner = z * w + z * np.tan(half_width) * u + z * np.tan(half_height) * v
    chk.ok(sensors.sees(0, corner), "the exact corner must be visible")
    chk.ok(not sensors.sees(0, corner + 1.0e-3 * z * np.tan(half_height) * v),
           "just past the corner in height must not be visible")

    # Behind the sensor, the mirrored direction must never be visible however close in.
    for _ in range(200):
        offset = rng.normal(size=3)
        offset -= 2.0 * np.dot(offset, w) * w if np.dot(offset, w) > 0 else 0.0
        behind = -abs(np.dot(offset, w)) * w + offset - np.dot(offset, w) * w
        chk.ok(not sensors.sees(0, behind), "nothing behind the boresight plane may be visible")

    chk.ok(not sensors.sees(0, sensor_position),
           "an object exactly at the sensor is on the boresight plane, not in front of it")


# ---------------------------------------------------------------------------------------------
# A4: range bounds, and the unpointed mode
# ---------------------------------------------------------------------------------------------


def check_range_and_unpointed(chk: Checker, rng) -> None:
    fov = lmb.SensorFovConfig(min_range=1.0e5, max_range=2.0e6,
                              half_width=np.deg2rad(5.0), half_height=np.deg2rad(5.0))
    sensors = lmb.SensorArray(fov)
    sensors.add("pointed", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0]))
    sensors.add_unpointed("omni", state((0.0, 0.0, 0.0)))

    boresight = np.asarray(sensors.boresight(0))
    for rho, expected, what in ((1.0e5, True, "exactly at min_range"),
                                (1.0e5 - 1.0, False, "just inside min_range"),
                                (2.0e6, True, "exactly at max_range"),
                                (2.0e6 + 1.0, False, "just beyond max_range"),
                                (1.0e6, True, "mid-band")):
        chk.ok(sensors.sees(0, rho * boresight) == expected, f"pointed sensor {what}")
        chk.ok(sensors.sees(1, rho * boresight) == expected, f"unpointed sensor {what}")

    # The unpointed sensor honours range in every direction; the pointed one does not.
    off_axis = 0
    for _ in range(500):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        target = 1.0e6 * direction
        chk.ok(sensors.sees(1, target), "an unpointed sensor must see any direction within range")
        if not sensors.sees(0, target):
            off_axis += 1
    chk.ok(off_axis > 400, f"only {off_axis} of 500 directions fell outside a 5-degree FOV")

    chk.ok(sensors.pointed(0) and not sensors.pointed(1), "pointed flags must round-trip")
    sensors.set_pointed(1, True)
    chk.ok(not sensors.sees(1, 1.0e6 * np.asarray(sensors.up(1))),
           "set_pointed(True) must re-enable the angular test")
    sensors.set_pointed(1, False)
    chk.ok(sensors.sees(1, 1.0e6 * np.asarray(sensors.up(1))),
           "set_pointed(False) must disable the angular test again")


# ---------------------------------------------------------------------------------------------
# A5: pointing -- roll continuity, point_at, bulk setters
# ---------------------------------------------------------------------------------------------


def check_pointing(chk: Checker, rng) -> None:
    sensors = lmb.SensorArray(lmb.SensorFovConfig(half_width=0.1, half_height=0.02))
    sensors.add("s0", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0]))

    # An explicitly non-orthogonal up must be orthogonalised, not rejected.
    sensors.set_pointing(0, np.array([2.0, 0.0, 0.0]), np.array([0.7, 0.0, 1.0]))
    b, v, u = (np.asarray(sensors.boresight(0)), np.asarray(sensors.up(0)),
               np.asarray(sensors.width_axis(0)))
    for name, vec in (("boresight", b), ("up", v), ("width", u)):
        chk.ok(abs(np.linalg.norm(vec) - 1.0) < 1e-12, f"{name} must be a unit vector")
    chk.ok(abs(np.dot(b, v)) < 1e-12, "up must be orthogonal to the boresight")
    chk.ok(np.allclose(np.cross(v, b), u, atol=1e-12), "width must be up x boresight")

    # Roll continuity: transporting through many small slews must not accumulate spin. Compare
    # against transporting in one step to the same final boresight.
    start = np.array([1.0, 0.0, 0.0])
    finish = np.array([1.0, 0.25, 0.1])
    finish /= np.linalg.norm(finish)

    stepwise = lmb.SensorArray(lmb.SensorFovConfig())
    stepwise.add("s", state((0.0, 0.0, 0.0)), start)
    stepwise.set_pointing(0, start, np.array([0.0, 0.0, 1.0]))
    steps = 64
    for k in range(1, steps + 1):
        interpolated = (1.0 - k / steps) * start + (k / steps) * finish
        stepwise.set_boresight(0, interpolated)

    one_shot = lmb.SensorArray(lmb.SensorFovConfig())
    one_shot.add("s", state((0.0, 0.0, 0.0)), start)
    one_shot.set_pointing(0, start, np.array([0.0, 0.0, 1.0]))
    one_shot.set_boresight(0, finish)

    chk.ok(np.allclose(np.asarray(stepwise.boresight(0)), np.asarray(one_shot.boresight(0)), atol=1e-12),
           "the two slews must land on the same boresight")
    chk.ok(np.allclose(np.asarray(stepwise.up(0)), np.asarray(one_shot.up(0)), atol=1e-9),
           "parallel transport must be path-independent enough that a slew does not spin the FOV")

    # A boresight reversal has nothing to transport and must still leave a valid frame.
    flipped = lmb.SensorArray(lmb.SensorFovConfig())
    flipped.add("s", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0]))
    flipped.set_boresight(0, np.array([-1.0, 0.0, 0.0]))
    fb, fv = np.asarray(flipped.boresight(0)), np.asarray(flipped.up(0))
    chk.ok(np.allclose(fb, [-1.0, 0.0, 0.0], atol=1e-12), "a reversed boresight must be stored")
    chk.ok(abs(np.linalg.norm(fv) - 1.0) < 1e-12 and abs(np.dot(fb, fv)) < 1e-12,
           "a reversed boresight must still leave an orthonormal frame")

    # point_at
    sensors.set_state(0, state((1.0e6, 2.0e6, -5.0e5)))
    target = np.array([3.0e6, -1.0e6, 4.0e5])
    sensors.point_at(0, target)
    expected = target - np.asarray(sensors.state(0))[:3]
    expected /= np.linalg.norm(expected)
    chk.ok(np.allclose(np.asarray(sensors.boresight(0)), expected, atol=1e-12),
           "point_at must aim the boresight at the target")
    chk.ok(sensors.sees(0, target), "a sensor pointed at a target must see it")
    sensors.set_boresight(0, np.array([1.0, 0.0, 0.0]))
    sensors.point_at(0, state(target, (9.0, 9.0, 9.0)))
    chk.ok(np.allclose(np.asarray(sensors.boresight(0)), expected, atol=1e-12),
           "point_at must ignore the velocity half of a 6-D state")

    # Bulk setters
    bulk = lmb.SensorArray(lmb.SensorFovConfig(half_width=0.05, half_height=0.05))
    for i in range(3):
        bulk.add(f"s{i}", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0]))
    states = rng.normal(size=(3, 6)) * 1.0e6
    boresights = rng.normal(size=(3, 3))
    bulk.set_states(states)
    bulk.set_boresights(boresights)
    for i in range(3):
        chk.ok(np.allclose(np.asarray(bulk.state(i)), states[i], atol=1e-9),
               f"set_states must write sensor {i}")
        expected_b = boresights[i] / np.linalg.norm(boresights[i])
        chk.ok(np.allclose(np.asarray(bulk.boresight(i)), expected_b, atol=1e-12),
               f"set_boresights must write sensor {i}")

    chk.raises(lambda: bulk.set_states(rng.normal(size=(2, 6))), ValueError, "states",
               "a short states array must be rejected")
    chk.raises(lambda: bulk.set_boresights(rng.normal(size=(3, 4))), ValueError, "boresights",
               "a mis-shaped boresights array must be rejected")


# ---------------------------------------------------------------------------------------------
# A6: coverage fractions against a NumPy count
# ---------------------------------------------------------------------------------------------


def make_track(states, weights=None):
    particles = []
    states = np.asarray(states, dtype=np.float64)
    if weights is None:
        weights = np.full(len(states), 1.0 / max(len(states), 1))
    for row, weight in zip(states, weights):
        particle = lmb.Particle()
        particle.state_vector = row.copy()
        particle.weight = float(weight)
        particles.append(particle)
    return lmb.Track(lmb.TrackLabel(), 0.8, particles)


def check_coverage(chk: Checker, rng) -> None:
    fov = lmb.SensorFovConfig(max_range=2.0e6, half_width=np.deg2rad(10.0), half_height=np.deg2rad(6.0))
    sensors = lmb.SensorArray(fov)
    sensors.add("a", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0]))
    sensors.add("b", state((0.0, 0.0, 0.0)), np.array([0.0, 1.0, 0.0]))
    sensors.add_unpointed("c", state((5.0e7, 0.0, 0.0)))  # far away: sees nothing here

    positions = rng.normal(size=(600, 3)) * 8.0e5
    states = np.concatenate([positions, rng.normal(size=(600, 3))], axis=1)
    weights = rng.random(600)
    weights /= weights.sum()
    track = make_track(states, weights)

    fractions = np.asarray(sensors.coverage_fractions(track))
    chk.ok(fractions.shape == (3,), f"coverage_fractions must be (S,), got {fractions.shape}")

    expected = np.zeros(3)
    union = 0.0
    for row, weight in zip(states, weights):
        visible_any = False
        for s in range(3):
            if sensors.sees(s, row):
                expected[s] += weight
                visible_any = True
        union += weight if visible_any else 0.0

    chk.ok(np.allclose(fractions, expected, atol=1e-12),
           f"coverage_fractions must match a NumPy count: {fractions} vs {expected}")
    chk.ok(abs(sensors.coverage_fraction(track) - union) < 1e-12,
           "coverage_fraction must be the union weight fraction")
    chk.ok(expected[0] > 1e-6 and expected[1] > 1e-6,
           "test is degenerate: at least two sensors must see part of the cloud")
    chk.ok(fractions[2] == 0.0, "a sensor with nothing in range must report zero coverage")

    # Sensors a and b are 90 degrees apart with 10-degree half-widths, so their volumes are
    # disjoint and the per-sensor fractions must add up to the union exactly.
    chk.ok(abs(fractions.sum() - sensors.coverage_fraction(track)) < 1e-12,
           "disjoint sensors' fractions must sum to the union fraction")

    # Uniform weights reduce to a plain count.
    uniform = make_track(states)
    counted = sum(1 for row in states if sensors.sees(0, row)) / len(states)
    chk.ok(abs(float(np.asarray(sensors.coverage_fractions(uniform))[0]) - counted) < 1e-12,
           "with uniform weights the fraction must be the plain particle count")

    # Degenerate clouds report no coverage rather than dividing by zero.
    empty = lmb.Track(lmb.TrackLabel(), 0.8, [])
    chk.ok(sensors.coverage_fraction(empty) == 0.0, "an empty cloud must report zero coverage")
    chk.ok(np.all(np.asarray(sensors.coverage_fractions(empty)) == 0.0),
           "an empty cloud must report zero per-sensor coverage")
    zero_weighted = make_track(states, np.zeros(len(states)))
    chk.ok(sensors.coverage_fraction(zero_weighted) == 0.0,
           "a zero-weight cloud must report zero coverage")

    # A cloud entirely inside one sensor is exactly 1.0 there and 1.0 in the union.
    boresight = np.asarray(sensors.boresight(0))
    tight = np.tile(np.concatenate([1.0e6 * boresight, np.zeros(3)]), (50, 1))
    tight_track = make_track(tight)
    tight_fractions = np.asarray(sensors.coverage_fractions(tight_track))
    chk.ok(tight_fractions[0] == 1.0, "a cloud wholly inside one sensor must report exactly 1.0")
    chk.ok(sensors.coverage_fraction(tight_track) == 1.0, "and exactly 1.0 for the union")


# ---------------------------------------------------------------------------------------------
# A7: array bookkeeping and validation
# ---------------------------------------------------------------------------------------------


def check_array_validation(chk: Checker, rng) -> None:
    sensors = lmb.SensorArray(lmb.SensorFovConfig())
    chk.ok(len(sensors) == 0 and sensors.size() == 0, "a new array must be empty")
    chk.ok(sensors.index_of("missing") == -1, "index_of must be -1 for an unknown id")
    chk.ok(sensors.visible_sensor(np.zeros(3)) == -1, "an empty array must see nothing")

    i0 = sensors.add("alpha", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0]))
    i1 = sensors.add_unpointed("beta", state((1.0, 2.0, 3.0)))
    chk.ok((i0, i1) == (0, 1), "add must return successive indices")
    chk.ok(len(sensors) == 2, "len must follow add")
    chk.ok(sensors.index_of("alpha") == 0 and sensors.index_of("beta") == 1,
           "index_of must resolve both ids")
    chk.ok(sensors.id(1) == "beta", "id must round-trip")

    chk.raises(lambda: sensors.add("alpha", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0])),
               ValueError, "duplicate", "a duplicate sensor id must be rejected")
    chk.raises(lambda: sensors.add("", state((0.0, 0.0, 0.0)), np.array([1.0, 0.0, 0.0])),
               ValueError, "empty", "an empty sensor id must be rejected")
    chk.raises(lambda: sensors.add("g", state((0.0, 0.0, 0.0)), np.zeros(3)),
               ValueError, "boresight", "a zero boresight must be rejected")
    chk.raises(lambda: sensors.add("g", state((0.0, 0.0, 0.0)), np.array([np.nan, 0.0, 0.0])),
               ValueError, "boresight", "a non-finite boresight must be rejected")
    chk.raises(lambda: sensors.add("g", state((0.0, 0.0, 0.0)), np.zeros(4)),
               ValueError, "boresight", "a 4-element boresight must be rejected")
    chk.raises(lambda: sensors.add("g", np.zeros(5), np.array([1.0, 0.0, 0.0])),
               ValueError, "state", "a 5-element sensor state must be rejected")
    chk.raises(lambda: sensors.add("g", np.full(6, np.nan), np.array([1.0, 0.0, 0.0])),
               ValueError, "state", "a non-finite sensor state must be rejected")
    # Every add above reused the id "g" on purpose: a rejected add must be atomic, so it must not
    # have been left registered by an earlier failure.
    chk.ok(len(sensors) == 2 and sensors.index_of("g") == -1,
           "a rejected add must leave the array untouched")

    chk.raises(lambda: sensors.set_pointing(0, np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])),
               ValueError, "parallel", "an up vector parallel to the boresight must be rejected")
    chk.raises(lambda: sensors.point_at(0, np.asarray(sensors.state(0))[:3]),
               ValueError, "sensor position", "pointing at the sensor's own position must be rejected")

    chk.raises(lambda: sensors.boresight(2), IndexError, "out of range",
               "an out-of-range index must be rejected")
    chk.raises(lambda: sensors.set_state(9, state((0.0, 0.0, 0.0))), IndexError, "out of range",
               "set_state must bounds-check")
    chk.raises(lambda: sensors.sees(7, np.zeros(3)), IndexError, "out of range",
               "sees must bounds-check")
    chk.raises(lambda: sensors.sees(0, np.zeros(4)), ValueError, "3 or 6",
               "a 4-element target must be rejected")

    del rng


# ---------------------------------------------------------------------------------------------


def run_all(seed: int) -> int:
    rng = np.random.default_rng(seed)
    chk = Checker()
    print(f"sensor field of view [seed {seed}]")
    for name, fn in (
        ("A1 configuration", check_config),
        ("A2 predicate vs reference", check_predicate_vs_reference),
        ("A3 field-of-view edges", check_edges),
        ("A4 range bounds and unpointed mode", check_range_and_unpointed),
        ("A5 pointing", check_pointing),
        ("A6 coverage fractions", check_coverage),
        ("A7 array bookkeeping", check_array_validation),
    ):
        before = chk.count
        fn(chk, rng)
        print(f"  {name}: {chk.count - before} assertions")
    return chk.count


def main() -> None:
    time_seed = int(time.time())
    total = 0
    for seed in (FIXED_SEED, time_seed):
        total += run_all(seed)
    if total <= 0:
        raise AssertionError("no assertions executed")
    print(f"PASS: test_sensor_fov ({total} assertions over seeds {FIXED_SEED}, {time_seed})")


if __name__ == "__main__":
    main()
