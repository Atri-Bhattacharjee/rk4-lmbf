"""Search-only tasking environment: the sensor bubbles, the debris inside them, and what has been seen.

The ring scenario of python/run_ring.py without the filter. Searching for objects nobody has seen
yet needs no particle filter: whether a sensor finds a new object depends only on where it pointed
and where the object was. And the truth is noise-free two-body motion that ignores the sensors, so
everything that can ever be detected is fixed before any policy acts. This module computes that
once, as a *pass table*, and an episode is then a lookup into it:

    config = SearchConfig(num_orbits=30)
    table = build_pass_table(make_scenario(config).states, config)   # < 1 s for 1000 objects
    env = SearchEnv(table, fov_half_angle_deg=20.0)
    result = env.run(policy)             # policy(env) -> one direction bin per sensor, every slot

Pass table. For every object, every stretch of time it spends inside a sensor's range bubble, as
relative positions in that sensor's local frame [radial, along-track, cross-track]. Two forms:

  samples    the object is inside the bubble at an integer step -- exactly the points at which
             run_ring tests for a detection;
  segments   the straight relative path between two consecutive steps, clipped to the bubble.

An object can only come within range of the equatorial ring near one of its two node crossings, so
the table is built from the closed-form two-body solution evaluated in a short window around each
crossing, not by stepping every object through the whole run.

Pointing. A sensor's field of view is the engine's rectangular pyramid (src/sensor_fov.h) with
equal half-angles, aimed at one of a fixed set of direction bins that tile the sky in the sensor's
local frame (DirectionBins). An action is one bin per sensor, held for a decision slot.

Detection models. "sample": the object is in the field of view at an integer step, as in run_ring.
"streak": the path it flew during the step crosses the field of view. At ~10 km/s a pass through a
20 km bubble lasts a few seconds and the line of sight swings through ~100 degrees, tens of degrees
between one-second samples; point samples then turn a narrow field of view into a lottery, so
"streak" is the default.

What a policy may read is the public state: the slot, the time, the sensors, and which objects
have been seen (SearchEnv.seen). SearchEnv.bubble_contents() is the truth -- every object in every
bubble this slot, seen or not -- and is for the oracle baseline and for building training labels.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from run_ring import DEFAULT_CSV, PASS_GAP, RingConfig, derive_seeds, load_catalogue
from simulation_common import MU_EARTH, R_EARTH

_RING = RingConfig()

# Bound on the object-sensor relative speed: how far outside a bubble a one-step path can start.
MAX_RELATIVE_SPEED = 20.0e3

# A node window longer than this (a nearly equatorial orbit) is replaced by a dense evaluation.
MAX_NODE_WINDOW = 600.0

DETECTION_MODELS = ("sample", "streak")


# =============================================================================
# Configuration and scenario
# =============================================================================


@dataclass
class SearchConfig:
    """The scenario: ring geometry, clock, and how the objects are drawn. Geometry defaults are
    run_ring's, so a default config with the same seed is the same truth as a ring run."""
    num_sensors: int = _RING.num_sensors
    sensor_altitude: float = _RING.sensor_altitude
    sensor_range: float = _RING.sensor_range
    num_objects: int = _RING.num_objects
    num_orbits: float = 30.0
    dt: float = _RING.dt
    seed: int = _RING.seed
    csv_path: str = str(DEFAULT_CSV)
    # Which catalogue rows objects are drawn from. "all" with rotate off is run_ring's sampling.
    # "train" / "test" split the catalogue by orbit family (see family_split).
    split: str = "all"
    # Turn each object about the Earth's axis by a random angle: a fresh node longitude with the
    # orbit's shape, inclination and node radii unchanged, so every seed is a new scenario.
    rotate: bool = False
    test_fraction: float = 0.2
    split_seed: int = 0

    @property
    def sensor_radius(self) -> float:
        return R_EARTH + self.sensor_altitude

    @property
    def orbit_period(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.sensor_radius**3 / MU_EARTH)

    @property
    def angular_rate(self) -> float:
        return 2.0 * np.pi / self.orbit_period

    @property
    def duration(self) -> float:
        return self.num_orbits * self.orbit_period

    @property
    def num_steps(self) -> int:
        return int(round(self.duration / self.dt))


@dataclass
class Scenario:
    states: np.ndarray        # (N, 6) ECI states at t = 0, m and m/s
    rows: np.ndarray          # catalogue row of each object
    object_ids: np.ndarray    # catalogue obj_id of each object


@lru_cache(maxsize=4)
def _catalogue(path: str):
    return load_catalogue(path)


@lru_cache(maxsize=4)
def _family_ids(path: str) -> np.ndarray:
    """Orbit family of each catalogue row: rows with identical (a, e, i, argument of perigee)."""
    keys = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            keys.append("|".join(row[name] for name in ("a_km", "e", "i_deg", "aop_deg")))
    return np.unique(np.asarray(keys), return_inverse=True)[1]


def family_split(path, test_fraction: float = 0.2, split_seed: int = 0):
    """(train rows, test rows) of the catalogue, split by orbit family.

    About a quarter of the catalogue's rows are copies of another row's orbit with a different node
    longitude and phase. Splitting by row would put a test object's orbit in the training set;
    splitting by family cannot.
    """
    family = _family_ids(str(path))
    order = np.random.default_rng(split_seed).permutation(family.max() + 1)
    in_test = np.zeros(family.max() + 1, dtype=bool)
    in_test[order[:int(round(test_fraction * len(order)))]] = True
    return np.flatnonzero(~in_test[family]), np.flatnonzero(in_test[family])


def make_scenario(config: SearchConfig, seed: int | None = None) -> Scenario:
    """Draw the objects of one episode. With split "all" and rotate off these are exactly the
    objects python/run_ring.py tracks for the same seed."""
    seed = config.seed if seed is None else seed
    ids, catalogue = _catalogue(str(config.csv_path))
    if config.split == "all":
        pool = np.arange(len(catalogue))
    elif config.split in ("train", "test"):
        train, test = family_split(config.csv_path, config.test_fraction, config.split_seed)
        pool = train if config.split == "train" else test
    else:
        raise ValueError(f"split must be 'all', 'train' or 'test', got {config.split!r}")
    if config.num_objects > len(pool):
        raise ValueError(f"asked for {config.num_objects} objects, split {config.split!r} has {len(pool)}")

    rng = np.random.default_rng(derive_seeds(seed)["sample"])
    rows = np.sort(pool[rng.choice(len(pool), size=config.num_objects, replace=False)])
    states = catalogue[rows].copy()
    if config.rotate:
        angle = rng.uniform(0.0, 2.0 * np.pi, size=len(rows))
        cos, sin = np.cos(angle), np.sin(angle)
        for first in (0, 3):
            x, y = states[:, first].copy(), states[:, first + 1].copy()
            states[:, first] = cos * x - sin * y
            states[:, first + 1] = sin * x + cos * y
    return Scenario(states=states, rows=rows, object_ids=ids[rows])


# =============================================================================
# Two-body motion in closed form
# =============================================================================


@dataclass
class Orbits:
    """The two-body orbit of each object, in the form position-at-time needs: the position is
    a (cos E - e) P + b sin E Q with E the eccentric anomaly, and E - e sin E = M0 + n t."""
    semi_major: np.ndarray       # a [m]
    semi_minor: np.ndarray       # b = a sqrt(1 - e^2) [m]
    eccentricity: np.ndarray
    mean_motion: np.ndarray      # n [rad/s]
    mean_anomaly: np.ndarray     # M0, at t = 0 [rad]
    periapsis: np.ndarray        # (N, 3) unit vector P, towards periapsis
    quadrature: np.ndarray       # (N, 3) unit vector Q, 90 degrees ahead of P in the orbit plane


def orbits_from_states(states: np.ndarray, epochs=None) -> Orbits:
    """The orbits through these (N, 6) states. ``epochs`` gives the time each state is at
    (default 0 for all); the orbit is the same either way, only its phase at t = 0 differs."""
    states = np.asarray(states, dtype=np.float64)
    position, velocity = states[:, :3], states[:, 3:]
    radius = np.linalg.norm(position, axis=1)
    momentum = np.cross(position, velocity)
    e_vector = np.cross(velocity, momentum) / MU_EARTH - position / radius[:, None]
    eccentricity = np.linalg.norm(e_vector, axis=1)
    semi_major = 1.0 / (2.0 / radius - np.einsum("ij,ij->i", velocity, velocity) / MU_EARTH)
    if np.any(semi_major <= 0.0) or np.any(eccentricity >= 1.0):
        raise ValueError("orbits_from_states: every object must be on a bound orbit")
    # A circular orbit has no periapsis; any direction in the plane serves, so take the position.
    circular = eccentricity < 1e-12
    periapsis = np.where(circular[:, None], position / radius[:, None],
                         e_vector / np.where(circular, 1.0, eccentricity)[:, None])
    normal = momentum / np.linalg.norm(momentum, axis=1)[:, None]
    quadrature = np.cross(normal, periapsis)
    semi_minor = semi_major * np.sqrt(1.0 - eccentricity**2)
    eccentric = np.arctan2(np.einsum("ij,ij->i", position, quadrature) / semi_minor,
                           np.einsum("ij,ij->i", position, periapsis) / semi_major + eccentricity)
    mean_motion = np.sqrt(MU_EARTH / semi_major**3)
    mean_anomaly = eccentric - eccentricity * np.sin(eccentric)
    if epochs is not None:
        mean_anomaly = mean_anomaly - mean_motion * np.asarray(epochs, dtype=np.float64)
    return Orbits(semi_major=semi_major, semi_minor=semi_minor, eccentricity=eccentricity,
                  mean_motion=mean_motion, mean_anomaly=mean_anomaly,
                  periapsis=periapsis, quadrature=quadrature)


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    """Eccentric anomaly E with E - e sin E = M, by Newton's method (vectorised, e < 1)."""
    mean = np.remainder(mean_anomaly + np.pi, 2.0 * np.pi) - np.pi
    eccentric = mean + eccentricity * np.sin(mean)
    for _ in range(12):
        change = ((eccentric - eccentricity * np.sin(eccentric) - mean)
                  / (1.0 - eccentricity * np.cos(eccentric)))
        eccentric = eccentric - change
        if np.max(np.abs(change), initial=0.0) < 1e-14:
            break
    return eccentric


def kepler_states(orbits: Orbits, index: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(K, 6) ECI states of objects ``index`` at ``times`` (two arrays of the same length)."""
    a, b, e = orbits.semi_major[index], orbits.semi_minor[index], orbits.eccentricity[index]
    eccentric = solve_kepler(orbits.mean_anomaly[index] + orbits.mean_motion[index] * times, e)
    cos, sin = np.cos(eccentric), np.sin(eccentric)
    rate = orbits.mean_motion[index] / (1.0 - e * cos)        # dE/dt
    p, q = orbits.periapsis[index], orbits.quadrature[index]
    out = np.empty((len(index), 6))
    out[:, :3] = (a * (cos - e))[:, None] * p + (b * sin)[:, None] * q
    out[:, 3:] = (-a * sin * rate)[:, None] * p + (b * cos * rate)[:, None] * q
    return out


def sensor_angles(config: SearchConfig, sensor: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Angle of each sensor from +x in the equatorial plane; sensor 0 starts on +x (ring_states)."""
    return 2.0 * np.pi * np.asarray(sensor) / config.num_sensors + config.angular_rate * np.asarray(times)


def local_offset(config: SearchConfig, position: np.ndarray, sensor: np.ndarray,
                 times: np.ndarray) -> np.ndarray:
    """ECI positions relative to a sensor, in its local frame [radial, along-track, cross-track]."""
    angle = sensor_angles(config, sensor, times)
    cos, sin = np.cos(angle), np.sin(angle)
    dx = position[:, 0] - config.sensor_radius * cos
    dy = position[:, 1] - config.sensor_radius * sin
    return np.column_stack([dx * cos + dy * sin, dy * cos - dx * sin, position[:, 2]])


def local_to_eci(vectors: np.ndarray, angle: float) -> np.ndarray:
    """Local-frame vectors [radial, along-track, cross-track] of a sensor at ``angle``, in ECI."""
    vectors = np.atleast_2d(vectors)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.column_stack([vectors[:, 0] * cos - vectors[:, 1] * sin,
                            vectors[:, 0] * sin + vectors[:, 1] * cos, vectors[:, 2]])


# =============================================================================
# Pass table
# =============================================================================


@dataclass
class PassTable:
    """Everything that can be detected in one episode. Rows are ordered by step.

    Positions are metres in the sensor's local frame [radial, along-track, cross-track]. A segment
    is the relative path over [step, step + 1]: ``segment_start`` and ``segment_end`` are its ends
    (either may lie outside the bubble) and ``segment_inside`` the fractions of it, 0..1, between
    which the object is inside the bubble.
    """
    config: SearchConfig
    num_objects: int
    sample_object: np.ndarray
    sample_sensor: np.ndarray
    sample_step: np.ndarray
    sample_position: np.ndarray
    segment_object: np.ndarray
    segment_sensor: np.ndarray
    segment_step: np.ndarray
    segment_start: np.ndarray
    segment_end: np.ndarray
    segment_inside: np.ndarray
    segment_pass: np.ndarray      # pass index of each segment (same object and sensor, no gap > PASS_GAP)

    @property
    def num_passes(self) -> int:
        return int(self.segment_pass.max()) + 1 if len(self.segment_pass) else 0

    def visible_objects(self, detection: str = "streak") -> np.ndarray:
        """Mask of the objects that ever enter a bubble: what sensors seeing in every direction find."""
        visible = np.zeros(self.num_objects, dtype=bool)
        visible[self.sample_object if detection == "sample" else self.segment_object] = True
        return visible


def _expand(counts: np.ndarray) -> np.ndarray:
    """0..counts[i]-1 for each i, concatenated."""
    total = int(counts.sum())
    return np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)


def _candidate_steps(orbits: Orbits, config: SearchConfig, reach: float, first_step: int, last_step: int):
    """(object, step) pairs, first_step <= step <= last_step, at which an object could be within
    ``reach`` of the ring.

    The ring lies in the equatorial plane, so that needs |z| <= reach, which only holds near a node
    crossing. For each crossing: its time, and a window around it long enough for the object to
    climb ``reach`` out of the plane. Crossings whose radius is too far from the ring's are dropped
    when the window is short enough for a straight-line estimate of the miss distance to be safe.
    """
    num_objects = len(orbits.semi_major)
    a, b, e = orbits.semi_major, orbits.semi_minor, orbits.eccentricity
    period = 2.0 * np.pi / orbits.mean_motion
    # z(E) = A cos E + B sin E - C, zero at E = phi +- acos(C / hypot(A, B)).
    A, B = a * orbits.periapsis[:, 2], b * orbits.quadrature[:, 2]
    C = a * e * orbits.periapsis[:, 2]
    amplitude = np.hypot(A, B)
    inclined = amplitude > 1e-9 * a
    safe = np.where(inclined, amplitude, 1.0)
    phi = np.arctan2(B, A)
    spread = np.arccos(np.clip(C / safe, -1.0, 1.0))

    objects, steps = [], []
    dense = ~inclined
    for sign in (1.0, -1.0):
        eccentric = phi + sign * spread
        cos, sin = np.cos(eccentric), np.sin(eccentric)
        rate = orbits.mean_motion / (1.0 - e * cos)
        radius = a * (1.0 - e * cos)
        vertical = (-a * sin * rate) * orbits.periapsis[:, 2] + (b * cos * rate) * orbits.quadrature[:, 2]
        radial = a * e * sin * rate
        window = 1.2 * reach / np.maximum(np.abs(vertical), 1e-9) + 2.0 * config.dt
        dense |= window > MAX_NODE_WINDOW
        # Straight-line closest approach to the ring circle, trusted only over a short window.
        miss = (np.abs(radius - config.sensor_radius) * np.abs(vertical)
                / np.maximum(np.hypot(radial, vertical), 1e-9))
        keep = inclined & ~dense & ((window > 30.0) | (miss <= reach + 5.0e3))

        first = np.remainder(eccentric - e * sin - orbits.mean_anomaly, 2.0 * np.pi) / orbits.mean_motion
        k_low = np.ceil((first_step * config.dt - window - first) / period)
        k_high = np.floor((last_step * config.dt + window - first) / period)
        count = np.where(keep, np.maximum(k_high - k_low + 1, 0), 0).astype(np.int64)
        index = np.repeat(np.arange(num_objects), count)
        crossing = first[index] + (k_low[index] + _expand(count)) * period[index]
        low = np.maximum(np.ceil((crossing - window[index]) / config.dt), first_step).astype(np.int64)
        high = np.minimum(np.floor((crossing + window[index]) / config.dt), last_step).astype(np.int64)
        length = np.maximum(high - low + 1, 0)
        objects.append(np.repeat(index, length))
        steps.append(np.repeat(low, length) + _expand(length))

    # Nearly equatorial orbits stay close to the plane for a long time: test every step.
    for index in np.flatnonzero(dense):
        objects.append(np.full(last_step - first_step + 1, index, dtype=np.int64))
        steps.append(np.arange(first_step, last_step + 1, dtype=np.int64))

    key = np.unique(np.concatenate(objects) * (last_step + 1) + np.concatenate(steps))
    return key // (last_step + 1), key % (last_step + 1)


def _inside_bubble(start: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
    """(K, 2) fractions of each segment start -> end between which it is within ``radius`` of the
    origin; low > high when it never is."""
    step = end - start
    a = np.einsum("ij,ij->i", step, step)
    half_b = np.einsum("ij,ij->i", start, step)
    c = np.einsum("ij,ij->i", start, start) - radius**2
    moving = a > 0.0
    root = np.sqrt(np.maximum(half_b**2 - a * c, 0.0))
    safe = np.where(moving, a, 1.0)
    low = np.where(moving, (-half_b - root) / safe, 0.0)
    high = np.where(moving, (-half_b + root) / safe, 1.0)
    missed = np.where(moving, half_b**2 - a * c < 0.0, c > 0.0)
    out = np.column_stack([np.maximum(low, 0.0), np.minimum(high, 1.0)])
    out[missed] = (1.0, 0.0)
    return out


def _near_ring(orbits: Orbits, config: SearchConfig, reach: float, first_step: int, last_step: int):
    """Every (object, step) in the step window at which the object is within ``reach`` of the ring
    circle, ordered by object then step: (object, step, time, ECI position, nearest sensor, offset
    from that sensor in its local frame, distance to it)."""
    obj, step = _candidate_steps(orbits, config, reach, first_step, last_step)
    times = step * config.dt
    position = kepler_states(orbits, obj, times)[:, :3]

    near = np.hypot(np.hypot(position[:, 0], position[:, 1]) - config.sensor_radius,
                    position[:, 2]) <= reach
    obj, step, times, position = obj[near], step[near], times[near], position[near]

    # Sensors are 2 pi r / N apart, far more than two bubbles, so only the nearest can be in range.
    spacing = 2.0 * np.pi / config.num_sensors
    bearing = np.arctan2(position[:, 1], position[:, 0]) - config.angular_rate * times
    sensor = np.rint(bearing / spacing).astype(np.int64) % config.num_sensors
    offset = local_offset(config, position, sensor, times)
    return obj, step, times, position, sensor, offset, np.linalg.norm(offset, axis=1)


def predict_samples(states: np.ndarray, epochs, config: SearchConfig, first_step: int, last_step: int):
    """Where states that are not at t = 0 will be seen: (index, sensor, step, local position) for
    every step in [first_step, last_step] at which the object through ``states[index]`` (which is
    at time ``epochs[index]``) is inside a sensor's bubble. Ordered by index, then step."""
    last_step = min(last_step, config.num_steps)
    empty = np.zeros(0, dtype=np.int64)
    if len(states) == 0 or last_step < first_step:
        return empty, empty, empty, np.zeros((0, 3))
    orbits = orbits_from_states(states, epochs)
    obj, step, _, _, sensor, offset, distance = _near_ring(orbits, config, config.sensor_range,
                                                           first_step, last_step)
    inside = distance <= config.sensor_range
    return obj[inside], sensor[inside], step[inside], offset[inside]


def build_pass_table(states: np.ndarray, config: SearchConfig) -> PassTable:
    """The pass table of objects with these (N, 6) ECI states at t = 0."""
    orbits = orbits_from_states(states)
    num_objects = len(states)
    reach = config.sensor_range + MAX_RELATIVE_SPEED * config.dt
    obj, step, times, position, sensor, offset, distance = _near_ring(orbits, config, reach, 0,
                                                                      config.num_steps)

    inside = distance <= config.sensor_range
    order = np.lexsort((obj[inside], sensor[inside], step[inside]))
    sample_object, sample_sensor = obj[inside][order], sensor[inside][order]
    sample_step, sample_position = step[inside][order], offset[inside][order]

    # Segments: consecutive steps of one object (the candidates are sorted by object, then step),
    # taken relative to whichever end's sensor is closer.
    first = np.flatnonzero((obj[1:] == obj[:-1]) & (step[1:] == step[:-1] + 1))
    second = first + 1
    segment_sensor = np.where(distance[first] <= distance[second], sensor[first], sensor[second])
    start = local_offset(config, position[first], segment_sensor, times[first])
    end = local_offset(config, position[second], segment_sensor, times[second])
    span = _inside_bubble(start, end, config.sensor_range)
    crosses = span[:, 0] <= span[:, 1]
    segment_object, segment_step = obj[first][crosses], step[first][crosses]
    segment_sensor, start, end, span = segment_sensor[crosses], start[crosses], end[crosses], span[crosses]

    # An object inside a bubble at the very last step has no path after it: a zero-length segment,
    # so that every sample is also the start of a segment.
    last = np.flatnonzero(sample_step == config.num_steps)
    segment_object = np.concatenate([segment_object, sample_object[last]])
    segment_step = np.concatenate([segment_step, sample_step[last]])
    segment_sensor = np.concatenate([segment_sensor, sample_sensor[last]])
    start = np.vstack([start, sample_position[last]])
    end = np.vstack([end, sample_position[last]])
    span = np.vstack([span, np.zeros((len(last), 2))])

    # Passes: runs of one object past one sensor with no gap longer than PASS_GAP.
    by_object = np.lexsort((segment_step, segment_object))
    o, s, k = segment_object[by_object], segment_sensor[by_object], segment_step[by_object]
    new_pass = np.ones(len(o), dtype=bool)
    new_pass[1:] = (o[1:] != o[:-1]) | (s[1:] != s[:-1]) | ((k[1:] - k[:-1]) * config.dt > PASS_GAP)
    segment_pass = np.empty(len(o), dtype=np.int64)
    segment_pass[by_object] = np.cumsum(new_pass) - 1

    order = np.lexsort((segment_object, segment_sensor, segment_step))
    return PassTable(
        config=config, num_objects=num_objects,
        sample_object=sample_object, sample_sensor=sample_sensor, sample_step=sample_step,
        sample_position=sample_position,
        segment_object=segment_object[order], segment_sensor=segment_sensor[order],
        segment_step=segment_step[order], segment_start=start[order], segment_end=end[order],
        segment_inside=span[order], segment_pass=segment_pass[order])


# =============================================================================
# Pointing: direction bins and the field of view
# =============================================================================


@dataclass
class DirectionBins:
    """The pointing choices: boresights that tile the sky, in the sensor's local frame.

    The grid is a cube's six faces, each cut n x n at equal angles. At 45 degrees the six faces
    are the fields of view exactly. Below that n = ceil(54 deg / half angle): the pyramids of two
    faces meet at an angle along the cube's edges, and a grid 20% finer than the field of view is
    what it takes to leave no direction uncovered. That gives 54 bins at 20 degrees, 216 at 10 and
    726 at 5. (width, up, boresight) is the engine's right-handed sensor frame
    (src/sensor_fov.h), with width = up x boresight.
    """
    half_angle: float           # rad, both half-width and half-height
    boresight: np.ndarray       # (B, 3)
    up: np.ndarray              # (B, 3)
    width: np.ndarray           # (B, 3)

    def __len__(self) -> int:
        return len(self.boresight)

    @property
    def tan_half_angle(self) -> float:
        return float(np.tan(self.half_angle))


def direction_bins(half_angle_deg: float) -> DirectionBins:
    if not 0.0 < half_angle_deg <= 45.0:
        raise ValueError(f"half_angle_deg must be in (0, 45], got {half_angle_deg}")
    n = 1 if half_angle_deg == 45.0 else int(np.ceil(54.0 / half_angle_deg - 1e-9))
    tangent = np.tan(np.deg2rad(-45.0 + (np.arange(n) + 0.5) * 90.0 / n))
    u, v = (grid.ravel() for grid in np.meshgrid(tangent, tangent, indexing="ij"))
    boresight, up = [], []
    # Each face: its axis and two tangent axes. Radial and along-track faces take cross-track as
    # their second axis; the two cross-track faces take along-track.
    for axis, first, second in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        for sign in (1.0, -1.0):
            direction = np.zeros((n * n, 3))
            direction[:, axis] = sign
            direction[:, first] = u
            direction[:, second] = v
            direction /= np.linalg.norm(direction, axis=1)[:, None]
            reference = np.zeros(3)
            reference[second] = 1.0
            face_up = reference - direction * direction[:, second:second + 1]
            boresight.append(direction)
            up.append(face_up / np.linalg.norm(face_up, axis=1)[:, None])
    boresight, up = np.vstack(boresight), np.vstack(up)
    return DirectionBins(half_angle=float(np.deg2rad(half_angle_deg)), boresight=boresight, up=up,
                         width=np.cross(up, boresight))


def point_hits(positions: np.ndarray, bins: DirectionBins) -> np.ndarray:
    """(K, B): is a point at this local position inside the field of view aimed at each bin.
    The angular part of SensorArray.sees; the range test is the caller's."""
    along = positions @ bins.boresight.T
    limit = along * bins.tan_half_angle
    return ((along > 0.0) & (np.abs(positions @ bins.width.T) <= limit)
            & (np.abs(positions @ bins.up.T) <= limit))


def segment_hits(start: np.ndarray, end: np.ndarray, inside: np.ndarray, bins: DirectionBins,
                 chunk: int = 4096) -> np.ndarray:
    """(K, B): does the path start -> end cross the field of view aimed at each bin while inside
    the bubble. The pyramid is four half-spaces through the sensor, each linear along the path, so
    each one cuts the path's parameter interval at most once."""
    tan = bins.tan_half_angle
    normals = np.stack([tan * bins.boresight - bins.width, tan * bins.boresight + bins.width,
                        tan * bins.boresight - bins.up, tan * bins.boresight + bins.up])   # (4, B, 3)
    out = np.empty((len(start), len(bins)), dtype=bool)
    for low_row in range(0, len(start), chunk):
        rows = slice(low_row, low_row + chunk)
        g0 = np.einsum("kj,fbj->fkb", start[rows], normals)
        g1 = np.einsum("kj,fbj->fkb", end[rows], normals)
        with np.errstate(divide="ignore", invalid="ignore"):
            cut = g0 / (g0 - g1)
        entering, leaving = g0 < 0.0, g1 < 0.0
        low = np.where(entering, np.where(leaving, np.inf, cut), 0.0).max(axis=0)
        high = np.where(leaving & ~entering, cut, 1.0).min(axis=0)
        out[rows] = np.maximum(low, inside[rows, :1]) <= np.minimum(high, inside[rows, 1:])
    return out


def aim(boresight: np.ndarray) -> DirectionBins:
    """Sensor frames for arbitrary local-frame boresights, (K, 3): the roll keeps ``up`` towards
    cross-track, or towards along-track when the boresight is itself cross-track. The half-angle is
    left at zero; pass the frames to ``point_hits`` through ``with_half_angle``."""
    boresight = np.atleast_2d(np.asarray(boresight, dtype=np.float64))
    boresight = boresight / np.linalg.norm(boresight, axis=1)[:, None]
    reference = np.zeros_like(boresight)
    reference[:, 2] = 1.0
    polar = np.abs(boresight[:, 2]) > 0.999
    reference[polar] = (0.0, 1.0, 0.0)
    up = reference - boresight * np.einsum("ij,ij->i", reference, boresight)[:, None]
    up /= np.linalg.norm(up, axis=1)[:, None]
    return DirectionBins(half_angle=0.0, boresight=boresight, up=up, width=np.cross(up, boresight))


def with_half_angle(frames: DirectionBins, half_angle_deg: float) -> DirectionBins:
    return DirectionBins(half_angle=float(np.deg2rad(half_angle_deg)), boresight=frames.boresight,
                         up=frames.up, width=frames.width)


def detection_rows(table: PassTable, bins: DirectionBins, detection: str):
    """(object, sensor, step, hits) of a table under a detection model: one row per step an
    object is in a bubble, with hits[row, bin] true when pointing at that bin detects it."""
    if detection == "sample":
        return (table.sample_object, table.sample_sensor, table.sample_step,
                point_hits(table.sample_position, bins))
    if detection == "streak":
        return (table.segment_object, table.segment_sensor, table.segment_step,
                segment_hits(table.segment_start, table.segment_end, table.segment_inside, bins))
    raise ValueError(f"detection must be one of {DETECTION_MODELS}, got {detection!r}")


@dataclass
class Visits:
    """A table reduced to what a slot-by-slot pointing policy can affect. One row per visit: an
    object in one sensor's bubble during one slot. ``hits[visit, bin]`` is true when holding that
    bin for the slot detects the object. Rows are ordered by object, then slot."""
    num_objects: int
    object: np.ndarray
    sensor: np.ndarray
    slot: np.ndarray
    hits: np.ndarray


def visit_hits(table: PassTable, bins: DirectionBins, slot_steps: int, detection: str = "streak") -> Visits:
    objects, sensors, steps, hits = detection_rows(table, bins, detection)
    slots = steps // slot_steps
    order = np.lexsort((sensors, slots, objects))
    objects, sensors, slots, hits = objects[order], sensors[order], slots[order], hits[order]
    first = np.ones(len(objects), dtype=bool)
    first[1:] = (objects[1:] != objects[:-1]) | (slots[1:] != slots[:-1]) | (sensors[1:] != sensors[:-1])
    start = np.flatnonzero(first)
    merged = (np.logical_or.reduceat(hits, start, axis=0) if len(start)
              else np.zeros((0, len(bins)), dtype=bool))
    return Visits(num_objects=table.num_objects, object=objects[start], sensor=sensors[start],
                  slot=slots[start], hits=merged)


# =============================================================================
# Environment
# =============================================================================


@dataclass
class SearchResult:
    first_detection: np.ndarray     # per object: time of its first detection [s], nan if never found
    visible: np.ndarray             # per object: did it ever enter a bubble
    duration: float
    custody_fraction: float         # share of sensor-slots spent on known objects (0 with custody off)

    @property
    def found(self) -> int:
        return int(np.sum(~np.isnan(self.first_detection)))

    @property
    def bound(self) -> int:
        """Objects found by sensors that see in every direction."""
        return int(np.sum(self.visible))

    def found_by(self, times) -> np.ndarray:
        """Number of objects found by each of ``times``: the discovery curve."""
        found = np.sort(self.first_detection[~np.isnan(self.first_detection)])
        return np.searchsorted(found, np.asarray(times, dtype=np.float64), side="right")

    @property
    def mean_time_to_discovery(self) -> float:
        """Mean first-detection time over every object that could be found, counting an object
        that never was as found at the end of the run. Lower is better; rewards finding early."""
        times = np.where(np.isnan(self.first_detection), self.duration, self.first_detection)
        return float(times[self.visible].mean()) if self.visible.any() else float("nan")


class SearchEnv:
    """One episode over a pass table. Every sensor holds one direction bin for a slot of
    ``slot`` seconds; an unseen object whose path crosses that field of view is found.

    custody: a sensor with an already-seen object in its bubble during the slot (or within
    ``custody_lead`` seconds after it) is given to that object and does no search that slot. The
    seen objects are known exactly here, standing in for a tracker; this is for measuring what
    custody costs search, not a model of the filter.
    """

    def __init__(self, table: PassTable, fov_half_angle_deg: float = 45.0, detection: str = "streak",
                 slot: float = 10.0, custody: bool = False, custody_lead: float = 0.0) -> None:
        if detection not in DETECTION_MODELS:
            raise ValueError(f"detection must be one of {DETECTION_MODELS}, got {detection!r}")
        self.table = table
        self.config = table.config
        self.detection = detection
        self.bins = direction_bins(fov_half_angle_deg)
        self.slot_steps = max(1, int(round(slot / self.config.dt)))
        self.num_slots = self.config.num_steps // self.slot_steps + 1
        self.num_sensors = self.config.num_sensors
        self.num_objects = table.num_objects
        self.custody = custody
        self.custody_slots = int(np.ceil(custody_lead / (self.slot_steps * self.config.dt)))

        self._object, self._sensor, self._step, self._hits = detection_rows(table, self.bins, detection)
        self._slot = self._step // self.slot_steps
        self._slot_start = np.searchsorted(self._slot, np.arange(self.num_slots + 1))
        self.visible = table.visible_objects(detection)
        self.reset()

    # --- state ---

    def reset(self) -> None:
        self.slot = 0
        self.seen = np.zeros(self.num_objects, dtype=bool)
        self.first_detection = np.full(self.num_objects, np.nan)
        self._custody_sensor_slots = 0

    @property
    def num_bins(self) -> int:
        return len(self.bins)

    @property
    def done(self) -> bool:
        return self.slot >= self.num_slots

    @property
    def time(self) -> float:
        """Start of the current slot [s]."""
        return self.slot * self.slot_steps * self.config.dt

    def sensor_angles(self) -> np.ndarray:
        """Angle of each sensor from +x at the start of the current slot [rad]."""
        return sensor_angles(self.config, np.arange(self.num_sensors), self.time)

    def bubble_contents(self):
        """TRUTH. Every object in a bubble during the current slot, one row per step it is there:
        (objects, sensors, hits), with hits[row, bin] true when pointing that sensor at that bin
        would detect the object at that step. Whether each is new is ``~env.seen[objects]``."""
        rows = slice(self._slot_start[self.slot], self._slot_start[self.slot + 1])
        return self._object[rows], self._sensor[rows], self._hits[rows]

    def _custody_sensors(self) -> np.ndarray:
        """Sensors with a known object in their bubble this slot or within the lead time."""
        last = min(self.slot + self.custody_slots + 1, self.num_slots)
        rows = slice(self._slot_start[self.slot], self._slot_start[last])
        busy = np.zeros(self.num_sensors, dtype=bool)
        busy[self._sensor[rows][self.seen[self._object[rows]]]] = True
        return busy

    # --- dynamics ---

    def step(self, bins) -> np.ndarray:
        """Hold ``bins`` (one direction bin per sensor) for the current slot. Returns the objects
        found for the first time, which is the search reward."""
        if self.done:
            raise RuntimeError("the episode is over; call reset()")
        bins = np.asarray(bins, dtype=np.int64)
        if bins.shape != (self.num_sensors,) or bins.min() < 0 or bins.max() >= self.num_bins:
            raise ValueError(f"bins must be {self.num_sensors} integers in [0, {self.num_bins})")
        found = np.zeros(0, dtype=np.int64)
        low, high = self._slot_start[self.slot], self._slot_start[self.slot + 1]
        searching = np.ones(self.num_sensors, dtype=bool)
        if self.custody:
            searching = ~self._custody_sensors()
            self._custody_sensor_slots += int(np.sum(~searching))
        if high > low:
            objects, sensors = self._object[low:high], self._sensor[low:high]
            hit = (~self.seen[objects] & searching[sensors]
                   & self._hits[np.arange(low, high), bins[sensors]])
            if hit.any():
                # Rows are in step order, so the first row of each object is its first detection.
                found, first = np.unique(objects[hit], return_index=True)
                self.seen[found] = True
                self.first_detection[found] = self._step[low:high][hit][first] * self.config.dt
        self.slot += 1
        return found

    def run(self, policy) -> SearchResult:
        """Play the episode from the start. ``policy(env)`` returns one bin per sensor."""
        self.reset()
        while not self.done:
            self.step(policy(self))
        return self.result()

    def run_schedule(self, schedule: np.ndarray) -> SearchResult:
        """Play a whole pointing schedule, (num_slots, num_sensors) bins, in one pass. The same
        result as stepping through it; needs custody off, since custody depends on what was found."""
        if self.custody:
            raise ValueError("run_schedule needs custody off; use run()")
        schedule = np.asarray(schedule, dtype=np.int64)
        if schedule.shape != (self.num_slots, self.num_sensors):
            raise ValueError(f"schedule must be {(self.num_slots, self.num_sensors)}, got {schedule.shape}")
        self.reset()
        hit = self._hits[np.arange(len(self._object)), schedule[self._slot, self._sensor]]
        first_step = np.full(self.num_objects, np.iinfo(np.int64).max)
        np.minimum.at(first_step, self._object[hit], self._step[hit])
        self.seen = first_step < np.iinfo(np.int64).max
        self.first_detection[self.seen] = first_step[self.seen] * self.config.dt
        self.slot = self.num_slots
        return self.result()

    def result(self) -> SearchResult:
        return SearchResult(
            first_detection=self.first_detection.copy(), visible=self.visible.copy(),
            duration=self.config.duration,
            custody_fraction=self._custody_sensor_slots / max(self.slot * self.num_sensors, 1))
