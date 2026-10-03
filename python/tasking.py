"""Sensor tasking for the pointed ring: what aims each sensor at every step of python/run_ring.py.

A tasker is handed to run_ring.run(config, tasker=...) with config.fov_half_angle_deg set. Its
``point`` is called at every step, after truth and sensors have moved and before anything is
detected, and sets every sensor's boresight and roll. Directions are worked out in each sensor's
local frame [radial, along-track, cross-track], the frame python/search_env.py uses, and turned
into ECI here.

  OracleTasker     knows the truth: every sensor looks straight at an object in its bubble. The
                   most a pointed sensor can do, and the check that pointing is wired up right.
  ScheduleTasker   search only: every sensor holds a direction bin for a slot, from a schedule
                   fixed before the run (random, one fixed bin, or odds tuned by search_sgd.py).
  CustodyTasker    wraps a search tasker. It predicts, from the filter's own particles, when and
                   where each known track will next cross a sensor's bubble, and at those steps
                   aims that sensor where the most of the track's cloud will be. A track is
                   re-predicted in the step it is born or updated, so a sensor that finds a new
                   object by search follows it for the rest of its pass.

Search does not depend on what has been detected (the objects are independent, and a look that
finds nothing and a look that finds something empty the same part of the sky), so its schedule is
drawn in advance. Custody is the part that reads the filter.
"""
from __future__ import annotations

import numpy as np

from run_ring import RingConfig, label_keys
from search_env import (SearchConfig, aim, build_pass_table, direction_bins, point_hits, predict_samples,
                        with_half_angle)
from search_sgd import SearchPolicy


def search_config_for(config: RingConfig) -> SearchConfig:
    """The search environment's view of a ring run: same ring, clock, objects and seed."""
    return SearchConfig(num_sensors=config.num_sensors, sensor_altitude=config.sensor_altitude,
                        sensor_range=config.sensor_range, num_objects=config.num_objects,
                        num_orbits=config.num_orbits, dt=config.dt, seed=config.seed,
                        csv_path=config.csv_path)


def apply_pointing(sensors, sensor_states: np.ndarray, boresight: np.ndarray, up: np.ndarray) -> None:
    """Aim every sensor from local-frame (S, 3) boresights and ups."""
    radial = sensor_states[:, :3] / np.linalg.norm(sensor_states[:, :3], axis=1)[:, None]
    along = np.column_stack([-radial[:, 1], radial[:, 0], np.zeros(len(radial))])

    def eci(local: np.ndarray) -> np.ndarray:
        out = local[:, :1] * radial + local[:, 1:2] * along
        out[:, 2] += local[:, 2]
        return out

    sensors.set_pointings(eci(boresight), eci(up))


class Tasker:
    """Aims every sensor each step. A subclass returns local-frame boresights and ups from
    ``directions``; ``counts`` tallies sensor-steps by what the sensor was doing."""

    def __init__(self, config: RingConfig) -> None:
        if config.fov_half_angle_deg is None:
            raise ValueError("a tasker needs pointed sensors: set config.fov_half_angle_deg")
        self.half_angle_deg = float(config.fov_half_angle_deg)
        self.search_config = search_config_for(config)
        self.num_sensors = config.num_sensors
        self.counts = {"search": 0, "custody": 0}

    def directions(self, **step):
        raise NotImplementedError

    def point(self, *, sensors, sensor_states, **step) -> None:
        boresight, up = self.directions(sensor_states=sensor_states, **step)
        apply_pointing(sensors, sensor_states, boresight, up)


class OracleTasker(Tasker):
    """Every sensor with an object in its bubble looks straight at it; the rest look outward."""

    def __init__(self, config: RingConfig) -> None:
        super().__init__(config)
        self._first_row = None

    def directions(self, *, step, truth, **_):
        if self._first_row is None:
            # Called first at step 0, when ``truth`` is the initial states: everything that will
            # ever be in a bubble follows from them.
            if step != 0:
                raise RuntimeError("OracleTasker must be used from the first step of a run")
            table = build_pass_table(truth, self.search_config)
            self._sensor, self._position = table.sample_sensor, table.sample_position
            self._first_row = np.searchsorted(table.sample_step, np.arange(self.search_config.num_steps + 2))
        boresight = np.tile([1.0, 0.0, 0.0], (self.num_sensors, 1))
        rows = slice(self._first_row[step], self._first_row[step + 1])
        boresight[self._sensor[rows]] = self._position[rows]
        frames = aim(boresight)
        occupied = len(np.unique(self._sensor[rows]))
        self.counts["custody"] += occupied
        self.counts["search"] += self.num_sensors - occupied
        return frames.boresight, frames.up


class ScheduleTasker(Tasker):
    """Search by a schedule fixed in advance: ``schedule[slot, sensor]`` is a direction bin, held
    for ``slot_steps`` steps. Playing the same schedule through SearchEnv(detection="sample") finds
    the same objects at the same times."""

    def __init__(self, config: RingConfig, schedule: np.ndarray, slot: float = 10.0) -> None:
        super().__init__(config)
        self.bins = direction_bins(self.half_angle_deg)
        self.slot_steps = max(1, int(round(slot / config.dt)))
        self.schedule = np.asarray(schedule, dtype=np.int64)
        expected = (self.search_config.num_steps // self.slot_steps + 1, self.num_sensors)
        if self.schedule.shape != expected:
            raise ValueError(f"schedule must be {expected}, got {self.schedule.shape}")
        if self.schedule.min() < 0 or self.schedule.max() >= len(self.bins):
            raise ValueError(f"schedule entries must be bins in [0, {len(self.bins)})")

    @staticmethod
    def num_slots(config: RingConfig, slot: float = 10.0) -> int:
        return search_config_for(config).num_steps // max(1, int(round(slot / config.dt))) + 1

    @classmethod
    def random(cls, config: RingConfig, seed: int, slot: float = 10.0) -> "ScheduleTasker":
        bins = len(direction_bins(config.fov_half_angle_deg))
        schedule = np.random.default_rng(seed).integers(0, bins, size=(cls.num_slots(config, slot), config.num_sensors))
        return cls(config, schedule, slot)

    @classmethod
    def fixed(cls, config: RingConfig, bin_index: int, slot: float = 10.0) -> "ScheduleTasker":
        return cls(config, np.full((cls.num_slots(config, slot), config.num_sensors), bin_index), slot)

    @classmethod
    def from_policy(cls, config: RingConfig, policy: SearchPolicy, seed: int) -> "ScheduleTasker":
        if policy.half_angle_deg != config.fov_half_angle_deg or policy.dt != config.dt:
            raise ValueError(f"policy is for a {policy.half_angle_deg} deg field of view at dt {policy.dt}, "
                             f"the run has {config.fov_half_angle_deg} deg at dt {config.dt}")
        schedule = policy.sample_schedule(cls.num_slots(config, policy.slot), config.num_sensors,
                                          np.random.default_rng(seed))
        return cls(config, schedule, policy.slot)

    def directions(self, *, step, **_):
        chosen = self.schedule[step // self.slot_steps]
        self.counts["search"] += self.num_sensors
        return self.bins.boresight[chosen], self.bins.up[chosen]


class CustodyTasker(Tasker):
    """Custody of known tracks on top of a search tasker.

    For every track at or above ``existence_threshold`` it keeps a prediction: ``particles`` of the
    track's cloud (taken evenly through its weights) carried forward in closed form, and the steps
    at which each lands in a sensor's bubble, over the next ``lookahead_orbits``. At a step where
    the predicted cloud weight in a sensor's bubble exceeds ``min_weight`` (by default: any predicted
    particle at all), that sensor leaves search and aims at the direction, among those of the
    predicted particles, whose field of view holds the most cloud weight.

    A track seen on one pass comes back as a cloud tens of kilometres long, of which a few percent
    crosses any one bubble. Hence the defaults: enough particles to resolve a few percent, and no
    threshold. Measured on 10 orbits at 45 degrees, known passes re-detected: 65% with 64 particles
    and a 5% threshold, 90% with 64 and none, 98% with 256 and none.

    Predictions are redone for a track in the step the filter births or updates it, and for every
    track once per orbit. Closed-form two-body motion is what the filter's own propagation amounts
    to here: its process noise is centimetres per minute.
    """

    def __init__(self, config: RingConfig, search: Tasker, particles: int = 256, lookahead_orbits: float = 1.25,
                 min_weight: float = 0.0, existence_threshold: float = 0.5) -> None:
        super().__init__(config)
        self.search = search
        self.particles = particles
        self.min_weight = min_weight
        self.existence_threshold = existence_threshold
        self.lookahead_steps = int(np.ceil(lookahead_orbits * self.search_config.orbit_period / config.dt))
        self.refresh_steps = int(self.search_config.orbit_period / config.dt)
        self._predicted = {}          # label key -> (sensor, step, position, weight) rows
        self._known = set()           # label keys seen so far
        self._next_refresh = 0
        self._dirty = True
        self.custody_log = []         # (step, sensor, cloud weight in the bubble, cloud weight in view)

    # --- predictions ---

    def _predict(self, tracker, keys, from_step: int) -> None:
        """Redo the predictions of the tracks with these label keys (None: every track)."""
        summary = tracker.track_summary(with_means=False)
        all_keys = label_keys(summary)
        self._known.update(all_keys.tolist())
        confirmed = summary["existence"] >= self.existence_threshold
        if keys is None:
            chosen = np.flatnonzero(confirmed)
            self._predicted = {}
        else:
            wanted = np.isin(all_keys, np.fromiter(keys, dtype=np.int64, count=len(keys)))
            chosen = np.flatnonzero(wanted & confirmed)
            for key in keys:
                self._predicted.pop(int(key), None)
        self._dirty = True
        if len(chosen) == 0:
            return
        sample = tracker.sample_particles(chosen.tolist(), self.particles)
        epochs = np.where(np.isnan(sample["propagated_time"]), from_step * self.search_config.dt,
                          sample["propagated_time"])
        index, sensor, step, position = predict_samples(
            sample["states"].reshape(-1, 6), np.repeat(epochs, self.particles), self.search_config,
            from_step, from_step + self.lookahead_steps)
        track = index // self.particles
        first = np.searchsorted(track, np.arange(len(chosen) + 1))
        for t, row in enumerate(chosen):
            rows = slice(first[t], first[t + 1])
            if first[t + 1] > first[t]:
                weight = np.full(first[t + 1] - first[t], summary["existence"][row] / self.particles)
                self._predicted[int(all_keys[row])] = (sensor[rows], step[rows], position[rows], weight)

    def _rows_at(self, step: int):
        if self._dirty:
            rows = list(self._predicted.values())
            if rows:
                sensor, when, position, weight = (np.concatenate([row[i] for row in rows]) for i in range(4))
                order = np.argsort(when, kind="stable")
                self._sensor, self._step = sensor[order], when[order]
                self._position, self._weight = position[order], weight[order]
            else:
                self._step = np.zeros(0, dtype=np.int64)
            self._dirty = False
        low, high = np.searchsorted(self._step, [step, step + 1])
        return slice(low, high)

    def after_update(self, *, step, tracker, records, measurements, **_) -> None:
        changed = set()
        if len(records["time"]):
            changed.update((records["birth_time"].astype(np.int64) * 1_000_000
                            + records["index"].astype(np.int64)).tolist())
        if measurements:
            # A measurement no track explains becomes a new track in this update.
            keys = label_keys(tracker.track_summary(with_means=False)).tolist()
            changed.update(key for key in keys if key not in self._known)
        if changed:
            self._predict(tracker, changed, step + 1)

    # --- pointing ---

    def directions(self, *, step, tracker, **context):
        if step >= self._next_refresh:
            self._predict(tracker, None, step)
            self._next_refresh = step + self.refresh_steps
        boresight, up = self.search.directions(step=step, tracker=tracker, **context)
        rows = self._rows_at(step)
        held = 0
        if rows.stop > rows.start:
            boresight, up = boresight.copy(), up.copy()
            sensor, position, weight = self._sensor[rows], self._position[rows], self._weight[rows]
            for s in np.unique(sensor):
                here = sensor == s
                if weight[here].sum() <= self.min_weight:
                    continue
                frames = with_half_angle(aim(position[here]), self.half_angle_deg)
                in_view = weight[here] @ point_hits(position[here], frames)
                best = int(np.argmax(in_view))
                boresight[s], up[s] = frames.boresight[best], frames.up[best]
                held += 1
                self.custody_log.append((step, int(s), float(weight[here].sum()), float(in_view[best])))
        self.counts["custody"] += held
        self.counts["search"] += self.num_sensors - held
        return boresight, up
