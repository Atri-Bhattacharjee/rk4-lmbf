"""Multi-sensor, pointed field-of-view driver.

The other drivers run a single sensor that sees everything. This one runs three bounded, pointed
sensors and re-aims each of them from Python at every timestep, which is the case the sensor
rework exists for:

- ``SensorFovConfig`` bounds range and the two angular half-widths. Those are global: every sensor
  in the array shares them.
- Each sensor carries its own state and its own boresight. ``SensorArray.point_at`` is called once
  per step per sensor -- that is the per-timestep pointing input.
- A ground-truth object only produces a measurement when some sensor can actually observe it, and
  the measurement is stamped with that sensor's id and state.
- Inside the filter each track's effective detection probability is ``P_D`` scaled by the fraction
  of its particle cloud inside the relevant sensor volume, so a track nobody is looking at keeps
  its existence probability instead of decaying.

The tasking policy here is deliberately trivial -- two sensors, dwelling on one object each and
rotating every DWELL_STEPS -- because the point is the plumbing, not the scheduler. Swap
``task_sensors`` for a real one. There are deliberately fewer sensors than objects, so at any
moment at least one object is unobservable once all three are up: watch its track hold its
existence probability flat instead of decaying, which is the behaviour the rework buys.

    python python/run_multisensor.py

Honours LMB_MULTISENSOR_STEPS and LMB_MULTISENSOR_PARTICLES (defaults 100 and 2000; the
production drivers use 10000 particles, which is slower than a demo wants).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simulation_common import (  # noqa: E402
    BIRTH_COVARIANCE_LOCAL,
    CLUTTER_INTENSITY,
    DT,
    FILTER_COVARIANCE,
    FILTER_SIGMAS,
    GOSPA_CUTOFF,
    GOSPA_PARAMS,
    K_BEST,
    NOISE_DECAY_RATE,
    NOISE_MIN_SCALE,
    ORBIT_RADIUS,
    P_BIRTH,
    P_DETECTION,
    P_SURVIVAL,
    PRUNE_THRESHOLD,
    Q_FILTER,
    SCENARIO,
    TRUTH_PROPAGATOR_NOISE,
    TRUTH_SIGMAS,
    V_CIRCULAR,
    lmb_engine,
    propagate_truth_state,
)

NUM_STEPS = int(os.environ.get("LMB_MULTISENSOR_STEPS", 100))
NUM_PARTICLES = int(os.environ.get("LMB_MULTISENSOR_PARTICLES", 2000))
SEED = 20260921

# Global to every sensor, as the configuration is shared. Rectangular: wider than it is tall, the
# way a focal plane usually is.
FOV = lmb_engine.SensorFovConfig(
    min_range=0.0,
    max_range=2.5e7,
    half_width=np.deg2rad(3.0),
    half_height=np.deg2rad(1.5),
)

# Deliberately fewer sensors than the three objects in SCENARIO, so something is always
# unobservable once they are all active.
NUM_SENSORS = 2

# Steps a sensor dwells on one object before rotating to the next.
DWELL_STEPS = 10


def sensor_orbit_state(phase_index: int) -> np.ndarray:
    """One of NUM_SENSORS observers spaced evenly around the same circular orbit."""
    angle = 2.0 * np.pi * phase_index / NUM_SENSORS
    return np.array([
        ORBIT_RADIUS * np.cos(angle),
        ORBIT_RADIUS * np.sin(angle),
        0.0,
        -V_CIRCULAR * np.sin(angle),
        V_CIRCULAR * np.cos(angle),
        0.0,
    ])


def build_sensors() -> lmb_engine.SensorArray:
    sensors = lmb_engine.SensorArray(FOV)
    for k in range(NUM_SENSORS):
        state = sensor_orbit_state(k)
        # An arbitrary initial boresight; task_sensors overwrites it on the first step.
        sensors.add(f"sensor_{k}", state, -state[:3])
    return sensors


def task_sensors(sensors, active_truths, step: int) -> None:
    """The per-timestep pointing input: a round-robin dwell over the active objects.

    Each sensor holds one object for DWELL_STEPS and then rotates to the next, offset from its
    neighbours so two sensors do not land on the same object. A sensor with nothing to look at
    keeps whatever boresight it had, which is what a real tasker would do between assignments.
    Re-pointing carries the roll over by parallel transport, so the rectangular footprint does not
    spin as the sensor slews.
    """
    for k in range(sensors.size()):
        if k >= len(active_truths):
            # More sensors than objects. Park the spare looking radially outward, away from
            # everything in orbit, so its volume cannot overlap a tasked sensor's -- the filter
            # assumes they are disjoint and the run reports it loudly when they are not.
            sensors.point_at(k, 2.0 * np.asarray(sensors.state(k))[:3])
            continue
        target_index = (step // DWELL_STEPS + k) % len(active_truths)
        sensors.point_at(k, active_truths[target_index][1][:3])


def generate_measurements(active_truths, sensors, current_time):
    """One measurement per detected object, attributed to the sensor that saw it."""
    measurements = []
    per_sensor_counts = np.zeros(sensors.size(), dtype=np.int64)
    overlaps = 0

    for _, truth_state in active_truths:
        # The demo assumes disjoint sensor volumes, as the filter does. Count violations rather
        # than silently letting visible_sensor's lowest-index tie-break hide them.
        if sum(1 for s in range(sensors.size()) if sensors.sees(s, truth_state)) > 1:
            overlaps += 1

        sensor_index = sensors.visible_sensor(truth_state)
        if sensor_index < 0:
            continue
        if np.random.random() > P_DETECTION:
            continue

        measurement = lmb_engine.Measurement.fromCartesian(
            truth_state, sensors.state(sensor_index)
        ).perturbed(np.random.normal(size=6) * TRUTH_SIGMAS)
        measurement.timestamp_ = current_time
        measurement.sensor_id_ = sensors.id(sensor_index)
        measurement.covariance_ = FILTER_COVARIANCE.copy()
        measurements.append(measurement)
        per_sensor_counts[sensor_index] += 1

    return measurements, per_sensor_counts, overlaps


def run(verbose: bool = True):
    np.random.seed(SEED)

    truth_propagator = lmb_engine.TwoBodyPropagator(TRUTH_PROPAGATOR_NOISE, seed=SEED)
    filter_propagator = lmb_engine.TwoBodyPropagator(Q_FILTER, seed=SEED)
    sensor_model = lmb_engine.InOrbitSensorModel(*FILTER_SIGMAS**2)
    birth_model = lmb_engine.AdaptiveBirthModel(NUM_PARTICLES, P_BIRTH, BIRTH_COVARIANCE_LOCAL, seed=SEED)
    tracker = lmb_engine.SMC_LMB_Tracker(
        filter_propagator, sensor_model, birth_model, P_SURVIVAL, K_BEST, PRUNE_THRESHOLD,
        CLUTTER_INTENSITY, P_DETECTION, NOISE_DECAY_RATE, NOISE_MIN_SCALE, seed=SEED,
    )

    sensors = build_sensors()
    sensor_states = [sensor_orbit_state(k) for k in range(NUM_SENSORS)]
    active_truths: list[tuple[int, np.ndarray]] = []

    gospa_history = np.zeros(NUM_STEPS)
    measurement_history = np.zeros((NUM_STEPS, NUM_SENSORS), dtype=np.int64)
    coverage_history = np.zeros(NUM_STEPS)
    unobservable_history = np.zeros(NUM_STEPS, dtype=np.int64)
    total_overlaps = 0

    for step in range(NUM_STEPS):
        current_time = step * DT

        if step > 0:
            for i, (obj_id, state) in enumerate(active_truths):
                active_truths[i] = (obj_id, propagate_truth_state(truth_propagator, state, DT))
            for k in range(NUM_SENSORS):
                sensor_states[k] = propagate_truth_state(truth_propagator, sensor_states[k], DT)
                sensors.set_state(k, sensor_states[k])

        for obj_id, birth_step, initial_state in SCENARIO:
            if step == birth_step:
                active_truths.append((obj_id, initial_state.copy()))

        # --- the per-timestep pointing input ---
        task_sensors(sensors, active_truths, step)

        measurements, counts, overlaps = generate_measurements(active_truths, sensors, current_time)
        measurement_history[step] = counts
        total_overlaps += overlaps

        if step > 0:
            tracker.predict(DT)

        tracker.update(measurements, sensors)

        tracks = tracker.get_tracks()
        truth_states = [state.copy() for (_, state) in active_truths]
        gospa_history[step] = lmb_engine.calculate_gospa_distance(tracks, truth_states, GOSPA_CUTOFF)

        track_coverage = [sensors.coverage_fraction(t) for t in tracks]
        if track_coverage:
            coverage_history[step] = float(np.mean(track_coverage))
        unobservable_history[step] = sum(1 for q in track_coverage if q == 0.0)

        if verbose and step % 10 == 0:
            existence = ", ".join(
                f"{t.existence_probability():.3f}{'*' if q == 0.0 else ''}"
                for t, q in zip(tracks, track_coverage)
            )
            print(
                f"  [Step {step:3d}] truths={len(active_truths)} tracks={len(tracks):2d} "
                f"meas={counts.sum()} (per sensor {counts.tolist()}) "
                f"coverage={coverage_history[step]:.2f} GOSPA={gospa_history[step]:8.1f} m "
                f"r=[{existence}]"
            )

    if total_overlaps:
        print(f"  WARNING: {total_overlaps} object-steps were visible to more than one sensor. "
              "The filter assumes disjoint sensor volumes; re-task so they do not overlap.")

    return gospa_history, measurement_history, coverage_history, unobservable_history


def main() -> None:
    print("=" * 72)
    print("SMC-LMB Multi-Sensor Field-of-View Demo")
    print("=" * 72)
    print(f"  Sensors: {NUM_SENSORS}, re-pointed every step")
    print(f"  FOV: {np.rad2deg(FOV.half_width):.1f} x {np.rad2deg(FOV.half_height):.1f} deg "
          f"half-angles, range <= {FOV.max_range / 1e3:.0f} km")
    print(f"  Steps: {NUM_STEPS}, DT: {DT}s, Particles: {NUM_PARTICLES}")
    print("=" * 72)

    gospa, measurements, coverage, unobservable = run()

    print("-" * 72)
    print(f"  Total measurements: {int(measurements.sum())} "
          f"(per sensor {measurements.sum(axis=0).tolist()})")
    print(f"  Mean GOSPA (last 20 steps): {np.mean(gospa[-20:]):.1f} m")
    print(f"  Track-steps with zero coverage: {int(unobservable.sum())} "
          "(marked * above; their existence probability is held, not decayed)")
    print(f"  Metric: GOSPA, {GOSPA_PARAMS}")

    time_axis = np.arange(NUM_STEPS)
    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    top.plot(time_axis, gospa, color="k", linewidth=1.8)
    top.set_ylabel("GOSPA (m)")
    top.set_title(f"Multi-sensor pointed field of view\n{GOSPA_PARAMS}", fontsize=12)
    top.grid(True, alpha=0.3)

    middle.stackplot(time_axis, *measurements.T,
                     labels=[f"sensor_{k}" for k in range(NUM_SENSORS)], alpha=0.85)
    middle.set_ylabel("Measurements")
    middle.legend(loc="upper left", ncol=NUM_SENSORS)
    middle.grid(True, alpha=0.3)

    bottom.plot(time_axis, coverage, color="#4c72b0", linewidth=1.8, label="mean track coverage")
    bottom.fill_between(time_axis, 0.0, (unobservable > 0).astype(float), step="mid",
                        color="#c44e52", alpha=0.18, label="some track unobservable")
    bottom.set_ylabel("Mean track coverage")
    bottom.set_xlabel("Time step")
    bottom.set_ylim([-0.05, 1.05])
    bottom.legend(loc="lower left")
    bottom.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "figure_multisensor.png")
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    print("\nMulti-sensor demo complete.")


if __name__ == "__main__":
    main()
