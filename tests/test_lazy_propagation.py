"""Time-consistent process noise and lazy propagation.

Lazy propagation (SMC_LMB_Tracker.set_lazy_propagation) steps a track's cloud with whatever step
length the moment calls for: the caller's fine step while a sensor might see it, coarse substeps of
up to max_pending otherwise. That is only sound if

  N1  the process noise is time-consistent -- sixty 1 s steps and one 60 s step inject the same
      noise -- which is what TwoBodyPropagator(noise_reference_dt=T) provides;
  L1  the "could any particle be inside a sensor volume?" test never says no when the answer is yes
      (a false no would score a stale cloud as unobservable and silently drop evidence);
  L2  a lazily stepped cloud is statistically the cloud an eager filter would have produced;
  L3  the bookkeeping holds: lagging tracks keep their existence exactly on steps nobody could see
      them, synchronize() and max_pending bring them current, and the per-call noise model is
      refused.

Usage:
    python tests/test_lazy_propagation.py
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

SEED = 20260930
MU = 3.986004418e14
RADIUS = 6.371e6 + 800.0e3
SPEED = np.sqrt(MU / RADIUS)
RANGE = 20.0e3
Q = np.diag([500.0**2] * 3 + [50.0**2] * 3)
REFERENCE_DT = 60.0
SIGMAS = np.array([5000.0, 500.0, 1e-2, 1e-2, 1e-3, 1e-3])


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


def particle(state, weight=1.0):
    p = lmb.Particle()
    p.state_vector = np.asarray(state, dtype=np.float64)
    p.weight = weight
    return p


def propagate(propagator, state, dt):
    return np.asarray(propagator.propagate(particle(state), dt, 0.0).state_vector)


def sensor_state(angle=0.0):
    return np.array([RADIUS * np.cos(angle), RADIUS * np.sin(angle), 0.0,
                     -SPEED * np.sin(angle), SPEED * np.cos(angle), 0.0])


# ---------------------------------------------------------------------------------------------
# N1: time-consistent process noise
# ---------------------------------------------------------------------------------------------


def check_noise_model(chk: Checker) -> None:
    per_call = lmb.TwoBodyPropagator(Q, seed=SEED)
    chk.ok(per_call.noise_reference_dt is None, "the default must stay the per-call model")
    for dt in (1.0, 60.0, 7.5):
        chk.ok(np.array_equal(per_call.step_noise_covariance(dt), Q),
               "per-call noise must be the configured matrix whatever dt is")
    chk.ok(per_call.noise_displacement_bound(60.0) is None,
           "per-call noise has no interval bound and must say so")
    chk.ok(lmb.TwoBodyPropagator(np.zeros((6, 6))).noise_displacement_bound(60.0) == 0.0,
           "a noiseless propagator moves nothing, whatever its model")

    consistent = lmb.TwoBodyPropagator(Q, seed=SEED, noise_reference_dt=REFERENCE_DT)
    chk.ok(consistent.noise_reference_dt == REFERENCE_DT, "noise_reference_dt must round-trip")

    # Closed form for a block-diagonal Q.
    for h in (1.0, 13.0, 60.0):
        qd = np.asarray(consistent.step_noise_covariance(h))
        qp, qv = 500.0**2 / REFERENCE_DT, 50.0**2 / REFERENCE_DT
        expected = np.zeros((6, 6))
        expected[:3, :3] = np.eye(3) * (qp * h + qv * h**3 / 3.0)
        expected[:3, 3:] = expected[3:, :3] = np.eye(3) * (qv * h**2 / 2.0)
        expected[3:, 3:] = np.eye(3) * (qv * h)
        chk.ok(np.allclose(qd, expected, rtol=1e-12, atol=0.0), f"Qd({h}) != the closed form")

    # Semigroup identity of the kinematic SDE: Qd(2h) = Qd(h) + Phi(h) Qd(h) Phi(h)^T.
    h = 17.0
    phi = np.eye(6)
    phi[:3, 3:] = np.eye(3) * h
    qd_h = np.asarray(consistent.step_noise_covariance(h))
    chk.ok(np.allclose(np.asarray(consistent.step_noise_covariance(2 * h)),
                       qd_h + phi @ qd_h @ phi.T, rtol=1e-12),
           "Qd must compose over consecutive steps, or step length would change the noise")

    bound = consistent.noise_displacement_bound(60.0)
    trace = np.trace(np.asarray(consistent.step_noise_covariance(60.0))[:3, :3])
    chk.ok(np.isclose(bound, 8.0 * np.sqrt(trace), rtol=1e-12), "bound must be 8 sigma of the trace")

    chk.raises(lambda: lmb.TwoBodyPropagator(Q, noise_reference_dt=0.0), ValueError,
               "noise_reference_dt", "a zero reference interval must be rejected")

    # Empirical: 60 x 1 s against 1 x 60 s, on a cloud started from one state.
    start = sensor_state(0.4)
    n = 20000
    fine = lmb.TwoBodyPropagator(Q, seed=SEED + 1, noise_reference_dt=REFERENCE_DT)
    coarse = lmb.TwoBodyPropagator(Q, seed=SEED + 2, noise_reference_dt=REFERENCE_DT)
    fine_states = np.empty((n, 6))
    coarse_states = np.empty((n, 6))
    for k in range(n):
        state = start.copy()
        for _ in range(60):
            state = propagate(fine, state, 1.0)
        fine_states[k] = state
        coarse_states[k] = propagate(coarse, start, 60.0)
    ratio = np.diag(np.cov(fine_states.T)) / np.diag(np.cov(coarse_states.T))
    # Sampling error of a variance ratio at n = 20000 is ~1.4%; the rest is gravity-gradient
    # coupling the kinematic model leaves out, which is well under 1% over 60 s.
    chk.ok(np.all(np.abs(ratio - 1.0) < 0.07),
           f"60 x 1 s and 1 x 60 s must spread a cloud alike, variance ratios {np.round(ratio, 3)}")
    print(f"    fine/coarse variance ratios over 60 s: {np.round(ratio, 3)}")


# ---------------------------------------------------------------------------------------------
# Tracker scaffolding
# ---------------------------------------------------------------------------------------------


def make_tracker(q=Q, reference_dt=REFERENCE_DT, seed=SEED, p_detection=0.999):
    propagator = lmb.TwoBodyPropagator(q, seed=seed, noise_reference_dt=reference_dt)
    sensor_model = lmb.InOrbitSensorModel(*SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(16, 0.9, np.diag([1000.0**2, 500.0**2, 1e-4**2, 1e-4**2,
                                                          5e-5**2, 5e-5**2]), seed=seed)
    return lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 1.0, 4, 1e-6, 1e-15,
                               p_detection, 0.0, 1.0, seed=seed)


def make_track(states, existence=0.9, index=0):
    weight = 1.0 / len(states)
    label = lmb.TrackLabel()
    label.birth_time = 0
    label.index = index
    return lmb.Track(label, existence, [particle(s, weight) for s in states])


def crossing_target(sensor_at_zero, crossing_time, miss_distance):
    """A polar-orbit object that passes `miss_distance` from the sensor at `crossing_time`."""
    exact = lmb.TwoBodyPropagator(np.zeros((6, 6)))
    sensor_then = propagate(exact, sensor_at_zero, crossing_time)
    position = sensor_then[:3] + np.array([0.0, 0.0, miss_distance])
    velocity = np.array([0.0, 0.0, SPEED]) + np.array([0.0, 150.0, 0.0])
    state = np.concatenate([position, velocity])
    return propagate(exact, state, -crossing_time)


# ---------------------------------------------------------------------------------------------
# L1 + L2: conservative observability test, and lazy == eager
# ---------------------------------------------------------------------------------------------


def check_conservative_reach(chk: Checker) -> None:
    """Step a lazy and an eager noiseless tracker side by side past a sensor.

    Without noise both clouds are deterministic, so the eager cloud says exactly when a particle is
    in range. Whenever it is, the lazy track must be current -- and when the cloud is far from the
    sensor, the lazy track must actually lag, or nothing is being saved.
    """
    rng = np.random.default_rng(SEED)
    sensor0 = sensor_state()
    target = crossing_target(sensor0, crossing_time=150.0, miss_distance=8e3)
    cloud = target + np.concatenate([rng.normal(0, 3e3, (64, 3)), rng.normal(0, 3.0, (64, 3))], axis=1)

    sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
    sensors.add_unpointed("s0", sensor0)

    # A vanishing P_D keeps the miss reweighting from resampling anything, so the two clouds stay
    # comparable particle for particle; the observability decisions are the same at any P_D > 0.
    zero = np.zeros((6, 6))
    lazy = make_tracker(q=zero, p_detection=1e-9)
    eager = make_tracker(q=zero, p_detection=1e-9)
    lazy.set_lazy_propagation(True, 60.0)
    lazy.set_tracks([make_track(cloud)])
    eager.set_tracks([make_track(cloud)])

    exact = lmb.TwoBodyPropagator(zero)
    sensor = sensor0.copy()
    in_range_steps = 0
    lagging_steps = 0
    for step in range(1, 301):
        sensor = propagate(exact, sensor, 1.0)
        sensors.set_state(0, sensor)
        lazy.predict(1.0)
        eager.predict(1.0)
        lazy.update([], sensors)
        eager.update([], sensors)

        eager_states = np.asarray(eager.get_tracks()[0].particle_states())
        in_range = np.any(np.linalg.norm(eager_states[:, :3] - sensor[:3], axis=1) <= RANGE)
        lazy_track = lazy.get_tracks()[0]
        if in_range:
            in_range_steps += 1
            chk.ok(lazy_track.propagated_time() == float(step),
                   f"step {step}: a particle is in range but the lazy cloud is stale at "
                   f"t={lazy_track.propagated_time()}")
        elif lazy_track.propagated_time() < step:
            lagging_steps += 1

    chk.ok(in_range_steps >= 2, f"scaffolding: the cloud must pass through the sensor ({in_range_steps} steps)")
    chk.ok(lagging_steps > 150, f"lazy propagation must actually skip work far from sensors "
                                f"(lagged on {lagging_steps} of 300 steps)")

    # L2 (deterministic): after synchronize the clouds differ only by RK4 step-size error.
    lazy.synchronize()
    lazy_states = np.asarray(lazy.get_tracks()[0].particle_states())
    eager_states = np.asarray(eager.get_tracks()[0].particle_states())
    lazy_sorted = lazy_states[np.lexsort(lazy_states.T[::-1])]
    eager_sorted = eager_states[np.lexsort(eager_states.T[::-1])]
    worst = float(np.max(np.linalg.norm(lazy_sorted[:, :3] - eager_sorted[:, :3], axis=1)))
    chk.ok(worst < 50.0, f"noiseless lazy and eager clouds must agree to RK4 error, worst {worst:.2f} m")
    print(f"    in range on {in_range_steps} steps, lagging on {lagging_steps}; "
          f"worst lazy/eager position gap {worst:.3f} m")


def check_statistical_equivalence(chk: Checker) -> None:
    """With noise, a cloud stepped lazily (60 s substeps) matches one stepped eagerly (1 s)."""
    rng = np.random.default_rng(SEED + 7)
    start = sensor_state(2.0) + np.concatenate([np.zeros(3), np.array([0.0, 0.0, 50.0])])
    cloud = np.repeat(start[None, :], 6000, axis=0)
    del rng

    sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
    sensors.add_unpointed("far", sensor_state(0.0))  # ~ 7000 km away throughout

    lazy = make_tracker(seed=SEED + 11)
    eager = make_tracker(seed=SEED + 12)
    lazy.set_lazy_propagation(True, 60.0)
    lazy.set_tracks([make_track(cloud)])
    eager.set_tracks([make_track(cloud)])
    for _ in range(300):
        lazy.predict(1.0)
        eager.predict(1.0)
        lazy.update([], sensors)
    lazy.synchronize()
    a = np.asarray(lazy.get_tracks()[0].particle_states())
    b = np.asarray(eager.get_tracks()[0].particle_states())
    mean_gap = np.abs(a.mean(axis=0) - b.mean(axis=0))
    sigma = np.sqrt(np.diag(np.cov(b.T)) * 2.0 / len(b))
    ratio = np.diag(np.cov(a.T)) / np.diag(np.cov(b.T))
    chk.ok(np.all(mean_gap < 6.0 * sigma + 1.0),
           f"lazy and eager means must agree within sampling error: gap {mean_gap}, sigma {sigma}")
    chk.ok(np.all(np.abs(ratio - 1.0) < 0.1), f"lazy/eager variance ratios {np.round(ratio, 3)}")
    print(f"    lazy/eager variance ratios after 300 s: {np.round(ratio, 3)}")


# ---------------------------------------------------------------------------------------------
# L3: bookkeeping
# ---------------------------------------------------------------------------------------------


def check_bookkeeping(chk: Checker) -> None:
    per_call = lmb.SMC_LMB_Tracker(
        lmb.TwoBodyPropagator(Q, seed=SEED), lmb.InOrbitSensorModel(*SIGMAS**2),
        lmb.AdaptiveBirthModel(8, 0.9, np.diag(SIGMAS**2), seed=SEED), 1.0, 4, 1e-6, 1e-15, 0.999,
        0.0, 1.0, seed=SEED)
    chk.raises(lambda: per_call.set_lazy_propagation(True), ValueError, "noise_reference_dt",
               "lazy propagation must refuse a per-call noise model")
    chk.raises(lambda: make_tracker().set_lazy_propagation(True, 0.0), ValueError, "max_pending",
               "a zero max_pending must be rejected")

    far = sensor_state(2.0)
    sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
    sensors.add_unpointed("s0", sensor_state(0.0))

    tracker = make_tracker()
    chk.ok(not tracker.lazy_propagation, "lazy propagation must be off by default")
    tracker.set_lazy_propagation(True, 60.0)
    chk.ok(tracker.lazy_propagation and tracker.max_pending == 60.0, "setters must round-trip")
    tracker.set_tracks([make_track(np.repeat(far[None, :], 8, axis=0), existence=0.37)])
    chk.ok(tracker.get_tracks()[0].propagated_time() == 0.0, "set_tracks must stamp unstamped tracks")

    for step in range(1, 60):
        tracker.predict(1.0)
        tracker.update([], sensors)
        track = tracker.get_tracks()[0]
        chk.ok(track.propagated_time() == 0.0, f"step {step}: an unobservable track must lag")
        chk.ok(track.existence_probability() == 0.37,
               f"step {step}: nobody could see it, so its existence must be exactly unchanged")
    tracker.predict(1.0)
    chk.ok(tracker.get_tracks()[0].propagated_time() == 60.0,
           "reaching max_pending must propagate the track")

    tracker.predict(1.0)
    tracker.predict(1.0)
    chk.ok(tracker.timestamp() == 62.0, "the clock must advance on every predict")
    chk.ok(tracker.get_tracks()[0].propagated_time() == 60.0, "scaffolding: lagging by 2 s")
    summary = tracker.track_summary()
    chk.ok(summary["propagated_time"][0] == 60.0 and summary["existence"][0] == 0.37,
           "track_summary must report the lag and the existence")
    tracker.synchronize()
    chk.ok(tracker.get_tracks()[0].propagated_time() == 62.0, "synchronize must bring tracks current")

    # A measurement no current cloud can explain is a birth, stamped at the clock.
    visible_target = sensor_state(0.0) + np.array([5e3, 0.0, 0.0, 0.0, 0.0, 7e3])
    measurement = lmb.Measurement.fromCartesian(visible_target, sensor_state(0.0))
    measurement.timestamp_ = 62.0
    measurement.sensor_id_ = "s0"
    measurement.covariance_ = np.diag(SIGMAS**2)
    tracker.predict(1.0)
    tracker.update([measurement], sensors)
    tracks = tracker.get_tracks()
    chk.ok(len(tracks) == 2, f"the unexplained measurement must spawn a track, got {len(tracks)} tracks")
    chk.ok(tracks[1].propagated_time() == 63.0, "a newborn track is current at its birth")
    chk.ok(tracks[0].existence_probability() == 0.37,
           "a lagging, unobservable track must be untouched by another object's measurement")

    tracker.set_lazy_propagation(False)
    chk.ok(all(t.propagated_time() == 63.0 for t in tracker.get_tracks()),
           "leaving lazy mode must synchronize every track")

    eager = make_tracker()
    eager.set_tracks([make_track(np.repeat(far[None, :], 4, axis=0))])
    eager.predict(5.0)
    chk.ok(eager.get_tracks()[0].propagated_time() == 5.0, "an eager predict must stamp the time")


def main() -> int:
    chk = Checker()
    print("lazy propagation")
    for name, fn in (
        ("N1 time-consistent noise", check_noise_model),
        ("L1 conservative observability, lazy == eager without noise", check_conservative_reach),
        ("L2 lazy == eager with noise", check_statistical_equivalence),
        ("L3 bookkeeping", check_bookkeeping),
    ):
        before = chk.count
        fn(chk)
        print(f"  {name}: {chk.count - before} assertions")
    print(f"PASS: test_lazy_propagation ({chk.count} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
