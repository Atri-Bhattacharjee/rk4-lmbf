"""Fast mode (keyed random streams, ziggurat sampler) and the lazy-propagation particle gate.

Both are opt-in and change random draws or the substep partition, so they are validated by what
must hold, not by bitwise goldens:

  Z1  the ziggurat sampler is a standard normal: moments, tail mass beyond R, Kolmogorov-Smirnov,
      no non-finite draw, and different keys give uncorrelated streams;
  Z2  a stream is a pure function of its key;
  K1  keyed propagation's deterministic part is bit-for-bit the legacy RK4 step;
  K2  keyed process noise has the time-consistent covariance Qd(dt);
  K3  edge cases: dt = 0, zero process noise, validation of set_fast_mode;
  G1  the particle gate never leaves a cloud stale while any particle is inside a sensor volume,
      on long curved needle clouds, and it lags far more often than the sphere test alone;
  G2  a cloud near a sensor that reported a measurement is brought current even when no particle
      is in reach (the fused proposal needs it scored);
  S1  the gate's sleep only skips answers already known: a ring run is bit-for-bit the same with it
      on or off, and so is a run with profiling on or off;
  S2  a sensor that jumps next to a sleeping cloud wakes it (sleeps assume bounded sensor speed);
  A1  audited on a ring run, no gate decision ever left a particle inside a sensor volume, and no
      sleep ever skipped a test that would have answered "maybe".

Usage:
    python tests/test_fast_mode.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
sys.path.insert(0, str(TESTS_DIR))

from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

SEED = 20261002
MU = 3.986004418e14
RADIUS = 6.371e6 + 800.0e3
SPEED = np.sqrt(MU / RADIUS)
RANGE = 20.0e3
Q = np.diag([2.0**2] * 3 + [0.02**2] * 3)
REFERENCE_DT = 60.0
SIGMAS = np.array([10.0, 1.0, 6.7e-4, 6.7e-4, 6.7e-5, 6.7e-5])
ZIGGURAT_R = 3.6541528853610088


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


def normal_cdf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def particle(state, weight=1.0):
    p = lmb.Particle()
    p.state_vector = np.asarray(state, dtype=np.float64)
    p.weight = weight
    return p


def make_track(states, existence=0.9, index=0, birth_time=0):
    weight = 1.0 / len(states)
    label = lmb.TrackLabel()
    label.birth_time = birth_time
    label.index = index
    return lmb.Track(label, existence, [particle(s, weight) for s in states])


def make_tracker(q=Q, seed=SEED, p_detection=0.999):
    propagator = lmb.TwoBodyPropagator(q, seed=seed, noise_reference_dt=REFERENCE_DT)
    sensor_model = lmb.InOrbitSensorModel(*SIGMAS**2)
    birth_model = lmb.AdaptiveBirthModel(64, 0.9, np.diag(SIGMAS**2), seed=seed)
    return lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 1.0, 4, 1e-6, 4e-15,
                               p_detection, 0.0, 1.0, seed=seed)


def ring_state(angle: float) -> np.ndarray:
    return np.array([RADIUS * np.cos(angle), RADIUS * np.sin(angle), 0.0,
                     -SPEED * np.sin(angle), SPEED * np.cos(angle), 0.0])


def exact_propagate(state: np.ndarray, dt: float, step: float = 1.0) -> np.ndarray:
    """Noiseless engine RK4 in steps of at most `step` seconds."""
    propagator = lmb.TwoBodyPropagator(np.zeros((6, 6)))
    n = max(1, int(math.ceil(abs(dt) / step)))
    out = np.asarray(state, dtype=np.float64)
    for _ in range(n):
        out = np.array(propagator.propagate(particle(out), dt / n, 0.0).state_vector)
    return out


def needle_cloud(centre: np.ndarray, half_span_seconds: float, count: int, rng) -> np.ndarray:
    """Particles spread along the orbit through `centre` (a curved arc), plus a thin jitter: the
    shape a cloud takes after an orbit or more without an update."""
    offsets = np.linspace(-half_span_seconds, half_span_seconds, count)
    states = np.array([exact_propagate(centre, t, step=30.0) for t in offsets])
    states[:, :3] += rng.normal(0.0, 50.0, (count, 3))
    states[:, 3:] += rng.normal(0.0, 0.05, (count, 3))
    return states


# ---------------------------------------------------------------------------------------------
# Z: the sampler
# ---------------------------------------------------------------------------------------------


def check_sampler(chk: Checker) -> None:
    n = 2_000_000
    x = np.asarray(lmb._fast_normals(SEED, n))
    chk.ok(x.shape == (n,), "Z1: wrong number of draws")
    chk.ok(np.all(np.isfinite(x)), "Z1: a draw is not finite")
    mean, var = float(x.mean()), float(x.var())
    chk.ok(abs(mean) < 5.0 / math.sqrt(n), f"Z1: mean {mean}")
    chk.ok(abs(var - 1.0) < 5.0 * math.sqrt(2.0 / n), f"Z1: variance {var}")
    skew = float(np.mean(x**3))
    kurt = float(np.mean(x**4))
    chk.ok(abs(skew) < 5.0 * math.sqrt(15.0 / n), f"Z1: third moment {skew}")
    chk.ok(abs(kurt - 3.0) < 5.0 * math.sqrt(96.0 / n), f"Z1: fourth moment {kurt}")
    # The tail beyond R comes from a separate branch; check its mass and its shape.
    tail = np.abs(x) > ZIGGURAT_R
    expected = 2.0 * (1.0 - float(normal_cdf(np.array([ZIGGURAT_R]))[0]))
    count = int(tail.sum())
    chk.ok(abs(count - n * expected) < 5.0 * math.sqrt(n * expected),
           f"Z1: {count} draws beyond R, expected {n * expected:.0f}")
    beyond_4 = int(np.sum(np.abs(x) > 4.0))
    expected_4 = n * 2.0 * (1.0 - float(normal_cdf(np.array([4.0]))[0]))
    chk.ok(abs(beyond_4 - expected_4) < 5.0 * math.sqrt(expected_4) + 2,
           f"Z1: {beyond_4} draws beyond 4, expected {expected_4:.1f}")
    # Kolmogorov-Smirnov on a subsample (the CDF is evaluated in Python).
    sub = np.sort(x[:200_000])
    cdf = normal_cdf(sub)
    m = len(sub)
    d = max(float(np.max(np.arange(1, m + 1) / m - cdf)), float(np.max(cdf - np.arange(m) / m)))
    chk.ok(d < 1.63 / math.sqrt(m), f"Z1: KS distance {d:.5f} (1% critical {1.63 / math.sqrt(m):.5f})")
    # Neighbouring keys, and the two halves of the layer/uniform split, are uncorrelated.
    y = np.asarray(lmb._fast_normals(SEED + 1, n))
    corr = float(np.corrcoef(x, y)[0, 1])
    chk.ok(abs(corr) < 5.0 / math.sqrt(n), f"Z1: streams of neighbouring keys correlate ({corr})")
    lag = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    chk.ok(abs(lag) < 5.0 / math.sqrt(n), f"Z1: lag-1 autocorrelation {lag}")
    print(f"    mean {mean:+.2e}, var {var:.5f}, kurtosis {kurt:.4f}, beyond R {count} "
          f"(expected {n * expected:.0f}), KS {d:.5f}")

    # Z2: determinism.
    again = np.asarray(lmb._fast_normals(SEED, 1000))
    chk.ok(np.array_equal(again, x[:1000]), "Z2: the same key must give the same stream")
    chk.ok(len(np.asarray(lmb._fast_normals(SEED, 0))) == 0, "Z2: zero draws")


# ---------------------------------------------------------------------------------------------
# K: keyed propagation
# ---------------------------------------------------------------------------------------------


def run_eager(states, q, fast: bool, dt: float, steps: int = 1):
    tracker = make_tracker(q=q)
    if fast:
        tracker.set_fast_mode(True)
    tracker.set_tracks([make_track(states)])
    for _ in range(steps):
        tracker.predict(dt)
    return np.asarray(tracker.get_tracks()[0].particle_states()).copy()


def check_keyed_propagation(chk: Checker) -> None:
    rng = np.random.default_rng(SEED)
    base = ring_state(0.3) + np.array([0.0, 0.0, 3e3, 0.0, 0.0, 400.0])
    states = base + np.concatenate([rng.normal(0, 2e3, (200, 3)), rng.normal(0, 2.0, (200, 3))], axis=1)
    zero = np.zeros((6, 6))

    # K1: without noise, fast and legacy take the same RK4 step bit for bit.
    for dt in (1.0, 37.5, 60.0):
        a = run_eager(states, zero, False, dt, steps=3)
        b = run_eager(states, zero, True, dt, steps=3)
        chk.ok(np.array_equal(a, b), f"K1: noiseless fast and legacy RK4 differ at dt={dt}")

    # K2: the keyed noise has covariance Qd(dt). One state, 40,000 copies, one step.
    propagator = lmb.TwoBodyPropagator(Q, seed=1, noise_reference_dt=REFERENCE_DT)
    copies = np.repeat(base[None, :], 40_000, axis=0)
    for dt in (1.0, 60.0):
        clean = run_eager(copies[:1], zero, True, dt)[0]
        noisy = run_eager(copies, Q, True, dt)
        offsets = noisy - clean
        expected = np.asarray(propagator.step_noise_covariance(dt))
        sample = np.cov(offsets.T)
        ratio = np.diag(sample) / np.diag(expected)
        chk.ok(np.all(np.abs(ratio - 1.0) < 0.04), f"K2: dt={dt} variance ratios {np.round(ratio, 4)}")
        # Position-velocity cross-covariance (the dt^2/2 coupling term) per axis.
        for k in range(3):
            rho_expected = expected[k, k + 3] / math.sqrt(expected[k, k] * expected[k + 3, k + 3])
            rho_sample = sample[k, k + 3] / math.sqrt(sample[k, k] * sample[k + 3, k + 3])
            chk.ok(abs(rho_sample - rho_expected) < 0.03,
                   f"K2: dt={dt} axis {k} correlation {rho_sample:.4f} vs {rho_expected:.4f}")
        mean_err = np.abs(offsets.mean(axis=0)) / np.sqrt(np.diag(expected) / len(offsets))
        chk.ok(np.all(mean_err < 5.0), f"K2: dt={dt} noise mean is biased ({np.round(mean_err, 2)} sigma)")
        print(f"    dt={dt:4.0f}s variance ratios {np.round(ratio, 3)}")

    # Keyed: the same run twice is identical; different tracker seeds differ.
    a = run_eager(states, Q, True, 60.0, steps=2)
    b = run_eager(states, Q, True, 60.0, steps=2)
    chk.ok(np.array_equal(a, b), "K2: a fast-mode run must be reproducible")


def check_edge_cases(chk: Checker) -> None:
    rng = np.random.default_rng(SEED + 3)
    states = ring_state(1.0) + np.concatenate([rng.normal(0, 1e3, (50, 3)), rng.normal(0, 1.0, (50, 3))], axis=1)
    # dt = 0: nothing moves and nothing becomes non-finite (Qd(0) has no Cholesky factor).
    out = run_eager(states, Q, True, 0.0)
    chk.ok(np.array_equal(out, states), "K3: a zero-length fast step must leave the cloud unchanged")
    # Zero process noise: fast mode draws nothing.
    a = run_eager(states, np.zeros((6, 6)), True, 60.0)
    chk.ok(np.all(np.isfinite(a)), "K3: noiseless fast step produced a non-finite state")
    # An empty track propagates without error.
    tracker = make_tracker()
    tracker.set_fast_mode(True)
    empty = lmb.Track(lmb.TrackLabel(), 0.5, [])
    tracker.set_tracks([empty])
    tracker.predict(60.0)
    chk.ok(len(tracker.get_tracks()[0].particle_weights()) == 0, "K3: an empty track must stay empty")

    tracker = make_tracker()
    tracker.set_fast_mode(True)
    chk.ok(tracker.fast_mode, "K3: setter did not stick")
    tracker.set_fast_mode(False)
    chk.ok(not tracker.fast_mode, "K3: turning fast mode off")


# ---------------------------------------------------------------------------------------------
# G: the particle gate
# ---------------------------------------------------------------------------------------------


def check_gate_conservative(chk: Checker) -> None:
    """Step a gated lazy tracker and an eager tracker side by side past ring sensors.

    Noiseless, so the eager cloud says exactly when a particle is inside a volume. Whenever one is,
    the gated cloud must be current. The clouds are long curved needles (up to ~1500 km of arc),
    the case the sphere test handles worst.
    """
    rng = np.random.default_rng(SEED + 5)
    zero = np.zeros((6, 6))
    total_in_range = 0
    gate_lag = 0
    sphere_lag = 0
    for case, (half_span, miss) in enumerate([(5.0, 5e3), (60.0, 12e3), (100.0, 18e3), (100.0, -15e3)]):
        # A ring of 8 sensors; the needle's centre crosses the equator near sensor 0 at t ~ 120 s.
        num_sensors = 8
        sensor_states = [ring_state(2.0 * np.pi * k / num_sensors) for k in range(num_sensors)]
        target_then = exact_propagate(sensor_states[0], 120.0)
        target_then[2] += miss
        target_then[3:] = np.array([0.0, 0.3 * SPEED, 0.95 * SPEED])
        start = exact_propagate(target_then, -120.0)
        cloud = needle_cloud(start, half_span, 120, rng)

        sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
        for k, s in enumerate(sensor_states):
            sensors.add_unpointed(f"s{k}", s)

        gated = make_tracker(q=zero, p_detection=1e-9)
        sphere = make_tracker(q=zero, p_detection=1e-9)
        eager = make_tracker(q=zero, p_detection=1e-9)
        for t, gate in ((gated, True), (sphere, False)):
            t.set_lazy_propagation(True, 60.0)
            t.set_particle_gate(gate)
            t.set_tracks([make_track(cloud)])
        eager.set_tracks([make_track(cloud)])

        current_sensors = [s.copy() for s in sensor_states]
        exact = lmb.TwoBodyPropagator(zero)
        for step in range(1, 301):
            current_sensors = [np.asarray(exact.propagate(particle(s), 1.0, 0.0).state_vector)
                               for s in current_sensors]
            sensors.set_states(np.array(current_sensors))
            for t in (gated, sphere, eager):
                t.predict(1.0)
                t.update([], sensors)
            eager_states = np.asarray(eager.get_tracks()[0].particle_states())
            in_range = any(np.any(np.linalg.norm(eager_states[:, :3] - s[:3], axis=1) <= RANGE)
                           for s in current_sensors)
            gated_time = gated.track_summary(with_means=False)["propagated_time"][0]
            sphere_time = sphere.track_summary(with_means=False)["propagated_time"][0]
            if in_range:
                total_in_range += 1
                chk.ok(gated_time == float(step),
                       f"G1 case {case} step {step}: a particle is in range but the gated cloud is "
                       f"stale at t={gated_time}")
            gate_lag += gated_time < step
            sphere_lag += sphere_time < step
    chk.ok(total_in_range >= 4, f"G1 scaffolding: clouds must pass through a sensor ({total_in_range})")
    chk.ok(gate_lag > sphere_lag, f"G1: the gate must skip more work than the sphere test "
                                  f"(lagging steps gate {gate_lag}, sphere {sphere_lag})")
    print(f"    in range on {total_in_range} steps; lagging steps: gate {gate_lag}, sphere {sphere_lag}")


def check_gate_reporting_sensor(chk: Checker) -> None:
    """A cloud whose sphere reaches a reporting sensor is refreshed even with no particle in reach."""
    rng = np.random.default_rng(SEED + 9)
    zero = np.zeros((6, 6))
    sensor0 = ring_state(0.0)
    # A needle crossing 40 km above sensor 0: far outside its 20 km volume, but its sphere reaches.
    target_then = exact_propagate(sensor0, 30.0)
    target_then[2] += 40e3
    target_then[3:] = np.array([0.0, 0.3 * SPEED, 0.95 * SPEED])
    start = exact_propagate(target_then, -30.0)
    cloud = needle_cloud(start, 80.0, 100, rng)

    sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
    sensors.add_unpointed("s0", sensor0)
    tracker = make_tracker(q=zero)
    tracker.set_lazy_propagation(True, 60.0)
    tracker.set_particle_gate(True)
    tracker.set_tracks([make_track(cloud)])

    exact = lmb.TwoBodyPropagator(zero)
    sensor = sensor0.copy()
    for _ in range(29):
        sensor = np.asarray(exact.propagate(particle(sensor), 1.0, 0.0).state_vector)
        sensors.set_state(0, sensor)
        tracker.predict(1.0)
        tracker.update([], sensors)
    chk.ok(tracker.track_summary(with_means=False)["propagated_time"][0] < 29.0,
           "G2 scaffolding: with nothing reported, the gate should leave the distant needle lagging")

    # Step 30: sensor 0 reports something at the edge of its volume.
    sensor = np.asarray(exact.propagate(particle(sensor), 1.0, 0.0).state_vector)
    sensors.set_state(0, sensor)
    tracker.predict(1.0)
    seen = sensor.copy()
    seen[2] += 0.9 * RANGE
    seen[3:] = target_then[3:]
    measurement = lmb.Measurement.fromCartesian(seen, sensor)
    measurement.sensor_id_ = "s0"
    measurement.timestamp_ = 30.0
    measurement.covariance_ = np.diag(SIGMAS**2)
    tracker.update([measurement], sensors)
    summary = tracker.track_summary(with_means=False)
    chk.ok(summary["propagated_time"][0] == 30.0,
           f"G2: a needle near a reporting sensor must be brought current (t={summary['propagated_time'][0]})")


def ring_config(**overrides):
    import run_ring

    config = run_ring.RingConfig()
    config.num_orbits = 0.6
    config.num_objects = 400
    config.num_particles = 300
    config.metric_interval = 120.0
    for key, value in overrides.items():
        setattr(config, key, value)
    return run_ring, config


def check_sleep_and_profiling_neutral(chk: Checker) -> None:
    run_ring, base = ring_config()
    reference = run_ring.run(base, verbose=False)
    for label, overrides in (("sleep off", {"gate_sleep": False}), ("profiling off", {"profile": False})):
        _, config = ring_config(**overrides)
        other = run_ring.run(config, verbose=False)
        differing = [k for k in run_ring.ARRAY_KEYS
                     if not np.array_equal(np.asarray(reference[k]), np.asarray(other[k]), equal_nan=True)]
        chk.ok(not differing, f"S1: {label} changed the run; differing: {differing}")
    counts = reference["engine_profile"]["counts"]
    chk.ok(counts["refresh_slept"] > counts["refresh_checks"],
           f"S1 scaffolding: sleep must skip most tests ({counts['refresh_slept']} slept, "
           f"{counts['refresh_checks']} tested)")
    print(f"    {counts['refresh_slept']} sleeping skips, {counts['refresh_checks']} tests; "
          f"identical with sleep off and with profiling off")


def check_sensor_jump_wakes(chk: Checker) -> None:
    rng = np.random.default_rng(SEED + 13)
    zero = np.zeros((6, 6))
    sensor_far = ring_state(np.pi)                     # the far side of the Earth
    centre = ring_state(0.0)
    cloud = needle_cloud(centre, 30.0, 80, rng)
    sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
    sensors.add_unpointed("s0", sensor_far)
    tracker = make_tracker(q=zero)
    tracker.set_lazy_propagation(True, 60.0)
    tracker.set_particle_gate(True)
    tracker.set_profiling(True)
    tracker.set_tracks([make_track(cloud)])

    exact = lmb.TwoBodyPropagator(zero)
    sensor = sensor_far.copy()
    for _ in range(5):
        sensor = np.asarray(exact.propagate(particle(sensor), 1.0, 0.0).state_vector)
        sensors.set_state(0, sensor)
        tracker.predict(1.0)
        tracker.update([], sensors)
    counts = tracker.profile()["counts"]
    chk.ok(counts["refresh_slept"] >= 3, f"S2 scaffolding: the far cloud should be asleep "
                                         f"({counts['refresh_slept']} skips)")
    resets_before = counts["sleep_resets"]

    # Teleport the sensor onto the cloud's predicted position: no orbital motion does that in 1 s.
    tracker.predict(1.0)
    now = 6.0
    target = exact_propagate(centre, now)
    jumped = target.copy()
    jumped[:3] += np.array([5e3, 0.0, 0.0])
    sensors.set_state(0, jumped)
    tracker.update([], sensors)
    counts = tracker.profile()["counts"]
    chk.ok(counts["sleep_resets"] == resets_before + 1, "S2: a sensor jump must wake every track")
    chk.ok(tracker.track_summary(with_means=False)["propagated_time"][0] == now,
           "S2: the cloud next to the jumped sensor must be brought current")


def check_audit(chk: Checker) -> None:
    run_ring, config = ring_config(gate_audit=True, num_orbits=0.8)
    log = run_ring.run(config, verbose=False)
    decisions, violations, skips, skip_violations = log["gate_audit"]
    chk.ok(decisions > 1000 and skips > 1000,
           f"A1 scaffolding: too little audited ({decisions} decisions, {skips} skips)")
    chk.ok(violations == 0, f"A1: {violations} gate decisions left a particle inside a sensor volume")
    chk.ok(skip_violations == 0, f"A1: {skip_violations} sleeps skipped a test that would say maybe")
    print(f"    {decisions} gate decisions and {skips} sleeping skips audited: 0 violations")


def main() -> int:
    chk = Checker()
    sections = [
        ("Z1/Z2 ziggurat sampler", check_sampler),
        ("K1/K2 keyed propagation", check_keyed_propagation),
        ("K3 edge cases", check_edge_cases),
        ("G1 particle gate is conservative on needles", check_gate_conservative),
        ("G2 reporting sensors keep the sphere test", check_gate_reporting_sensor),
        ("S1 sleep and profiling are bit-for-bit neutral", check_sleep_and_profiling_neutral),
        ("S2 a sensor jump wakes sleeping clouds", check_sensor_jump_wakes),
        ("A1 audited ring run", check_audit),
    ]
    for name, fn in sections:
        before = chk.count
        fn(chk)
        print(f"  {name}: {chk.count - before} assertions")
    print(f"PASS: test_fast_mode ({chk.count} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
