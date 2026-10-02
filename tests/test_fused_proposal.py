"""The fused proposal (SMC_LMB_Tracker.set_fused_proposal).

When the ordinary particle update for a (track, measurement) pair collapses -- the measurement is far
sharper than the gaps between the track's particles, so the particle sum is a handful of terms or
exactly zero -- the pair's detection component is drawn from where the track's cloud and the
measurement overlap, and importance-weighted with the exact likelihood. Checked here:

  F1  off, and on with ess_min = 0 (never triggers), the filter is bit-for-bit unchanged;
  F2  where the ordinary update is valid (a cloud only somewhat wider than the measurement), forcing
      the fused path gives the same association likelihood and posterior moments;
  F3  re-acquisition of a "needle": a measurement-sized cloud flown one orbit with no process noise.
      The ordinary update scores the returning object at exactly zero and births a new track; the
      fused path takes the detection and lands on the truth. Its association likelihood is checked
      on a Gaussian needle of the same covariance (18 km long, ~0.02 thick), where the exact answer
      is closed-form -- the regime where the ordinary particle sum collapses;
  F4  a cloud whose bounding sphere reaches the sensor with no particle inside is still a candidate;
  F5  the Gaussian fallback (forced) gives a likelihood close to the importance-sampled one;
  F6  argument validation and diagnostics.

The association likelihood L is not exposed directly; for a single track and a single measurement
the posterior existence r' fixes it: r' = (A + r (1 - P_D q)) / (A + 1 - r P_D q), A = r P_D L / kappa.

Usage:
    python tests/test_fused_proposal.py
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
MIN_RADIUS = 6.371e6 + 100.0e3
RADIUS = 6.371e6 + 800.0e3
SPEED = np.sqrt(MU / RADIUS)
RANGE = 20.0e3
SIGMAS = np.array([10.0, 1.0, 6.7e-4, 6.7e-4, 6.7e-5, 6.7e-5])   # balanced sensor
P_D = 0.9
R0 = 0.6


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


# ---------------------------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------------------------


def rk4(states, dt):
    """The engine's two-body RK4 step, vectorised (same model as src/two_body_propagator.cpp)."""
    def derivative(x):
        r = np.linalg.norm(x[:, :3], axis=1)
        rs = np.maximum(r, MIN_RADIUS)
        out = np.empty_like(x)
        out[:, :3] = x[:, 3:]
        out[:, 3:] = -MU * (x[:, :3] / r[:, None]) / (rs**2)[:, None]
        return out
    k1 = derivative(states)
    k2 = derivative(states + 0.5 * dt * k1)
    k3 = derivative(states + 0.5 * dt * k2)
    k4 = derivative(states + dt * k3)
    return states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def fly(states, seconds, step=10.0):
    steps = int(round(abs(seconds) / step))
    h = np.sign(seconds) * step
    for _ in range(steps):
        states = rk4(states, h)
    return states


def make_tracker(kappa, seed=SEED):
    propagator = lmb.TwoBodyPropagator(np.zeros((6, 6)), seed=seed)
    sensor_model = lmb.InOrbitSensorModel(*SIGMAS**2)
    birth = lmb.AdaptiveBirthModel(64, 0.5, np.diag(SIGMAS**2), seed=seed)
    return lmb.SMC_LMB_Tracker(propagator, sensor_model, birth, 1.0, 16, 0.0, kappa, P_D, 0.0, 1.0,
                               seed=seed)


def make_track(states):
    weight = 1.0 / len(states)
    particles = []
    for state in states:
        particle = lmb.Particle()
        particle.state_vector = state
        particle.weight = weight
        particles.append(particle)
    label = lmb.TrackLabel()
    label.birth_time = 0
    label.index = 7
    return lmb.Track(label, R0, particles)


def sensor_near(truth, offset_m, rng):
    """An unpointed ring-like sensor `offset_m` from the truth position."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    position = truth[:3] + offset_m * direction
    velocity = np.cross([0.0, 0.0, 1.0], position)
    velocity *= SPEED / np.linalg.norm(velocity)
    state = np.concatenate([position, velocity])
    sensors = lmb.SensorArray(lmb.SensorFovConfig(max_range=RANGE))
    sensors.add_unpointed("s0", state)
    return sensors, state


def measure(truth, sensor_state, rng):
    measurement = lmb.Measurement.fromCartesian(truth, sensor_state).perturbed(rng.normal(size=6) * SIGMAS)
    measurement.timestamp_ = 0.0
    measurement.sensor_id_ = "s0"
    measurement.covariance_ = np.diag(SIGMAS**2)
    return measurement


def run_update(states, measurement, sensors, kappa, fused=None, ess_min=20.0, neighbours=0,
               fallback_ess_min=20.0):
    tracker = make_tracker(kappa)
    if fused is not None:
        tracker.set_fused_proposal(fused, ess_min, neighbours, fallback_ess_min)
    tracker.set_record_diagnostics(True)
    tracker.set_tracks([make_track(states)])
    tracker.update([measurement], sensors)
    return tracker


def implied_likelihood(r_new, q, kappa):
    """Back out L from the single-track, single-measurement posterior existence."""
    b = 1.0 - R0 * P_D * q
    a = (r_new * b - R0 * (1.0 - P_D * q)) / (1.0 - r_new)
    return a * kappa / (R0 * P_D)


def moments(states):
    mean = states.mean(axis=0)
    return mean, np.cov(states.T)


def jacobian(x, measurement, sensor_state):
    obs = measurement.observation()
    f = lambda y: np.asarray(lmb.local_residual(obs, lmb.observe(y, sensor_state)))
    J = np.zeros((6, 6))
    steps = [1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2]
    for k in range(6):
        e = np.zeros(6)
        e[k] = steps[k]
        J[:, k] = -(f(x + e) - f(x - e)) / (2 * steps[k])
    return J


# ---------------------------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------------------------


def check_off_is_unchanged(chk, rng):
    truth = np.array([RADIUS, 0.0, 0.0, 0.0, 0.0, SPEED])
    sensors, sensor_state = sensor_near(truth, 12e3, rng)
    cloud = truth + np.concatenate([rng.normal(0, 400.0, (500, 3)), rng.normal(0, 20.0, (500, 3))], axis=1)
    measurement = measure(truth, sensor_state, rng)
    base = run_update(cloud, measurement, sensors, 1e-12)
    never = run_update(cloud, measurement, sensors, 1e-12, fused=True, ess_min=0.0)
    for a, b in zip(base.get_tracks(), never.get_tracks()):
        chk.ok(a.existence_probability() == b.existence_probability(),
               "ess_min = 0 must never trigger: existence differs")
        chk.ok(np.array_equal(np.asarray(a.particle_states()), np.asarray(b.particle_states())),
               "ess_min = 0 must never trigger: particles differ")
    chk.ok(not make_tracker(1e-12).fused_proposal, "the fused proposal must be off by default")


# ---------------------------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------------------------


def check_agrees_where_ordinary_is_valid(chk, rng):
    """A cloud ~2x the measurement in every direction: both updates are valid estimators."""
    truth = np.array([RADIUS, 0.0, 0.0, 0.0, 0.0, SPEED])
    sensors, sensor_state = sensor_near(truth, 12e3, rng)
    J = jacobian(truth, measure(truth, sensor_state, rng), sensor_state)
    sigma_z = np.linalg.inv(J) @ np.diag(SIGMAS**2) @ np.linalg.inv(J).T
    n = 20000
    prior = rng.multivariate_normal(truth, 4.0 * sigma_z, size=n)
    measurement = measure(truth, sensor_state, rng)
    q = sensors.coverage_fraction(make_track(prior))
    # kappa near the expected r P_D L, so existence lands mid-range and L can be read back from it.
    x_z = np.asarray(measurement.toCartesian())
    s_cov = 5.0 * sigma_z
    d = x_z - truth
    l_guess = np.exp(-0.5 * d @ np.linalg.solve(s_cov, d)) / np.sqrt(np.linalg.det(2 * np.pi * s_cov)) \
        / abs(np.linalg.det(J))
    kappa = R0 * P_D * l_guess
    ordinary = run_update(prior, measurement, sensors, kappa)
    ess = ordinary.take_diagnostics()["ess"][0]
    chk.ok(ess > 300, f"scaffolding: the ordinary update must be healthy here, ESS {ess:.0f}")
    fused = run_update(prior, measurement, sensors, kappa, fused=True, ess_min=1e12)
    diag = fused.take_diagnostics()
    chk.ok(diag["fused_components"][0] == 1 and diag["fallback_components"][0] == 0,
           "forcing ess_min must take the fused (importance-sampled) path, not the fallback")
    l_ordinary = implied_likelihood(ordinary.get_tracks()[0].existence_probability(), q, kappa)
    l_fused = implied_likelihood(fused.get_tracks()[0].existence_probability(), q, kappa)
    chk.ok(abs(l_fused / l_ordinary - 1.0) < 0.15,
           f"association likelihoods must agree: fused {l_fused:.4g} vs ordinary {l_ordinary:.4g}")
    m_o, c_o = moments(np.asarray(ordinary.get_tracks()[0].particle_states()))
    m_f, c_f = moments(np.asarray(fused.get_tracks()[0].particle_states()))
    sd = np.sqrt(np.diag(c_o))
    chk.ok(np.all(np.abs(m_f - m_o) < 0.5 * sd), f"posterior means differ by {np.abs(m_f - m_o) / sd} sd")
    ratio = np.diag(c_f) / np.diag(c_o)
    chk.ok(np.all((ratio > 0.6) & (ratio < 1.6)), f"posterior variance ratios {np.round(ratio, 2)}")
    print(f"    L fused/ordinary {l_fused / l_ordinary:.3f}; variance ratios {np.round(ratio, 2)}")


# ---------------------------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------------------------


def needle_scene(rng, n=2000):
    """A measurement-sized cloud on a polar 800 km orbit, flown one orbit with no process noise."""
    start = np.array([RADIUS, 0.0, 0.0, 0.0, 0.0, SPEED])
    start_cov = np.diag([10.0**2] * 3 + [1.0**2] * 3)
    period = 2.0 * np.pi * np.sqrt(RADIUS**3 / MU)
    cloud0 = rng.multivariate_normal(start, start_cov, size=n)
    cloud = fly(cloud0, period)
    truth0 = rng.multivariate_normal(start, start_cov)
    truth = fly(truth0[None, :], period)[0]
    return cloud, truth, start, start_cov, period


def closed_form_likelihood(mean, cov, measurement):
    """Exact L for a Gaussian prior N(mean, cov): N(x_z; mean, cov + Sigma_z) / |det J|.

    Exact up to linearising the measurement over its own extent, which is all the integrand
    covers when the prior is longer than the measurement in some directions and thinner in others.
    """
    sensor_state = np.asarray(measurement.sensor_state_)
    x_z = np.asarray(measurement.toCartesian())
    J = jacobian(x_z, measurement, sensor_state)
    j_inv = np.linalg.inv(J)
    sigma_z = j_inv @ np.diag(SIGMAS**2) @ j_inv.T
    s_cov = cov + sigma_z
    d = x_z - mean
    sign, logdet = np.linalg.slogdet(2 * np.pi * s_cov)
    return float(np.exp(-0.5 * d @ np.linalg.solve(s_cov, d) - 0.5 * logdet) / abs(np.linalg.det(J)))


def check_needle_reacquisition(chk, rng):
    # (a) The flown needle: does the returning object get taken, and does the track land on it?
    cloud, truth, start, start_cov, period = needle_scene(rng)
    sensors, sensor_state = sensor_near(truth, 12e3, rng)
    measurement = measure(truth, sensor_state, rng)
    q = sensors.coverage_fraction(make_track(cloud))
    chk.ok(q > 0.0, "scaffolding: some of the needle must be inside the sensor volume")
    kappa = 1e-3

    ordinary = run_update(cloud, measurement, sensors, kappa)
    chk.ok(len(ordinary.get_tracks()) == 2,
           f"without the fused proposal the returning object must be born anew, got {len(ordinary.get_tracks())} tracks")

    fused = run_update(cloud, measurement, sensors, kappa, fused=True)
    tracks = fused.get_tracks()
    diag = fused.take_diagnostics()
    chk.ok(len(tracks) == 1, f"with the fused proposal the track must take the detection, got {len(tracks)} tracks")
    chk.ok(diag["fused_components"][0] == 1 and diag["fallback_components"][0] == 0,
           "the fused path (not the fallback) must have produced the detection component")
    chk.ok(diag["fused_ess"][0] > 0.1 * len(cloud), f"fused ESS {diag['fused_ess'][0]:.0f} of {len(cloud)}")
    states = np.asarray(tracks[0].particle_states())
    error = np.linalg.norm(states[:, :3].mean(axis=0) - truth[:3])
    spread = np.sqrt(np.trace(np.cov(states[:, :3].T)))
    chk.ok(error < 100.0, f"the re-acquired track must land on the truth, position error {error:.1f} m")
    chk.ok(error < 4.0 * spread + 1.0, f"position error {error:.1f} m against a cloud spread of {spread:.1f} m")

    # (b) The likelihood itself, against an exact answer: a Gaussian needle with the flown needle's
    #     covariance (18 km long, ~0.02 thick), so the closed form above is exact.
    mean = cloud.mean(axis=0)
    cov = np.cov(cloud.T)
    gaussian_cloud = rng.multivariate_normal(mean, cov, size=len(cloud))
    truth_g = rng.multivariate_normal(mean, cov)
    sensors_g, sensor_state_g = sensor_near(truth_g, 12e3, rng)
    measurement_g = measure(truth_g, sensor_state_g, rng)
    q_g = sensors_g.coverage_fraction(make_track(gaussian_cloud))
    l_exact = closed_form_likelihood(mean, cov, measurement_g)
    kappa_g = R0 * P_D * l_exact     # existence lands mid-range, so L can be read back from it
    ordinary_g = run_update(gaussian_cloud, measurement_g, sensors_g, kappa_g)
    l_ordinary = implied_likelihood(ordinary_g.get_tracks()[0].existence_probability(), q_g, kappa_g)
    fused_g = run_update(gaussian_cloud, measurement_g, sensors_g, kappa_g, fused=True)
    diag_g = fused_g.take_diagnostics()
    l_fused = implied_likelihood(fused_g.get_tracks()[0].existence_probability(), q_g, kappa_g)
    chk.ok(diag_g["fused_components"][0] == 1, "the Gaussian needle must also take the fused path")
    chk.ok(l_ordinary < 0.1 * l_exact,
           f"scaffolding: the ordinary particle sum must collapse here ({l_ordinary:.3g} vs exact {l_exact:.3g})")
    # Kernel smoothing of a 2000-particle cloud bounds the agreement: over 8 seeds the ratio was
    # 0.65-1.11 (median 0.91), a scatter that is irrelevant to decisions won by factors of 1e16.
    chk.ok(0.5 < l_fused / l_exact < 2.0,
           f"fused association likelihood {l_fused:.4g} vs exact {l_exact:.4g} (ratio {l_fused / l_exact:.2f})")
    print(f"    flown needle {np.ptp(cloud[:, :3], axis=0).max() / 1e3:.1f} km long: fused ESS "
          f"{diag['fused_ess'][0]:.0f}/{len(cloud)}, re-acquired position error {error:.1f} m")
    print(f"    Gaussian needle: L ordinary/exact {l_ordinary / l_exact:.2g}, fused/exact {l_fused / l_exact:.2f}")
    return cloud, truth, sensors, sensor_state, measurement, q, kappa


# ---------------------------------------------------------------------------------------------
# F4, F5
# ---------------------------------------------------------------------------------------------


def check_reach_only(chk, rng):
    """A sparse, smooth cloud that happens to have no particle inside the sensor volume.

    (Not a cloud with a hole carved where the sensor is: there the density really is ~0 and the
    filter is right to refuse the match.)
    """
    truth = np.array([RADIUS, 0.0, 0.0, 0.0, 0.0, SPEED])
    sensors, sensor_state = sensor_near(truth, 12.0e3, rng)
    spread = np.concatenate([np.full(3, 40e3), np.full(3, 5.0)])
    for _ in range(200):
        cloud = truth + rng.normal(size=(40, 6)) * spread
        if sensors.coverage_fraction(make_track(cloud)) == 0.0:
            break
    chk.ok(sensors.coverage_fraction(make_track(cloud)) == 0.0, "scaffolding: no particle may be inside")
    measurement = measure(truth, sensor_state, rng)
    ordinary = run_update(cloud, measurement, sensors, 1e-30)
    chk.ok(len(ordinary.get_tracks()) == 2, "without the fused proposal the pair is ruled out (birth)")
    fused = run_update(cloud, measurement, sensors, 1e-30, fused=True)
    diag = fused.take_diagnostics()
    chk.ok(len(diag["time"]) == 1 and diag["fused_components"][0] == 1,
           "a reach-only cloud must be scored with the fused proposal")
    chk.ok(len(fused.get_tracks()) == 1, "the reach-only track must take the detection")
    chk.ok(fused.get_tracks()[0].existence_probability() > R0,
           "the reach-only track must gain existence from the detection it explains")


def check_fallback(chk, scene):
    cloud, truth, sensors, sensor_state, measurement, q, kappa = scene
    normal = run_update(cloud, measurement, sensors, kappa, fused=True)
    forced = run_update(cloud, measurement, sensors, kappa, fused=True, fallback_ess_min=1e12)
    diag = forced.take_diagnostics()
    chk.ok(diag["fallback_components"][0] == 1,
           "an unreachable fallback_ess_min must force the Gaussian fallback")
    l_normal = implied_likelihood(normal.get_tracks()[0].existence_probability(), q, kappa)
    l_forced = implied_likelihood(forced.get_tracks()[0].existence_probability(), q, kappa)
    chk.ok(0.5 < l_forced / l_normal < 2.0,
           f"fallback likelihood {l_forced:.4g} vs importance-sampled {l_normal:.4g}")
    certain = run_update(cloud, measurement, sensors, kappa * 1e-9, fused=True, fallback_ess_min=1e12)
    states = np.asarray(certain.get_tracks()[0].particle_states())
    error = np.linalg.norm(states[:, :3].mean(axis=0) - truth[:3])
    chk.ok(error < 200.0, f"the fallback posterior must also land on the truth ({error:.1f} m)")


def check_validation(chk):
    tracker = make_tracker(1e-3)
    chk.raises(lambda: tracker.set_fused_proposal(True, -1.0), ValueError, "ess_min", "negative ess_min")
    chk.raises(lambda: tracker.set_fused_proposal(True, 20.0, 3), ValueError, "neighbours", "too few neighbours")
    chk.raises(lambda: tracker.set_fused_proposal(True, 20.0, 0, -1.0), ValueError, "fallback_ess_min",
               "negative fallback_ess_min")
    tracker.set_fused_proposal(True)
    chk.ok(tracker.fused_proposal, "the setter must round-trip")


def main() -> int:
    chk = Checker()
    rng = np.random.default_rng(SEED)
    print("fused proposal")
    before = chk.count
    check_off_is_unchanged(chk, rng)
    print(f"  F1 off / never-triggered is unchanged: {chk.count - before} assertions")
    before = chk.count
    check_agrees_where_ordinary_is_valid(chk, rng)
    print(f"  F2 agrees with the ordinary update where both are valid: {chk.count - before} assertions")
    before = chk.count
    scene = check_needle_reacquisition(chk, rng)
    print(f"  F3 needle re-acquisition against the exact flown density: {chk.count - before} assertions")
    before = chk.count
    check_reach_only(chk, rng)
    print(f"  F4 reach-only cloud: {chk.count - before} assertions")
    before = chk.count
    check_fallback(chk, scene)
    print(f"  F5 Gaussian fallback: {chk.count - before} assertions")
    before = chk.count
    check_validation(chk)
    print(f"  F6 validation: {chk.count - before} assertions")
    print(f"PASS: test_fused_proposal ({chk.count} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
