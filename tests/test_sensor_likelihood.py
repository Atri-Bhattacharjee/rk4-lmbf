"""InOrbitSensorModel likelihood against an independent NumPy reference.

The reference residual/density live in tests/reference_geometry.py (plain-projection atan2 log
map, Rodrigues exponential map, closed-form transport, correlation-form Gaussian) and never call
lmb_engine. Tolerances follow the plan; where ECI-scale roundoff (positions ~7e6 m, ulp ~1e-9 m)
makes a literal tolerance unattainable, a documented roundoff model is used instead.
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
import reference_geometry as ref  # noqa: E402

lmb = import_lmb_engine()

EPS = np.finfo(float).eps
FIXED_SEED = 20260906
PI = np.pi


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def particle_at(state) -> "lmb.Particle":
    particle = lmb.Particle()
    particle.state_vector = np.asarray(state, dtype=float)
    particle.weight = 1.0
    return particle


def random_scene(rng: np.random.Generator, angular_speed: float | None = None):
    """(target_state, sensor_state) with LEO sensor and a target 2e5..2e6 m away; |u_dot| ~ angular_speed."""
    sensor_dir = ref.random_unit_vectors(rng, 1)[0]
    sensor_pos = 6.771e6 * sensor_dir
    sensor_vel = 7.67e3 * ref.unit(np.cross(sensor_dir, ref.random_unit_vectors(rng, 1)[0]))
    los = ref.random_unit_vectors(rng, 1)[0]
    rho = rng.uniform(2.0e5, 2.0e6)
    speed = angular_speed if angular_speed is not None else 10.0 ** rng.uniform(-4.0, -2.5)
    transverse = ref.unit(np.cross(los, ref.random_unit_vectors(rng, 1)[0])) * speed * rho
    radial = rng.uniform(-2000.0, 2000.0) * los
    target = np.concatenate([sensor_pos + rho * los, sensor_vel + radial + transverse])
    return target, np.concatenate([sensor_pos, sensor_vel])


def measurement_from(target, sensor, covariance) -> "lmb.Measurement":
    measurement = lmb.Measurement.fromCartesian(target, sensor)
    measurement.covariance_ = covariance
    return measurement


def as_ref_obs(measurement) -> ref.LosObs:
    return ref.LosObs(measurement.range_, measurement.range_rate_, measurement.los_, measurement.los_rate_)


def random_covariance(rng: np.random.Generator, sigmas, dense: bool) -> np.ndarray:
    sigmas = np.asarray(sigmas, dtype=float)
    if not dense:
        return np.diag(sigmas**2)
    a = rng.normal(size=(6, 6))
    raw = a @ a.T + np.eye(6)
    d = np.sqrt(np.diag(raw))
    correlation = raw / np.outer(d, d)
    return np.outer(sigmas, sigmas) * correlation


def residual_tolerances(target, sensor, obs) -> np.ndarray:
    """Roundoff model for a residual between two evaluations of the same geometry in double."""
    position_scale = np.linalg.norm(target[:3])
    velocity_scale = np.linalg.norm(target[3:])
    rate_scale = velocity_scale / obs.range + np.linalg.norm(obs.los_rate)
    return 8.0 * EPS * np.array([position_scale, velocity_scale, 1.0, 1.0, rate_scale, rate_scale])


def check_basis_specification(chk: Checker, rng: np.random.Generator) -> None:
    for u in ref.random_unit_vectors(rng, 200):
        basis = lmb.tangent_basis(u)
        spec = ref.tangent_basis_spec(u)
        chk.ok(np.max(np.abs(basis - spec)) <= 1e-15, "tangent_basis differs from its specification")


def check_independent_reference(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 1: C++ likelihood vs NumPy reference for 2000 pairs, gamma in [1e-4, 0.3]."""
    gammas = []
    for index in range(2000):
        dense = index % 2 == 1
        target, sensor = random_scene(rng)
        gamma_target = 10.0 ** rng.uniform(-4.0, np.log10(0.3))
        k = rng.uniform(0.5, 3.0)
        sigmas = np.array([
            10.0 ** rng.uniform(np.log10(50.0), 4.0),
            10.0 ** rng.uniform(-1.0, 2.0),
            gamma_target / k,
            gamma_target / k,
            10.0 ** rng.uniform(-7.0, -3.0),
            10.0 ** rng.uniform(-7.0, -3.0),
        ])
        covariance = random_covariance(rng, sigmas, dense)
        if dense:
            scale = np.sqrt(np.diag(covariance))
            condition = np.linalg.cond(covariance / np.outer(scale, scale))
            chk.ok(condition <= 1e6, f"dense correlation matrix condition number {condition:.3e} > 1e6")
        measurement = measurement_from(target, sensor, covariance)
        obs = as_ref_obs(measurement)

        xi = rng.normal(size=6)
        eps = np.linalg.cholesky(covariance) @ xi
        angular = np.linalg.norm(eps[2:4])
        eps[2:4] *= gamma_target / angular
        particle_state = ref.to_cartesian_ref(ref.perturbed_ref(obs, eps), sensor)
        particle = particle_at(particle_state)

        predicted_ref = ref.observe_ref(particle_state, sensor)
        residual_ref = ref.local_residual_ref(obs, predicted_ref)
        gammas.append(float(ref.geodesic_angle_ld(obs.los, predicted_ref.los)))
        log_l_ref = ref.gaussian_log_density(residual_ref, covariance)
        l_ref = np.exp(log_l_ref)
        chk.ok(np.isfinite(l_ref) and l_ref > 0.0, "reference likelihood underflowed; test configuration error")

        l_cpp = lmb.InOrbitSensorModel(*sigmas**2).calculate_likelihood(particle, measurement)
        chk.ok(abs(l_cpp - l_ref) <= 1e-10 * l_ref,
               f"likelihood mismatch rel {abs(l_cpp - l_ref) / l_ref:.3e} (dense={dense}, gamma={gammas[-1]:.2e})")

        predicted_cpp = lmb.observe(particle_state, sensor)
        residual_cpp = lmb.local_residual(measurement.observation(), predicted_cpp)
        tolerance = np.maximum(residual_tolerances(target, sensor, obs), 1e-11 * np.abs(residual_ref))
        chk.ok(np.all(np.abs(residual_cpp - residual_ref) <= tolerance),
               f"residual mismatch {residual_cpp - residual_ref} > {tolerance}")
    gammas = np.array(gammas)
    chk.ok(gammas.min() < 3e-4 and gammas.max() > 0.1, "gamma sweep did not cover [1e-4, 0.3]")


def check_peak(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 2: particle at the measurement's own Cartesian state gives the density peak."""
    for index in range(50):
        target, sensor = random_scene(rng)
        sigmas = np.array([100.0, 1.0, 1e-4, 1e-4, 1e-5, 1e-5])
        covariance = random_covariance(rng, sigmas, dense=index % 2 == 1)
        measurement = measurement_from(target, sensor, covariance)
        particle = particle_at(measurement.toCartesian())
        peak = (2.0 * PI) ** -3 * np.linalg.det(covariance) ** -0.5
        l_cpp = lmb.InOrbitSensorModel(*sigmas**2).calculate_likelihood(particle, measurement)
        chk.ok(abs(l_cpp - peak) <= 1e-13 * peak, f"peak likelihood {l_cpp} vs {peak}")
        residual = lmb.local_residual(measurement.observation(), lmb.observe(particle.state_vector, sensor))
        # "Exactly zero" is unattainable for the range component: r_s + rho*u - r_s at |r| ~ 7e6 m rounds
        # at the 1e-9 m level. Roundoff model: 8 eps x magnitude of the underlying quantity.
        tolerance = residual_tolerances(target, sensor, as_ref_obs(measurement))
        chk.ok(np.all(np.abs(residual) <= tolerance), f"peak residual {residual} exceeds roundoff model {tolerance}")


def rotate_scene(q, target, sensor, measurement):
    rotated_target = np.concatenate([q @ target[:3], q @ target[3:]])
    rotated_sensor = np.concatenate([q @ sensor[:3], q @ sensor[3:]])
    rotated = lmb.Measurement()
    rotated.range_ = measurement.range_
    rotated.range_rate_ = measurement.range_rate_
    rotated.los_ = q @ measurement.los_
    rotated.los_rate_ = q @ measurement.los_rate_
    rotated.covariance_ = measurement.covariance_
    rotated.sensor_state_ = rotated_sensor
    return rotated_target, rotated_sensor, rotated


def check_rotation_invariance(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 3: likelihood is invariant under rigid rotations of the whole scene (pole proof).

    Rotating ECI states of magnitude |r| ~ 7e6 m (ulp ~1e-9 m) and |v| ~ 7e3 m/s perturbs the relative
    geometry by ~eps*|r| in position, hence the measured/predicted directions by ~eps*|r|/rho (up to
    ~6e-15 rad for |r|/rho ~ 27) regardless of the likelihood implementation. The tolerance on L is
    therefore 1e-12 plus the first-order propagation of that roundoff through the quadratic form,
    sum_k |r_k| delta_k / sigma_k^2 with delta = 4 eps [|r|, |v|, |r|/rho, |r|/rho, |v|/rho + |udot|, ...].
    Measured without this term: up to 2.0e-12 relative for |r|/rho = 27 (literal 1e-12 unattainable).
    sigma = (2000 m, 5 m/s, 2e-3 rad, 2e-3 rad, 2e-5 rad/s, 2e-5 rad/s), residual ~ (100, 1, 2e-3, 2e-5).
    """
    sigmas = np.array([2000.0, 5.0, 2e-3, 2e-3, 2e-5, 2e-5])
    covariance = np.diag(sigmas**2)
    sensor_model = lmb.InOrbitSensorModel(*sigmas**2)
    for _ in range(5):
        target, sensor = random_scene(rng, angular_speed=1e-3)
        measurement = measurement_from(target, sensor, covariance)
        obs = as_ref_obs(measurement)
        eps = np.array([100.0, 1.0, 1.5e-3, -1.0e-3, 1.5e-5, -1.0e-5]) * rng.choice([-1.0, 1.0], size=6)
        particle_state = ref.to_cartesian_ref(ref.perturbed_ref(obs, eps), sensor)
        l_base = sensor_model.calculate_likelihood(particle_at(particle_state), measurement)
        residual_base = lmb.local_residual(measurement.observation(), lmb.observe(particle_state, sensor))

        position_scale = np.linalg.norm(target[:3])
        velocity_scale = np.linalg.norm(target[3:])
        rate_scale = velocity_scale / obs.range + np.linalg.norm(obs.los_rate)
        delta = 4.0 * EPS * np.array([position_scale, velocity_scale, position_scale / obs.range,
                                      position_scale / obs.range, rate_scale, rate_scale])
        l_tolerance = 1e-12 + float(np.sum(np.abs(residual_base) * delta / sigmas**2))
        angle_norm = np.linalg.norm(residual_base[2:4])
        rate_norm = np.linalg.norm(residual_base[4:6])
        angle_tolerance = 2e-13 * angle_norm + 2.0 * delta[2]
        rate_tolerance = 2e-13 * rate_norm + 2.0 * delta[4]

        rotations = [ref.rotation_matrix(ref.random_unit_vectors(rng, 1)[0], rng.uniform(0.0, 2.0 * PI))
                     for _ in range(200)]
        z = np.array([0.0, 0.0, 1.0])
        to_plus_z = ref.rotation_taking(measurement.los_, z)
        to_minus_z = ref.rotation_taking(measurement.los_, -z)
        tilt = ref.rotation_matrix(np.array([1.0, 0.0, 0.0]), 1e-12)
        rotations += [to_plus_z, to_minus_z, tilt @ to_plus_z]
        for q in rotations:
            rotated_target, rotated_sensor, rotated = rotate_scene(q, particle_state, sensor, measurement)
            l_rot = sensor_model.calculate_likelihood(particle_at(rotated_target), rotated)
            chk.ok(abs(l_rot - l_base) <= l_tolerance * l_base,
                   f"rotation changed L by {abs(l_rot - l_base) / l_base:.3e} > {l_tolerance:.3e} "
                   f"(los_z={rotated.los_[2]:.3f})")
            residual_rot = lmb.local_residual(rotated.observation(), lmb.observe(rotated_target, rotated_sensor))
            chk.ok(abs(residual_rot[0] - residual_base[0]) <= 2.0 * delta[0], "range residual not rotation invariant")
            chk.ok(abs(residual_rot[1] - residual_base[1]) <= 2.0 * delta[1], "range-rate residual not rotation invariant")
            chk.ok(abs(np.linalg.norm(residual_rot[2:4]) - angle_norm) <= angle_tolerance,
                   f"angular residual norm not rotation invariant: "
                   f"{abs(np.linalg.norm(residual_rot[2:4]) - angle_norm):.3e} > {angle_tolerance:.3e}")
            chk.ok(abs(np.linalg.norm(residual_rot[4:6]) - rate_norm) <= rate_tolerance,
                   "rate residual norm not rotation invariant")
        # The pole cases must have put the measured direction on the z-axis.
        chk.ok(abs(abs((to_plus_z @ measurement.los_)[2]) - 1.0) <= 1e-15, "pole rotation did not reach +z")


def check_chi_square(chk: Checker, rng: np.random.Generator, sigmas, dense: bool, label: str) -> None:
    """Assertion 4: d^2 = -2(log L - log_norm) of particles drawn from N(0, R) around the measurement is chi^2_6."""
    n = 20000
    covariance = random_covariance(rng, sigmas, dense)
    target, sensor = random_scene(rng, angular_speed=1e-3)
    measurement = measurement_from(target, sensor, covariance)
    obs = as_ref_obs(measurement)
    sensor_model = lmb.InOrbitSensorModel(*sigmas**2)
    log_norm = -3.0 * np.log(2.0 * PI) - 0.5 * np.linalg.slogdet(covariance)[1]
    chol = np.linalg.cholesky(covariance)
    d2 = np.empty(n)
    for i in range(n):
        eps = chol @ rng.normal(size=6)
        state = ref.to_cartesian_ref(ref.perturbed_ref(obs, eps), sensor)
        likelihood = sensor_model.calculate_likelihood(particle_at(state), measurement)
        d2[i] = -2.0 * (np.log(likelihood) - log_norm)
    chk.ok(np.all(np.isfinite(d2)), f"[{label}] non-finite d^2")
    mean = d2.mean()
    var = d2.var(ddof=1)
    chk.ok(abs(mean - 6.0) <= 5.0 * np.sqrt(12.0 / n), f"[{label}] mean d^2 = {mean:.4f} not ~6")
    chk.ok(abs(var - 12.0) <= 5.0 * np.sqrt(2.0 * 144.0 / n), f"[{label}] var d^2 = {var:.4f} not ~12")
    for x in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        empirical = np.mean(d2 <= x)
        analytic = ref.chi_square_6_cdf(x)
        bound = 5.0 * np.sqrt(analytic * (1.0 - analytic) / n)
        chk.ok(abs(empirical - analytic) <= bound,
               f"[{label}] ECDF({x}) = {empirical:.4f} vs chi2_6 {analytic:.4f} (bound {bound:.4f})")
    print(f"    [{label}] mean d^2 = {mean:.4f}, var = {var:.3f}")


def check_rate_sensitivity(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 5: shifting one measured quantity by k sigma changes -2 log L by exactly k^2."""
    sigmas = np.array([100.0, 1.0, 1e-3, 1e-3, 1e-5, 1e-5])
    sensor_model = lmb.InOrbitSensorModel(*sigmas**2)
    for _ in range(20):
        target, sensor = random_scene(rng, angular_speed=1e-3)
        measurement = measurement_from(target, sensor, np.diag(sigmas**2))
        particle = particle_at(measurement.toCartesian())
        log_l0 = np.log(sensor_model.calculate_likelihood(particle, measurement))
        basis = lmb.tangent_basis(measurement.los_)
        for k in (1.0, 2.0, 3.0):
            shifted = {
                "rate e1": lambda m: setattr(m, "los_rate_", m.los_rate_ + k * sigmas[4] * basis[:, 0]),
                "rate e2": lambda m: setattr(m, "los_rate_", m.los_rate_ + k * sigmas[5] * basis[:, 1]),
                "range": lambda m: setattr(m, "range_", m.range_ + k * sigmas[0]),
                "range rate": lambda m: setattr(m, "range_rate_", m.range_rate_ + k * sigmas[1]),
            }
            for name, apply in shifted.items():
                shifted_measurement = measurement_from(target, sensor, np.diag(sigmas**2))
                apply(shifted_measurement)
                log_lk = np.log(sensor_model.calculate_likelihood(particle, shifted_measurement))
                delta = -2.0 * (log_lk - log_l0)
                chk.ok(abs(delta - k * k) <= 1e-9, f"{name}: -2 dlogL = {delta} for k={k}")


def check_no_wrapping(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 6: a predicted direction 3.0 rad away gives a finite likelihood equal to the reference."""
    sigmas = np.array([100.0, 1.0, 1.0, 1.0, 1e-3, 1e-3])
    covariance = np.diag(sigmas**2)
    sensor_model = lmb.InOrbitSensorModel(*sigmas**2)
    for _ in range(20):
        target, sensor = random_scene(rng, angular_speed=1e-3)
        measurement = measurement_from(target, sensor, covariance)
        obs = as_ref_obs(measurement)
        angle = rng.uniform(0.0, 2.0 * PI)
        eps = np.array([0.0, 0.0, 3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.0, 0.0])
        state = ref.to_cartesian_ref(ref.perturbed_ref(obs, eps), sensor)
        l_cpp = sensor_model.calculate_likelihood(particle_at(state), measurement)
        residual_ref = ref.local_residual_ref(obs, ref.observe_ref(state, sensor))
        chk.ok(abs(np.linalg.norm(residual_ref[2:4]) - 3.0) <= 1e-9, "reference geometry did not reach gamma = 3")
        l_ref = np.exp(ref.gaussian_log_density(residual_ref, covariance))
        chk.ok(np.isfinite(l_cpp) and l_cpp > 0.0, "likelihood at gamma = 3 not finite/positive")
        chk.ok(abs(l_cpp - l_ref) <= 1e-10 * l_ref, f"gamma = 3 likelihood {l_cpp} vs reference {l_ref}")


def check_default_covariance(chk: Checker) -> None:
    """Assertion 7."""
    args = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    chk.ok(np.array_equal(lmb.InOrbitSensorModel(*args).defaultCovariance(), np.diag(args)),
           "defaultCovariance must equal diag(ctor args)")
    default = lmb.InOrbitSensorModel().defaultCovariance()
    chk.ok(np.all(np.diag(default) > 0.0) and np.array_equal(default, default.T), "default covariance not symmetric positive")
    np.linalg.cholesky(default)
    chk.ok(default.shape == (6, 6), "default covariance must be 6x6")


def check_cache_path(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 8: the tracker's cached path equals the direct likelihood."""
    sigmas = np.array([100.0, 1.0, 1e-3, 1e-3, 1e-5, 1e-5])
    sensor_model = lmb.InOrbitSensorModel(*sigmas**2)
    birth_model = lmb.AdaptiveBirthModel(10, 0.5, np.diag(sigmas**2), seed=1)
    tracker = lmb.SMC_LMB_Tracker(lmb.TwoBodyPropagator(np.zeros((6, 6))), sensor_model, birth_model,
                                  0.99, 2, 0.001, 1e-9, 0.99, 0.0, 1.0)
    for index in range(20):
        target, sensor = random_scene(rng, angular_speed=1e-3)
        covariance = random_covariance(rng, sigmas, dense=index % 2 == 1)
        measurement = measurement_from(target, sensor, covariance)
        eps = np.linalg.cholesky(covariance) @ rng.normal(size=6)
        state = ref.to_cartesian_ref(ref.perturbed_ref(as_ref_obs(measurement), eps), sensor)
        particle = particle_at(state)
        label = lmb.TrackLabel()
        track = lmb.Track(label, 0.9, [particle])
        direct = sensor_model.calculate_likelihood(particle, measurement)
        cached = tracker.compute_association_likelihood(track, measurement)
        chk.ok(abs(cached - direct) <= 1e-12 * direct, f"cache path {cached} vs direct {direct}")


def run_all(seed: int) -> int:
    print(f"--- seed {seed} ---")
    rng = np.random.default_rng(seed)
    chk = Checker()
    steps = [
        ("basis specification", lambda: check_basis_specification(chk, rng)),
        ("A1 independent reference", lambda: check_independent_reference(chk, rng)),
        ("A2 peak", lambda: check_peak(chk, rng)),
        ("A3 rotation invariance", lambda: check_rotation_invariance(chk, rng)),
        ("A4 chi-square narrow (dense R)", lambda: check_chi_square(
            chk, rng, np.array([100.0, 1.0, 1e-3, 1e-3, 1e-5, 1e-5]), True, "narrow")),
        ("A4 chi-square wide (sigma_theta = 0.05)", lambda: check_chi_square(
            chk, rng, np.array([100.0, 1.0, 0.05, 0.05, 1e-6, 1e-6]), False, "wide")),
        ("A5 rate sensitivity", lambda: check_rate_sensitivity(chk, rng)),
        ("A6 no wrapping", lambda: check_no_wrapping(chk, rng)),
        ("A7 default covariance", lambda: check_default_covariance(chk)),
        ("A8 cache path", lambda: check_cache_path(chk, rng)),
    ]
    for name, fn in steps:
        before = chk.count
        fn()
        print(f"  {name}: {chk.count - before} assertions")
    return chk.count


def main() -> None:
    time_seed = int(time.time())
    total = sum(run_all(seed) for seed in (FIXED_SEED, time_seed))
    if total <= 0:
        raise AssertionError("no assertions executed")
    print(f"PASS: test_sensor_likelihood ({total} assertions over seeds {FIXED_SEED}, {time_seed})")


if __name__ == "__main__":
    main()
