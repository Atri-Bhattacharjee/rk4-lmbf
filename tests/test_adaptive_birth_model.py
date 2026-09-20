"""Statistical and structural tests for AdaptiveBirthModel (local tangent-frame Gaussian birth).

Birth samples eps ~ N(0, R_birth) in the measured direction's tangent frame, applies them with the
sphere exponential map and converts to ECI. The tests recover R_birth from the C++ local residual
(independently verified in tests/test_sensor_likelihood.py), check the physical spread against
rho*sigma, and prove the old circular-speed fan is gone. Tolerances follow the plan.
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

FIXED_SEED = 20260907
MU_EARTH_TEST = 3.986004418e14  # m^3/s^2, the test's own value for the old fan's circular speed
R_BIRTH_DIAG = np.diag(np.array([1000.0, 500.0, 1e-4, 1e-4, 5e-5, 5e-5]) ** 2)
N_RECOVERY = 50_000


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def test_scene(rng: np.random.Generator, range_rate: float = 6000.0):
    """LEO sensor; target 1e6 m away with purely radial relative velocity (los_rate = 0).

    The line of sight is drawn perpendicular to the sensor velocity, so the target's inertial speed is
    sqrt(v_s^2 + range_rate^2) ~ 9.7 km/s independent of the random directions. That is > 4 birth sigmas
    (sigma_speed ~ sigma_range_rate * range_rate / |v| ~ 310 m/s) above the largest circular speed
    sqrt(mu/r) (~8.3 km/s at the smallest |r| = 5.77e6 m in this geometry) that the removed fan used to
    impose, so assertion 5 is sharp. A random LOS would not do: for cos(angle(v_s, los)) ~ -0.33 the
    target speed lands exactly on the circular speed.
    """
    sensor_dir = ref.random_unit_vectors(rng, 1)[0]
    sensor_pos = 6.771e6 * sensor_dir
    sensor_vel = 7.67e3 * ref.unit(np.cross(sensor_dir, ref.random_unit_vectors(rng, 1)[0]))
    los = ref.unit(np.cross(sensor_vel, ref.random_unit_vectors(rng, 1)[0]))
    rho = 1.0e6
    target = np.concatenate([sensor_pos + rho * los, sensor_vel + range_rate * los])
    sensor = np.concatenate([sensor_pos, sensor_vel])
    return target, sensor


def make_measurement(target, sensor, covariance=R_BIRTH_DIAG) -> "lmb.Measurement":
    measurement = lmb.Measurement.fromCartesian(target, sensor)
    measurement.covariance_ = covariance
    measurement.timestamp_ = 0.0
    measurement.sensor_id_ = "birth-test"
    return measurement


def particle_states(track) -> np.ndarray:
    return np.array([p.state_vector for p in track.particles()])


def particle_weights(track) -> np.ndarray:
    return np.array([p.weight for p in track.particles()])


def birth_states(covariance, measurement, n: int, seed: int):
    model = lmb.AdaptiveBirthModel(n, 0.5, covariance, seed=seed)
    tracks = model.generate_new_tracks([measurement], 0.0)
    return tracks[0]


def local_residuals(measurement, states: np.ndarray) -> np.ndarray:
    observed = measurement.observation()
    sensor = measurement.sensor_state_
    return np.array([lmb.local_residual(observed, lmb.observe(state, sensor)) for state in states])


def dense_birth_covariance() -> np.ndarray:
    sigmas = np.sqrt(np.diag(R_BIRTH_DIAG))
    correlation = np.full((6, 6), 0.5)
    np.fill_diagonal(correlation, 1.0)
    return np.outer(sigmas, sigmas) * correlation


def check_determinism(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 1."""
    target, sensor = test_scene(rng)
    measurement = make_measurement(target, sensor)
    a = particle_states(birth_states(R_BIRTH_DIAG, measurement, 2000, seed=11))
    b = particle_states(birth_states(R_BIRTH_DIAG, measurement, 2000, seed=11))
    c = particle_states(birth_states(R_BIRTH_DIAG, measurement, 2000, seed=12))
    chk.ok(np.array_equal(a, b), "same seed must give bitwise-identical particles")
    chk.ok(not np.array_equal(a, c), "different seeds must give different particles")
    chk.ok(a.shape == (2000, 6), f"unexpected particle array shape {a.shape}")
    # Two unseeded models (random_device) must also differ from each other.
    d = particle_states(lmb.AdaptiveBirthModel(2000, 0.5, R_BIRTH_DIAG).generate_new_tracks([measurement], 0.0)[0])
    e = particle_states(lmb.AdaptiveBirthModel(2000, 0.5, R_BIRTH_DIAG).generate_new_tracks([measurement], 0.0)[0])
    chk.ok(not np.array_equal(d, e), "unseeded models must not be identical")


def check_degenerate(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 2: 1e-30 covariance collapses every particle onto to_cartesian(meas.observation())."""
    target, sensor = test_scene(rng)
    measurement = make_measurement(target, sensor)
    states = particle_states(birth_states(np.eye(6) * 1e-30, measurement, 500, seed=3))
    expected = lmb.to_cartesian(measurement.observation(), sensor)
    scale = np.array([np.linalg.norm(expected[:3])] * 3 + [np.linalg.norm(expected[3:])] * 3)
    chk.ok(np.all(np.abs(states - expected) <= 1e-9 * scale), "degenerate birth must reproduce the measurement state")
    chk.ok(np.all(np.abs(measurement.toCartesian() - expected) <= 1e-12 * scale),
           "Measurement.toCartesian disagrees with to_cartesian")


def check_covariance_recovery(chk: Checker, rng: np.random.Generator, covariance: np.ndarray, label: str) -> np.ndarray:
    """Assertion 3: sample statistics of the C++ local residual recover R_birth."""
    n = N_RECOVERY
    target, sensor = test_scene(rng)
    measurement = make_measurement(target, sensor, covariance)
    track = birth_states(covariance, measurement, n, seed=int(rng.integers(0, 2**63 - 1)))
    states = particle_states(track)
    residuals = local_residuals(measurement, states)
    chk.ok(np.all(np.isfinite(residuals)), f"[{label}] non-finite residuals")
    sigma = np.sqrt(np.diag(covariance))
    mean = residuals.mean(axis=0)
    chk.ok(np.all(np.abs(mean) <= 5.0 * sigma / np.sqrt(n)), f"[{label}] residual mean {mean / sigma} sigma off zero")
    sample_cov = np.cov(residuals, rowvar=False, ddof=1)
    variance_rel_err = np.abs(np.diag(sample_cov) / np.diag(covariance) - 1.0)
    chk.ok(np.all(variance_rel_err <= 5.0 * np.sqrt(2.0 / n)),
           f"[{label}] variance relative error {variance_rel_err} > {5.0 * np.sqrt(2.0 / n):.4f}")
    if label == "diagonal":
        correlation = sample_cov / np.outer(sigma, sigma)
        off = correlation[~np.eye(6, dtype=bool)]
        chk.ok(np.all(np.abs(off) <= 5.0 / np.sqrt(n)), f"[{label}] max |corr| {np.abs(off).max():.4f} > {5.0 / np.sqrt(n):.4f}")
    bound = 5.0 * np.sqrt((np.outer(np.diag(covariance), np.diag(covariance)) + covariance**2) / n)
    chk.ok(np.all(np.abs(sample_cov - covariance) <= bound), f"[{label}] sample covariance outside Wishart bound")
    print(f"    [{label}] max variance rel err {variance_rel_err.max():.4f}, "
          f"max |S-R|/bound {np.max(np.abs(sample_cov - covariance) / bound):.3f}")
    return states


def check_physical_spread_and_no_fan(chk: Checker, rng: np.random.Generator) -> None:
    """Assertions 4 and 5 on one 50 000-particle draw."""
    n = N_RECOVERY
    target, sensor = test_scene(rng)
    measurement = make_measurement(target, sensor)
    # fromCartesian leaves an ECI-roundoff LOS rate of ~eps*|v|/rho ~ 1e-15 rad/s; zero it exactly so the
    # transverse velocity of every particle is birth noise alone.
    chk.ok(np.linalg.norm(measurement.los_rate_) <= 1e-14, "test scene must have (numerically) zero LOS rate")
    measurement.los_rate_ = np.zeros(3)
    states = particle_states(birth_states(R_BIRTH_DIAG, measurement, n, seed=77))
    u_m = measurement.los_
    rho = measurement.range_
    sigma_theta = 1e-4
    sigma_omega = 5e-5

    rel_pos = states[:, :3] - sensor[:3]
    rel_vel = states[:, 3:] - sensor[3:]
    transverse_pos = rel_pos - np.outer(rel_pos @ u_m, u_m)
    transverse_vel = rel_vel - np.outer(rel_vel @ u_m, u_m)
    rms_pos = np.sqrt(np.mean(np.sum(transverse_pos**2, axis=1)))
    rms_vel = np.sqrt(np.mean(np.sum(transverse_vel**2, axis=1)))
    expected_pos = rho * sigma_theta * np.sqrt(2.0)
    expected_vel = rho * sigma_omega * np.sqrt(2.0)
    chk.ok(abs(rms_pos / expected_pos - 1.0) <= 0.03, f"cross-range position RMS {rms_pos:.2f} vs {expected_pos:.2f}")
    chk.ok(abs(rms_vel / expected_vel - 1.0) <= 0.03, f"transverse velocity RMS {rms_vel:.3f} vs {expected_vel:.3f}")

    transverse_speed = np.linalg.norm(transverse_vel, axis=1)
    fraction_far = np.mean(transverse_speed > 4.0 * rho * sigma_omega)
    chk.ok(fraction_far < 0.01, f"{fraction_far:.4f} of particles have transverse speed > 4 rho sigma_omega")
    # Unimodal at zero: the transverse speed of a 2-D isotropic Gaussian is Rayleigh; its mode is at
    # rho*sigma_omega and its median at rho*sigma_omega*sqrt(2 ln 2). Check the median to 3%.
    median = np.median(transverse_speed)
    chk.ok(abs(median / (rho * sigma_omega * np.sqrt(2.0 * np.log(2.0))) - 1.0) <= 0.03,
           f"transverse speed median {median:.3f} not Rayleigh")

    inertial_speed = np.linalg.norm(states[:, 3:], axis=1)
    circular_speed = np.sqrt(MU_EARTH_TEST / np.linalg.norm(states[:, :3], axis=1))
    # Scene precondition: the truth speed must sit >= 4 birth sigmas above the 1% band around the largest
    # circular speed reachable by any particle, otherwise this assertion could not distinguish fan from no fan.
    truth_speed = np.linalg.norm(target[3:])
    sigma_speed = 500.0 * abs(measurement.range_rate_) / truth_speed
    chk.ok(truth_speed - 4.0 * sigma_speed > 1.01 * circular_speed.max(),
           f"scene not separable: truth speed {truth_speed:.0f} m/s, sigma {sigma_speed:.0f} m/s, "
           f"max circular {circular_speed.max():.0f} m/s")
    within_one_percent = np.abs(inertial_speed / circular_speed - 1.0) <= 0.01
    chk.ok(not np.any(within_one_percent),
           f"{np.sum(within_one_percent)} particles sit within 1% of the old circular-speed cluster")
    chk.ok(abs(np.mean(inertial_speed) / truth_speed - 1.0) <= 0.01, "mean particle speed should follow the measurement")


def check_structure(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 6."""
    measurements = []
    for _ in range(4):
        target, sensor = test_scene(rng)
        measurements.append(make_measurement(target, sensor))
    model = lmb.AdaptiveBirthModel(1234, 0.37, R_BIRTH_DIAG, seed=5)
    tracks = model.generate_new_tracks(measurements, 42.0)
    chk.ok(len(tracks) == 4, f"expected one track per measurement, got {len(tracks)}")
    for index, track in enumerate(tracks):
        weights = particle_weights(track)
        chk.ok(len(weights) == 1234, "wrong particle count")
        chk.ok(abs(weights.sum() - 1.0) <= 1e-12, f"weights sum to {weights.sum()}")
        chk.ok(np.all(weights == 1.0 / 1234), "weights must all equal 1/N")
        chk.ok(track.existence_probability() == 0.37, "existence probability must equal ctor arg")
        chk.ok(track.label().index == index, f"label.index {track.label().index} != {index}")
        chk.ok(track.label().birth_time == 42, "label.birth_time must equal current_time")
        states = particle_states(track)
        chk.ok(np.all(np.isfinite(states)), "non-finite birth states")
        ranges = np.linalg.norm(states[:, :3] - measurements[index].sensor_state_[:3], axis=1)
        chk.ok(np.all(ranges > 0.0), "birth ranges must be positive")
    chk.ok(np.array_equal(model.birth_covariance_local(), R_BIRTH_DIAG), "birth_covariance_local getter mismatch")
    chk.ok(model.generate_new_tracks([], 0.0) == [], "no measurements must give no tracks")

    # Range redraw/clamp path: a birth range sigma ten times the range still yields valid states.
    target, sensor = test_scene(rng)
    measurement = make_measurement(target, sensor)
    wide = R_BIRTH_DIAG.copy()
    wide[0, 0] = (10.0 * measurement.range_) ** 2
    states = particle_states(birth_states(wide, measurement, 5000, seed=9))
    ranges = np.linalg.norm(states[:, :3] - sensor[:3], axis=1)
    chk.ok(np.all(np.isfinite(states)) and np.all(ranges > 0.0), "wide-range birth produced invalid states")


def run_all(seed: int) -> int:
    print(f"--- seed {seed} ---")
    rng = np.random.default_rng(seed)
    chk = Checker()
    steps = [
        ("A1 determinism", lambda: check_determinism(chk, rng)),
        ("A2 degenerate covariance", lambda: check_degenerate(chk, rng)),
        ("A3 covariance recovery (diagonal)", lambda: check_covariance_recovery(chk, rng, R_BIRTH_DIAG, "diagonal")),
        ("A3 covariance recovery (dense)", lambda: check_covariance_recovery(chk, rng, dense_birth_covariance(), "dense")),
        ("A4/A5 physical spread, no fan", lambda: check_physical_spread_and_no_fan(chk, rng)),
        ("A6 structure", lambda: check_structure(chk, rng)),
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
    print(f"PASS: test_adaptive_birth_model ({total} assertions over seeds {FIXED_SEED}, {time_seed})")


if __name__ == "__main__":
    main()
