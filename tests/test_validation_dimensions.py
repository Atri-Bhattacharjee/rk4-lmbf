"""Input validation of Measurement and model constructors through the Python bindings.

Every invalid input must raise ValueError (pybind maps std::invalid_argument) with a message
that names the offending field, through both the sensor-model and the birth-model entry points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))
from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

SENSOR = np.array([6.771e6, 0.0, 0.0, 0.0, 7672.0, 0.0])
TARGET = np.array([6.771e6 + 4.0e5, 3.0e5, 2.0e5, -600.0, 7300.0, 400.0])
COVARIANCE = np.diag([50.0**2, 1.0**2, 1e-4**2, 1e-4**2, 1e-5**2, 1e-5**2])
BIRTH_COVARIANCE = np.diag([1000.0**2, 500.0**2, 1e-4**2, 1e-4**2, 5e-5**2, 5e-5**2])

assertion_count = 0


def check(condition, message: str) -> None:
    global assertion_count
    assertion_count += 1
    if not bool(condition):
        raise AssertionError(message)


def expect_value_error(action, field: str) -> None:
    """Run `action`; require a ValueError whose message names `field`."""
    global assertion_count
    assertion_count += 1
    try:
        action()
    except ValueError as error:
        if field not in str(error):
            raise AssertionError(f"ValueError message {str(error)!r} does not name {field!r}") from None
        return
    raise AssertionError(f"expected ValueError naming {field!r}, nothing was raised")


def valid_measurement() -> "lmb.Measurement":
    measurement = lmb.Measurement.fromCartesian(TARGET, SENSOR)
    measurement.covariance_ = COVARIANCE
    measurement.timestamp_ = 12.0
    measurement.sensor_id_ = "sensor-a"
    return measurement


def particle_at(state) -> "lmb.Particle":
    particle = lmb.Particle()
    particle.state_vector = np.asarray(state, dtype=float)
    particle.weight = 1.0
    return particle


def entry_points():
    sensor_model = lmb.InOrbitSensorModel(*np.diag(COVARIANCE))
    birth_model = lmb.AdaptiveBirthModel(50, 0.5, BIRTH_COVARIANCE, seed=1)
    particle = particle_at(TARGET)
    return [
        ("calculate_likelihood", lambda m: sensor_model.calculate_likelihood(particle, m)),
        ("generate_new_tracks", lambda m: birth_model.generate_new_tracks([m], 0.0)),
    ]


def test_valid_measurement_passes() -> None:
    measurement = valid_measurement()
    for name, entry in entry_points():
        result = entry(measurement)
        check(result is not None, f"{name} rejected a valid measurement")
    likelihood = lmb.InOrbitSensorModel(*np.diag(COVARIANCE)).calculate_likelihood(particle_at(TARGET), measurement)
    check(np.isfinite(likelihood) and likelihood > 0.0, "valid measurement must have a finite positive likelihood")

    obs = measurement.observation()
    copy = lmb.Measurement()
    copy.setObservation(obs)
    check(copy.range_ == measurement.range_ and copy.range_rate_ == measurement.range_rate_,
          "setObservation must round-trip range fields bitwise")
    check(np.array_equal(copy.los_, measurement.los_) and np.array_equal(copy.los_rate_, measurement.los_rate_),
          "setObservation must round-trip los fields bitwise")
    check(np.array_equal(measurement.observation().los, obs.los), "observation() must be deterministic")
    check(np.max(np.abs(measurement.toCartesian() - TARGET)) <= 1e-9 * np.linalg.norm(TARGET),
          "toCartesian must invert fromCartesian")
    check(measurement.covariance_.shape == (6, 6), "covariance_ must be 6x6")
    check(lmb.MEAS_DIM == 6, "MEAS_DIM must be 6")


def test_measurement_field_rejections() -> None:
    def with_range(value):
        m = valid_measurement()
        m.range_ = value
        return m

    def with_los_scale(scale):
        m = valid_measurement()
        m.los_ = m.los_ * scale
        return m

    def with_non_tangent_rate():
        m = valid_measurement()
        m.los_rate_ = m.los_rate_ + 1e-3 * m.los_
        return m

    def with_non_symmetric_covariance():
        m = valid_measurement()
        cov = COVARIANCE.copy()
        cov[0, 1] = 5.0
        m.covariance_ = cov
        return m

    def with_zero_variance():
        m = valid_measurement()
        cov = COVARIANCE.copy()
        cov[4, 4] = 0.0
        m.covariance_ = cov
        return m

    def with_nan_sensor_state():
        m = valid_measurement()
        state = SENSOR.copy()
        state[2] = np.nan
        m.sensor_state_ = state
        return m

    def with_nan_range_rate():
        m = valid_measurement()
        m.range_rate_ = np.nan
        return m

    cases = [
        ("range_", lambda: with_range(0.0)),
        ("range_", lambda: with_range(-1.0)),
        ("range_", lambda: with_range(np.nan)),
        ("range_rate_", with_nan_range_rate),
        ("los_", lambda: with_los_scale(1.0 + 1e-3)),
        ("los_rate_", with_non_tangent_rate),
        ("covariance_", with_non_symmetric_covariance),
        ("covariance_", with_zero_variance),
        ("sensor_state_", with_nan_sensor_state),
    ]
    for field, build in cases:
        for name, entry in entry_points():
            expect_value_error(lambda: entry(build()), field)

    def with_negative_eigenvalue():
        m = valid_measurement()
        cov = COVARIANCE.copy()
        cov[0, 1] = cov[1, 0] = 2.0 * np.sqrt(cov[0, 0] * cov[1, 1])  # correlation 2 -> indefinite
        m.covariance_ = cov
        return m

    for name, entry in entry_points():
        if name == "calculate_likelihood" or lmb.VALIDATION_ENABLED:
            # The likelihood cache factors the (dense) covariance and always rejects an indefinite one;
            # the birth model only does so in debug/validation builds (it never uses covariance_).
            expect_value_error(lambda: entry(with_negative_eigenvalue()), "covariance_")
        else:
            entry(with_negative_eigenvalue())
            check(True, "release-build birth accepts an indefinite measurement covariance (documented)")


def test_assignment_dimension_rejections() -> None:
    def assign_cov_4x4():
        m = valid_measurement()
        m.covariance_ = np.eye(4)

    def assign_cov_6x5():
        m = valid_measurement()
        m.covariance_ = np.ones((6, 5))

    def assign_sensor_state_5():
        m = valid_measurement()
        m.sensor_state_ = np.zeros(5)

    def assign_los_2():
        m = valid_measurement()
        m.los_ = np.array([1.0, 0.0])

    def assign_los_rate_4():
        m = valid_measurement()
        m.los_rate_ = np.zeros(4)

    expect_value_error(assign_cov_4x4, "covariance_")
    expect_value_error(assign_cov_6x5, "covariance_")
    expect_value_error(assign_sensor_state_5, "sensor_state_")
    expect_value_error(assign_los_2, "los_")
    expect_value_error(assign_los_rate_4, "los_rate_")
    expect_value_error(lambda: lmb.Measurement.fromCartesian(np.zeros(5), SENSOR), "target_state")
    expect_value_error(lambda: lmb.Measurement.fromCartesian(TARGET, np.zeros(7)), "sensor_state")
    expect_value_error(lambda: valid_measurement().perturbed(np.zeros(4)), "eps")
    expect_value_error(lambda: lmb.observe(np.zeros(3), SENSOR), "target_state")
    expect_value_error(lambda: lmb.perturbed(lmb.LosObservation(), np.zeros(5)), "eps")


def test_model_constructor_rejections() -> None:
    expect_value_error(lambda: lmb.AdaptiveBirthModel(100, 0.5, np.eye(4)), "birth_covariance")
    non_spd = BIRTH_COVARIANCE.copy()
    non_spd[2, 2] = -1e-8
    expect_value_error(lambda: lmb.AdaptiveBirthModel(100, 0.5, non_spd), "birth_covariance")
    non_symmetric = BIRTH_COVARIANCE.copy()
    non_symmetric[0, 1] = 1.0
    expect_value_error(lambda: lmb.AdaptiveBirthModel(100, 0.5, non_symmetric), "birth_covariance")
    expect_value_error(lambda: lmb.AdaptiveBirthModel(0, 0.5, BIRTH_COVARIANCE), "particles_per_track")
    expect_value_error(lambda: lmb.InOrbitSensorModel(1.0, 1.0, 1.0, 0.0, 1.0, 1.0), "variances")
    expect_value_error(lambda: lmb.InOrbitSensorModel(1.0, 1.0, 1.0, 1.0, -1.0, 1.0), "variances")


def test_tracker_update_validates() -> None:
    propagator = lmb.TwoBodyPropagator(np.zeros((6, 6)))
    sensor_model = lmb.InOrbitSensorModel(*np.diag(COVARIANCE))
    birth_model = lmb.AdaptiveBirthModel(50, 0.5, BIRTH_COVARIANCE, seed=3)
    tracker = lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99, 2, 0.001, 1e-9, 0.99, 0.0, 1.0)
    bad = valid_measurement()
    bad.range_ = -5.0
    expect_value_error(lambda: tracker.update([bad]), "range_")
    tracker.update([valid_measurement()])
    check(len(tracker.get_tracks()) == 1, "tracker must birth one track from one valid measurement")


def test_sensor_array_dimension_rejections() -> None:
    """Dimension and finiteness of everything Python hands the sensor array each timestep."""
    sensors = lmb.SensorArray(lmb.SensorFovConfig())
    sensors.add("sensor-a", SENSOR, np.array([1.0, 0.0, 0.0]))

    expect_value_error(lambda: sensors.add("sensor-b", np.zeros(5), np.array([1.0, 0.0, 0.0])), "state")
    expect_value_error(lambda: sensors.add("sensor-b", np.zeros(7), np.array([1.0, 0.0, 0.0])), "state")
    expect_value_error(lambda: sensors.add("sensor-b", np.full(6, np.inf), np.array([1.0, 0.0, 0.0])), "state")
    expect_value_error(lambda: sensors.add("sensor-b", SENSOR, np.zeros(2)), "boresight")
    expect_value_error(lambda: sensors.add("sensor-b", SENSOR, np.zeros(6)), "boresight")
    expect_value_error(lambda: sensors.add("sensor-b", SENSOR, np.full(3, np.nan)), "boresight")

    expect_value_error(lambda: sensors.set_state(0, np.zeros(3)), "state")
    expect_value_error(lambda: sensors.set_state(0, np.full(6, np.nan)), "state")
    expect_value_error(lambda: sensors.set_boresight(0, np.zeros(4)), "boresight")
    expect_value_error(lambda: sensors.set_boresight(0, np.zeros(3)), "boresight")
    expect_value_error(lambda: sensors.set_pointing(0, np.array([1.0, 0.0, 0.0]), np.zeros(3)), "up")
    expect_value_error(lambda: sensors.set_pointing(0, np.array([1.0, 0.0, 0.0]),
                                                    np.array([3.0, 0.0, 0.0])), "up")
    expect_value_error(lambda: sensors.point_at(0, np.zeros(4)), "target_position")
    expect_value_error(lambda: sensors.point_at(0, np.full(3, np.nan)), "target_position")
    expect_value_error(lambda: sensors.sees(0, np.zeros(4)), "target")

    # A rejected call must not have half-applied itself.
    check(len(sensors) == 1, "rejected adds must not grow the array")
    check(np.allclose(np.asarray(sensors.boresight(0)), [1.0, 0.0, 0.0]),
          "rejected pointing calls must leave the boresight alone")
    check(np.allclose(np.asarray(sensors.state(0)), SENSOR),
          "rejected set_state calls must leave the state alone")


def test_tracker_update_resolves_sensors() -> None:
    """update(measurements, sensors) must refuse a measurement it cannot attribute."""
    propagator = lmb.TwoBodyPropagator(np.zeros((6, 6)))
    sensor_model = lmb.InOrbitSensorModel(*np.diag(COVARIANCE))
    birth_model = lmb.AdaptiveBirthModel(50, 0.5, BIRTH_COVARIANCE, seed=3)
    tracker = lmb.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99, 2, 0.001, 1e-9, 0.99, 0.0, 1.0)

    sensors = lmb.SensorArray(lmb.SensorFovConfig())
    sensors.add_unpointed("sensor-a", SENSOR)

    # A birth step (no tracks yet) does not consult the array, so seed a track first.
    tracker.update([valid_measurement()], sensors)
    check(len(tracker.get_tracks()) == 1, "one valid measurement must birth one track")

    stray = valid_measurement()
    stray.sensor_id_ = "sensor-z"
    expect_value_error(lambda: tracker.update([stray], sensors), "sensor-z")

    bad = valid_measurement()
    bad.range_ = -5.0
    expect_value_error(lambda: tracker.update([bad], sensors), "range_")


def main() -> None:
    test_valid_measurement_passes()
    test_measurement_field_rejections()
    test_assignment_dimension_rejections()
    test_model_constructor_rejections()
    test_tracker_update_validates()
    test_sensor_array_dimension_rejections()
    test_tracker_update_resolves_sensors()
    check(assertion_count > 0, "no assertions executed")
    print(f"PASS: test_validation_dimensions ({assertion_count} assertions, validation_enabled={lmb.VALIDATION_ENABLED})")


if __name__ == "__main__":
    main()
