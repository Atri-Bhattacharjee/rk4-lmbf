"""Introspection tests for the lmb_engine Python API after the tangent-plane rework.

Checks that every new name exists and behaves, that every removed name is gone, and that the
InOrbitSensorModel constructor exposes the six variance arguments by name in frame order.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "python"))

from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

SENSOR_CTOR_ARGS = ("range_var", "range_rate_var", "angle_var_1", "angle_var_2", "angle_rate_var_1", "angle_rate_var_2")


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def expect_raises(chk: Checker, exc_type, fn, label: str, message_contains: str):
    """Run fn; require exc_type whose message contains message_contains. Returns the error."""
    chk.count += 1
    try:
        fn()
    except exc_type as error:
        if message_contains not in str(error):
            raise AssertionError(f"{label}: {exc_type.__name__} message {str(error)!r} lacks {message_contains!r}") from None
        return error
    raise AssertionError(f"{label}: expected {exc_type.__name__}")


def check_names(chk: Checker) -> None:
    module_names = ("Measurement", "LosObservation", "AdaptiveBirthModel", "InOrbitSensorModel", "tangent_basis",
                    "exp_map", "log_map", "parallel_transport", "observe", "to_cartesian", "perturbed",
                    "local_residual", "angular_coordinates", "from_angles_and_rates", "MEAS_DIM", "VALIDATION_ENABLED",
                    "SensorFovConfig", "SensorArray", "SENSOR_MAX_HALF_ANGLE", "SENSOR_DEFAULT_HALF_ANGLE")
    for name in module_names:
        chk.ok(hasattr(lmb, name), f"lmb_engine.{name} missing")
    chk.ok(lmb.MEAS_DIM == 6, f"MEAS_DIM must be 6, got {lmb.MEAS_DIM}")
    chk.ok(isinstance(lmb.VALIDATION_ENABLED, bool), "VALIDATION_ENABLED must be a bool")

    measurement_names = ("range_", "range_rate_", "los_", "los_rate_", "covariance_", "timestamp_", "sensor_id_",
                         "sensor_state_", "fromCartesian", "fromAnglesAndRates", "toCartesian", "perturbed",
                         "angularCoordinates", "observation", "setObservation")
    for name in measurement_names:
        chk.ok(hasattr(lmb.Measurement, name), f"Measurement.{name} missing")
    for removed in ("cartesianToMeasurement", "measurementToCartesian", "value_"):
        chk.ok(not hasattr(lmb.Measurement, removed), f"Measurement.{removed} must be removed")

    for name in ("calculate_likelihood", "predictObservation", "defaultCovariance"):
        chk.ok(hasattr(lmb.InOrbitSensorModel, name), f"InOrbitSensorModel.{name} missing")
    chk.ok(not hasattr(lmb.InOrbitSensorModel, "convertParticleToMeasurement"),
           "InOrbitSensorModel.convertParticleToMeasurement must be removed")
    for name in ("generate_new_tracks", "birth_covariance_local"):
        chk.ok(hasattr(lmb.AdaptiveBirthModel, name), f"AdaptiveBirthModel.{name} missing")
    for name in ("range", "range_rate", "los", "los_rate"):
        chk.ok(hasattr(lmb.LosObservation, name), f"LosObservation.{name} missing")


def check_sensor_ctor(chk: Checker) -> None:
    expect_raises(chk, TypeError, lambda: lmb.InOrbitSensorModel(1.0, 1.0, 1.0, 1.0), "4-arg InOrbitSensorModel ctor",
                  "incompatible constructor arguments")
    doc = lmb.InOrbitSensorModel.__init__.__doc__ or ""
    positions = [doc.find(name) for name in SENSOR_CTOR_ARGS]
    chk.ok(all(p >= 0 for p in positions), f"ctor docstring must list {SENSOR_CTOR_ARGS}; got:\n{doc}")
    chk.ok(positions == sorted(positions), f"ctor argument names out of frame order in docstring:\n{doc}")
    model = lmb.InOrbitSensorModel(range_var=1.0, range_rate_var=2.0, angle_var_1=3.0, angle_var_2=4.0,
                                   angle_rate_var_1=5.0, angle_rate_var_2=6.0)
    chk.ok(np.array_equal(np.diag(model.defaultCovariance()), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
           "named ctor args must map to defaultCovariance diagonal in order")
    default = lmb.InOrbitSensorModel().defaultCovariance()
    chk.ok(np.allclose(np.diag(default), [50.0**2, 1.0, 1e-8, 1e-8, 1e-10, 1e-10]),
           f"default variances changed unexpectedly: {np.diag(default)}")
    for text in ("m^2", "(m/s)^2", "rad^2", "(rad/s)^2"):
        chk.ok(text in doc, f"ctor docstring must state units ({text})")


def check_birth_ctor(chk: Checker) -> None:
    doc = lmb.AdaptiveBirthModel.__init__.__doc__ or ""
    for text in ("seed", "birth_covariance_local", "tangent frame", "d_range", "d_omega2"):
        chk.ok(text in doc, f"AdaptiveBirthModel ctor docstring must mention {text!r}")
    chk.ok("fan" not in doc.lower() and "circular" not in doc.lower(), "AdaptiveBirthModel docstring still mentions the fan")
    covariance = np.diag([1e6, 2.5e5, 1e-8, 1e-8, 2.5e-9, 2.5e-9])
    model_kw = lmb.AdaptiveBirthModel(particles_per_track=10, initial_existence_probability=0.5,
                                      birth_covariance_local=covariance, seed=3)
    model_pos = lmb.AdaptiveBirthModel(10, 0.5, covariance, 3)
    model_none = lmb.AdaptiveBirthModel(10, 0.5, covariance, seed=None)
    chk.ok(np.array_equal(model_kw.birth_covariance_local(), covariance), "birth covariance getter mismatch")
    chk.ok(np.array_equal(model_pos.birth_covariance_local(), covariance), "positional ctor mismatch")
    chk.ok(model_none.birth_covariance_local().shape == (6, 6), "seed=None ctor failed")
    expect_raises(chk, TypeError, lambda: lmb.AdaptiveBirthModel(10, 0.5), "birth ctor without covariance",
                  "incompatible constructor arguments")
    expect_raises(chk, TypeError, lambda: lmb.AdaptiveBirthModel(10, 0.5, covariance, seed=-1),
                  "negative seed must be rejected (uint64)", "incompatible constructor arguments")


def check_measurement_roundtrip(chk: Checker) -> None:
    sensor = np.array([6.771e6, 0.0, 0.0, 0.0, 7672.0, 0.0])
    target = np.array([7.071e6, 2.0e5, 1.0e5, -500.0, 7400.0, 300.0])
    measurement = lmb.Measurement.fromCartesian(target, sensor)
    chk.ok(isinstance(measurement, lmb.Measurement), "fromCartesian must return a Measurement")
    chk.ok(np.allclose(measurement.sensor_state_, sensor), "fromCartesian must store the sensor state")
    chk.ok(measurement.covariance_.shape == (6, 6), "default covariance must be 6x6")
    # The covariance is the producer's responsibility: a fresh Measurement carries a zero covariance
    # and is rejected by every consumer until covariance_ is set.
    chk.ok(np.array_equal(measurement.covariance_, np.zeros((6, 6))), "unset covariance must be zero")
    unset_particle = lmb.Particle()
    unset_particle.state_vector = target
    unset_particle.weight = 1.0
    expect_raises(chk, ValueError, lambda: lmb.InOrbitSensorModel().calculate_likelihood(unset_particle, measurement),
                  "likelihood with unset covariance", "covariance")
    measurement.covariance_ = lmb.InOrbitSensorModel().defaultCovariance()
    back = measurement.toCartesian()
    scale = np.array([np.linalg.norm(target[:3])] * 3 + [np.linalg.norm(target[3:])] * 3)
    chk.ok(np.all(np.abs(back - target) <= 4.0 * np.finfo(float).eps * scale), f"round trip error {back - target}")

    obs = measurement.observation()
    chk.ok(isinstance(obs, lmb.LosObservation), "observation() must return LosObservation")
    chk.ok(obs.range == measurement.range_ and obs.range_rate == measurement.range_rate_, "observation fields mismatch")
    chk.ok(np.array_equal(obs.los, measurement.los_) and np.array_equal(obs.los_rate, measurement.los_rate_),
           "observation vectors mismatch")
    other = lmb.Measurement()
    other.setObservation(obs)
    chk.ok(other.range_ == measurement.range_ and np.array_equal(other.los_, measurement.los_), "setObservation mismatch")

    eps = np.array([10.0, 0.1, 1e-4, -2e-4, 1e-6, 2e-6])
    perturbed = measurement.perturbed(eps)
    chk.ok(isinstance(perturbed, lmb.Measurement), "perturbed must return a Measurement")
    chk.ok(perturbed.range_ == measurement.range_ + 10.0, "perturbed range")
    chk.ok(perturbed.sensor_id_ == measurement.sensor_id_ and np.array_equal(perturbed.covariance_, measurement.covariance_),
           "perturbed must copy metadata")
    residual = lmb.local_residual(measurement.observation(), perturbed.observation())
    chk.ok(np.allclose(residual, -eps, rtol=1e-9, atol=1e-13), f"local_residual(m, perturbed) = {residual} != -eps")

    angles = measurement.angularCoordinates()
    chk.ok(angles.shape == (4,), "angularCoordinates must be length 4")
    rebuilt = lmb.Measurement.fromAnglesAndRates(measurement.range_, measurement.range_rate_, *angles, sensor)
    chk.ok(np.allclose(rebuilt.los_, measurement.los_, atol=1e-15) and np.allclose(rebuilt.los_rate_, measurement.los_rate_,
                                                                                      rtol=1e-9, atol=1e-18),
           "fromAnglesAndRates(angularCoordinates()) must reproduce the measurement")
    with_kwargs = lmb.Measurement.fromAnglesAndRates(range=measurement.range_, range_rate=measurement.range_rate_,
                                                     azimuth=angles[0], elevation=angles[1], azimuth_rate=angles[2],
                                                     elevation_rate=angles[3], sensor_state=sensor)
    chk.ok(np.array_equal(with_kwargs.los_, rebuilt.los_), "fromAnglesAndRates keyword names")

    particle = lmb.Particle()
    particle.state_vector = target
    particle.weight = 1.0
    predicted = lmb.InOrbitSensorModel().predictObservation(particle, sensor)
    chk.ok(isinstance(predicted, lmb.LosObservation), "predictObservation must return LosObservation")
    chk.ok(predicted.range == obs.range and np.array_equal(predicted.los, obs.los), "predictObservation mismatch")
    likelihood = lmb.InOrbitSensorModel().calculate_likelihood(particle, measurement)
    chk.ok(np.isfinite(likelihood) and likelihood > 0.0, "likelihood not finite/positive")


def check_helpers_signatures(chk: Checker) -> None:
    for name in ("tangent_basis", "exp_map", "log_map", "parallel_transport", "observe", "to_cartesian", "perturbed",
                 "local_residual", "angular_coordinates", "from_angles_and_rates"):
        fn = getattr(lmb, name)
        chk.ok(callable(fn) and (fn.__doc__ or "").strip() != "", f"{name} must be callable with a docstring")
    u = np.array([0.0, 0.0, 1.0])
    chk.ok(lmb.tangent_basis(u).shape == (3, 2), "tangent_basis must return 3x2")
    chk.ok(lmb.log_map(u, u).shape == (3,), "log_map must return 3-vector")
    chk.ok(lmb.exp_map(u, np.zeros(3)).shape == (3,), "exp_map must return 3-vector")
    chk.ok(np.array_equal(lmb.parallel_transport(u, u, np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0]),
           "identity transport")


def check_sensor_array_surface(chk: Checker) -> None:
    """Pin the sensor/field-of-view surface Python drives every timestep."""
    config_names = ("min_range", "max_range", "half_width", "half_height")
    for name in config_names:
        chk.ok(hasattr(lmb.SensorFovConfig, name), f"SensorFovConfig.{name} missing")

    array_names = ("add", "add_unpointed", "size", "index_of", "id", "state", "boresight", "up",
                   "width_axis", "pointed", "set_state", "set_states", "set_boresight",
                   "set_boresights", "set_pointing", "point_at", "set_pointed", "sees",
                   "visible_sensor", "coverage_fractions", "coverage_fraction", "fov_config")
    for name in array_names:
        chk.ok(hasattr(lmb.SensorArray, name), f"SensorArray.{name} missing")

    chk.ok(0.0 < lmb.SENSOR_DEFAULT_HALF_ANGLE < lmb.SENSOR_MAX_HALF_ANGLE,
           "SENSOR_DEFAULT_HALF_ANGLE must sit inside (0, SENSOR_MAX_HALF_ANGLE)")
    chk.ok(abs(lmb.SENSOR_MAX_HALF_ANGLE - np.pi / 2) < 1e-15,
           "SENSOR_MAX_HALF_ANGLE must be pi/2")

    # A default config paired with add_unpointed is the omniscient sensor: keep that documented
    # contract executable, since every existing driver and fixture depends on it.
    sensors = lmb.SensorArray()
    chk.ok(len(sensors) == 0, "SensorArray() must default to an empty array")
    sensors.add_unpointed("sensor_0", np.zeros(6))
    chk.ok(sensors.sees(0, np.array([1.0e12, -3.0e11, 7.0e10])),
           "a default unpointed sensor must see anything at any range")
    chk.ok(sensors.visible_sensor(np.array([1.0e12, -3.0e11, 7.0e10])) == 0,
           "visible_sensor must resolve the default unpointed sensor")

    # update is overloaded; both arities must be reachable from Python.
    doc = (lmb.SMC_LMB_Tracker.update.__doc__ or "")
    chk.ok("measurements: list" in doc.replace("List", "list") or "measurements" in doc,
           "SMC_LMB_Tracker.update must document its arguments")
    chk.ok(doc.count("update(") >= 2,
           f"SMC_LMB_Tracker.update must expose both overloads, docstring was:\n{doc}")

    index = sensors.add("pointed", np.zeros(6), np.array([0.0, 0.0, 2.0]))
    chk.ok(index == 1, "add must return the new sensor's index")
    chk.ok(np.allclose(np.asarray(sensors.boresight(1)), [0.0, 0.0, 1.0]),
           "add must normalise the boresight")
    chk.ok(sensors.index_of("pointed") == 1 and sensors.index_of("absent") == -1,
           "index_of must resolve known ids and return -1 otherwise")


def main() -> None:
    chk = Checker()
    steps = [
        ("names", lambda: check_names(chk)),
        ("sensor array surface", lambda: check_sensor_array_surface(chk)),
        ("sensor ctor", lambda: check_sensor_ctor(chk)),
        ("birth ctor", lambda: check_birth_ctor(chk)),
        ("measurement API", lambda: check_measurement_roundtrip(chk)),
        ("geometry helpers", lambda: check_helpers_signatures(chk)),
    ]
    for name, fn in steps:
        before = chk.count
        fn()
        print(f"  {name}: {chk.count - before} assertions")
    if chk.count <= 0:
        raise AssertionError("no assertions executed")
    print(f"PASS: test_bindings_api ({chk.count} assertions)")


if __name__ == "__main__":
    main()
