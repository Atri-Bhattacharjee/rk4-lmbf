"""Independent reference implementations used by the measurement-model tests.

Nothing in this module calls lmb_engine. Every function is either exact
(Fraction/Decimal) or evaluated in 80-bit long double with formulas that differ
from the C++ implementation (cross products instead of projections, Rodrigues
rotations instead of exponential maps), so the tests compare the engine against
an independent reference rather than against itself.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

LD = np.longdouble


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def rodrigues_ld(vector, axis, angle) -> np.ndarray:
    """Rotate `vector` about unit `axis` by `angle` radians in long double (Rodrigues' formula)."""
    v = np.asarray(vector, dtype=LD)
    k = np.asarray(axis, dtype=LD)
    k = k / np.sqrt(np.sum(k * k))
    a = LD(angle)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    return v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (LD(1) - cos_a)


def rotation_matrix(axis, angle) -> np.ndarray:
    """3x3 rotation matrix (double) about unit `axis` by `angle` radians."""
    k = unit(axis)
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def rotation_taking(from_dir, to_dir) -> np.ndarray:
    """Rotation matrix that maps unit from_dir onto unit to_dir (minimal rotation)."""
    f = unit(from_dir)
    t = unit(to_dir)
    axis = np.cross(f, t)
    sin_angle = np.linalg.norm(axis)
    cos_angle = float(np.dot(f, t))
    if sin_angle < 1e-15:
        if cos_angle > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = unit(np.cross(f, helper))
        return rotation_matrix(axis, np.pi)
    return rotation_matrix(axis / sin_angle, float(np.arctan2(sin_angle, cos_angle)))


def geodesic_angle_ld(from_dirs, to_dirs) -> np.ndarray:
    """Geodesic angle between the given (double) vectors, evaluated in long double.

    Uses atan2(|f x (t - f)|, f . t): the cross product of f with the exact-ish difference
    t - f equals f x t and avoids cancellation, so the result is accurate to ~1e-18 relative
    down to angles of a few ulps. Broadcasts over leading axes.
    """
    f = np.asarray(from_dirs, dtype=LD)
    t = np.asarray(to_dirs, dtype=LD)
    difference = t - f
    cross = np.cross(f, difference)
    sine = np.sqrt(np.sum(cross * cross, axis=-1))
    cosine = np.sum(f * t, axis=-1)
    return np.arctan2(sine, cosine)


def tangent_component_ld(from_dirs, to_dirs) -> np.ndarray:
    """Unit direction (long double) in the tangent plane at f pointing toward t, via (f x t) x f."""
    f = np.asarray(from_dirs, dtype=LD)
    t = np.asarray(to_dirs, dtype=LD)
    cross = np.cross(f, t - f)
    tangent = np.cross(cross, f)
    return tangent / np.sqrt(np.sum(tangent * tangent, axis=-1, keepdims=True))


def _decimal_atan_nonneg(x: Decimal) -> Decimal:
    """atan(x) for x >= 0 with the active Decimal precision (argument halving + Taylor series)."""
    halvings = 0
    one = Decimal(1)
    while x > Decimal("0.05"):
        x = x / (one + (one + x * x).sqrt())
        halvings += 1
    x_sq = x * x
    term = x
    total = Decimal(0)
    k = 0
    threshold = Decimal(10) ** (-(getcontext().prec + 2))
    while abs(term) > threshold:
        total += term / (2 * k + 1)
        term = -term * x_sq
        k += 1
    return total * (2 ** halvings)


def decimal_pi() -> Decimal:
    return 16 * _decimal_atan_nonneg(Decimal(1) / 5) - 4 * _decimal_atan_nonneg(Decimal(1) / 239)


def decimal_atan2(y: Decimal, x: Decimal) -> Decimal:
    """atan2 for y >= 0 using the active Decimal precision."""
    if x > 0:
        return _decimal_atan_nonneg(y / x)
    if x < 0:
        return decimal_pi() - _decimal_atan_nonneg(y / (-x))
    return decimal_pi() / 2


def geodesic_angle_exact(from_dir, to_dir, digits: int = 50) -> float:
    """Geodesic angle between two double vectors using exact rational cross/dot products
    and a `digits`-digit Decimal atan2. Returns a float (correctly rounded to ~1e-16)."""
    getcontext().prec = digits
    f = [Fraction(float(x)) for x in from_dir]
    t = [Fraction(float(x)) for x in to_dir]
    cross = [
        f[1] * t[2] - f[2] * t[1],
        f[2] * t[0] - f[0] * t[2],
        f[0] * t[1] - f[1] * t[0],
    ]
    sine_sq = sum(c * c for c in cross)
    cosine = sum(a * b for a, b in zip(f, t))
    sine = (Decimal(sine_sq.numerator) / Decimal(sine_sq.denominator)).sqrt()
    cosine_dec = Decimal(cosine.numerator) / Decimal(cosine.denominator)
    return float(decimal_atan2(sine, cosine_dec))


def relative_observation_ld(target_state, sensor_state):
    """(range, range_rate, u, u_dot) of target relative to sensor in long double, from first principles."""
    x = np.asarray(target_state, dtype=LD)
    s = np.asarray(sensor_state, dtype=LD)
    p = x[:3] - s[:3]
    v = x[3:] - s[3:]
    rho = np.sqrt(np.sum(p * p))
    u = p / rho
    rho_dot = np.dot(u, v)
    u_dot = (v - rho_dot * u) / rho
    return rho, rho_dot, u, u_dot


def azimuth_elevation_ld(u):
    u = np.asarray(u, dtype=LD)
    horizontal = np.sqrt(u[0] * u[0] + u[1] * u[1])
    return np.arctan2(u[1], u[0]), np.arctan2(u[2], horizontal)


def gaussian_log_density(residual, covariance) -> float:
    """log N(residual; 0, covariance) with the covariance factored in correlation form.

    R = D C D with D = sqrt(diag R); Cholesky of C (well scaled even when R mixes m^2 and (rad/s)^2),
    then log det R = log det C + 2 sum log D and the quadratic form is solved on the scaled residual.
    """
    residual = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    dim = residual.size
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    chol = np.linalg.cholesky(correlation)
    scaled = residual / scale
    y = np.linalg.solve(chol, scaled)
    mahalanobis = float(y @ y)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol)))) + 2.0 * float(np.sum(np.log(scale)))
    return -0.5 * dim * np.log(2.0 * np.pi) - 0.5 * logdet - 0.5 * mahalanobis


# ---------------------------------------------------------------------------
# Independent (double precision, NumPy) implementation of the measurement model,
# written from the plan's formulas. Used as the reference for the C++ likelihood.
# ---------------------------------------------------------------------------

class LosObs:
    """Plain container mirroring lmb_engine.LosObservation without depending on it."""

    def __init__(self, range_m, range_rate, los, los_rate):
        self.range = float(range_m)
        self.range_rate = float(range_rate)
        self.los = np.asarray(los, dtype=float)
        self.los_rate = np.asarray(los_rate, dtype=float)


def tangent_basis_spec(u) -> np.ndarray:
    """Specification of the deterministic tangent basis: e1 from the least-aligned axis, e2 = u x e1."""
    u = np.asarray(u, dtype=float)
    axis = int(np.argmin(np.abs(u)))
    helper = np.zeros(3)
    helper[axis] = 1.0
    e1 = helper - np.dot(helper, u) * u
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 = e2 / np.linalg.norm(e2)
    return np.column_stack([e1, e2])


def log_map_ref(from_dir, to_dir) -> np.ndarray:
    """atan2-based log map, plain projection form w = t - (f . t) f (adequate for gamma >= 1e-6)."""
    f = np.asarray(from_dir, dtype=float)
    t = np.asarray(to_dir, dtype=float)
    c = float(np.clip(np.dot(f, t), -1.0, 1.0))
    w = t - c * f
    s = float(np.linalg.norm(w))
    if s == 0.0:
        return np.zeros(3)
    return (np.arctan2(s, c) / s) * w


def exp_map_ref(u, tangent) -> np.ndarray:
    """Rodrigues rotation of u about the axis u x a by |a| radians (independent of the sinc form)."""
    u = np.asarray(u, dtype=float)
    a = np.asarray(tangent, dtype=float)
    alpha = float(np.linalg.norm(a))
    if alpha == 0.0:
        return u.copy()
    axis = np.cross(u, a) / alpha
    return rotation_matrix(axis, alpha) @ u


def parallel_transport_ref(from_dir, to_dir, tangent) -> np.ndarray:
    f = np.asarray(from_dir, dtype=float)
    t = np.asarray(to_dir, dtype=float)
    v = np.asarray(tangent, dtype=float)
    return v - (np.dot(t, v) / (1.0 + np.dot(f, t))) * (f + t)


def observe_ref(target_state, sensor_state) -> LosObs:
    x = np.asarray(target_state, dtype=float)
    s = np.asarray(sensor_state, dtype=float)
    p = x[:3] - s[:3]
    v = x[3:] - s[3:]
    rho = float(np.linalg.norm(p))
    u = p / rho
    rho_dot = float(np.dot(u, v))
    u_dot = (v - rho_dot * u) / rho
    return LosObs(rho, rho_dot, u, u_dot)


def to_cartesian_ref(obs: LosObs, sensor_state) -> np.ndarray:
    s = np.asarray(sensor_state, dtype=float)
    position = s[:3] + obs.range * obs.los
    velocity = s[3:] + obs.range_rate * obs.los + obs.range * obs.los_rate
    return np.concatenate([position, velocity])


def perturbed_ref(obs: LosObs, eps) -> LosObs:
    eps = np.asarray(eps, dtype=float)
    basis = tangent_basis_spec(obs.los)
    new_los = exp_map_ref(obs.los, eps[2] * basis[:, 0] + eps[3] * basis[:, 1])
    rate_at_origin = obs.los_rate + eps[4] * basis[:, 0] + eps[5] * basis[:, 1]
    new_rate = parallel_transport_ref(obs.los, new_los, rate_at_origin)
    new_rate = new_rate - np.dot(new_rate, new_los) * new_los
    return LosObs(obs.range + eps[0], obs.range_rate + eps[1], new_los, new_rate)


def local_residual_ref(measured: LosObs, predicted: LosObs) -> np.ndarray:
    basis = tangent_basis_spec(measured.los)
    direction_log = log_map_ref(measured.los, predicted.los)
    transported = parallel_transport_ref(predicted.los, measured.los, predicted.los_rate)
    rate_difference = measured.los_rate - transported
    return np.array([
        measured.range - predicted.range,
        measured.range_rate - predicted.range_rate,
        -np.dot(basis[:, 0], direction_log),
        -np.dot(basis[:, 1], direction_log),
        np.dot(basis[:, 0], rate_difference),
        np.dot(basis[:, 1], rate_difference),
    ])


def chi_square_6_cdf(x: float) -> float:
    """Closed-form CDF of the chi-square distribution with 6 degrees of freedom."""
    half = 0.5 * x
    return 1.0 - np.exp(-half) * (1.0 + half + half * half / 2.0)
