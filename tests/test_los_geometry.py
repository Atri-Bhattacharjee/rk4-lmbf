"""Stress tests for the line-of-sight geometry in src/los_geometry.h.

Priority is the gamma -> 0 regime: a converged track puts every particle within
micro-radians of the measured direction, so the near-identity branch of the
logarithmic map is the hot path, not an edge case. All references come from
tests/reference_geometry.py (long double / exact rational arithmetic) and never
from the functions under test.
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

LD = np.longdouble
PI = np.pi
FIXED_SEED = 20260905


class Checker:
    def __init__(self) -> None:
        self.count = 0

    def ok(self, condition, message: str) -> None:
        self.count += 1
        if not bool(condition):
            raise AssertionError(message)


def make_observation(range_m, range_rate, los, los_rate):
    obs = lmb.LosObservation()
    obs.range = float(range_m)
    obs.range_rate = float(range_rate)
    obs.los = np.asarray(los, dtype=float)
    obs.los_rate = np.asarray(los_rate, dtype=float)
    return obs


def random_tangent(rng: np.random.Generator, u, magnitude: float) -> np.ndarray:
    """Tangent vector at unit u, projected in long double so that (t . u)/|t| is at the 1e-16 level."""
    while True:
        raw = rng.normal(size=3)
        if np.linalg.norm(raw - np.dot(raw, u) * u) > 0.3 * np.linalg.norm(raw):
            break
    raw_ld = np.asarray(raw, dtype=LD)
    u_ld = np.asarray(u, dtype=LD)
    tangent = raw_ld - (np.dot(raw_ld, u_ld) / np.dot(u_ld, u_ld)) * u_ld
    tangent = tangent / np.sqrt(np.sum(tangent * tangent))
    return np.asarray(LD(magnitude) * tangent, dtype=float)


def random_leo_pair(rng: np.random.Generator):
    """Random (target_state, sensor_state) with LEO-scale geometry and up to 15 km/s relative speed."""
    sensor_dir = ref.random_unit_vectors(rng, 1)[0]
    sensor_pos = 6.771e6 * sensor_dir
    tangent = ref.unit(np.cross(sensor_dir, ref.random_unit_vectors(rng, 1)[0]))
    sensor_vel = 7.67e3 * tangent
    rel_dir = ref.random_unit_vectors(rng, 1)[0]
    rel_range = rng.uniform(1.0e5, 3.0e6)
    rel_vel = rng.uniform(0.0, 1.5e4) * ref.random_unit_vectors(rng, 1)[0]
    target = np.concatenate([sensor_pos + rel_range * rel_dir, sensor_vel + rel_vel])
    sensor = np.concatenate([sensor_pos, sensor_vel])
    return target, sensor


def check_tangent_basis(chk: Checker, rng: np.random.Generator) -> None:
    axes = [np.eye(3)[i] * sign for i in range(3) for sign in (1.0, -1.0)]
    for u in list(ref.random_unit_vectors(rng, 200)) + axes:
        basis = lmb.tangent_basis(u)
        e1, e2 = basis[:, 0], basis[:, 1]
        chk.ok(abs(np.dot(e1, u)) <= 1e-15, f"e1 not tangent at {u}")
        chk.ok(abs(np.dot(e2, u)) <= 1e-15, f"e2 not tangent at {u}")
        chk.ok(abs(np.linalg.norm(e1) - 1.0) <= 1e-15, "e1 not unit")
        chk.ok(abs(np.linalg.norm(e2) - 1.0) <= 1e-15, "e2 not unit")
        chk.ok(abs(np.dot(e1, e2)) <= 1e-15, "e1, e2 not orthogonal")
        chk.ok(abs(np.dot(np.cross(e1, e2), u) - 1.0) <= 1e-15, "basis not right-handed")


def check_log_identity(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 1: log_map(u, u) is exactly zero (bitwise == 0), never NaN."""
    axes = [np.eye(3)[i] * sign for i in range(3) for sign in (1.0, -1.0)]
    for u in list(ref.random_unit_vectors(rng, 1000)) + axes:
        result = lmb.log_map(u, u)
        chk.ok(np.all(result == 0.0), f"log_map(u,u) != 0 for u={u}: {result}")


def check_log_sweep(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 2: |log_map| vs the geodesic angle between the actual double vectors."""
    gammas = np.concatenate([np.logspace(-16, np.log10(PI - 1e-3), 400), [0.5e-3, 1e-3, 2e-3]])
    exact_checks = 0
    for u_index, u_m in enumerate(ref.random_unit_vectors(rng, 20)):
        basis = lmb.tangent_basis(u_m)
        e1, e2 = basis[:, 0], basis[:, 1]
        for gamma_index, gamma in enumerate(gammas):
            u_p = np.asarray(ref.rodrigues_ld(u_m, e2, gamma), dtype=float)
            log_vec = lmb.log_map(u_m, u_p)
            chk.ok(np.all(np.isfinite(log_vec)), f"non-finite log_map at gamma={gamma}")
            if np.array_equal(u_p, u_m):
                chk.ok(np.all(log_vec == 0.0), "identical vectors must give exact zero")
                continue
            gamma_ref = float(ref.geodesic_angle_ld(u_m, u_p))
            norm = float(np.linalg.norm(log_vec))
            rel_err = abs(norm - gamma_ref) / gamma_ref
            tolerance = 1e-13 if gamma_ref >= 1e-3 else 2e-15
            chk.ok(rel_err <= tolerance,
                   f"|log_map| rel err {rel_err:.3e} > {tolerance} at gamma={gamma_ref:.3e}")
            direction_ref = np.asarray(ref.tangent_component_ld(u_m, u_p), dtype=float)
            direction_err = np.linalg.norm(log_vec / norm - direction_ref)
            chk.ok(direction_err <= 1e-12, f"log_map direction err {direction_err:.3e} at gamma={gamma_ref:.3e}")
            if gamma_ref >= 1e-3:
                chk.ok(np.linalg.norm(log_vec / norm - e1) <= 1e-12, "log_map direction differs from nominal e1")
            # Validate the long-double reference itself against exact rational arithmetic on a subset.
            if u_index < 2 and gamma_index % 10 == 0:
                gamma_exact = ref.geodesic_angle_exact(u_m, u_p)
                chk.ok(abs(gamma_ref - gamma_exact) <= 4e-16 * gamma_exact,
                       f"long-double reference disagrees with exact reference at gamma={gamma_exact:.3e}")
                exact_checks += 1
    chk.ok(exact_checks >= 80, "too few exact-reference cross-checks executed")


def _normalised_log_error(u_m, e2, gamma_nominal) -> float:
    u_p = np.asarray(ref.rodrigues_ld(u_m, e2, gamma_nominal), dtype=float)
    gamma_ref = float(ref.geodesic_angle_ld(u_m, u_p))
    return float(np.linalg.norm(lmb.log_map(u_m, u_p))) / gamma_ref - 1.0


def check_branch_continuity(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 3: no jump across the series/closed-form switches.

    log_map: the output is a small vector with full relative precision, so the normalised error
    (|log|/gamma_ref - 1) must agree on both sides of s = 1e-3 to 1e-15.

    exp_map: the output is a unit vector whose components carry ~1e-16 absolute rounding, i.e. an
    angular resolution of ~1e-12 relative at alpha = 1e-4, so a 1e-15 normalised-error criterion is
    not resolvable through the output for any implementation. Continuity is instead checked at the
    resolution the output supports: both sides must match the Rodrigues reference to 1e-15 per
    component and the output difference across the switch must equal the reference difference to
    4e-16 per component. A missing alpha^2/6 series term would show up as a ~2e-13 component error.
    """
    for u in ref.random_unit_vectors(rng, 20):
        basis = lmb.tangent_basis(u)
        e1, e2 = basis[:, 0], basis[:, 1]
        s_switch = 1e-3
        errors = [_normalised_log_error(u, e2, float(np.arcsin(s_switch * (1.0 + sign * 1e-9))))
                  for sign in (-1.0, 1.0)]
        chk.ok(abs(errors[0] - errors[1]) <= 1e-15,
               f"log_map error jump {abs(errors[0] - errors[1]):.3e} across series switch")

        alpha_switch = 1e-4
        alphas = [alpha_switch * (1.0 + sign * 1e-9) for sign in (-1.0, 1.0)]
        moved = [lmb.exp_map(u, alpha * e1) for alpha in alphas]
        expected_ld = [ref.rodrigues_ld(u, e2, alpha) for alpha in alphas]
        for got, exp_ld in zip(moved, expected_ld):
            chk.ok(np.max(np.abs(got - np.asarray(exp_ld, dtype=float))) <= 1e-15,
                   "exp_map differs from Rodrigues next to the series switch")
        got_difference = moved[1] - moved[0]
        expected_difference = np.asarray(expected_ld[1] - expected_ld[0], dtype=float)
        chk.ok(np.max(np.abs(got_difference - expected_difference)) <= 4e-16,
               f"exp_map output jump across series switch: {got_difference} vs {expected_difference}")


def check_exp_map(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 4: unit norm, exact identity at zero, agreement with Rodrigues rotation."""
    magnitudes = [0.0, 1e-14, 1e-9, 1e-6, 1e-4, 1e-3, 0.1, 1.0, 3.0]
    for u in ref.random_unit_vectors(rng, 50):
        basis = lmb.tangent_basis(u)
        e1, e2 = basis[:, 0], basis[:, 1]
        chk.ok(np.array_equal(lmb.exp_map(u, np.zeros(3)), u), "exp_map(u, 0) must return u bitwise")
        for magnitude in magnitudes:
            moved = lmb.exp_map(u, magnitude * e1)
            chk.ok(abs(np.linalg.norm(moved) - 1.0) <= 4e-16, f"exp_map norm off by {np.linalg.norm(moved) - 1.0:.3e}")
            expected = np.asarray(ref.rodrigues_ld(u, e2, magnitude), dtype=float)
            chk.ok(np.max(np.abs(moved - expected)) <= 1e-15,
                   f"exp_map differs from Rodrigues by {np.max(np.abs(moved - expected)):.3e} at |a|={magnitude}")


def check_round_trips(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 5: log(exp(a)) == a and exp(log(v)) == v.

    The intermediate unit vector carries ~1e-16 absolute rounding, which the log map amplifies by
    its condition number gamma / sin(gamma) in the direction orthogonal to a (21.5 at gamma = 3).
    The tolerance therefore uses max(1, |a|, |a|/sin|a|) rather than max(1, |a|) alone.
    """
    magnitudes = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.1, 1.0, 3.0]
    for u in ref.random_unit_vectors(rng, 40):
        basis = lmb.tangent_basis(u)
        for magnitude in magnitudes:
            angle = rng.uniform(0.0, 2.0 * PI)
            a = magnitude * (np.cos(angle) * basis[:, 0] + np.sin(angle) * basis[:, 1])
            recovered = lmb.log_map(u, lmb.exp_map(u, a))
            err = np.max(np.abs(recovered - a))
            conditioning = magnitude / np.sin(magnitude) if magnitude > 0.0 else 1.0
            tolerance = 1e-15 * max(1.0, magnitude, conditioning)
            chk.ok(err <= tolerance, f"log(exp(a)) - a = {err:.3e} > {tolerance:.3e} at |a|={magnitude}")
    for u in ref.random_unit_vectors(rng, 1000):
        basis = lmb.tangent_basis(u)
        angle = rng.uniform(0.0, 2.0 * PI)
        axis = np.cos(angle) * basis[:, 0] + np.sin(angle) * basis[:, 1]
        v = np.asarray(ref.rodrigues_ld(u, axis, rng.uniform(0.0, PI - 1e-3)), dtype=float)
        rebuilt = lmb.exp_map(u, lmb.log_map(u, v))
        err = np.max(np.abs(rebuilt - v))
        # ~10 chained floating-point operations (dot, subtract, norm, atan2, divide, multiply,
        # cos, sin, multiply-add, normalise) each contribute up to 0.5 ulp of O(1) quantities;
        # 2.5e-15 (~11 ulp) is the accumulated bound. Measured maximum over 2e5 samples: 1.22e-15.
        chk.ok(err <= 2.5e-15, f"exp(log(v)) - v = {err:.3e}")


def check_converged_cloud(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 6: 1e5 particles within 1e-12..1e-6 rad; finite, accurate, strictly monotone."""
    per_center = 2000
    all_norms = []
    all_refs = []
    for u_m in ref.random_unit_vectors(rng, 50):
        basis = lmb.tangent_basis(u_m)
        gammas = 10.0 ** rng.uniform(-12.0, -6.0, size=per_center)
        angles = rng.uniform(0.0, 2.0 * PI, size=per_center)
        axes = np.outer(np.cos(angles), basis[:, 0]) + np.outer(np.sin(angles), basis[:, 1])
        u_ps = np.empty((per_center, 3))
        for i in range(per_center):
            u_ps[i] = np.asarray(ref.rodrigues_ld(u_m, axes[i], gammas[i]), dtype=float)
        gamma_refs = np.asarray(ref.geodesic_angle_ld(np.broadcast_to(u_m, u_ps.shape), u_ps), dtype=float)
        logs = np.array([lmb.log_map(u_m, u_ps[i]) for i in range(per_center)])
        chk.ok(np.all(np.isfinite(logs)), "non-finite log_map in converged cloud")
        norms = np.linalg.norm(logs, axis=1)
        rel_err = np.abs(norms - gamma_refs) / gamma_refs
        chk.ok(np.max(rel_err) <= 2e-15, f"converged-cloud |log_map| rel err {np.max(rel_err):.3e}")
        all_norms.append(norms)
        all_refs.append(gamma_refs)
    norms = np.concatenate(all_norms)
    refs = np.concatenate(all_refs)
    chk.ok(norms.size == 100000, "expected 1e5 samples")
    order = np.argsort(refs)
    sorted_refs = refs[order]
    sorted_norms = norms[order]
    distinct = np.diff(sorted_refs) > 0
    chk.ok(np.all(np.diff(sorted_norms)[distinct] > 0), "|log_map| not strictly monotone in gamma (plateaus)")


def check_antipodal(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 7: antipodal inputs give a finite result of norm pi."""
    for u in ref.random_unit_vectors(rng, 100):
        result = lmb.log_map(u, -u)
        chk.ok(np.all(np.isfinite(result)), "antipodal log_map not finite")
        chk.ok(abs(np.linalg.norm(result) - PI) <= 1e-15 * PI, f"antipodal norm {np.linalg.norm(result)}")
        chk.ok(abs(np.dot(result, u)) <= 1e-15 * PI, "antipodal result not tangent")
        e1 = lmb.tangent_basis(u)[:, 0]
        near = ref.unit(-u + 1e-12 * e1)
        result = lmb.log_map(u, near)
        chk.ok(np.all(np.isfinite(result)), "near-antipodal log_map not finite")
        chk.ok(PI - 1e-6 <= np.linalg.norm(result) <= PI, f"near-antipodal norm {np.linalg.norm(result)}")


def check_parallel_transport(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 8: tangency, norm preservation, identity limit, analytic meridian case.

    The closed-form transport subtracts ((u_m . t) / (1 + u_p . u_m)) (u_p + u_m) from t, so its
    rounding error is amplified by 1 / (1 + cos gamma), which is the intrinsic conditioning of
    transport toward the antipode (the geodesic itself becomes undefined there). The tolerance for
    random pairs therefore carries that factor; it is 1 for the tracker's gamma << 1 regime.
    """
    for _ in range(1000):
        u_p, u_m = ref.random_unit_vectors(rng, 2)
        t = random_tangent(rng, u_p, rng.uniform(1e-6, 1e3))
        t_prime = lmb.parallel_transport(u_p, u_m, t)
        scale = np.linalg.norm(t)
        conditioning = max(1.0, 1.0 / (1.0 + float(np.dot(u_p, u_m))))
        chk.ok(abs(np.dot(t_prime, u_m)) <= 1e-15 * scale * conditioning, "transported vector not tangent")
        chk.ok(abs(np.linalg.norm(t_prime) - scale) <= 1e-15 * scale * conditioning, "transport changed the norm")
    for _ in range(1000):
        # Tracker regime: gamma <= 0.1 rad, where the conditioning factor is ~1.
        u_p = ref.random_unit_vectors(rng, 1)[0]
        basis = lmb.tangent_basis(u_p)
        angle = rng.uniform(0.0, 2.0 * PI)
        axis = np.cos(angle) * basis[:, 0] + np.sin(angle) * basis[:, 1]
        u_m = np.asarray(ref.rodrigues_ld(u_p, axis, 10.0 ** rng.uniform(-9.0, -1.0)), dtype=float)
        t = random_tangent(rng, u_p, rng.uniform(1e-6, 1e3))
        t_prime = lmb.parallel_transport(u_p, u_m, t)
        scale = np.linalg.norm(t)
        chk.ok(abs(np.dot(t_prime, u_m)) <= 1e-15 * scale, "transported vector not tangent (small gamma)")
        chk.ok(abs(np.linalg.norm(t_prime) - scale) <= 1e-15 * scale, "transport changed the norm (small gamma)")
    for gamma in (1e-8, 1e-9, 1e-10):
        for u_p in ref.random_unit_vectors(rng, 20):
            basis = lmb.tangent_basis(u_p)
            u_m = np.asarray(ref.rodrigues_ld(u_p, basis[:, 1], gamma), dtype=float)
            t = rng.uniform(0.1, 10.0) * basis[:, 0]
            t_prime = lmb.parallel_transport(u_p, u_m, t)
            chk.ok(np.linalg.norm(t_prime - t) <= 2.0 * gamma * np.linalg.norm(t), "transport not near identity")
    for gamma in (1e-4, 1e-3, 1e-2):
        # Transport differs from plain projection by (u_m . t)(c u_m - u_p)/(1 + c), i.e. at most
        # ~gamma^2 |t| / 2; check the second-order agreement where it is resolvable.
        for u_p in ref.random_unit_vectors(rng, 20):
            basis = lmb.tangent_basis(u_p)
            u_m = np.asarray(ref.rodrigues_ld(u_p, basis[:, 1], gamma), dtype=float)
            t = rng.uniform(0.1, 10.0) * basis[:, 0]
            t_prime = lmb.parallel_transport(u_p, u_m, t)
            projection = t - np.dot(t, u_m) * u_m
            chk.ok(np.linalg.norm(t_prime - projection) <= 0.6 * gamma * gamma * np.linalg.norm(t),
                   "transport disagrees with projection beyond O(gamma^2)")
    z = np.array([0.0, 0.0, 1.0])
    x = np.array([1.0, 0.0, 0.0])
    # Along the meridian from the pole to the equator, the tangent that points toward the
    # equator (x at the pole) ends up pointing along -z at the equator.
    t_prime = lmb.parallel_transport(z, x, x)
    chk.ok(np.max(np.abs(t_prime - (-z))) <= 1e-15, f"meridian transport gave {t_prime}")


def check_observe_and_inverse(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 9: forward/inverse round trip, tangency, agreement with first-principles references."""
    for _ in range(1000):
        target, sensor = random_leo_pair(rng)
        obs = lmb.observe(target, sensor)
        rebuilt = lmb.to_cartesian(obs, sensor)
        state_scale = np.linalg.norm(target)
        chk.ok(np.max(np.abs(rebuilt - target)) <= 1e-9 * state_scale, "to_cartesian(observe(x)) != x")
        chk.ok(np.max(np.abs(rebuilt[3:] - target[3:])) <= 1e-9 * np.linalg.norm(target[3:]),
               "velocity round trip too loose")
        chk.ok(abs(np.dot(obs.los, obs.los_rate)) <= 1e-15 * np.linalg.norm(obs.los_rate), "los_rate not tangent")
        chk.ok(abs(np.linalg.norm(obs.los) - 1.0) <= 4e-16, "los not unit")

        rho, rho_dot, u_ref, u_dot_ref = ref.relative_observation_ld(target, sensor)
        chk.ok(abs(obs.range - float(rho)) <= 1e-12 * float(rho), "range mismatch")
        chk.ok(abs(obs.range_rate - float(rho_dot)) <= 1e-12 * max(1.0, abs(float(rho_dot))), "range_rate mismatch")
        chk.ok(np.max(np.abs(obs.los - np.asarray(u_ref, dtype=float))) <= 1e-15, "los mismatch")

        dt = LD(1e-3)
        p = np.asarray(target[:3], LD) - np.asarray(sensor[:3], LD)
        v = np.asarray(target[3:], LD) - np.asarray(sensor[3:], LD)
        u_plus = (p + v * dt) / np.sqrt(np.sum((p + v * dt) ** 2))
        u_minus = (p - v * dt) / np.sqrt(np.sum((p - v * dt) ** 2))
        u_dot_fd = np.asarray((u_plus - u_minus) / (2 * dt), dtype=float)
        speed = np.linalg.norm(u_dot_fd)
        chk.ok(np.max(np.abs(obs.los_rate - u_dot_fd)) <= 1e-6 * speed, "los_rate disagrees with finite difference")
        chk.ok(np.max(np.abs(obs.los_rate - np.asarray(u_dot_ref, dtype=float))) <= 1e-12 * speed,
               "los_rate disagrees with long-double reference")


def check_perturbation_identity(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 10: local_residual(obs, perturbed(obs, eps)) == -eps and tangency of the result.

    Per-component tolerance follows the roundoff model 1e-13 * max(1, |eps_k|, scale_k) where
    scale_k is the magnitude of the underlying quantity (range for d_range, etc.), because the
    range residual is a difference of two O(range) numbers.
    """
    for _ in range(1000):
        target, sensor = random_leo_pair(rng)
        obs = lmb.observe(target, sensor)
        angular_magnitude = 10.0 ** rng.uniform(-12.0, 0.0)
        angle = rng.uniform(0.0, 2.0 * PI)
        eps = np.array([
            rng.normal(0.0, 1000.0),
            rng.normal(0.0, 500.0),
            angular_magnitude * np.cos(angle),
            angular_magnitude * np.sin(angle),
            rng.normal(0.0, 1e-4),
            rng.normal(0.0, 1e-4),
        ])
        moved = lmb.perturbed(obs, eps)
        chk.ok(abs(np.dot(moved.los, moved.los_rate)) <= 1e-15 * max(1e-300, np.linalg.norm(moved.los_rate)),
               "perturbed los_rate not tangent")
        chk.ok(abs(np.linalg.norm(moved.los) - 1.0) <= 4e-16, "perturbed los not unit")
        residual = lmb.local_residual(obs, moved)
        rate_scale = np.linalg.norm(obs.los_rate)
        scales = np.array([obs.range, abs(obs.range_rate), 1.0, 1.0, rate_scale, rate_scale])
        tolerance = 1e-13 * np.maximum(1.0, np.maximum(np.abs(eps), scales))
        err = np.abs(residual + eps)
        chk.ok(np.all(err <= tolerance), f"residual + eps = {err} exceeds {tolerance}")


def check_angular_coordinates(chk: Checker, rng: np.random.Generator) -> None:
    """Assertion 11: derived az/el and their rates against long-double finite differences."""
    count = 0
    while count < 1000:
        target, sensor = random_leo_pair(rng)
        obs = lmb.observe(target, sensor)
        if abs(np.degrees(np.arcsin(obs.los[2]))) >= 80.0:
            continue
        count += 1
        angles = lmb.angular_coordinates(obs)
        az_ref, el_ref = ref.azimuth_elevation_ld(obs.los)
        chk.ok(abs(angles[0] - float(az_ref)) <= 1e-12, "azimuth mismatch")
        chk.ok(abs(angles[1] - float(el_ref)) <= 1e-12, "elevation mismatch")

        dt = LD(1e-4)
        p = np.asarray(target[:3], LD) - np.asarray(sensor[:3], LD)
        v = np.asarray(target[3:], LD) - np.asarray(sensor[3:], LD)
        az_plus, el_plus = ref.azimuth_elevation_ld(p + v * dt)
        az_minus, el_minus = ref.azimuth_elevation_ld(p - v * dt)
        d_az = az_plus - az_minus
        if d_az > PI:
            d_az -= 2 * PI
        if d_az < -PI:
            d_az += 2 * PI
        az_rate_fd = float(d_az / (2 * dt))
        el_rate_fd = float((el_plus - el_minus) / (2 * dt))
        speed = np.linalg.norm(obs.los_rate)
        chk.ok(abs(angles[2] - az_rate_fd) <= 1e-6 * speed, f"azimuth rate {angles[2]} vs fd {az_rate_fd}")
        chk.ok(abs(angles[3] - el_rate_fd) <= 1e-6 * speed, f"elevation rate {angles[3]} vs fd {el_rate_fd}")

        rebuilt = lmb.from_angles_and_rates(obs.range, obs.range_rate, angles[0], angles[1], angles[2], angles[3])
        chk.ok(rebuilt.range == obs.range and rebuilt.range_rate == obs.range_rate, "range fields not preserved")
        chk.ok(np.max(np.abs(rebuilt.los - obs.los)) <= 1e-12, "from_angles_and_rates los mismatch")
        chk.ok(np.max(np.abs(rebuilt.los_rate - obs.los_rate)) <= 1e-12 * speed, "from_angles_and_rates los_rate mismatch")


def run_all(seed: int) -> int:
    print(f"--- seed {seed} ---")
    rng = np.random.default_rng(seed)
    chk = Checker()
    for name, fn in [
        ("tangent basis", check_tangent_basis),
        ("A1 log identity", check_log_identity),
        ("A2 log sweep", check_log_sweep),
        ("A3 branch continuity", check_branch_continuity),
        ("A4 exp map", check_exp_map),
        ("A5 round trips", check_round_trips),
        ("A6 converged cloud", check_converged_cloud),
        ("A7 antipodal", check_antipodal),
        ("A8 parallel transport", check_parallel_transport),
        ("A9 observe / to_cartesian", check_observe_and_inverse),
        ("A10 perturbation identity", check_perturbation_identity),
        ("A11 angular coordinates", check_angular_coordinates),
    ]:
        before = chk.count
        fn(chk, rng)
        print(f"  {name}: {chk.count - before} assertions")
    return chk.count


def main() -> None:
    time_seed = int(time.time())
    total = 0
    for seed in (FIXED_SEED, time_seed):
        total += run_all(seed)
    if total <= 0:
        raise AssertionError("no assertions executed")
    print(f"PASS: test_los_geometry ({total} assertions over seeds {FIXED_SEED}, {time_seed})")


if __name__ == "__main__":
    main()
