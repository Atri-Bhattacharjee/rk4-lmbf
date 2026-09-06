#pragma once

/**
 * @file los_geometry.h
 * @brief Singularity-free line-of-sight (LOS) measurement geometry.
 *
 * A sensor observation is represented as
 *
 *   range      rho   (m)      = |p|,              p = r_target - r_sensor
 *   range_rate rhodot (m/s)   = u . v,            v = v_target - v_sensor
 *   los        u     (unit)   = p / rho
 *   los_rate   udot  (rad/s)  = (v - rhodot u) / rho     (udot . u = 0)
 *
 * and the exact inverse is  p = rho u,  v = rhodot u + rho udot.
 *
 * Noise and residuals live in the 6-D *local tangent frame* at the measured
 * direction u_m with the deterministic orthonormal basis (e1, e2) = tangentBasis(u_m):
 *
 *   [ d_rho, d_rhodot, d_theta1, d_theta2, d_omega1, d_omega2 ]
 *   [   m  ,   m/s   ,   rad   ,   rad   ,  rad/s  ,  rad/s  ]
 *
 * Angular perturbations are applied with the sphere exponential map and angular
 * residuals are read with the logarithmic map, so there is no azimuth/elevation
 * pole anywhere on the sphere. The only ill-defined configuration is the
 * antipodal one (predicted direction exactly opposite the measured one), which
 * is 180 degrees away from any plausible association and handled deterministically.
 *
 * Numerical hot path: a converged track has all of its particles within
 * micro-radians of the measured direction, so the gamma -> 0 branch of the log
 * map is the common case. It uses atan2 (not acos) and a short power series for
 * gamma / sin(gamma) so that it is accurate to machine precision all the way to
 * gamma == 0, where it returns exactly zero.
 */

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace los {

using Vector6 = Eigen::Matrix<double, 6, 1>;
using TangentBasis = Eigen::Matrix<double, 3, 2>;

constexpr double kPi = 3.14159265358979323846;
//! Below this |a| the exp-map sinc uses its series (truncation ~ a^6/5040 < 2e-28).
constexpr double kSmallAngle = 1e-4;
//! Below this sin(gamma) the log-map gamma/sin(gamma) uses its series (next term ~ 5 s^6/112 < 4e-20).
constexpr double kSmallSine = 1e-3;
//! Ranges at or below this are treated as degenerate (no direction information).
constexpr double kRangeEpsilon = 1e-10;

struct LosObservation {
    double range = 0.0;
    double range_rate = 0.0;
    Eigen::Vector3d los = Eigen::Vector3d::UnitX();
    Eigen::Vector3d los_rate = Eigen::Vector3d::Zero();
};

/**
 * @brief Deterministic right-handed orthonormal basis (e1, e2) of the tangent plane at unit u.
 *
 * e1 is the normalised projection of the coordinate axis least aligned with u; e2 = u x e1.
 */
inline TangentBasis tangentBasis(const Eigen::Vector3d& u) {
    int axis = 0;
    u.cwiseAbs().minCoeff(&axis);
    const Eigen::Vector3d helper = Eigen::Vector3d::Unit(axis);
    const Eigen::Vector3d e1 = (helper - helper.dot(u) * u).normalized();
    const Eigen::Vector3d e2 = u.cross(e1).normalized();
    TangentBasis basis;
    basis.col(0) = e1;
    basis.col(1) = e2;
    return basis;
}

/**
 * @brief Sphere exponential map: move from unit u along tangent vector a by |a| radians.
 *
 * u' = cos(|a|) u + (sin(|a|)/|a|) a, renormalised to remove roundoff.
 */
inline Eigen::Vector3d expMap(const Eigen::Vector3d& u, const Eigen::Vector3d& tangent) {
    const double alpha_sq = tangent.squaredNorm();
    if (alpha_sq == 0.0) {
        return u;
    }
    const double alpha = std::sqrt(alpha_sq);
    double sinc = 0.0;
    if (alpha < kSmallAngle) {
        sinc = 1.0 - alpha_sq / 6.0 + alpha_sq * alpha_sq / 120.0;
    } else {
        sinc = std::sin(alpha) / alpha;
    }
    const Eigen::Vector3d moved = std::cos(alpha) * u + sinc * tangent;
    return moved.normalized();
}

/**
 * @brief Sphere logarithmic map: tangent vector at unit `from` pointing to unit `to`,
 *        with |result| = geodesic angle gamma.
 *
 * The tangent component of `to` at `from` is formed from the difference d = to - from as
 * w = d - (from . d) from, which has |w| = sin(gamma). Forming it this way (rather than as
 * to - (from . to) from) keeps the relative rounding error of w at machine precision for every
 * gamma, because d is computed without cancellation against O(1) quantities. w is exactly zero
 * when to == from. gamma is computed with atan2(|w|, from . to), which keeps full precision at
 * gamma ~ 0 where acos(from . to) would lose half of the digits.
 */
inline Eigen::Vector3d logMap(const Eigen::Vector3d& from, const Eigen::Vector3d& to) {
    const Eigen::Vector3d difference = to - from;
    // Dividing by |from|^2 removes the parallel residue that a unit vector's rounding
    // (|from|^2 = 1 + O(1e-16)) would otherwise leave in w; it is exactly zero at the antipode.
    const Eigen::Vector3d w = difference - (from.dot(difference) / from.squaredNorm()) * from;
    const double s = w.norm();
    const double c = std::clamp(from.dot(to), -1.0, 1.0);

    if (c < 0.0) {
        // Beyond 90 degrees. The direction of w is pure rounding noise once s is at the 1e-16
        // level, i.e. within ~1e-14 rad of the antipode, so return the conventional pi * e1 there.
        if (s < 1e-14) {
            return kPi * tangentBasis(from).col(0);
        }
        return (std::atan2(s, c) / s) * w;
    }

    if (s < kSmallSine) {
        // gamma / sin(gamma) = 1 + s^2/6 + 3 s^4/40 + O(s^6), evaluated in s = sin(gamma).
        const double s_sq = s * s;
        const double ratio = 1.0 + s_sq / 6.0 + 3.0 * s_sq * s_sq / 40.0;
        return ratio * w;
    }
    return (std::atan2(s, c) / s) * w;
}

/**
 * @brief Parallel transport of a tangent vector at unit `from` to the tangent plane at unit `to`
 *        along the connecting geodesic.
 *
 * t' = t - ((to . t) / (1 + from . to)) (from + to). Norm-preserving, exactly tangent at `to`,
 * identity as gamma -> 0. Only singular at the antipode, where plain projection is used.
 */
inline Eigen::Vector3d parallelTransport(const Eigen::Vector3d& from,
                                         const Eigen::Vector3d& to,
                                         const Eigen::Vector3d& tangent) {
    const double denom = 1.0 + from.dot(to);
    if (denom < 1e-12) {
        return tangent - tangent.dot(to) * to;
    }
    return tangent - (to.dot(tangent) / denom) * (from + to);
}

//! Forward model: relative geometry of `target` as seen from `sensor` (both 6-D ECI [r; v]).
inline LosObservation observe(const Vector6& target, const Vector6& sensor) {
    const Eigen::Vector3d p = target.head<3>() - sensor.head<3>();
    const Eigen::Vector3d v = target.tail<3>() - sensor.tail<3>();
    LosObservation obs;
    obs.range = p.norm();
    if (obs.range <= kRangeEpsilon) {
        obs.range_rate = 0.0;
        obs.los = Eigen::Vector3d::UnitX();
        obs.los_rate = Eigen::Vector3d::Zero();
        return obs;
    }
    obs.los = p / obs.range;
    obs.range_rate = obs.los.dot(v);
    obs.los_rate = (v - obs.range_rate * obs.los) / obs.range;
    obs.los_rate -= obs.los_rate.dot(obs.los) * obs.los;
    return obs;
}

//! Exact inverse model: 6-D ECI state of the observed object.
inline Vector6 toCartesian(const LosObservation& obs, const Vector6& sensor) {
    Vector6 state;
    state.head<3>() = sensor.head<3>() + obs.range * obs.los;
    state.tail<3>() = sensor.tail<3>() + obs.range_rate * obs.los + obs.range * obs.los_rate;
    return state;
}

/**
 * @brief Apply a 6-D local-frame perturbation [d_rho, d_rhodot, d_theta1, d_theta2, d_omega1, d_omega2]
 *        expressed in tangentBasis(obs.los).
 *
 * The angular-rate perturbation is added in the tangent plane at the original direction and then
 * parallel-transported to the perturbed direction, so localResidual(obs, perturbed(obs, eps)) == -eps.
 */
inline LosObservation perturbed(const LosObservation& obs, const Vector6& eps) {
    const TangentBasis basis = tangentBasis(obs.los);
    LosObservation result;
    result.range = obs.range + eps(0);
    result.range_rate = obs.range_rate + eps(1);
    const Eigen::Vector3d angular_step = eps(2) * basis.col(0) + eps(3) * basis.col(1);
    result.los = expMap(obs.los, angular_step);
    const Eigen::Vector3d rate_at_origin = obs.los_rate + eps(4) * basis.col(0) + eps(5) * basis.col(1);
    result.los_rate = parallelTransport(obs.los, result.los, rate_at_origin);
    result.los_rate -= result.los_rate.dot(result.los) * result.los;
    return result;
}

/**
 * @brief Residual (measured - predicted) in the local tangent frame of the measured direction.
 *
 * Direction residual: the predicted direction has chart coordinates B_m^T log_{u_m}(u_p) and the
 * measured one has coordinates 0, hence the minus sign. Rate residual: the predicted angular rate
 * is transported into the measured tangent plane before subtraction.
 */
inline Vector6 localResidual(const LosObservation& measured, const LosObservation& predicted) {
    const TangentBasis basis = tangentBasis(measured.los);
    const Eigen::Vector3d direction_log = logMap(measured.los, predicted.los);
    const Eigen::Vector3d transported_rate = parallelTransport(predicted.los, measured.los, predicted.los_rate);
    const Eigen::Vector3d rate_difference = measured.los_rate - transported_rate;

    Vector6 residual;
    residual(0) = measured.range - predicted.range;
    residual(1) = measured.range_rate - predicted.range_rate;
    residual(2) = -basis.col(0).dot(direction_log);
    residual(3) = -basis.col(1).dot(direction_log);
    residual(4) = basis.col(0).dot(rate_difference);
    residual(5) = basis.col(1).dot(rate_difference);
    return residual;
}

/**
 * @brief Derived ECI-axis spherical angles [azimuth, elevation, azimuth_rate, elevation_rate].
 *
 * For display and interoperability only; these coordinates are singular on the ECI z-axis and are
 * never used for noise, residuals, or covariance. At the pole the azimuth rate is reported as 0 and
 * the elevation rate as -sign(z) |udot| (motion away from the pole decreases |elevation|).
 */
inline Eigen::Vector4d angularCoordinates(const LosObservation& obs) {
    const Eigen::Vector3d& u = obs.los;
    const Eigen::Vector3d& udot = obs.los_rate;
    const double horizontal_sq = u.x() * u.x() + u.y() * u.y();
    const double horizontal = std::sqrt(horizontal_sq);

    Eigen::Vector4d angles;
    angles(0) = std::atan2(u.y(), u.x());
    angles(1) = std::atan2(u.z(), horizontal);
    if (horizontal < 1e-12) {
        angles(2) = 0.0;
        angles(3) = -std::copysign(udot.norm(), u.z());
        return angles;
    }
    angles(2) = (u.x() * udot.y() - u.y() * udot.x()) / horizontal_sq;
    const double horizontal_rate = (u.x() * udot.x() + u.y() * udot.y()) / horizontal;
    angles(3) = horizontal * udot.z() - u.z() * horizontal_rate;
    return angles;
}

//! Exact construction from ECI-axis spherical angles and their rates (inverse of angularCoordinates).
inline LosObservation fromAnglesAndRates(double range, double range_rate,
                                         double azimuth, double elevation,
                                         double azimuth_rate, double elevation_rate) {
    const double cos_az = std::cos(azimuth);
    const double sin_az = std::sin(azimuth);
    const double cos_el = std::cos(elevation);
    const double sin_el = std::sin(elevation);

    LosObservation obs;
    obs.range = range;
    obs.range_rate = range_rate;
    obs.los = Eigen::Vector3d(cos_el * cos_az, cos_el * sin_az, sin_el);
    const Eigen::Vector3d e_az(-sin_az, cos_az, 0.0);
    const Eigen::Vector3d e_el(-sin_el * cos_az, -sin_el * sin_az, cos_el);
    obs.los_rate = azimuth_rate * cos_el * e_az + elevation_rate * e_el;
    return obs;
}

}  // namespace los
