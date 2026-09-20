"""Welch's t-test and the two-sample Kolmogorov-Smirnov test, implemented on top of NumPy.

The project does not depend on SciPy and this harness is not a good reason to add one, so the two
tests the statistical gate needs are implemented here. Both reduce to one special function:

* Welch's t-test needs the Student-t survival function, which is the regularized incomplete beta
  via the identity ``P(|T| > t) = I_{df/(df + t^2)}(df/2, 1/2)``.
* The KS test needs the Kolmogorov limiting distribution, which is an alternating exponential
  series that converges in a handful of terms.

``self_test`` checks both against independently known values and is called by the harness before
any real comparison, so a bug in this file surfaces as a failure here rather than as a
silently-passing gate.
"""

from __future__ import annotations

import math

import numpy as np

_BETACF_MAX_ITERATIONS = 300
_BETACF_EPS = 3.0e-16
_BETACF_TINY = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function, evaluated by Lentz's method."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETACF_TINY:
        d = _BETACF_TINY
    d = 1.0 / d
    h = d

    for m in range(1, _BETACF_MAX_ITERATIONS + 1):
        m2 = 2 * m

        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + numerator / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        h *= d * c

        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + numerator / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < _BETACF_EPS:
            return h

    raise RuntimeError(f"incomplete beta continued fraction failed to converge for a={a}, b={b}, x={x}")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)``, the regularized incomplete beta function."""
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must lie in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_prefactor = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    prefactor = math.exp(log_prefactor)

    # The continued fraction converges quickly only on one side of this point; reflect otherwise.
    if x < (a + 1.0) / (a + b + 2.0):
        return prefactor * _betacf(a, b, x) / a
    return 1.0 - prefactor * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_statistic: float, degrees_of_freedom: float) -> float:
    """``P(|T| >= |t|)`` for a Student-t variate with the given degrees of freedom."""
    if degrees_of_freedom <= 0.0:
        raise ValueError(f"degrees of freedom must be positive, got {degrees_of_freedom}")
    if not math.isfinite(t_statistic):
        return 0.0
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic * t_statistic)
    return regularized_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x)


def welch_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float, float]:
    """Two-sided Welch t-test. Returns ``(t_statistic, degrees_of_freedom, p_value)``.

    Welch's test does not assume equal variances, which matters here because a change to the RNG
    stream can plausibly shift spread as well as location.
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("Welch's t-test needs at least two observations per sample")

    n_a, n_b = a.size, b.size
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    scaled_a = var_a / n_a
    scaled_b = var_b / n_b
    denominator = scaled_a + scaled_b

    # Two zero-variance samples: identical means agree perfectly, different means disagree totally.
    if denominator == 0.0:
        return (0.0, float(n_a + n_b - 2), 1.0) if float(np.mean(a)) == float(np.mean(b)) else (
            math.inf,
            float(n_a + n_b - 2),
            0.0,
        )

    t_statistic = (float(np.mean(a)) - float(np.mean(b))) / math.sqrt(denominator)
    degrees_of_freedom = denominator * denominator / (
        scaled_a * scaled_a / (n_a - 1) + scaled_b * scaled_b / (n_b - 1)
    )
    return t_statistic, degrees_of_freedom, student_t_two_sided_p(t_statistic, degrees_of_freedom)


def kolmogorov_sf(lam: float) -> float:
    """``Q(lambda) = 2 * sum_{j>=1} (-1)^(j-1) exp(-2 j^2 lambda^2)``, the Kolmogorov limiting tail."""
    if lam <= 0.0:
        return 1.0

    exponent = -2.0 * lam * lam
    total = 0.0
    sign = 2.0
    previous_magnitude = 0.0
    for j in range(1, 101):
        term = sign * math.exp(exponent * j * j)
        total += term
        magnitude = abs(term)
        if magnitude <= 1e-8 * previous_magnitude or magnitude <= 1e-16 * abs(total):
            break
        sign = -sign
        previous_magnitude = magnitude

    return min(max(total, 0.0), 1.0)


def ks_2samp(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns ``(statistic, p_value)``.

    The p-value uses the standard finite-sample-corrected asymptotic form,
    ``Q((sqrt(n_eff) + 0.12 + 0.11/sqrt(n_eff)) * D)``. It is approximate, which is fine for a
    gate stated as ``p > 0.01``.
    """
    a = np.sort(np.asarray(sample_a, dtype=np.float64))
    b = np.sort(np.asarray(sample_b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        raise ValueError("KS test needs non-empty samples")

    pooled = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pooled, side="right") / a.size
    cdf_b = np.searchsorted(b, pooled, side="right") / b.size
    statistic = float(np.max(np.abs(cdf_a - cdf_b)))

    effective_n = math.sqrt(a.size * b.size / (a.size + b.size))
    p_value = kolmogorov_sf((effective_n + 0.12 + 0.11 / effective_n) * statistic)
    return statistic, p_value


def self_test() -> int:
    """Validate both tests against independently known values. Returns the assertion count."""
    checks = 0

    def close(actual: float, expected: float, tolerance: float, what: str) -> None:
        nonlocal checks
        checks += 1
        if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(f"statistics self-test failed: {what}: got {actual!r}, expected {expected!r}")

    # I_x(a, b) reference values.
    close(regularized_incomplete_beta(1.0, 1.0, 0.5), 0.5, 1e-12, "I_0.5(1, 1)")
    close(regularized_incomplete_beta(2.0, 3.0, 0.5), 0.6875, 1e-12, "I_0.5(2, 3)")
    close(regularized_incomplete_beta(0.5, 0.5, 0.25), 1.0 / 3.0, 1e-12, "I_0.25(0.5, 0.5)")
    close(regularized_incomplete_beta(5.0, 2.0, 0.9), 0.885735, 1e-6, "I_0.9(5, 2)")

    # Two-sided Student-t tail probabilities.
    close(student_t_two_sided_p(0.0, 10.0), 1.0, 1e-12, "t=0, df=10")
    close(student_t_two_sided_p(1.0, 1.0), 0.5, 1e-12, "t=1, df=1 (Cauchy)")
    close(student_t_two_sided_p(2.0, 10.0), 0.0733889, 1e-6, "t=2, df=10")
    close(student_t_two_sided_p(2.228138852, 10.0), 0.05, 1e-7, "t=t_crit(0.05, 10)")
    close(student_t_two_sided_p(1.959963985, 1.0e7), 0.05, 1e-4, "t=1.96, df->inf")

    # Welch's t-test against a hand-computable case: equal sizes and equal variances reduce to
    # Student's t with df = n_a + n_b - 2.
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    t_statistic, degrees_of_freedom, p_value = welch_t_test(a, b)
    close(t_statistic, -1.0, 1e-12, "Welch t statistic")
    close(degrees_of_freedom, 8.0, 1e-12, "Welch degrees of freedom")
    close(p_value, 0.3465935, 1e-6, "Welch p-value")
    close(welch_t_test(a, a)[2], 1.0, 1e-12, "Welch on identical samples")

    # Kolmogorov tail reference values.
    close(kolmogorov_sf(1.0), 0.2699996, 1e-6, "Q_KS(1)")
    close(kolmogorov_sf(1.94947), 0.001, 2e-6, "Q_KS(1.94947)")
    close(kolmogorov_sf(0.0), 1.0, 1e-12, "Q_KS(0)")

    # KS statistic on disjoint samples is exactly 1 and must be decisively rejected.
    statistic, p_value = ks_2samp(np.arange(50.0), np.arange(50.0) + 1000.0)
    close(statistic, 1.0, 1e-12, "KS statistic on disjoint samples")
    checks += 1
    if p_value >= 1e-6:
        raise AssertionError(f"statistics self-test failed: KS on disjoint samples gave p={p_value}")

    # KS on identical samples must not reject.
    statistic, p_value = ks_2samp(np.arange(50.0), np.arange(50.0))
    close(statistic, 0.0, 1e-12, "KS statistic on identical samples")
    close(p_value, 1.0, 1e-12, "KS p-value on identical samples")

    # A calibration check: two genuinely identical distributions should reject at the 1% level
    # close to 1% of the time, not far more often.
    rng = np.random.default_rng(20260908)
    rejections = 0
    trials = 400
    for _ in range(trials):
        x = rng.normal(size=48)
        y = rng.normal(size=48)
        if welch_t_test(x, y)[2] < 0.01 or ks_2samp(x, y)[1] < 0.01:
            rejections += 1
    checks += 1
    if rejections > 0.06 * trials:
        raise AssertionError(
            f"statistics self-test failed: null rejection rate {rejections}/{trials} is far above the "
            "~2% expected from two tests at the 1% level"
        )

    return checks


if __name__ == "__main__":
    print(f"PASS: statistics_helpers self-test ({self_test()} assertions)")
