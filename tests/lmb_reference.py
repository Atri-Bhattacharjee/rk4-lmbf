"""Brute-force reference for one SMC-LMB update step.

Every joint association hypothesis is enumerated explicitly -- no ranked assignment, no grouping --
and the LMB posterior is assembled straight from the definitions (Reuter, Vo, Vo & Dietmayer, "The
Labeled Multi-Bernoulli Filter", IEEE TSP 2014), with a state-dependent detection probability
P_D(x) = P_D * visible(x):

    eta_i(j) = r_i <p_i, P_D g_j> / kappa           track i produced measurement j
    eta_i(0) = 1 - r_i <p_i, P_D>                    track i not detected (missed, or absent)
    w(theta) ∝ prod_i eta_i(theta(i))                over injective theta: tracks -> {0} ∪ measurements

    rho_i    = r_i (1 - <p_i, P_D>) / (1 - r_i <p_i, P_D>)
    r_i'     = sum_theta w(theta) [theta(i) != 0 ? 1 : rho_i]
    p_i'(x)  ∝ sum_theta w(theta) [ theta(i) = j : p_i(x) P_D(x) g_j(x) / <p_i, P_D g_j>
                                    theta(i) = 0 : rho_i p_i(x) (1 - P_D(x)) / (1 - <p_i, P_D>) ]

Inputs are plain NumPy arrays, and the per-particle likelihoods and visibility come from the
engine's *public* primitives (InOrbitSensorModel.calculate_likelihood, SensorArray.sees), so
agreement with SMC_LMB_Tracker.update is evidence about the update's algebra rather than a
restatement of it. Enumeration is exponential; keep scenes to a handful of tracks and measurements.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackInput:
    existence: float
    states: np.ndarray        # (n, 6)
    weights: np.ndarray       # (n,)


@dataclass
class ReferencePosterior:
    existence: np.ndarray             # (N,) posterior existence per input track
    mixture: list                     # N arrays of normalised posterior particle weights
    active: np.ndarray                # (N,) bool: some particle visible to some sensor
    best_assignment: tuple            # theta of the heaviest hypothesis, -1 = undetected
    hypothesis_weights: np.ndarray    # normalised, one per enumerated hypothesis


def particle_visibility(lmb, sensors, states):
    """Per particle: lowest index of a sensor that sees it, or -1. None -> legacy (sees all, index -1)."""
    if sensors is None:
        return None
    visible = np.full(len(states), -1, dtype=np.int64)
    for p, state in enumerate(states):
        for k in range(sensors.size()):
            if sensors.sees(k, state):
                visible[p] = k
                break
    return visible


def particle_likelihoods(lmb, sensor_model, states, measurement):
    out = np.empty(len(states))
    for p, state in enumerate(states):
        particle = lmb.Particle()
        particle.state_vector = state
        particle.weight = 1.0
        out[p] = sensor_model.calculate_likelihood(particle, measurement)
    return out


def enumerate_hypotheses(num_tracks, num_meas):
    """Every injective map from tracks to {-1} ∪ range(num_meas)."""
    choices = [-1] + list(range(num_meas))
    for theta in itertools.product(choices, repeat=num_tracks):
        used = [j for j in theta if j >= 0]
        if len(used) == len(set(used)):
            yield theta


def posterior(lmb, sensor_model, tracks, measurements, sensors, p_detection, clutter_intensity):
    """Exact LMB posterior for `tracks` given `measurements`, observed through `sensors` (or None)."""
    num_tracks = len(tracks)
    num_meas = len(measurements)
    meas_sensor = []
    for measurement in measurements:
        meas_sensor.append(-1 if sensors is None else sensors.index_of(measurement.sensor_id_))

    normalised = []
    visibility = []
    coverage = np.zeros(num_tracks)
    coverage_by_sensor = []
    for i, track in enumerate(tracks):
        weights = np.asarray(track.weights, dtype=np.float64)
        wn = weights / weights.sum()
        normalised.append(wn)
        vis = particle_visibility(lmb, sensors, track.states)
        visibility.append(vis)
        if vis is None:
            coverage[i] = 1.0
            coverage_by_sensor.append(None)
        else:
            coverage[i] = float(wn[vis >= 0].sum())
            coverage_by_sensor.append(
                np.array([float(wn[vis == k].sum()) for k in range(sensors.size())]))

    active = coverage > 0.0

    likelihood = np.zeros((num_tracks, num_meas))
    per_particle_g = [[None] * num_meas for _ in range(num_tracks)]
    observable = np.zeros((num_tracks, num_meas), dtype=bool)
    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            s = meas_sensor[j]
            if visibility[i] is None:
                mask = np.ones(len(track.states), dtype=bool)
                observable[i, j] = True
            else:
                mask = visibility[i] == s
                observable[i, j] = bool(coverage_by_sensor[i][s] > 0.0)
            g = particle_likelihoods(lmb, sensor_model, track.states, measurement)
            g = np.where(mask, g, 0.0)
            per_particle_g[i][j] = g
            likelihood[i, j] = float(np.sum(normalised[i] * g))

    r = np.array([t.existence for t in tracks], dtype=np.float64)
    pd_mass = p_detection * coverage
    eta_miss = 1.0 - r * pd_mass
    eta_det = np.where(observable, r[:, None] * p_detection * likelihood / clutter_intensity, 0.0)
    rho = np.where(eta_miss > 0.0, r * (1.0 - pd_mass) / np.where(eta_miss > 0.0, eta_miss, 1.0), 0.0)

    hypotheses = list(enumerate_hypotheses(num_tracks, num_meas))
    log_w = np.empty(len(hypotheses))
    for h, theta in enumerate(hypotheses):
        total = 0.0
        for i, j in enumerate(theta):
            factor = eta_miss[i] if j < 0 else eta_det[i, j]
            total += np.log(factor) if factor > 0.0 else -np.inf
        log_w[h] = total
    finite = np.isfinite(log_w)
    weights = np.zeros(len(hypotheses))
    weights[finite] = np.exp(log_w[finite] - log_w[finite].max())
    weights /= weights.sum()

    existence = np.zeros(num_tracks)
    mixture = []
    for i, track in enumerate(tracks):
        n = len(track.states)
        wn = normalised[i]
        vis_any = np.ones(n, dtype=bool) if visibility[i] is None else visibility[i] >= 0
        miss_density = wn * np.where(vis_any, 1.0 - p_detection, 1.0)
        miss_norm = 1.0 - pd_mass[i]
        miss_density = miss_density / miss_norm if miss_norm > 0.0 else np.zeros(n)

        mix = np.zeros(n)
        for h, theta in enumerate(hypotheses):
            j = theta[i]
            if j < 0:
                existence[i] += weights[h] * rho[i]
                mix += weights[h] * rho[i] * miss_density
            else:
                existence[i] += weights[h]
                if likelihood[i, j] > 0.0:
                    mix += weights[h] * wn * per_particle_g[i][j] / likelihood[i, j]
        total = mix.sum()
        mixture.append(mix / total if total > 0.0 else np.full(n, 1.0 / n))

    best = hypotheses[int(np.argmax(weights))]
    return ReferencePosterior(existence=existence, mixture=mixture, active=active,
                              best_assignment=best, hypothesis_weights=weights)


def expected_multiplicity_ok(original_states, resampled_states, mixture, slack=1.0 + 1e-9):
    """Systematic resampling keeps particle p between floor and ceil of N * W_p copies.

    Returns (ok, worst_excess) where worst_excess is max_p |count_p - N W_p| - slack (<= 0 passes).
    Particles are identified by exact state equality, so the input cloud must have distinct states.
    """
    n = len(original_states)
    counts = np.zeros(n)
    for state in resampled_states:
        matches = np.flatnonzero(np.all(original_states == state, axis=1))
        if matches.size != 1:
            return False, np.inf
        counts[matches[0]] += 1
    deviation = np.abs(counts - n * np.asarray(mixture))
    worst = float(np.max(deviation) - slack)
    return worst <= 0.0, worst
