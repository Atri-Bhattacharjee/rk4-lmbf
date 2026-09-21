"""
SMC-LMB Filter Simulation Harness (Single Run)

This script validates the Sequential Monte Carlo Labeled Multi-Bernoulli filter
implementation by simulating a multi-object space debris tracking scenario.

Scenario:
- 3 LEO objects with staggered birth times (steps 0, 30, 50)
- Sensor on a circular 400 km orbit, propagated alongside the truths
- 100 time steps at 60-second intervals

Measurements are (range, range rate, line-of-sight unit vector, line-of-sight angular rate) with a
6x6 noise covariance in the local tangent frame of the measured direction (see the constants below).

The simulation uses a "dual-noise" strategy:
- Truth generation uses low noise (high precision sensor)
- Filter model uses inflated noise (wide acceptance gate for birth convergence)

Shared configuration and run_single_simulation live in simulation_common.py. Names required by
tests (FILTER_SIGMAS, SCENARIO, …) are re-exported here unchanged.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from simulation_common import *  # noqa: F401,F403
from simulation_common import NUM_STEPS, DT, NUM_PARTICLES, GOSPA_PARAMS, run_single_simulation


def main():
    """
    Single simulation driver.

    Runs one simulation and generates an GOSPA distance plot (solid black line).
    """
    print("=" * 60)
    print("SMC-LMB Single Run Analysis")
    print("=" * 60)
    print("Configuration:")
    print(f"  Steps: {NUM_STEPS}, DT: {DT}s")
    print(f"  Particles: {NUM_PARTICLES}")
    print("=" * 60)

    print("\nRunning simulation...")
    gospa_results, _, components = run_single_simulation(
        verbose=False, collect_track_errors=False, collect_components=True
    )

    print(f"Run complete - Final GOSPA: {gospa_results[-1]:.1f}m")
    print(f"  Mean GOSPA (last 20 steps): {np.mean(gospa_results[-20:]):.1f} m")
    print(f"  Metric: GOSPA, {GOSPA_PARAMS}")

    print("\nGenerating GOSPA plot...")

    # Stacked in the p-th-power domain, where the three components are exactly additive and sum to
    # GOSPA**p. Unnormalised GOSPA steps up whenever an object is born -- it grows as sqrt(k) with
    # cardinality even under perfect tracking -- so plotting the total alone reads as degradation.
    # Stacking makes the cardinality contribution explicit instead.
    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.arange(NUM_STEPS)
    localisation = np.asarray(components["localisation"], dtype=np.float64)
    missed = np.asarray(components["missed"], dtype=np.float64)
    false_positive = np.asarray(components["false_positive"], dtype=np.float64)

    ax.stackplot(
        time_axis,
        localisation,
        missed,
        false_positive,
        labels=["Localisation", "Missed truths", "False tracks"],
        colors=["#4c72b0", "#dd8452", "#c44e52"],
        alpha=0.85,
    )
    ax.plot(time_axis, np.asarray(gospa_results) ** 2, color="k", linewidth=2.0,
            label="Total (GOSPA$^2$)")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("GOSPA$^2$ cost (m$^2$)", fontsize=12)
    ax.set_title(f"GOSPA error decomposition\n{GOSPA_PARAMS}", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, NUM_STEPS - 1])
    ax.set_ylim([0, np.max(np.asarray(gospa_results) ** 2) * 1.1])
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "figure_gospa_single_run.png")
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.show()

    print("\nSingle run analysis complete.")
    return gospa_results


if __name__ == "__main__":
    main()
