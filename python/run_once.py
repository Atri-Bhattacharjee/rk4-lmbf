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
from simulation_common import NUM_STEPS, DT, NUM_PARTICLES, run_single_simulation


def main():
    """
    Single simulation driver.

    Runs one simulation and generates an OSPA distance plot (solid black line).
    """
    print("=" * 60)
    print("SMC-LMB Single Run Analysis")
    print("=" * 60)
    print("Configuration:")
    print(f"  Steps: {NUM_STEPS}, DT: {DT}s")
    print(f"  Particles: {NUM_PARTICLES}")
    print("=" * 60)

    print("\nRunning simulation...")
    ospa_results, _ = run_single_simulation(verbose=False, collect_track_errors=False)

    print(f"Run complete - Final OSPA: {ospa_results[-1]:.1f}m")
    print(f"  Mean OSPA (last 20 steps): {np.mean(ospa_results[-20:]):.1f} m")

    print("\nGenerating OSPA plot...")

    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.arange(NUM_STEPS)
    ax.plot(time_axis, ospa_results, color="k", linewidth=2.0, label="OSPA Distance")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("OSPA Distance (m)", fontsize=12)
    ax.set_title("OSPA Distance Plot", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, NUM_STEPS - 1])
    ax.set_ylim([0, np.max(ospa_results) * 1.1])
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "figure_ospa_single_run.png")
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.show()

    print("\nSingle run analysis complete.")
    return ospa_results


if __name__ == "__main__":
    main()
