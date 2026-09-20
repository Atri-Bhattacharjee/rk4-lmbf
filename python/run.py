"""
SMC-LMB Filter Simulation Harness

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

Shared configuration and run_single_simulation live in simulation_common.py.
Monte Carlo runs use ProcessPoolExecutor with explicit per-run derived seeds.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from simulation_common import *  # noqa: F401,F403
from simulation_common import (
    NUM_STEPS,
    DT,
    NUM_PARTICLES,
    resolve_max_workers,
    run_monte_carlo,
)

# Number of Monte Carlo runs; override with LMB_NUM_RUNS (e.g. for smoke tests)
NUM_MONTE_CARLO = int(os.environ.get("LMB_NUM_RUNS", "20"))


def _resolve_master_seed() -> int:
    env = os.environ.get("LMB_MC_SEED")
    if env is not None and str(env).strip() != "":
        return int(env)
    # One entropy draw for the whole batch; per-run seeds are derived from this.
    return int(np.random.SeedSequence().entropy)


def main():
    """
    Monte Carlo simulation driver.

    Runs NUM_MONTE_CARLO independent simulations and generates:
    - Figure 1: All individual runs overlaid (thin cyan lines)
    - Figure 2: Average performance (thick black line)
    - Figure 3: Component error for Object 1 from run 0
    """
    master_seed = _resolve_master_seed()
    max_workers = resolve_max_workers()

    print("=" * 60)
    print("SMC-LMB Monte Carlo Analysis")
    print("=" * 60)
    print("Configuration:")
    print(f"  Monte Carlo Runs: {NUM_MONTE_CARLO}")
    print(f"  Steps per Run: {NUM_STEPS}, DT: {DT}s")
    print(f"  Particles: {NUM_PARTICLES}")
    print(f"  Master seed: {master_seed}")
    print(f"  Workers: {max_workers}" + (" (serial)" if max_workers == 1 else ""))
    print("=" * 60)

    print("\nRunning Monte Carlo simulations...")
    completed = {"n": 0}

    def handle_complete(run_index, ospa_results):
        completed["n"] += 1
        print(
            f"Run {run_index + 1}/{NUM_MONTE_CARLO} complete - Final OSPA: {ospa_results[-1]:.1f}m "
            f"({completed['n']}/{NUM_MONTE_CARLO} finished)"
        )

    all_run_data, representative_errors, _run_seeds = run_monte_carlo(
        NUM_MONTE_CARLO,
        master_seed=master_seed,
        max_workers=max_workers,
        on_run_complete=handle_complete,
    )
    mean_ospa = np.mean(all_run_data, axis=0)

    print("\n" + "-" * 60)
    print("Monte Carlo Statistics:")
    print(f"  Mean Final OSPA: {np.mean(all_run_data[:, -1]):.1f} m")
    print(f"  Std Final OSPA: {np.std(all_run_data[:, -1]):.1f} m")
    print(f"  Mean OSPA (last 20 steps, averaged): {np.mean(mean_ospa[-20:]):.1f} m")
    print("-" * 60)

    print("\nGenerating Figure 1 (Individual Runs)...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    time_axis = np.arange(NUM_STEPS)
    for i, run_data in enumerate(all_run_data):
        label = "Individual Runs" if i == 0 else None
        ax1.plot(
            time_axis,
            run_data,
            color="#00CED1",
            linewidth=0.5,
            alpha=0.4,
            label=label,
        )
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("OSPA Distance (m)", fontsize=12)
    ax1.set_title(f"OSPA Distance Plot of {NUM_MONTE_CARLO} Runs", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, NUM_STEPS - 1])
    ax1.set_ylim([0, np.max(all_run_data) * 1.1])
    plt.tight_layout()
    output_path_1 = os.path.join(os.path.dirname(__file__), "run_figure_1_individual_runs.png")
    plt.savefig(output_path_1, dpi=150)
    print(f"  Saved: {output_path_1}")

    print("Generating Figure 2 (Average Performance)...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(
        time_axis,
        mean_ospa,
        color="k",
        linewidth=2.0,
        label=f"Average of {NUM_MONTE_CARLO} Runs",
    )
    ax2.set_xlabel("Time Step", fontsize=12)
    ax2.set_ylabel("Average OSPA Distance (m)", fontsize=12)
    ax2.set_title(f"Average OSPA Performance Across {NUM_MONTE_CARLO} Runs", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, NUM_STEPS - 1])
    ax2.set_ylim([0, np.max(mean_ospa) * 1.1])
    plt.tight_layout()
    output_path_2 = os.path.join(os.path.dirname(__file__), "run_figure_2_average_performance.png")
    plt.savefig(output_path_2, dpi=150)
    print(f"  Saved: {output_path_2}")

    print("Generating Figure 3 (Component Error for Object 1)...")
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    component_titles = [
        "X Error (m)",
        "Y Error (m)",
        "Z Error (m)",
        "Vx Error (m/s)",
        "Vy Error (m/s)",
        "Vz Error (m/s)",
    ]
    for idx in range(6):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        ax.plot(time_axis, representative_errors[:, idx], color="b", linewidth=1.0, label="Error")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(component_titles[idx], fontsize=12)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.grid(True, alpha=0.3)
    fig3.suptitle(
        "Filter State Component Error for Object 1 (Representative Run)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path_3 = os.path.join(os.path.dirname(__file__), "run_figure_3_component_error.png")
    plt.savefig(output_path_3, dpi=150)
    print(f"  Saved: {output_path_3}")

    plt.show()
    print("\nMonte Carlo analysis complete.")
    return all_run_data, mean_ospa


if __name__ == "__main__":
    all_data, mean_data = main()
