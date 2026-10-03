"""Evaluation figures for a ring run (python/run_ring.py).

Every figure is scored against the objects detected at least once: an object no sensor has seen is
not something the filter could have tracked. Each figure has a CSV twin with the plotted numbers.

    make_all_plots(log, output_dir)               # from run_ring.run()
    python python/evaluation_plots.py path/to/ring_log.npz [--output DIR]

Figures (PNG, light surface):
  1 gospa.png              GOSPA total, and its alpha = 2 decomposition (localisation / missed / false)
  2 cardinality.png        detected objects vs estimated count (sum of r, and tracks with r >= 0.5)
  3 object_error.png       position error of each object's track over time, median and IQR
  4 error_vs_staleness.png error against time since the object was last detected
  5 track_lifecycle.png    existence probability of every track, matched vs never matched
  6 tracking_fraction.png  fraction of detected objects with a GOSPA-matched estimate
  7 nees.png               position NEES of matched pairs against the chi-square(3) 95% band
  8 run_summary.png        detections per sensor, pass outcomes, wall time per phase
  9 ess.png                effective sample size of each posterior, detection vs miss-only updates

Pointing-policy comparison (python/run_tasked.py), scored against the objects that could be found:

    make_tasking_plots(results_dir)
    python python/evaluation_plots.py --tasked python/results/tasked

  tasking_policies.png     per field of view: objects found, objects tracked at the end, and known
                           objects re-detected when they come back, for every policy
  tasking_discovery.png    objects found against time, per field of view
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill, light mode), used in fixed slot order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # slots 1-3: blue, orange, aqua
BAND = "#f0efec"                             # neutral gray, for reference bands
DPI = 150
LINE = 1.0                                   # points; ~2 px at 150 dpi
MARKER = 4.0                                 # points; ~8 px at 150 dpi

# chi-square(3) central 95% interval, for position NEES
CHI2_3_LO, CHI2_3_HI = 0.2158, 9.3484

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.labelcolor": INK_SECONDARY,
    "lines.linewidth": LINE,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "font.family": "sans-serif",
}


def _minutes(seconds):
    return np.asarray(seconds, dtype=np.float64) / 60.0


def _write_csv(path: Path, header, rows) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _subtitle(ax, text):
    """A secondary line under the left title; the title is lifted to make room for it."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=16)
    ax.text(0.0, 1.01, text, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="bottom")


def _binned(x, y, edges):
    """Median and IQR of y in each x bin (bins with < 3 points are NaN)."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(len(centres), np.nan)
    lo = np.full(len(centres), np.nan)
    hi = np.full(len(centres), np.nan)
    for b in range(len(centres)):
        inside = (x >= edges[b]) & (x < edges[b + 1]) & np.isfinite(y)
        if np.sum(inside) >= 3:
            lo[b], median[b], hi[b] = np.percentile(y[inside], [25, 50, 75])
    return centres, median, lo, hi


# --------------------------------------------------------------------------------------------


def plot_gospa(log, out: Path) -> list[Path]:
    t = _minutes(log["time"])
    km2 = 1e-6
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.35})
    top.plot(t, np.asarray(log["gospa"]) / 1e3, color=SERIES[0])
    top.set_ylabel("GOSPA (km)")
    top.set_title("GOSPA against objects detected at least once", loc="left")
    _subtitle(top, log.get("gospa_params", ""))

    parts = [np.asarray(log[k]) * km2 for k in ("localisation", "missed", "false_positive")]
    labels = ("localisation", "missed objects", "false tracks")
    cumulative = np.zeros_like(t)
    for part, label, colour in zip(parts, labels, SERIES):
        upper = cumulative + part
        bottom.fill_between(t, cumulative, upper, color=colour, alpha=0.18, linewidth=0)
        bottom.plot(t, upper, color=colour, label=label)
        cumulative = upper
    bottom.set_ylabel("cost (km², stacked)")
    bottom.set_xlabel("time (min)")
    bottom.set_title("Decomposition: the three costs sum to GOSPA²", loc="left")
    bottom.legend(loc="upper left", ncol=3)
    paths = [_save(fig, out / "gospa.png")]
    paths.append(_write_csv(out / "gospa.csv",
                            ["time_s", "gospa_m", "localisation_m2", "missed_m2", "false_m2"],
                            zip(log["time"], log["gospa"], log["localisation"], log["missed"],
                                log["false_positive"])))
    return paths


def plot_cardinality(log, out: Path, threshold: float) -> list[Path]:
    t = _minutes(log["time"])
    truth = np.asarray(log["num_truths"])
    sum_r = np.asarray(log["existence_sum"])
    confirmed = np.asarray(log["num_estimates"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True,
                                      gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.35})
    top.plot(t, truth, color=SERIES[0], label="objects detected so far")
    top.plot(t, sum_r, color=SERIES[1], label="sum of existence probabilities")
    top.plot(t, confirmed, color=SERIES[2], label=f"tracks with r ≥ {threshold:g}")
    top.set_ylabel("count")
    top.set_title("Cardinality", loc="left")
    top.legend(loc="upper left")

    bottom.axhline(0.0, color=AXIS, linewidth=0.8)
    bottom.plot(t, sum_r - truth, color=SERIES[1], label="sum of r − detected")
    bottom.plot(t, confirmed - truth, color=SERIES[2], label=f"r ≥ {threshold:g} − detected")
    bottom.set_ylabel("estimate − truth")
    bottom.set_xlabel("time (min)")
    bottom.set_title("Cardinality error (positive: too many tracks)", loc="left")
    bottom.legend(loc="upper left", ncol=2)
    paths = [_save(fig, out / "cardinality.png")]
    paths.append(_write_csv(out / "cardinality.csv",
                            ["time_s", "detected_objects", "existence_sum", "confirmed_tracks", "all_tracks"],
                            zip(log["time"], truth, sum_r, confirmed, log["num_tracks"])))
    return paths


def plot_object_error(log, out: Path, cutoff: float) -> list[Path]:
    rows = np.asarray(log["object_rows"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_yscale("log")
    if len(rows):
        for o in np.unique(rows[:, 1]):
            mine = rows[rows[:, 1] == o]
            ax.plot(_minutes(mine[:, 0]), mine[:, 2] / 1e3, color=MUTED, alpha=0.25, linewidth=0.5)
        times = np.unique(rows[:, 0])
        med, lo, hi = [], [], []
        for tt in times:
            values = rows[rows[:, 0] == tt, 2]
            values = values[np.isfinite(values)]
            if len(values) >= 3:
                q = np.percentile(values, [25, 50, 75])
            else:
                q = [np.nan] * 3
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        ax.fill_between(_minutes(times), np.asarray(lo) / 1e3, np.asarray(hi) / 1e3, color=SERIES[0],
                        alpha=0.12, linewidth=0, label="interquartile range")
        ax.plot(_minutes(times), np.asarray(med) / 1e3, color=SERIES[0], label="median over objects")
    events = np.asarray(log["detection_events"])
    if len(events):
        ymin = ax.get_ylim()[0] if len(rows) else 1e-3
        ax.plot(_minutes(events[:, 0]), np.full(len(events), ymin), "|", color=INK_SECONDARY,
                markersize=6, alpha=0.5, label="detections")
    ax.axhline(cutoff / 1e3, color=AXIS, linewidth=0.8)
    ax.text(0.995, cutoff / 1e3, f" GOSPA cutoff {cutoff / 1e3:g} km", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("position error of the object's track (km)")
    ax.set_title("Per-object track error (each gray line is one detected object)", loc="left")
    _subtitle(ax, "track = the one last born from that object's detections; not truncated at the cutoff")
    ax.legend(loc="upper left")
    paths = [_save(fig, out / "object_error.png")]
    paths.append(_write_csv(out / "object_errors.csv",
                            ["time_s", "object", "track_error_m", "gospa_matched_error_m",
                             "since_last_detection_s", "nees"], rows.tolist()))
    return paths


def plot_error_vs_staleness(log, out: Path) -> list[Path]:
    rows = np.asarray(log["object_rows"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_yscale("log")
    if len(rows):
        since = _minutes(rows[:, 4])
        error = rows[:, 2] / 1e3
        ok = np.isfinite(error) & (error > 0)
        ax.scatter(since[ok], error[ok], s=6, color=SERIES[0], alpha=0.25, linewidths=0,
                   label="object × sample")
        if np.any(ok):
            edges = np.linspace(0.0, max(since[ok].max(), 1.0) + 1e-9, 25)
            centres, median, lo, hi = _binned(since[ok], error[ok], edges)
            ax.fill_between(centres, lo, hi, color=SERIES[1], alpha=0.12, linewidth=0,
                            label="interquartile range")
            ax.plot(centres, median, color=SERIES[1], label="binned median")
    ax.set_xlabel("time since the object was last detected (min)")
    ax.set_ylabel("position error of the object's track (km)")
    ax.set_title("How fast tracks go stale between detections", loc="left")
    ax.legend(loc="upper left")
    return [_save(fig, out / "error_vs_staleness.png")]


def plot_track_lifecycle(log, out: Path) -> list[Path]:
    rows = np.asarray(log["track_rows"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    final_time = float(np.max(log["time"])) if len(log["time"]) else 0.0
    matched_count = 0
    unmatched_count = 0
    if len(rows):
        for key in np.unique(rows[:, 1]):
            mine = rows[rows[:, 1] == key]
            ever = bool(np.any(mine[:, 3] >= 0))
            colour = SERIES[0] if ever else SERIES[1]
            matched_count += ever
            unmatched_count += not ever
            ax.plot(_minutes(mine[:, 0]), mine[:, 2], color=colour, alpha=0.35, linewidth=0.6)
            ax.plot(_minutes(mine[0, 0]), mine[0, 2], "o", color=colour, markersize=2.5, alpha=0.6)
            if mine[-1, 0] < final_time:
                ax.plot(_minutes(mine[-1, 0]), mine[-1, 2], "x", color=INK_SECONDARY, markersize=3,
                        alpha=0.7)
    # Legend proxies (the per-track lines are too many to label).
    ax.plot([], [], color=SERIES[0], label=f"matched to an object at some sample ({matched_count})")
    ax.plot([], [], color=SERIES[1], label=f"never matched ({unmatched_count})")
    ax.plot([], [], "o", color=MUTED, markersize=2.5, label="first sample")
    ax.plot([], [], "x", color=INK_SECONDARY, markersize=3, label="last sample before pruning")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("existence probability r")
    ax.set_title("Track lifecycles", loc="left")
    _subtitle(ax, "sampled at the metric interval; a track born and pruned between samples is not shown")
    ax.legend(loc="lower left", fontsize=7)
    paths = [_save(fig, out / "track_lifecycle.png")]
    paths.append(_write_csv(out / "track_lifecycle.csv",
                            ["time_s", "label_key", "existence", "matched_object"], rows.tolist()))
    return paths


def plot_tracking_fraction(log, out: Path) -> list[Path]:
    t = _minutes(log["time"])
    truths = np.asarray(log["num_truths"], dtype=np.float64)
    fraction = np.divide(np.asarray(log["num_assigned"], dtype=np.float64), truths,
                         out=np.full_like(truths, np.nan), where=truths > 0)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 5.4), sharex=True,
                                      gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.35})
    top.plot(t, fraction, color=SERIES[0])
    top.set_ylim(-0.02, 1.02)
    top.set_ylabel("fraction")
    top.set_title("Tracking fraction: detected objects with a matched estimate", loc="left")
    _subtitle(top, f"matched = GOSPA-assigned within the cutoff ({log.get('gospa_cutoff', 0) / 1e3:g} km)")
    finite = np.isfinite(fraction)
    if np.any(finite):
        last = np.flatnonzero(finite)[-1]
        top.annotate(f"{fraction[last]:.2f}", (t[last], fraction[last]), xytext=(4, 0),
                     textcoords="offset points", fontsize=8, color=INK, va="center")
    bottom.plot(t, truths, color=SERIES[0])
    bottom.set_ylabel("detected objects")
    bottom.set_xlabel("time (min)")
    bottom.set_title("Denominator: objects detected at least once", loc="left")
    return [_save(fig, out / "tracking_fraction.png")]


def plot_nees(log, out: Path) -> list[Path]:
    rows = np.asarray(log["object_rows"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_yscale("log")
    ax.axhspan(CHI2_3_LO, CHI2_3_HI, color=BAND, zorder=0, label="χ²(3) central 95%")
    ax.axhline(3.0, color=AXIS, linewidth=0.8)
    inside_text = "no matched pairs"
    if len(rows):
        nees = rows[:, 5]
        ok = np.isfinite(nees) & (nees > 0)
        if np.any(ok):
            ax.scatter(_minutes(rows[ok, 0]), nees[ok], s=6, color=SERIES[0], alpha=0.35, linewidths=0,
                       label="matched object × sample")
            inside = np.mean((nees[ok] >= CHI2_3_LO) & (nees[ok] <= CHI2_3_HI))
            inside_text = (f"{inside * 100:.0f}% of {int(np.sum(ok))} matched pairs inside the band "
                           f"(95% for a consistent filter); median NEES {np.median(nees[ok]):.2g} vs 3")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("position NEES")
    ax.set_title("Consistency: is the filter's stated uncertainty honest?", loc="left")
    _subtitle(ax, inside_text)
    ax.legend(loc="upper left")
    return [_save(fig, out / "nees.png")]


def plot_run_summary(log, out: Path) -> list[Path]:
    per_sensor = np.asarray(log["detections_per_sensor"])
    outcomes = dict(log.get("pass_outcomes", {}))
    timers = dict(log.get("timers", {}))
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(8.5, 8.0),
                                  gridspec_kw={"height_ratios": [1.2, 1, 1], "hspace": 0.6})
    a.bar(np.arange(len(per_sensor)), per_sensor, width=0.7, color=SERIES[0])
    a.set_xlabel("sensor index")
    a.set_ylabel("detections")
    a.set_title(f"Detections per sensor ({int(per_sensor.sum())} total, "
                f"{int(np.sum(per_sensor > 0))} of {len(per_sensor)} sensors saw something)", loc="left")

    order = ["first detection", "re-acquired", "re-acquired, extra birth", "taken by another track",
             "duplicate birth", "lost, re-born"]
    names = [name for name in order if name in outcomes] + [n for n in outcomes if n not in order]
    values = [outcomes[n] for n in names]
    b.barh(names, values, height=0.3, color=SERIES[0])
    for y, v in enumerate(values):
        b.text(v, y, f" {v}", va="center", fontsize=8, color=INK)
    b.invert_yaxis()
    b.set_xlabel("passes")
    b.set_title("Pass outcomes (a pass = detections of one object < 60 s apart)", loc="left")
    b.grid(axis="y", visible=False)

    phases = sorted(timers, key=timers.get, reverse=True)
    seconds = [timers[p] for p in phases]
    c.barh(phases, seconds, height=0.4, color=SERIES[0])
    for y, v in enumerate(seconds):
        c.text(v, y, f" {v:.1f} s", va="center", fontsize=8, color=INK)
    c.invert_yaxis()
    c.set_xlabel("wall time (s)")
    c.set_title(f"Wall time by phase ({log.get('wall_seconds', sum(seconds)):.1f} s total)", loc="left")
    c.grid(axis="y", visible=False)
    return [_save(fig, out / "run_summary.png")]


def plot_ess(log, out: Path) -> list[Path]:
    records = np.asarray(log.get("posterior_records", np.zeros((0, 6))))
    records = records.reshape(-1, records.shape[1] if records.ndim == 2 and records.shape[1] else 6)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    edges = np.logspace(-4, 0, 41)
    detection = records[records[:, 2] >= 0.5, 1]
    miss = records[records[:, 2] < 0.5, 1]
    for values, colour, label in ((detection, SERIES[0], "updates with a detection"),
                                  (miss, SERIES[1], "miss-only updates")):
        if len(values):
            ax.hist(np.clip(values, edges[0], 1.0), bins=edges, color=colour, alpha=0.35,
                    histtype="stepfilled", linewidth=0)
            ax.hist(np.clip(values, edges[0], 1.0), bins=edges, color=colour, histtype="step",
                    linewidth=LINE, label=f"{label} ({len(values)}, median {np.median(values):.3g})")
    threshold = float(log.get("config", {}).get("regularization_ess_threshold", 0.5))
    ax.axvline(threshold, color=AXIS, linewidth=0.8)
    ax.text(threshold, 1.0, " regularization threshold", transform=ax.get_xaxis_transform(),
            fontsize=7, color=INK_SECONDARY, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("effective sample size / particles")
    ax.set_ylabel("posterior updates")
    ax.set_title("Particle degeneracy at each update", loc="left")
    regularized = records[:, 3].mean() if len(records) else 0.0
    _subtitle(ax, f"ESS = 1 / sum w^2 of the posterior weights; {regularized * 100:.0f}% of updates regularized")
    ax.legend(loc="upper left")
    paths = [_save(fig, out / "ess.png")]
    paths.append(_write_csv(out / "posterior_updates.csv",
                            ["time_s", "ess_fraction", "detection_mass", "regularized", "fused_components",
                             "fallbacks"][:records.shape[1]], records.tolist()))
    return paths


def make_all_plots(log: dict, output_dir) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = log.get("config", {})
    threshold = float(config.get("existence_threshold", 0.5))
    cutoff = float(log.get("gospa_cutoff", 10e3))
    paths: list[Path] = []
    with plt.rc_context(STYLE):
        paths += plot_gospa(log, out)
        paths += plot_cardinality(log, out, threshold)
        paths += plot_object_error(log, out, cutoff)
        paths += plot_error_vs_staleness(log, out)
        paths += plot_track_lifecycle(log, out)
        paths += plot_tracking_fraction(log, out)
        paths += plot_nees(log, out)
        paths += plot_run_summary(log, out)
        paths += plot_ess(log, out)
    return paths


# =============================================================================
# Pointing-policy comparison
# =============================================================================

# Bars top to bottom. The proposed tasker takes the accent; the others are context.
POLICY_ORDER = ("oracle", "sgd+custody", "sgd", "random+custody", "random")
POLICY_LABEL = {"oracle": "oracle", "sgd+custody": "tuned schedule + custody", "sgd": "tuned schedule only",
                "random+custody": "random + custody", "random": "random only", "all-seeing": "all-seeing sensors"}
PROPOSED = "sgd+custody"
# Lines keep one colour per policy across panels (slots 1-3, fixed order).
DISCOVERY_SERIES = (("sgd+custody", SERIES[0]), ("random+custody", SERIES[1]), ("oracle", SERIES[2]))


def _load_tasking(results_dir: Path):
    summary = json.loads((results_dir / "summary.json").read_text())
    rows = {(row["fov_half_angle_deg"], row["policy"]): row for row in summary["summary"]}
    angles = sorted({angle for angle, _ in rows if angle is not None}, reverse=True)
    return summary, rows, angles


def plot_tasking_policies(results_dir: Path, out: Path) -> list[Path]:
    _, rows, angles = _load_tasking(results_dir)
    reference = rows.get((None, "all-seeing"))
    metrics = (
        ("Objects found", "share of the objects that could be found",
         lambda row: row["found_share"], 1.0),
        ("Objects tracked at the end", "share with an estimate within the GOSPA cutoff",
         lambda row: row["final_tracking_fraction"], reference["final_tracking_fraction"] if reference else None),
        ("Known objects re-detected", "share of passes by an already-found object with a detection",
         lambda row: row["known_pass_detection_rate"], 1.0),
    )
    policies = [policy for policy in POLICY_ORDER if any((angle, policy) in rows for angle in angles)]
    fig, axes = plt.subplots(len(metrics), len(angles), figsize=(2.6 * len(angles) + 1.6, 1.9 * len(metrics) + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    table = []
    for r, (title, subtitle, value, mark) in enumerate(metrics):
        for c, angle in enumerate(angles):
            ax = axes[r, c]
            for y, policy in enumerate(policies):
                row = rows.get((angle, policy))
                if row is None or not np.isfinite(value(row)):
                    continue
                share = float(value(row))
                ax.barh(y, share, height=0.5, color=SERIES[0] if policy == PROPOSED else AXIS)
                ax.text(share + 0.02, y, f"{share:.0%}", va="center", fontsize=7, zorder=3,
                        color=INK if policy == PROPOSED else INK_SECONDARY,
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})
                table.append((angle, policy, title, share))
            if mark is not None:
                ax.axvline(mark, color=MUTED, linewidth=0.6)
                ax.text(mark, -0.75, "all-seeing ", ha="right", va="center", fontsize=6.5, color=MUTED)
            ax.set_axisbelow(True)
            ax.set_xlim(0.0, 1.18)
            ax.set_ylim(len(policies) - 0.4, -1.1)
            ax.set_xticks([0.0, 0.5, 1.0], ["0", "50%", "100%"])
            ax.set_yticks(range(len(policies)), [POLICY_LABEL[policy] for policy in policies])
            ax.grid(axis="y", visible=False)
            ax.tick_params(length=0)
            if r == 0:
                ax.set_title(f"±{angle:g}° field of view", loc="left", fontsize=9)
        axes[r, 0].annotate(f"{title}\n", xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-118, 2 if r else 18),
                            textcoords="offset points", fontsize=9, fontweight="bold", color=INK, va="bottom")
        axes[r, 0].annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-118, 2 if r else 18),
                            textcoords="offset points", fontsize=7.5, color=INK_SECONDARY, va="bottom")
    fig.subplots_adjust(hspace=0.55, wspace=0.12)
    paths = [_save(fig, out / "tasking_policies.png")]
    paths.append(_write_csv(out / "tasking_policies.csv", ["fov_half_angle_deg", "policy", "metric", "share"], table))
    return paths


def plot_tasking_discovery(results_dir: Path, out: Path) -> list[Path]:
    summary, rows, angles = _load_tasking(results_dir)
    fig, axes = plt.subplots(1, len(angles), figsize=(2.9 * len(angles) + 0.6, 3.0), sharey=True, squeeze=False)
    table = []
    for c, angle in enumerate(angles):
        ax = axes[0, c]
        for policy, colour in DISCOVERY_SERIES:
            curves = []
            for record in summary["records"]:
                if record["policy"] != policy or record["fov_half_angle_deg"] != angle:
                    continue
                path = results_dir / f"{policy}_fov{angle:g}_seed{record['seed']}.npz"
                with np.load(path, allow_pickle=False) as data:
                    first = np.sort(data["first_detection"][~np.isnan(data["first_detection"])])
                    period = data["time"][-1] / record["orbits"]
                    orbits = np.linspace(0.0, record["orbits"], 121)
                    curves.append(np.searchsorted(first, orbits * period, side="right") / max(len(data["findable"]), 1))
            if not curves:
                continue
            mean = np.mean(curves, axis=0)
            ax.plot(orbits, mean, color=colour, linewidth=1.2, label=POLICY_LABEL[policy])
            table += [(angle, policy, float(o), float(m)) for o, m in zip(orbits, mean)]
            if policy == PROPOSED:
                ax.plot(orbits[-1], mean[-1], "o", color=colour, markersize=MARKER, markeredgecolor=SURFACE,
                        markeredgewidth=1.0)
                ax.annotate(f"{mean[-1]:.0%}", (orbits[-1], mean[-1]), xytext=(-4, 6), textcoords="offset points",
                            ha="right", fontsize=8, color=INK)
        ax.set_title(f"±{angle:g}° field of view", loc="left", fontsize=9)
        ax.set_xlabel("orbits")
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    axes[0, 0].set_ylabel("objects found, share of those that could be")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1.06), ncol=len(labels))
    fig.subplots_adjust(wspace=0.1)
    paths = [_save(fig, out / "tasking_discovery.png")]
    paths.append(_write_csv(out / "tasking_discovery.csv", ["fov_half_angle_deg", "policy", "orbits", "share_found"], table))
    return paths


def make_tasking_plots(results_dir, output_dir=None) -> list[Path]:
    results_dir = Path(results_dir)
    out = Path(output_dir) if output_dir else results_dir
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        return plot_tasking_policies(results_dir, out) + plot_tasking_discovery(results_dir, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-plot a saved ring run, or a policy comparison")
    parser.add_argument("log", nargs="?", help="ring_log.npz written by run_ring.py")
    parser.add_argument("--tasked", help="results folder written by run_tasked.py")
    parser.add_argument("--output", help="directory for the figures (default: next to the log)")
    args = parser.parse_args()
    if args.tasked:
        for path in make_tasking_plots(args.tasked, args.output):
            print(f"wrote {path}")
        return
    if not args.log:
        parser.error("give a ring_log.npz, or --tasked RESULTS_DIR")
    from run_ring import load

    log = load(args.log)
    output = Path(args.output) if args.output else Path(args.log).parent
    for path in make_all_plots(log, output):
        print(f"wrote {path}")
    print(json.dumps(log.get("pass_outcomes", {})))


if __name__ == "__main__":
    main()
