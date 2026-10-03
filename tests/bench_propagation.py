"""Cost of one particle propagation step, with and without process noise.

Times SMC_LMB_Tracker.predict in eager mode over a few 1000-particle clouds at the ring's geometry
(800 km circular orbit), once with the ring's time-consistent process noise and once with none. The
difference is what the noise draws cost. Public bindings only; not part of CI.

    python tests/bench_propagation.py
    python tests/bench_propagation.py --tracks 20 --repeats 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lmb_engine_loader import import_lmb_engine  # noqa: E402

lmb = import_lmb_engine()

MU = 3.986004418e14
RADIUS = 6.371e6 + 800.0e3
RING_Q = np.diag([2.0**2] * 3 + [0.02**2] * 3)


def make_tracks(num_tracks: int, num_particles: int, rng: np.random.Generator):
    speed = np.sqrt(MU / RADIUS)
    tracks = []
    for t in range(num_tracks):
        angle = 2.0 * np.pi * t / max(num_tracks, 1)
        centre = np.array([RADIUS * np.cos(angle), RADIUS * np.sin(angle), 0.0,
                           -speed * np.sin(angle), speed * np.cos(angle), 0.0])
        spread = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0])
        particles = []
        for _ in range(num_particles):
            p = lmb.Particle()
            p.state_vector = centre + rng.normal(size=6) * spread
            p.weight = 1.0 / num_particles
            particles.append(p)
        label = lmb.TrackLabel()
        label.birth_time = 0
        label.index = t
        tracks.append(lmb.Track(label, 0.9, particles))
    return tracks


def time_predict(q: np.ndarray, tracks, dt: float, repeats: int, fast: bool) -> float:
    """Median ns per particle-step of predict(dt)."""
    propagator = lmb.TwoBodyPropagator(q, seed=1, noise_reference_dt=60.0)
    sensor_model = lmb.InOrbitSensorModel(100.0, 1.0, 1e-6, 1e-6, 1e-8, 1e-8)
    birth = lmb.AdaptiveBirthModel(10, 0.5, np.eye(6), seed=2)
    tracker = lmb.SMC_LMB_Tracker(propagator, sensor_model, birth, 0.999, 1, 1e-3, 1e-15, 0.99,
                                  seed=3)
    if fast:
        tracker.set_fast_mode(True)
    tracker.set_tracks(tracks)
    steps = sum(len(t.particle_weights()) for t in tracks)
    tracker.predict(dt)  # warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        tracker.predict(dt)
        samples.append((time.perf_counter() - start) * 1e9 / steps)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tracks", type=int, default=10)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    tracks = make_tracks(args.tracks, args.particles, np.random.default_rng(7))
    print(f"{args.tracks} tracks x {args.particles} particles, median of {args.repeats}")
    for fast in (False, True):
        label = "fast" if fast else "legacy"
        for dt in (1.0, 60.0):
            noisy = time_predict(RING_Q, tracks, dt, args.repeats, fast)
            quiet = time_predict(np.zeros((6, 6)), tracks, dt, args.repeats, fast)
            share = (noisy - quiet) / noisy if noisy > 0 else float("nan")
            print(f"  {label:6s} dt={dt:4.0f}s  with noise {noisy:7.1f} ns  without {quiet:7.1f} ns  "
                  f"noise share {share:5.1%}")


if __name__ == "__main__":
    main()
