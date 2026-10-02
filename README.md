# SMC-LMB Tracker

[![CI](https://github.com/Atri-Bhattacharjee/rk4-lmbf/actions/workflows/ci.yml/badge.svg)](https://github.com/Atri-Bhattacharjee/rk4-lmbf/actions/workflows/ci.yml)

A C++/Python project for Sequential Monte Carlo Labeled Multi-Bernoulli (SMC-LMB) space debris tracking. The core filter runs in C++ (Eigen + pybind11) and is driven from Python for simulation and validation.

## Prerequisites

| Tool | Version |
|------|---------|
| Git | any recent |
| C++ compiler | C++17 (GCC 11+, Clang 14+, or MSVC 2019+) |
| CMake | 3.15+ |
| Python | 3.10+ |

Native and Python dependencies:

- **Eigen3** — linear algebra (install via your OS package manager)
- **pybind11, numpy, matplotlib** — installed via pip (see below)

> **Note:** The bundled `external/vcpkg` submodule is **not** used for the default build. Install native dependencies with your system package manager (Linux/macOS) or a standalone vcpkg install (Windows).

## Quick start

```bash
git clone https://github.com/Atri-Bhattacharjee/rk4-lmbf.git
cd rk4-lmbf

./scripts/build.sh                # Linux / macOS — creates venv, installs deps, builds
# .\scripts\build.ps1             # Windows PowerShell

source venv/bin/activate          # Windows: venv\Scripts\activate
python python/run_once.py         # fast single-run smoke test
python python/run.py              # full Monte Carlo analysis (slow)
```

Manual build (same steps as the helper scripts):

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

source ./scripts/cmake-venv-args.sh   # Windows: see scripts/cmake-venv-args.ps1
cmake --preset release \
  -DPython_EXECUTABLE="$CMAKE_VENV_PYTHON" \
  -Dpybind11_DIR="$CMAKE_PYBIND11_DIR"
cmake --build --preset release
```

## Platform-specific native dependencies

Install **Eigen3** and a C++ toolchain before running CMake. Always activate your Python virtual environment before configuring so CMake finds pybind11 from pip.

### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev python3-venv python3-pip
```

### macOS

```bash
xcode-select --install            # if needed
brew install cmake eigen
```

### Windows

1. Install [Visual Studio 2022](https://visualstudio.microsoft.com/) with the **Desktop development with C++** workload.
2. Install [CMake](https://cmake.org/download/).
3. Install Eigen3, for example with [vcpkg](https://vcpkg.io/):

   ```powershell
   git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
   C:\vcpkg\bootstrap-vcpkg.bat
   C:\vcpkg\vcpkg install eigen3:x64-windows
   ```

4. Configure with the vcpkg toolchain and venv Python (adjust paths as needed):

   ```powershell
   .\venv\Scripts\Activate.ps1
   $cmakeVenvArgs = & .\scripts\cmake-venv-args.ps1
   cmake --preset release @cmakeVenvArgs `
     -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
   cmake --build --preset release
   ```

   Or set `Eigen3_DIR` if Eigen is installed elsewhere.

## Build options

### CMake presets (recommended)

| Preset | Use case |
|--------|----------|
| `release` | Normal use (default) |
| `debug` | Debugging with symbols |
| `asan` | AddressSanitizer + UndefinedBehaviorSanitizer; drive it via `./scripts/asan-test.sh` |

```bash
source ./scripts/cmake-venv-args.sh   # Windows: see scripts/cmake-venv-args.ps1
cmake --preset release \
  -DPython_EXECUTABLE="$CMAKE_VENV_PYTHON" \
  -Dpybind11_DIR="$CMAKE_PYBIND11_DIR"
cmake --build --preset release

cmake --preset debug \
  -DPython_EXECUTABLE="$CMAKE_VENV_PYTHON" \
  -Dpybind11_DIR="$CMAKE_PYBIND11_DIR"
cmake --build --preset debug
```

Built extensions are written to:

- `python/lmb_engine/Release/` — recommended
- `python/lmb_engine/Debug/`
- `python/lmb_engine/Asan/` — sanitizer build, kept separate so it never shadows the others

Optional CMake flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `LMB_ENGINE_ENABLE_VALIDATION` | `OFF` | Keep hot-path validation checks in Release builds |
| `LMB_ENGINE_SANITIZE` | `OFF` | Instrument with ASan + UBSan. Only the extension is instrumented, so the ASan runtime must be `LD_PRELOAD`ed at run time; use `./scripts/asan-test.sh`, which handles that. |

### Manual CMake (without presets)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"      # Windows: cmake --build . -j %NUMBER_OF_PROCESSORS%
cd ..
```

### Important: use one Python for build and run

The extension is tied to the Python version used during CMake configure (e.g. `lmb_engine.cpython-312-…so`). Activate the same `venv` before both `cmake` and `python`.

## Running

All simulation scripts must be run from the repo root with the same `venv` used at build time. They import the C++ extension via `python/lmb_engine_loader.py`.

### Single run (fast)

```bash
source venv/bin/activate
python python/run_once.py
```

Writes `python/figure_gospa_single_run.png`.

### Monte Carlo simulation

```bash
source venv/bin/activate
python python/run.py
```

Runs 20 Monte Carlo trials and writes PNG figures under `python/`:

- `run_figure_1_individual_runs.png`
- `run_figure_2_average_performance.png`
- `run_figure_3_component_error.png`

(`python/2026_ieee_aerospace.py` writes the same three without the `run_` prefix.)

### Paper reproduction

`python/2026_ieee_aerospace.py` is the IEEE Aerospace 2026 configuration (same harness as `run.py` with `K_BEST=100`). It produces the same three figure outputs as the Monte Carlo script.

Optional verbose import logging (works with any simulation script):

```bash
LMB_ENGINE_VERBOSE=1 python python/run_once.py
```

Both Monte Carlo scripts honour `LMB_NUM_RUNS` (e.g. `LMB_NUM_RUNS=1 python python/run.py` for a quick smoke run).

## Accuracy metric

Tracking accuracy is measured with **GOSPA** (Generalized Optimal Sub-Pattern Assignment;
Rahmathullah, Garcia-Fernandez & Svensson, FUSION 2017), implemented in `src/metrics.cpp` and
exposed as `lmb_engine.calculate_gospa_distance` / `calculate_gospa_components`.

| Parameter | Value | Note |
|-----------|-------|------|
| `p` | 2 | compile-time constant, not an argument |
| `alpha` | 2 | fixed — the error decomposition is only valid at 2 |
| `c` (cutoff) | 10 km | `lmb_engine.GOSPA_DEFAULT_CUTOFF`, the single source in the repo |
| base distance | position only (first 3 components), metres | |
| normalisation | none | |

Two consequences worth knowing before reading a number:

- **It is unnormalised**, so it grows as `sqrt(k)` with the number of objects and is *not* bounded
  by `c`. The tight bound is `c * sqrt((m + n) / 2)`. Expect the curve to step up at each birth
  even under perfect tracking; `python/run_once.py` plots the decomposition stacked so that the
  cardinality contribution is visible rather than mistaken for degradation.
- **At `alpha = 2` it decomposes exactly** into localisation, missed-truth and false-track costs,
  which sum to `GOSPA**p`. That is the reason it replaced OSPA here: OSPA folds localisation and
  cardinality error into one number, so it can say that two sensor-tasking policies differ but not
  how. `calculate_gospa_components` returns the breakdown.

The cutoff is scaled for the production driver (10k particles, ~2.6 km mean error, ~2% of steps at
the cutoff). The 200-particle test harness errs by more than `c` at most steps, which weakens
`tests/test_statistical_equivalence.py` — see the KNOWN WEAKNESS block in that file.

## Measurement model

A measurement is a line-of-sight observation, not an azimuth/elevation tuple:

| Field | Type | Meaning |
|-------|------|---------|
| `range_` | float | Sensor-to-object distance (m), > 0 |
| `range_rate_` | float | Radial relative speed (m/s) |
| `los_` | 3-vector | Unit line-of-sight direction in ECI |
| `los_rate_` | 3-vector | Line-of-sight angular rate (rad/s), orthogonal to `los_` |
| `covariance_` | 6x6 | Noise covariance in the **local tangent frame** of `los_` |
| `sensor_state_` | 6-vector | Sensor ECI state `[x, y, z, vx, vy, vz]` |

`sensor_id_` is no longer decorative: with a `SensorArray` it is what ties a measurement back to the
sensor that produced it, and therefore to the field of view the association is scored against. See
[Sensors, pointing and field of view](#sensors-pointing-and-field-of-view).

The exact conversion is `r = r_s + range * los`, `v = v_s + range_rate * los + range * los_rate`
(`Measurement.fromCartesian` / `Measurement.toCartesian`), which is smooth everywhere on the sphere: there is
no azimuth/elevation singularity, no angle wrapping and no `cos(el)` division anywhere in the filter.

`covariance_` is authoritative for the likelihood. It is expressed in the deterministic tangent basis
`(e1, e2) = tangent_basis(los_)` and ordered as

```
[d_range (m), d_range_rate (m/s), d_theta1 (rad), d_theta2 (rad), d_omega1 (rad/s), d_omega2 (rad/s)]
```

with variances in `[m^2, (m/s)^2, rad^2, rad^2, (rad/s)^2, (rad/s)^2]`. The same frame is used by every
producer and consumer:

- **Simulation** applies noise with `Measurement.perturbed(eps)` (sphere exponential map for the direction,
  parallel transport for the rate), so an angular sigma of `1e-6 rad` is exactly that at every direction.
- **Likelihood** (`InOrbitSensorModel`) forms the 6-D residual with `local_residual` (log map + parallel
  transport) and evaluates a Gaussian with `covariance_`; the six constructor variances only define
  `defaultCovariance()`.
- **Birth** (`AdaptiveBirthModel`) samples `eps ~ N(0, birth_covariance_local)` in this frame and maps
  each sample exactly to ECI, so new-track particles spread in range, angle and rate, never in ECI x/y/z.
  The constructor takes an optional `seed` for reproducible particle streams.

`Measurement.angularCoordinates()` / `Measurement.fromAnglesAndRates(...)` derive ECI-axis azimuth/elevation
(and rates) for display or interoperability only; they are singular on the z-axis and are not used by the filter.

The angular-rate sigmas in the simulation scripts (`TRUTH_SIGMA_ANGLE_RATE`, `FILTER_SIGMA_ANGLE_RATE`) are
placeholders pending sensor characterisation.

### Tests

Build the extension first, then:

```bash
source venv/bin/activate
./scripts/ci-test.sh              # Linux / macOS
# .\scripts\ci-test.ps1           # Windows PowerShell
```

Or run individual scripts:

```bash
source venv/bin/activate
python tests/test_two_body_propagator_multistep.py
python tests/assignments.py
python tests/test_los_geometry.py          # sphere geometry primitives vs independent references
python tests/test_validation_dimensions.py # input validation and error messages
python tests/test_sensor_likelihood.py     # likelihood vs NumPy reference, rotation invariance, chi-square
python tests/test_sensor_fov.py            # visibility predicate vs an atan2 reference, pointing, coverage
python tests/test_fov_detection_probability.py  # FOV-scaled P_D vs an independent existence-update reference
python tests/test_adaptive_birth_model.py  # birth covariance recovery and spread statistics
python tests/test_bindings_api.py          # Python API surface
python tests/statistics_helpers.py         # self-test of the Welch/KS implementations
python tests/test_particle_statistics.py   # cloud statistics vs NumPy, zero-copy view aliasing
python tests/test_invariants.py            # per-step structural invariants of a seeded run
python tests/test_golden_invariance.py     # digest vs the committed fixture (see below)
python tests/test_end_to_end.py            # short tracker runs, incl. a pole-aligned scene
```

`LMB_ENGINE_BUILD=Debug` (or `Release`, or `Asan`) forces the loader to pick a specific build
directory.

### Determinism and regression harness

`TwoBodyPropagator`, `AdaptiveBirthModel` and `SMC_LMB_Tracker` all take an optional `seed`. With
those set plus `np.random.seed`, a whole simulation becomes a pure function of one integer, which
is what the regression harness is built on. Debug, Release and the sanitizer build all produce
bitwise-identical results *on one toolchain*.

```bash
python tests/test_golden_invariance.py               # comparison against tests/fixtures/
python tests/test_golden_invariance.py --exact       # force bitwise
python tests/test_golden_invariance.py --rtol 1e-9   # tolerant, for a deliberate FP-order change
python tests/test_golden_invariance.py --portable    # force the platform-independent subset
python tests/test_golden_invariance.py --write       # regenerate the fixtures
python tests/test_statistical_equivalence.py         # 48 seeds/arm, Welch t + KS on mean GOSPA
python tests/bench_engine.py --json before.json      # record hot-path timings and peak RSS
python tests/bench_engine.py --compare before.json   # diff against a recorded set
./scripts/gate.sh                                    # Release + Debug suites, statistics, benchmark
./scripts/asan-test.sh                               # ASan + UBSan build and suite
```

#### Bitwise reproduction is per-platform

**Standing rule for CI/CD:** never require bitwise identity on macOS (or any non-reference
runner). New tests that pin absolute float digests, GOSPA byte heads, or `np.array_equal` against a
Linux-captured fixture must gate on the reference platform and use portable / `rtol` / statistical
checks everywhere else — including the macOS GitHub Actions job. Bitwise gates belong only on
x86-64 Linux / libstdc++.

The committed golden fixtures encode one toolchain's bit pattern, so the bitwise gate is limited to
the platform they were written on (x86-64 Linux / libstdc++) and `test_golden_invariance.py` selects
its mode accordingly: bitwise there, `--portable` everywhere else. Two things make the bit pattern a
property of the toolchain rather than of the filter:

* **The RNG stream.** `std::mt19937_64` is specified bit-for-bit, but `std::normal_distribution` and
  `std::uniform_real_distribution` are not. libstdc++ and the MSVC STL agree; libc++ (macOS) draws a
  different sequence from the same seed, and a run diverges on its first birth.
* **Floating-point rounding.** Clang contracts `a * b + c` into a single-rounding `fma` wherever the
  ISA has one (always, on arm64), Apple's libm is not bit-identical to glibc's, and Eigen reduces in
  NEON lane order rather than SSE lane order.

The first of those is a different draw from the same distribution, not a rounding difference, so no
`--rtol` absorbs it. Portable mode therefore checks what holds anywhere — the NumPy-driven
measurement counts, that the filter is still tracking, and run-to-run repeatability — and prints the
observed drift for the log. The numerical gate off the reference platform is
`test_statistical_equivalence.py`, which `ci-test.sh` and `ci-test.ps1` run there for exactly that
reason (at `--alpha 1e-4`, since it runs on every PR). On the reference platform it stays out of
`ci-test.sh`: the bitwise gate already covers it, and in Debug it costs minutes.

### Smoke import

```bash
python -c "import sys; sys.path.insert(0, 'python'); from lmb_engine_loader import import_lmb_engine; import_lmb_engine(); print('ok')"
```

## Sensors, pointing and field of view

A sensor is a 6-D ECI state plus an orientation. Several can be active at once, each with its own
pointing, and Python is expected to re-point them at every timestep. Range bounds and the angular
half-widths are **global**: they live on the `SensorFovConfig` the `SensorArray` holds, and every
sensor in that array shares them.

```python
fov = lmb_engine.SensorFovConfig(
    min_range=0.0,                      # m, inclusive
    max_range=2.5e7,                    # m, inclusive (default +inf)
    half_width=np.deg2rad(3.0),         # rad, about the frame's width axis
    half_height=np.deg2rad(1.5),        # rad, about the frame's height axis
)

sensors = lmb_engine.SensorArray(fov)
sensors.add("sensor_0", state_0, boresight_0)   # pointed
sensors.add_unpointed("survey", state_1)        # range-only, ignores the FOV

for step in range(num_steps):
    sensors.set_state(0, propagated_state)      # or sensors.set_states(all_states)   (S, 6)
    sensors.point_at(0, target_position)        # or sensors.set_boresight(0, direction)
    ...
    tracker.update(measurements, sensors)
```

### The visibility predicate

For a pointed sensor, `(width, height, boresight)` is a right-handed orthonormal triad with
`height = up` and `width = up x boresight`, so `width x height = boresight`. A target at relative
position `d = r_target - r_sensor` is visible when

```
min_range <= |d| <= max_range
d . boresight > 0
|d . width|  <= (d . boresight) * tan(half_width)
|d . height| <= (d . boresight) * tan(half_height)
```

That is a rectangular pyramid in the tangent-plane (pinhole) convention, the natural shape for a
focal-plane detector. Half-angles must lie in `(0, pi/2)`; the two tangents are cached whenever the
config changes, so the hot path does no trigonometry per particle and there is no `acos`. An
**unpointed** sensor skips the three angular lines entirely and is bounded by range alone.

`SensorArray.sees` is the single source of truth for this test. The simulation side calls it to
decide whether a ground-truth object produces a measurement at all, and the filter calls it on
every particle, so the two cannot drift apart.

### Pointing

| Call | Effect |
|------|--------|
| `set_boresight(i, b)` | Re-point. `b` is normalised; the roll is carried over by parallel transport (`los::parallelTransport`), the minimal rotation, so a slewing footprint does not spin about its own axis. |
| `set_pointing(i, b, up)` | Re-point with an explicit roll. `up` is orthogonalised against `b` and must not be parallel to it. |
| `point_at(i, position)` | Aim at an ECI position (3-vector or 6-D state), keeping the roll. |
| `set_state(i, s)` / `set_states(S, 6)` | Move the sensors. |
| `set_boresights((S, 3))` | Re-point every sensor in one call. |
| `set_pointed(i, bool)` | Turn the angular test on or off for one sensor. |

A near-antipodal flip leaves nothing to transport and falls back to the deterministic
`los::tangentBasis` frame. Both `add` and every pointing call are atomic: a rejected call leaves
the array exactly as it was.

### Effective detection probability

This is the part that changes the filter's arithmetic. The detection probability is
**state-dependent**: `P_D(x) = P_D * visible(x)`, where `visible(x)` asks whether a particle at `x`
is inside some sensor's volume. For a track `i` with normalised particle weights `w_p`, existence
`r_i`, and a measurement `j` produced by sensor `s(j)`, the update's hypothesis factors are the LMB
ones (Reuter, Vo, Vo & Dietmayer, *The Labeled Multi-Bernoulli Filter*, IEEE TSP 2014):

```
L_ij       = sum_p w_p * [particle p visible to s(j)] * g_j(x_p)     # only what that sensor can see
eta_i(j)   = r_i * P_D * L_ij / kappa                                # track i produced measurement j
eta_i(0)   = 1 - r_i * P_D * q_union[i]                              # not detected: missed, or absent
rho_i      = r_i (1 - P_D q_union[i]) / (1 - r_i P_D q_union[i])     # existence given "not detected"
r_i'       = sum_j c_ij + c_i0 * rho_i                               # c = hypothesis-weight marginals
```

`q_union[i]` is the weight-fraction of the cloud inside any volume. The posterior cloud mixes the
detection posteriors (each normalised over the visible particles) with the miss posterior, which
reweights every particle by `1 - P_D(x_p)` — so the part of a cloud a sensor looked at and saw
nothing in is **emptied out**, not just discounted in existence. The marginals `c_ij`, `c_i0`
already carry the `eta` factors; nothing multiplies them by `P_D * L / kappa` again.
`tests/test_existence_enumeration.py` checks all of this against an exhaustive enumeration of the
joint hypotheses (`tests/lmb_reference.py`).

Consequences:

- **A track outside every field of view is left out of the update entirely.** Its only hypothesis
  is "not detected" with `P_D(x) = 0` everywhere on its cloud, whose posterior is its prior — so it
  keeps its existence probability and cloud exactly, and it never enters the ranked assignment. The
  assignment is therefore the size of the handful of tracks near a sensor, not the catalogue.
- **A track invisible to sensor s cannot claim s's measurements.** Those pairs are written into the
  cost matrix as `INF_COST` and skip their likelihood pass.
- **A step in which no sensor reported anything is informative.** `update(measurements, sensors)`
  applies the miss-only update to every observable track rather than returning early.

Measurements are traced back to their sensor through `Measurement.sensor_id_`, which must match an
id in the array. A `sensor_id_` the array does not know is a `ValueError`, not a guess — guessing
would score the association against the wrong volume.

Query the coverage from Python with `sensors.coverage_fractions(track)` (shape `(S,)`) and
`sensors.coverage_fraction(track)` (the union). `SensorArray.coverage` skips sensors that cannot
reach a cloud's bounding sphere before its per-particle pass, which changes no result.

### Back-compatibility and the disjointness assumption

`tracker.update(measurements)` — the single-argument overload — treats every track as fully
observable and an empty step as a no-op. A `SensorArray` built from a default `SensorFovConfig` plus
`add_unpointed` (what `python/run.py`, `python/run_once.py` and `python/2026_ieee_aerospace.py` use)
reproduces that path bit for bit: full coverage is reported as exactly `1.0`.

The committed golden fixtures still pass bitwise after the existence/mixture correction above, but
not because the arithmetic is unchanged: at `P_D = 0.999999999` and `kappa = 1e-15` one hypothesis
always carries all the weight and every existence probability saturates at `1.0`, where the old and
the corrected formulas agree. The goldens are blind to association scaling for the same reason, which
is why the enumeration test exists.

**Sensor volumes are assumed disjoint.** The filter is not built to have one object reported by two
sensors in the same step, and nothing here tries to. `visible_sensor` breaks a tie by returning the
lowest index; `python/run_multisensor.py` counts and reports violations rather than hiding them.

### Multi-sensor demo

```bash
python python/run_multisensor.py        # LMB_MULTISENSOR_STEPS, LMB_MULTISENSOR_PARTICLES
```

Two bounded, pointed sensors re-aimed every step against three objects, so something is always
unobservable — its track holds its existence probability flat while it is. Writes
`python/figure_multisensor.png`.

## Lazy propagation and time-consistent process noise

`TwoBodyPropagator(Q, seed, noise_reference_dt=None)` keeps the historical per-call noise: `Q` is
added in full on every `propagate()`, whatever `dt` is. Pass `noise_reference_dt=T` and `Q` is read
as the covariance accumulated over `T` seconds of continuous white noise on `[r, v]` with `r' = v`;
a step of `dt` then draws from

```
Qd(dt) = [ Dpp dt + (Dpv + Dvp) dt²/2 + Dvv dt³/3 ,  Dpv dt + Dvv dt²/2 ]      D = Q / T
         [ Dvp dt + Dvv dt²/2                     ,  Dvv dt              ]
```

so sixty 1 s steps and one 60 s step spread a cloud alike. Velocity noise now also diffuses into
position within a step, so a `Q` tuned for the per-call model at `dt = T` spreads clouds faster
under this one — retune when switching.

`tracker.set_lazy_propagation(True, max_pending=60.0)` (requires the time-consistent model) makes
`predict(dt)` advance only the clock and the survival probability. A track's cloud is propagated

- when `update(measurements, sensors)` finds that some particle of it could be inside a sensor
  volume (a conservative test on the cloud's bounding spheres, the gravity terms over the lag, and
  the propagator's `noise_displacement_bound`);
- when it has lagged the clock by `max_pending` seconds, in substeps of at most `max_pending`;
- on `tracker.synchronize()`.

`get_tracks()` may therefore return tracks that lag the clock: read `track.propagated_time()`, or
call `synchronize()` first. `tracker.track_summary()` returns labels, existence, propagation times
and (optionally) mean states as NumPy arrays without copying any particle cloud.

The bounding-sphere test is only as tight as the cloud: a cloud smeared along its orbit has a
bounding sphere that nearly always reaches some sensor, and is then propagated at the fine step.

### Regularization

`tracker.set_regularization(True, bandwidth_scale=1.0, ess_threshold=0.5)` jitters a freshly
resampled cloud whenever its posterior effective sample size fell below `ess_threshold * N`:
`x <- m + a (x - m) + h L eps`, with `m`, `L L^T` the weighted posterior mean and covariance,
`a = sqrt(1 - h^2)` (Liu & West kernel shrinkage: mean and covariance are kept, so repeated
resampling does not inflate the cloud) and `h` the regularized-particle-filter bandwidth
`bandwidth_scale * (4 / (N (d + 2)))^(1/(d+4))`, `d = 6` (Musso, Oudjane & Le Gland 2001). It
restores the diversity resampling destroys. It cannot rescue a posterior that has collapsed onto
one particle -- a one-particle posterior has no spread to jitter with -- which is what a
near-full-state measurement at close range does to a 1000-particle cloud.
`tracker.set_record_diagnostics(True)` / `tracker.take_diagnostics()` expose the ESS and detection
mass of every posterior, which is how to see that happening.

### Fused proposal (re-acquisition)

`tracker.set_fused_proposal(True, ess_min=20, neighbours=0, fallback_ess_min=20)`, off by default.

**The problem it fixes.** The ordinary update scores a track's particles against a measurement and
averages the scores. That only works if some particle lands within a few noise widths of the
measurement in every one of the six channels. When a track comes back past a sensor after an orbit,
its cloud is a long thin needle in position-velocity space (tens of km long, but thinner than the
measurement in some velocity directions -- the flow stretches it along the orbit and, with little
process noise, squeezes it elsewhere to keep its volume). No particle lands near the measurement,
every score rounds to exactly zero, and the returning object is born as a new track.

**What it does.** For each (track, measurement) pair the ordinary update runs first. If its
effective sample size is below `ess_min` (it has collapsed or underflowed), the pair's detection
component is rebuilt where the needle and the measurement overlap:

1. express the measurement as a Gaussian in state space (its Jacobian, by central differences);
2. fit a kernel density to the track's particles nearest the measured state (`neighbours`, 0 = max(30,
   5% of the cloud), used to set the kernel width; the kernel sum runs over every particle that can
   reach the measurement's footprint);
3. sample from the Gaussian product of the two, widened 1.5x, and weight each draw by
   kernel density x exact likelihood / proposal density.

The association likelihood becomes the average of those weights instead of a sum that underflows. A
cloud whose bounding sphere reaches the sensor with no particle inside is scored the same way. If the
fused weights collapse too (ESS below `fallback_ess_min`), the component falls back to the Gaussian
product with a closed-form likelihood -- an approximation kept as a last resort.

Measured on the ring (100 sensors, 1000 objects, 1000 particles): repeat passes re-acquired went from
0/17 to 17/17 (2 orbits) and 77/77 (5 orbits); births equal objects detected; no detection taken by
another object's track across kappa from 4e-19 to 4e-3. `tests/test_fused_proposal.py` checks it
against the ordinary update where both are valid, and against a closed-form answer on a needle where
the ordinary sum is exactly zero.

**Assignment solver fix.** The Munkres core assumed non-negative costs (its reduction only subtracts a
positive minimum, and its first step stars exact zeros), so a row like `[-62, 0]` came back assigned to
the 0. The wrapper in `src/assignment.cpp` now shifts the matrix to be non-negative first; the optimal
assignment cannot change. Detection costs are routinely negative, and a track no sensor can see has a
miss cost of exactly 0.

## Ring scenario: many sensors, sampled debris

```bash
python python/run_ring.py                          # 100 sensors, 1000 objects, 2 orbits, 1000 particles
python python/run_ring.py --orbits 5 --particles 2000 --seed 7
python python/evaluation_plots.py python/results/ring_seed20260930/ring_log.npz   # re-plot a run
```

`N` range-only sensors on a circular equatorial 800 km orbit (20 km range) against objects sampled
per run from `python/data/eci_800km-altitude_20km-range_randomized_phase.csv`, on a 1 s clock with
lazy propagation, regularization and the fused proposal. Defaults: sensor noise 10 m, 1 m/s,
6.7e-4 rad, 6.7e-5 rad/s (angles matched to range at ~15 km; `--tight-angles` for the 1 urad
placeholder), filter at 1x truth, kappa 4e-15 (derived in `RingConfig`). Truth, sensors and
detection are propagated in NumPy with the engine's RK4 model. Tuning lives in the `RingConfig` block
of `python/run_ring.py`; useful flags: `--sigma-scale`, `--truth-sigmas`, `--kappa`, `--no-fused`,
`--no-regularization`. The truth has no process noise, so the filter's process noise only keeps
particles diverse; with `--tight-angles` it dominates the cloud's growth between passes and makes
the filter underconfident (NEES ~0.3) unless reduced ~100x.

Outputs go to `python/results/ring_seed<seed>/` (gitignored): `ring_log.npz`, `summary.json`, and
nine figures with CSV twins — GOSPA and its decomposition, cardinality, per-object track error, error
against time since last detection, track lifecycles, tracking fraction, position NEES against the
χ²(3) band, a run summary (detections per sensor, pass outcomes, wall time per phase), and ESS per
update. Everything is scored against objects detected at least once. `summary.json` also counts
detections taken by the object's own track versus another object's track.

## Project layout

```
rk4-lmbf/
├── CMakeLists.txt              # pybind11 extension target (lmb_engine)
├── CMakePresets.json           # release / debug configure & build presets
├── pyproject.toml              # setuptools package metadata
├── setup.py                    # setuptools entry point
├── requirements.txt            # Python dependencies (pybind11, numpy, matplotlib)
├── LICENSE
├── .gitmodules                 # optional external/ submodules
├── .github/
│   └── workflows/
│       └── ci.yml              # Linux, macOS, Windows build + test
├── scripts/
│   ├── gate.sh                 # full pre-commit gate: both builds, statistics, benchmark
│   ├── asan-test.sh            # ASan + UBSan build and suite
│   ├── build.sh                # Linux/macOS: venv + configure + build
│   ├── build.ps1               # Windows build helper
│   ├── cmake-venv-args.sh      # resolve venv Python & pybind11 for CMake
│   ├── cmake-venv-args.ps1     # Windows variant
│   ├── ci-test.sh              # Linux/macOS CI validation suite
│   └── ci-test.ps1             # Windows CI validation suite
├── src/                        # C++ SMC-LMB filter engine
│   ├── main.cpp                # pybind11 module bindings
│   ├── smc_lmb_tracker.{h,cpp} # core LMB filter
│   ├── los_geometry.h          # tangent basis, exp/log maps, transport, observe/toCartesian
│   ├── adaptive_birth_model.{h,cpp}
│   ├── in_orbit_sensor_model.{h,cpp}
│   ├── sensor_fov.h            # sensor pointing, rectangular FOV, per-track coverage fractions
│   ├── two_body_propagator.{h,cpp}
│   ├── assignment.{h,cpp}      # K-best data association (Munkres LAP)
│   ├── metrics.{h,cpp}
│   ├── particle_statistics.h   # weighted mean/covariance of a particle cloud
│   ├── munkres.{h,cpp}         # linear assignment (header-included)
│   ├── matrix.{h,cpp}          # matrix utilities (header-included)
│   ├── datatypes.h
│   ├── models.h
│   └── validation.h
├── python/
│   ├── lmb_engine_loader.py    # locates & imports compiled extension
│   ├── run_once.py             # single-run simulation + GOSPA plot
│   ├── run.py                  # Monte Carlo simulation (20 runs)
│   ├── 2026_ieee_aerospace.py  # paper config (K_BEST=100)
│   ├── run_multisensor.py      # multi-sensor pointed FOV demo
│   ├── run_ring.py             # N-sensor equatorial ring vs sampled debris (lazy propagation)
│   ├── evaluation_plots.py     # figures + CSV tables for a ring run
│   ├── data/                   # debris catalogue samples (ECI states)
│   └── lmb_engine/             # built extension output (.so / .pyd)
│       ├── Release/            # recommended built extension
│       ├── Debug/
│       └── Asan/               # sanitizer build (scripts/asan-test.sh)
├── tests/
│   ├── test_two_body_propagator_multistep.py
│   ├── assignments.py
│   ├── reference_geometry.py   # independent long-double/NumPy references used by the tests
│   ├── test_los_geometry.py
│   ├── test_validation_dimensions.py
│   ├── test_sensor_likelihood.py
│   ├── test_sensor_fov.py      # pointing and visibility geometry
│   ├── test_fov_detection_probability.py  # FOV-dependent P_D in the filter
│   ├── lmb_reference.py        # exhaustive-enumeration reference for one LMB update
│   ├── test_existence_enumeration.py      # update() against that reference
│   ├── test_clutter_scaling.py # kappa enters the detection cost exactly once
│   ├── test_lazy_propagation.py           # time-consistent noise, lazy == eager
│   ├── test_regularization.py  # kernel jitter keeps posterior moments, restores diversity
│   ├── test_fused_proposal.py  # re-acquisition: fused proposal vs ordinary update and exact answers
│   ├── test_adaptive_birth_model.py
│   ├── test_bindings_api.py
│   ├── test_end_to_end.py
│   ├── harness_scenario.py     # fully-seeded scenario runner + digest, shared by the harnesses
│   ├── statistics_helpers.py   # Welch t-test and two-sample KS, implemented on NumPy
│   ├── test_particle_statistics.py     # cloud statistics and zero-copy view aliasing
│   ├── test_invariants.py      # per-step structural invariants
│   ├── test_golden_invariance.py       # bitwise/rtol digest regression
│   ├── test_statistical_equivalence.py # distributional equivalence vs a committed baseline
│   ├── bench_engine.py         # hot-path timings and peak RSS
│   └── fixtures/               # committed golden digests, statistical baseline, bench baseline
└── external/                   # optional git submodules (not required for default build)
    ├── astro/                  # openastro propagation library
    ├── sgp4/                   # SGP4 propagator
    └── vcpkg/                  # vendored vcpkg (Windows CI only)
```

Build artifacts also land in `build/` (CMake binary dir) and `venv/` (local Python environment); both are gitignored.

## Troubleshooting

### `ImportError: Could not find a built lmb_engine extension`

Build the project first:

```bash
./scripts/build.sh                # Linux / macOS
# .\scripts\build.ps1             # Windows
```

Or configure manually with the venv Python (see [Quick start](#quick-start)).

### `Could NOT find Eigen3`

Install Eigen3 for your platform (see [Platform-specific native dependencies](#platform-specific-native-dependencies)).

### `Could NOT find pybind11`

Install pybind11 in the active virtual environment, then re-run CMake using the venv Python:

```bash
pip install -r requirements.txt
source ./scripts/cmake-venv-args.sh
cmake --preset release \
  -DPython_EXECUTABLE="$CMAKE_VENV_PYTHON" \
  -Dpybind11_DIR="$CMAKE_PYBIND11_DIR"
```

On CI or when `actions/setup-python` is used, CMake may otherwise pick the hosted Python instead of your venv unless `Python_EXECUTABLE` and `pybind11_DIR` are set explicitly (the helper scripts above do this automatically).

### Wrong Python version / import fails after rebuild

Activate the same `venv` used during `cmake`, then run Python. Delete `build/` and reconfigure if you switched Python versions.

### Windows: `.pyd` not found

Ensure you built **Release** (or **Debug** if using that preset). The loader checks both `python/lmb_engine/Release/` and `python/lmb_engine/Debug/`.

## Continuous Integration

CI runs on every push and pull request to `main` on **Linux**, **macOS**, and **Windows** (Python 3.12). Each job:

1. Installs native dependencies (Eigen via apt/brew/vcpkg)
2. Builds the Release extension with CMake presets
3. Runs `./scripts/ci-test.sh` (smoke import, propagator, assignment, geometry, validation, likelihood, birth, API and end-to-end tests)

**Do not add bitwise float / digest assertions that must pass on the macOS CI job.** libc++ draws a
different `normal_distribution` stream from the same seed, so absolute bit patterns from Linux
fixtures will fail there by design. Keep bitwise checks behind the reference-platform gate (x86-64
Linux); on macOS use `--portable`, `rtol`, or `test_statistical_equivalence.py`. See
[Bitwise reproduction is per-platform](#bitwise-reproduction-is-per-platform).

The full Monte Carlo simulation (`python/run.py`) is intentionally excluded from CI because it is too slow for routine checks.

Run the same validation locally after building:

```bash
./scripts/build.sh && ./scripts/ci-test.sh
```

## Optional git submodules

The repo includes optional submodules under `external/` (`astro`, `sgp4`, `vcpkg`). They are **not required** for the default SMC-LMB build. To fetch them:

```bash
git submodule update --init --recursive
```

## License

See [LICENSE](LICENSE).
