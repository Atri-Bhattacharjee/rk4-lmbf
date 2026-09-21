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
