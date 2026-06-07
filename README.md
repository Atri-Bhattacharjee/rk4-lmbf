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

Optional CMake flag:

| Variable | Default | Description |
|----------|---------|-------------|
| `LMB_ENGINE_ENABLE_VALIDATION` | `OFF` | Keep hot-path validation checks in Release builds |

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

Writes `python/figure_ospa_single_run.png`.

### Monte Carlo simulation

```bash
source venv/bin/activate
python python/run.py
```

Runs 20 Monte Carlo trials and writes PNG figures under `python/`:

- `figure_1_individual_runs.png`
- `figure_2_average_performance.png`
- `figure_3_component_error.png`

### Paper reproduction

`python/2026_ieee_aerospace.py` is the IEEE Aerospace 2026 configuration (same harness as `run.py` with `K_BEST=100`). It produces the same three figure outputs as the Monte Carlo script.

Optional verbose import logging (works with any simulation script):

```bash
LMB_ENGINE_VERBOSE=1 python python/run_once.py
```

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
```

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
│   ├── build.sh                # Linux/macOS: venv + configure + build
│   ├── build.ps1               # Windows build helper
│   ├── cmake-venv-args.sh      # resolve venv Python & pybind11 for CMake
│   ├── cmake-venv-args.ps1     # Windows variant
│   ├── ci-test.sh              # Linux/macOS CI validation suite
│   └── ci-test.ps1             # Windows CI validation suite
├── src/                        # C++ SMC-LMB filter engine
│   ├── main.cpp                # pybind11 module bindings
│   ├── smc_lmb_tracker.{h,cpp} # core LMB filter
│   ├── adaptive_birth_model.{h,cpp}
│   ├── in_orbit_sensor_model.{h,cpp}
│   ├── two_body_propagator.{h,cpp}
│   ├── assignment.{h,cpp}      # K-best data association (Munkres LAP)
│   ├── metrics.{h,cpp}
│   ├── munkres.{h,cpp}         # linear assignment (header-included)
│   ├── matrix.{h,cpp}          # matrix utilities (header-included)
│   ├── datatypes.h
│   ├── models.h
│   └── validation.h
├── python/
│   ├── lmb_engine_loader.py    # locates & imports compiled extension
│   ├── run_once.py             # single-run simulation + OSPA plot
│   ├── run.py                  # Monte Carlo simulation (20 runs)
│   ├── 2026_ieee_aerospace.py  # paper config (K_BEST=100)
│   └── lmb_engine/             # built extension output (.so / .pyd)
│       ├── Release/            # recommended built extension
│       └── Debug/
├── tests/
│   ├── test_two_body_propagator_multistep.py
│   └── assignments.py
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
3. Runs `./scripts/ci-test.sh` (smoke import, propagator test, assignment tests)

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
