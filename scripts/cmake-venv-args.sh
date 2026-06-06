#!/usr/bin/env bash
# Resolve CMake cache variables for the project venv Python and pip-installed pybind11.
# Source this script:  source ./scripts/cmake-venv-args.sh
# Or print args:       ./scripts/cmake-venv-args.sh
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE_VENV_PYTHON="$_ROOT/venv/bin/python"

if [[ ! -x "$CMAKE_VENV_PYTHON" ]]; then
  echo "error: $CMAKE_VENV_PYTHON not found; create and populate venv before configuring CMake" >&2
  exit 1
fi

CMAKE_PYBIND11_DIR="$("$CMAKE_VENV_PYTHON" -m pybind11 --cmakedir)"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf '%s\n' "-DPython_EXECUTABLE=$CMAKE_VENV_PYTHON" "-Dpybind11_DIR=$CMAKE_PYBIND11_DIR"
fi
