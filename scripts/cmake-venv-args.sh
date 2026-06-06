#!/usr/bin/env bash
# Print CMake cache arguments for the project venv Python and pip-installed pybind11.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: $PYTHON not found; create and populate venv before configuring CMake" >&2
  exit 1
fi

PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
printf '%s\n' "-DPython_EXECUTABLE=$PYTHON" "-Dpybind11_DIR=$PYBIND11_DIR"
