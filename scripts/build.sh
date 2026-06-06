#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

mapfile -t CMAKE_VENV_ARGS < <("$ROOT/scripts/cmake-venv-args.sh")
cmake --preset release "${CMAKE_VENV_ARGS[@]}"
cmake --build --preset release

echo ""
echo "Build complete. Run the simulation with:"
echo "  source venv/bin/activate"
echo "  python python/run.py"
