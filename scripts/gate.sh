#!/usr/bin/env bash
# Full pre-commit gate for one optimization phase.
#
# Runs, in order:
#   1. a Release build and the full suite against it
#   2. a Debug build and the full suite against it   (the plan requires both)
#   3. the statistical-equivalence harness
#   4. the benchmark, diffed against tests/fixtures/bench_phase0.json
#
# The sanitizer pass is separate because it needs a third build and its own LD_PRELOAD; run
# ./scripts/asan-test.sh for the phases that call for it (2 and 4 in particular).
#
# Usage:
#   ./scripts/gate.sh                 # bitwise golden comparison (the default for most phases)
#   ./scripts/gate.sh --rtol 1e-9     # tolerant golden comparison (Phase 2 only)
#
# The gate assumes the reference platform (x86-64 Linux / libstdc++), which is where the golden
# fixtures are bitwise reproducible. Run with no arguments elsewhere and the golden step drops to its
# portable mode on its own; step 3 is then the gate that is doing the numerical work.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GOLDEN_ARGS=()
if [[ $# -gt 0 ]]; then
  GOLDEN_ARGS=("$@")
fi

if [[ -d venv ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/cmake-venv-args.sh"

for preset in release debug; do
  echo "==============================================================="
  echo "== building and testing: $preset"
  echo "==============================================================="
  cmake --preset "$preset" \
    -DPython_EXECUTABLE="$CMAKE_VENV_PYTHON" \
    -Dpybind11_DIR="$CMAKE_PYBIND11_DIR" >/dev/null
  cmake --build --preset "$preset" --parallel "$(nproc)" >/dev/null

  config="Release"
  [[ "$preset" == "debug" ]] && config="Debug"
  LMB_ENGINE_BUILD="$config" ./scripts/ci-test.sh
done

echo "==============================================================="
echo "== golden invariance (explicit mode)"
echo "==============================================================="
LMB_ENGINE_BUILD=Release python tests/test_golden_invariance.py "${GOLDEN_ARGS[@]}"

echo "==============================================================="
echo "== statistical equivalence"
echo "==============================================================="
LMB_ENGINE_BUILD=Release python tests/test_statistical_equivalence.py

echo "==============================================================="
echo "== benchmark vs the Phase 0 baseline"
echo "==============================================================="
LMB_ENGINE_BUILD=Release python tests/bench_engine.py --compare tests/fixtures/bench_phase0.json

echo ""
echo "Gate complete. Remember ./scripts/asan-test.sh for phases that change memory ownership."
