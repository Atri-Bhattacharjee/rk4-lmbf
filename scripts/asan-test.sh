#!/usr/bin/env bash
# Build the extension with AddressSanitizer + UndefinedBehaviorSanitizer and run the suite under it.
#
# Only the extension is instrumented; CPython is not. The ASan runtime therefore has to be the first
# thing loaded into the process, which is what the LD_PRELOAD below achieves. Without it the loader
# rejects the module with "ASan runtime does not come first in initial library list".
#
# detect_leaks is off on purpose: CPython and NumPy retain allocations for the process lifetime by
# design, so leak checking here reports hundreds of interpreter internals and nothing about this
# project. The phases this configuration exists for (2 and 4, which change memory ownership) are
# after use-after-free, buffer overflow and use-after-move, all of which ASan still catches.
#
# Usage: ./scripts/asan-test.sh [test file ...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d venv ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/cmake-venv-args.sh"

echo "== configuring and building the instrumented extension =="
cmake --preset asan \
  -DPython_EXECUTABLE="$CMAKE_VENV_PYTHON" \
  -Dpybind11_DIR="$CMAKE_PYBIND11_DIR" >/dev/null
cmake --build --preset asan --parallel "$(nproc)"

COMPILER="${CXX:-c++}"
ASAN_RUNTIME="$("$COMPILER" -print-file-name=libasan.so)"
if [[ ! -f "$ASAN_RUNTIME" ]]; then
  echo "error: could not locate libasan.so via '$COMPILER -print-file-name=libasan.so'" >&2
  echo "       got: $ASAN_RUNTIME" >&2
  exit 1
fi

# libstdc++ has to be preloaded too. ASan intercepts __cxa_throw, and when libstdc++ is only pulled
# in later as a dependency of the extension, the interceptor cannot resolve the real symbol and
# aborts on the first C++ exception -- which the validation tests raise deliberately.
STDCXX_RUNTIME="$("$COMPILER" -print-file-name=libstdc++.so.6)"
if [[ ! -f "$STDCXX_RUNTIME" ]]; then
  # Not every toolchain reports an absolute path here; the bare soname is fine for the loader.
  STDCXX_RUNTIME="libstdc++.so.6"
fi

export LD_PRELOAD="$ASAN_RUNTIME:$STDCXX_RUNTIME"
export LMB_ENGINE_BUILD=Asan
export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:strict_string_checks=1:detect_stack_use_after_return=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"

TESTS=("$@")
if [[ ${#TESTS[@]} -eq 0 ]]; then
  TESTS=(
    tests/test_two_body_propagator_multistep.py
    tests/assignments.py
    tests/test_los_geometry.py
    tests/test_validation_dimensions.py
    tests/test_sensor_likelihood.py
    tests/test_adaptive_birth_model.py
    tests/test_bindings_api.py
    tests/test_invariants.py
    tests/test_golden_invariance.py
  )
fi

echo ""
echo "== running the suite under ASan + UBSan =="
echo "   preload: $LD_PRELOAD"
for test_file in "${TESTS[@]}"; do
  echo "-- $test_file"
  python "$test_file"
done

echo ""
echo "ASan + UBSan clean."
