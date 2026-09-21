#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d venv ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Set LMB_ENGINE_BUILD=Debug|Release to pin the extension build the tests import.
python -c "import sys; sys.path.insert(0, 'python'); from lmb_engine_loader import import_lmb_engine; m = import_lmb_engine(); print('ok', m.__file__, 'validation' if m.VALIDATION_ENABLED else 'no-validation')"
python tests/test_two_body_propagator_multistep.py
python tests/assignments.py
# Pure arithmetic, sub-second. Runs early so a broken metric fails here rather than after
# the multi-minute golden and statistical stages.
python tests/test_gospa_metric.py
python tests/test_los_geometry.py
python tests/test_local_residual_cached_basis.py
# Sensor pointing / field-of-view geometry, then the FOV-scaled detection probability it
# feeds. Both are seconds-scale, and a broken visibility predicate should fail here rather
# than as a mystery in the golden stage.
python tests/test_sensor_fov.py
python tests/test_fov_detection_probability.py
python tests/test_validation_dimensions.py
python tests/test_sensor_likelihood.py
python tests/test_adaptive_birth_model.py
python tests/test_bindings_api.py
python tests/statistics_helpers.py
python tests/test_particle_statistics.py
python tests/test_invariants.py
python tests/test_mixture_grouping.py
# Selects its own mode: bitwise against the committed fixtures on the platform that wrote them
# (x86-64 Linux / libstdc++), portable everywhere else. See the PLATFORM GATING block in the test.
python tests/test_golden_invariance.py
python tests/test_run_once_api_surface.py
# ProcessPoolExecutor Monte Carlo: correctness + speedup. Too expensive in Debug (full 10k-particle
# runs); Release is the meaningful gate for the parallel path.
if [[ "${LMB_ENGINE_BUILD:-Release}" == "Debug" ]]; then
  echo "test_monte_carlo_parallel.py: skipped, it costs minutes in Debug; run it against Release"
else
  python tests/test_monte_carlo_parallel.py
fi
python tests/test_end_to_end.py

# test_statistical_equivalence.py replaces the bitwise gate off the reference platform, where the
# STL's random distributions draw a different stream from the same seed and no tolerance on the
# golden digest is meaningful. It compares mean-GOSPA distributions over 48 independent scenarios per
# arm, which is the right question for a stream difference. On the reference platform the bitwise
# gate already covers it, so it is skipped there; in Debug it costs minutes, so it belongs against
# Release.
REFERENCE_PLATFORM=0
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64 | Linux/amd64) REFERENCE_PLATFORM=1 ;;
esac

if [[ "$REFERENCE_PLATFORM" == "1" ]]; then
  echo "test_statistical_equivalence.py: skipped, the bitwise golden gate covers this platform"
elif [[ "${LMB_ENGINE_BUILD:-Release}" == "Debug" ]]; then
  echo "test_statistical_equivalence.py: skipped, it costs minutes in Debug; run it against Release"
else
  # alpha 1e-4 rather than the default 0.01: this runs on every PR, and at 0.01 each of the three
  # comparisons rejects a healthy platform 1% of the time. A port that is actually broken misses by
  # many sigma and still fails.
  python tests/test_statistical_equivalence.py --alpha 1e-4
fi
