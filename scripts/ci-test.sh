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
python tests/test_los_geometry.py
python tests/test_validation_dimensions.py
python tests/test_sensor_likelihood.py
python tests/test_adaptive_birth_model.py
python tests/test_bindings_api.py
python tests/statistics_helpers.py
python tests/test_particle_statistics.py
python tests/test_invariants.py
python tests/test_golden_invariance.py
python tests/test_end_to_end.py

# test_statistical_equivalence.py is deliberately not run here. It costs ~20 s in Release and
# several minutes in Debug, and it is only the right gate for phases that change the RNG stream or
# floating-point summation order. Run it via ./scripts/gate.sh, or directly, for those phases.
