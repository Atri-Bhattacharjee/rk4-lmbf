$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path "venv") {
    & ".\venv\Scripts\Activate.ps1"
}

python -c "import sys; sys.path.insert(0, 'python'); from lmb_engine_loader import import_lmb_engine; import_lmb_engine(); print('ok')"
python tests/test_two_body_propagator_multistep.py
python tests/assignments.py
python tests/test_gospa_metric.py
python tests/test_los_geometry.py
python tests/test_local_residual_cached_basis.py
python tests/test_validation_dimensions.py
python tests/test_sensor_likelihood.py
python tests/test_adaptive_birth_model.py
python tests/test_bindings_api.py
python tests/statistics_helpers.py
python tests/test_particle_statistics.py
python tests/test_invariants.py
# Selects its own mode: bitwise against the committed fixtures on the platform that wrote them
# (x86-64 Linux / libstdc++), portable here. See the PLATFORM GATING block in the test.
python tests/test_golden_invariance.py
python tests/test_run_once_api_surface.py
# ProcessPoolExecutor Monte Carlo: correctness + speedup. Too expensive in Debug.
if ($env:LMB_ENGINE_BUILD -eq "Debug") {
    Write-Host "test_monte_carlo_parallel.py: skipped, it costs minutes in Debug; run it against Release"
} else {
    python tests/test_monte_carlo_parallel.py
}
python tests/test_end_to_end.py

# test_statistical_equivalence.py replaces the bitwise gate here, since Windows is never the
# reference platform: the STL's random distributions may draw a different stream from the same seed
# and no tolerance on the golden digest is meaningful. It compares mean-GOSPA distributions over 48
# independent scenarios per arm. It costs minutes in Debug, so it is skipped there and belongs
# against Release.
if ($env:LMB_ENGINE_BUILD -eq "Debug") {
    Write-Host "test_statistical_equivalence.py: skipped, it costs minutes in Debug; run it against Release"
} else {
    # alpha 1e-4 rather than the default 0.01: this runs on every PR, and at 0.01 each of the three
    # comparisons rejects a healthy platform 1% of the time. A port that is actually broken misses by
    # many sigma and still fails.
    python tests/test_statistical_equivalence.py --alpha 1e-4
}
