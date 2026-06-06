$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path "venv") {
    & ".\venv\Scripts\Activate.ps1"
}

python -c "import sys; sys.path.insert(0, 'python'); from lmb_engine_loader import import_lmb_engine; import_lmb_engine(); print('ok')"
python tests/test_two_body_propagator_multistep.py
python tests/assignments.py
