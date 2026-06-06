$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "venv")) {
    python -m venv venv
}

& ".\venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
pip install -r requirements.txt

cmake --preset release
cmake --build --preset release

Write-Host ""
Write-Host "Build complete. Run the simulation with:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python python\run.py"
