$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "venv Python not found at $Python; create and populate venv before configuring CMake"
}

$Pybind11Dir = & $Python -m pybind11 --cmakedir
"-DPython_EXECUTABLE=$Python"
"-Dpybind11_DIR=$Pybind11Dir"
