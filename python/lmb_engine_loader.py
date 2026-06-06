"""Locate and import the compiled lmb_engine pybind11 extension."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from pathlib import Path

_BUILD_DIRS = ("Release", "Debug")
_MODULE_NAME = "lmb_engine"


def _python_package_dir() -> Path:
    return Path(__file__).resolve().parent


def _module_search_roots() -> tuple[Path, ...]:
    package_dir = _python_package_dir()
    return (package_dir / "lmb_engine",)


def _has_extension_module(directory: Path) -> bool:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if (directory / f"{_MODULE_NAME}{suffix}").exists():
            return True

    # Windows: pybind11 may emit lmb_engine.pyd without a tagged suffix.
    if (directory / f"{_MODULE_NAME}.pyd").exists():
        return True

    return False


def find_extension_dir(*, prefer: str = "Release") -> Path:
    """Return the directory containing a built lmb_engine extension."""
    search_order = (prefer,) + tuple(build_dir for build_dir in _BUILD_DIRS if build_dir != prefer)

    for root in _module_search_roots():
        for build_dir in search_order:
            candidate = root / build_dir
            if candidate.is_dir() and _has_extension_module(candidate):
                return candidate

    roots = ", ".join(str(root) for root in _module_search_roots())
    raise ImportError(
        "Could not find a built lmb_engine extension.\n"
        f"Searched under: {roots}\n"
        "Build the project first (Release is recommended):\n"
        "  cmake --preset release\n"
        "  cmake --build --preset release\n"
        "Or run: ./scripts/build.sh  (Linux/macOS) / scripts\\build.ps1  (Windows)\n"
        "Use the same Python interpreter for building and running."
    )


def import_lmb_engine(*, prefer: str = "Release", verbose: bool = False):
    """Import lmb_engine after locating the compiled extension."""
    extension_dir = find_extension_dir(prefer=prefer)
    extension_path = str(extension_dir)
    if extension_path not in sys.path:
        sys.path.insert(0, extension_path)

    module = importlib.import_module(_MODULE_NAME)

    if verbose:
        print(f"Loaded lmb_engine from {module.__file__}")

    return module
