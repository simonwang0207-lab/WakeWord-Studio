"""Phase 10 dependency declaration and import-startup smoke tests."""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _declared_project_dependencies() -> set[str]:
    document = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = document["project"]["dependencies"]
    return {
        dependency.split(";", 1)[0]
        .split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        for dependency in dependencies
    }


def test_phase10_direct_dependencies_are_declared() -> None:
    declared = _declared_project_dependencies()
    assert {"numpy", "pypinyin", "pyyaml"} <= declared


def test_webapp_import_startup_smoke() -> None:
    sys.modules.pop("wakeword_studio.webapp", None)
    module = importlib.import_module("wakeword_studio.webapp")
    assert callable(module.serve)


if __name__ == "__main__":
    test_phase10_direct_dependencies_are_declared()
    test_webapp_import_startup_smoke()
    print("PHASE10_DEPENDENCY_STARTUP_SMOKE=PASS")
