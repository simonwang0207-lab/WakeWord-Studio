"""WakeWord Studio core interfaces, imported lazily across isolated environments."""

from __future__ import annotations

from typing import Any

__all__ = ["DatasetManifest", "DetectionLogic", "WakeWordBackend"]


def __getattr__(name: str) -> Any:
    if name == "DatasetManifest":
        from .dataset.manifest import DatasetManifest

        return DatasetManifest
    if name == "DetectionLogic":
        from .runtime.detection_logic import DetectionLogic

        return DetectionLogic
    if name == "WakeWordBackend":
        from .backends.base import WakeWordBackend

        return WakeWordBackend
    raise AttributeError(name)
