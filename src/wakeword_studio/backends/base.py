"""Model-neutral backend contract used by training orchestration and runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class BackendEvaluation:
    sample_count: int
    scores: list[float]
    metrics: dict[str, float]


@dataclass(slots=True)
class ExportArtifact:
    path: Path
    bytes: int
    full_int8: bool
    operators: tuple[str, ...] = ()


class WakeWordBackend(ABC):
    """Stable boundary: no UI or runtime code may depend on model internals."""

    @abstractmethod
    def train(self, manifest_path: Path, config_path: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, manifest_path: Path) -> BackendEvaluation:
        raise NotImplementedError

    @abstractmethod
    def export(self, destination: Path | None = None) -> ExportArtifact:
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def stream_scores(self, pcm16: np.ndarray) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def reset_stream(self) -> None:
        raise NotImplementedError

