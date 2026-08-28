"""LiveKit RepCNN backend; shares the same contract as microWakeWord."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .base import BackendEvaluation, ExportArtifact, WakeWordBackend
from ..dataset.manifest import DatasetManifest


class RepCNNBackend(WakeWordBackend):
    def __init__(self, keyword: str = "你好，青小甲"):
        self.keyword = keyword
        self.model_path: Path | None = None
        self._stream = None

    def train(self, manifest_path: Path, config_path: Path) -> Path:
        manifest = DatasetManifest.load(manifest_path)
        errors = manifest.validate(manifest_path)
        if errors:
            raise ValueError("Invalid DatasetManifest: " + "; ".join(errors))
        executable = shutil.which("livekit-wakeword")
        if not executable:
            raise RuntimeError("livekit-wakeword executable is not on PATH")
        subprocess.run([executable, "train", str(config_path)], check=True, env=os.environ.copy())
        return config_path

    def evaluate(self, manifest_path: Path) -> BackendEvaluation:
        if self._stream is None:
            raise RuntimeError("Call load() first")
        # Runtime evaluation is intentionally backend-neutral and delegated to the common harness.
        manifest = DatasetManifest.load(manifest_path)
        return BackendEvaluation(len(manifest.records), [], {"evaluation_delegated": 1.0})

    def export(self, destination: Path | None = None) -> ExportArtifact:
        if self.model_path is None:
            raise RuntimeError("Call load() first")
        path = self.model_path
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            path = destination
        return ExportArtifact(path, path.stat().st_size, "full_int8" in path.name)

    def load(self, model_path: Path) -> None:
        from livekit.embedded_wakeword.inference.model import StreamingWakeWordModel

        self.model_path = model_path.resolve()
        self._stream = StreamingWakeWordModel(self.model_path)

    def reset_stream(self) -> None:
        if self.model_path is not None:
            self.load(self.model_path)

    def stream_scores(self, pcm16: np.ndarray) -> dict[str, float]:
        if self._stream is None:
            raise RuntimeError("Call load() first")
        audio = np.asarray(pcm16, dtype=np.float32) / 32768.0
        score = self._stream.predict_streaming(audio)
        return {} if score is None else {self.keyword: float(score)}

