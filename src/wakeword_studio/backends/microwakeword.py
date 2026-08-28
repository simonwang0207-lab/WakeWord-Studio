"""microWakeWord/MixedNet backend implementation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .base import BackendEvaluation, ExportArtifact, WakeWordBackend
from ..dataset.manifest import DatasetManifest
from ..frontends import load_inference_audio


class MicroWakeWordBackend(WakeWordBackend):
    def __init__(self, keyword: str = "你好，青小甲", upstream_root: Path | None = None):
        self.keyword = keyword
        self.upstream_root = upstream_root
        self.model_path: Path | None = None
        self._model = None
        self._audio = np.zeros(0, dtype=np.int16)
        self._feature_cursor = 0
        self._feature_remainder = np.empty((0, 40), dtype=np.float32)

    def train(self, manifest_path: Path, config_path: Path) -> Path:
        manifest = DatasetManifest.load(manifest_path)
        errors = manifest.validate(manifest_path)
        if errors:
            raise ValueError("Invalid DatasetManifest: " + "; ".join(errors))
        if manifest.wake_word != self.keyword:
            raise ValueError("Manifest wake word does not match backend keyword")
        env = os.environ.copy()
        if self.upstream_root:
            env["PYTHONPATH"] = str(self.upstream_root) + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable,
            "-m",
            "microwakeword.train",
            "--training_config",
            str(config_path),
            "--model_name",
            "mixednet",
            "--train",
            "1",
            "--test_tflite_streaming_quantized",
            "1",
        ]
        subprocess.run(command, check=True, env=env)
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return Path(config["train_dir"])

    def evaluate(self, manifest_path: Path) -> BackendEvaluation:
        if self._model is None:
            raise RuntimeError("Call load() first")
        manifest = DatasetManifest.load(manifest_path)
        root = Path(manifest.root)
        scores: list[float] = []
        positives: list[float] = []
        negatives: list[float] = []
        for row in manifest.records:
            audio = load_inference_audio(root / row.audio_path)
            self.reset_stream()
            row_scores = self._model.predict_clip(audio.astype(np.float32), step_ms=10)
            score = float(max(row_scores, default=0.0))
            scores.append(score)
            (positives if row.label == "positive" else negatives).append(score)
        metrics = {
            "positive_mean": float(np.mean(positives)) if positives else 0.0,
            "negative_mean": float(np.mean(negatives)) if negatives else 0.0,
            "positive_max": max(positives, default=0.0),
            "negative_max": max(negatives, default=0.0),
        }
        return BackendEvaluation(len(scores), scores, metrics)

    def export(self, destination: Path | None = None) -> ExportArtifact:
        if self.model_path is None:
            raise RuntimeError("Call load() first")
        path = self.model_path
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            path = destination
        return ExportArtifact(path=path, bytes=path.stat().st_size, full_int8=True)

    def load(self, model_path: Path) -> None:
        from microwakeword.inference import Model

        self.model_path = model_path.resolve()
        self._model = Model(str(self.model_path), stride=3)
        self.reset_stream()

    def reset_stream(self) -> None:
        self._audio = np.zeros(0, dtype=np.int16)
        self._feature_cursor = 0
        self._feature_remainder = np.empty((0, 40), dtype=np.float32)
        if self.model_path is not None:
            from microwakeword.inference import Model

            self._model = Model(str(self.model_path), stride=3)

    def stream_scores(self, pcm16: np.ndarray) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call load() first")
        from microwakeword.audio.audio_utils import generate_features_for_clip

        frame = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        self._audio = np.concatenate((self._audio, frame))
        # Recompute the frontend track and emit only newly stable 10 ms feature rows.
        features = generate_features_for_clip(self._audio.astype(np.float32) / 32768.0, step_ms=10)
        new_features = features[self._feature_cursor :]
        self._feature_cursor = len(features)
        pending = np.concatenate((self._feature_remainder, new_features), axis=0)
        usable = len(pending) - len(pending) % 3
        if usable == 0:
            self._feature_remainder = pending
            return {}
        predictions = self._model.predict_spectrogram(pending[:usable])
        self._feature_remainder = pending[usable:]
        return {self.keyword: float(predictions[-1])} if predictions else {}
