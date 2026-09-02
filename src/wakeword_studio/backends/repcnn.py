"""Full-INT8 RepCNN backend with a configurable two-second rolling window."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .base import BackendEvaluation, ExportArtifact, WakeWordBackend
from ..dataset.manifest import DatasetManifest
from ..frontends import PymicroFrontend
from ..runtime.score_smoothing import RollingScoreSmoother
from ..tflite_runtime import create_tflite_interpreter


class RepCNNBackend(WakeWordBackend):
    def __init__(
        self,
        keyword: str = "你好，青小甲",
        *,
        hop_seconds: float = 0.20,
        sample_rate_hz: int = 16_000,
        window_seconds: float = 2.0,
        smoothing_mode: str = "raw",
        smoothing_window: int = 3,
        hybrid_max_weight: float = 0.5,
    ):
        if not 0.20 <= hop_seconds <= 1.0:
            raise ValueError("RepCNN hop_seconds must be between 0.20 and 1.0")
        self.keyword = keyword
        self.hop_seconds = float(hop_seconds)
        self.sample_rate_hz = int(sample_rate_hz)
        self.window_seconds = float(window_seconds)
        if abs(self.window_seconds - 2.0) > 1e-9:
            raise ValueError("RepCNN deployment window must remain exactly 2.0 seconds")
        self.smoother = RollingScoreSmoother(
            smoothing_mode,
            window_size=smoothing_window,
            hybrid_max_weight=hybrid_max_weight,
        )
        self.last_raw_score = 0.0
        self.last_smoothed_score = 0.0
        self.model_path: Path | None = None
        self._frontend = None
        self._interpreter = None
        self._input = None
        self._output = None
        self._audio = np.zeros(0, dtype=np.int16)
        self._samples_since_inference = 0

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
        if self._interpreter is None:
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
        self.model_path = model_path.resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self._interpreter = create_tflite_interpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()
        inputs = self._interpreter.get_input_details()
        outputs = self._interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("RepCNN deployment must have exactly one input and output")
        self._input = inputs[0]
        self._output = outputs[0]
        if self._input["shape"].tolist() != [1, 99, 40]:
            raise RuntimeError(f"Unexpected RepCNN input shape: {self._input['shape'].tolist()}")
        if self._output["shape"].tolist() != [1, 1]:
            raise RuntimeError(f"Unexpected RepCNN output shape: {self._output['shape'].tolist()}")
        if np.dtype(self._input["dtype"]) != np.dtype(np.int8):
            raise RuntimeError("Live RepCNN model input must be full INT8")
        if np.dtype(self._output["dtype"]) != np.dtype(np.int8):
            raise RuntimeError("Live RepCNN model output must be full INT8")
        self._frontend = PymicroFrontend(
            sample_rate=self.sample_rate_hz,
            window_size_ms=30,
            window_step_ms=20,
            num_channels=40,
        )
        self.reset_stream()

    def reset_stream(self) -> None:
        self._audio = np.zeros(0, dtype=np.int16)
        self._samples_since_inference = 0
        self.smoother.reset()
        self.last_raw_score = 0.0
        self.last_smoothed_score = 0.0

    def score_state(self) -> dict[str, object]:
        """Observable state for UI/log integrations without GUI coupling."""

        return {
            "raw_score": self.last_raw_score,
            "decision_score": self.last_smoothed_score,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "smoothing": self.smoother.state(),
        }

    def stream_scores(self, pcm16: np.ndarray) -> dict[str, float]:
        if self._interpreter is None or self._frontend is None:
            raise RuntimeError("Call load() first")
        frame = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        if not len(frame):
            return {}
        self._audio = np.concatenate((self._audio, frame))
        self._samples_since_inference += len(frame)
        window_samples = int(round(self.sample_rate_hz * self.window_seconds))
        if len(self._audio) > window_samples:
            self._audio = self._audio[-window_samples:]
        hop_samples = int(round(self.sample_rate_hz * self.hop_seconds))
        if len(self._audio) < window_samples or self._samples_since_inference < hop_samples:
            return {}
        # A pre-roll replay can be much larger than one hop, but it represents a
        # single current two-second decision window. Start the next hop afresh.
        self._samples_since_inference = 0

        feature = np.asarray(
            self._frontend(self._audio.astype(np.float32) / 32768.0)[0], dtype=np.float32
        )
        if feature.shape != (99, 40):
            raise RuntimeError(f"RepCNN frontend produced unexpected shape: {feature.shape}")
        input_scale, input_zero_point = self._input["quantization"]
        output_scale, output_zero_point = self._output["quantization"]
        if input_scale <= 0 or output_scale <= 0:
            raise RuntimeError("RepCNN INT8 model lacks scalar quantization metadata")
        quantized = np.clip(
            np.rint(feature / float(input_scale) + int(input_zero_point)), -128, 127
        ).astype(np.int8)[np.newaxis, ...]
        self._interpreter.set_tensor(self._input["index"], quantized)
        self._interpreter.invoke()
        raw = int(np.asarray(self._interpreter.get_tensor(self._output["index"])).reshape(-1)[0])
        score = float(output_scale) * (raw - int(output_zero_point))
        self.last_raw_score = score
        self.last_smoothed_score = self.smoother.update(score)
        return {self.keyword: self.last_smoothed_score}
