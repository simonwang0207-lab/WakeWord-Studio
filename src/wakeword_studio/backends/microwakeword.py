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


def _generate_features_for_clip(audio_samples: np.ndarray) -> np.ndarray:
    """Run the same 16 kHz/10 ms microfrontend used by formal Model A evaluation."""

    from pymicro_features import MicroFrontend

    samples = np.asarray(audio_samples).reshape(-1)
    if samples.dtype in (np.float32, np.float64):
        samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    elif samples.dtype != np.int16:
        samples = samples.astype(np.int16)

    audio_bytes = samples.tobytes()
    frontend = MicroFrontend()
    process_samples = getattr(
        frontend,
        "ProcessSamples",
        getattr(frontend, "process_samples", None),
    )
    if process_samples is None:
        raise RuntimeError("Unsupported pymicro-features MicroFrontend API")

    features: list[object] = []
    audio_index = 0
    packet_bytes = 160 * 2
    # Keep the upstream microWakeWord boundary rule byte-for-byte compatible.
    while audio_index + packet_bytes < len(audio_bytes):
        result = process_samples(audio_bytes[audio_index : audio_index + packet_bytes])
        audio_index += int(result.samples_read) * 2
        if result.features:
            features.append(result.features)
    return np.asarray(features, dtype=np.float32).reshape(-1, 40)


class _TFLiteStreamingModel:
    """Minimal deployment adapter matching the frozen evaluator's TFLite path."""

    def __init__(self, model_path: Path, *, stride: int = 3):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Model A deployment must have exactly one input and output")
        self.input = inputs[0]
        self.output = outputs[0]
        self.input_feature_slices = int(self.input["shape"][1])
        self.stride = int(stride)
        if self.input["shape"].tolist() != [1, 3, 40]:
            raise RuntimeError(f"Unexpected Model A input shape: {self.input['shape'].tolist()}")
        if self.output["shape"].tolist() != [1, 1]:
            raise RuntimeError(f"Unexpected Model A output shape: {self.output['shape'].tolist()}")
        if np.dtype(self.input["dtype"]) != np.dtype(np.int8):
            raise RuntimeError("Model A deployment input must be INT8")
        if np.dtype(self.output["dtype"]) != np.dtype(np.uint8):
            raise RuntimeError("Model A deployment output must be UINT8")
        self.input_scale, self.input_zero_point = self.input["quantization"]
        self.output_scale, self.output_zero_point = self.output["quantization"]
        if self.input_scale <= 0 or self.output_scale <= 0:
            raise RuntimeError("Model A deployment lacks scalar quantization metadata")

    def predict_clip(self, audio: np.ndarray, step_ms: int = 10) -> list[float]:
        if int(step_ms) != 10:
            raise ValueError("Frozen Model A frontend requires a 10 ms feature step")
        return self.predict_spectrogram(_generate_features_for_clip(audio))

    def predict_spectrogram(self, spectrogram: np.ndarray) -> list[float]:
        features = np.asarray(spectrogram)
        if np.issubdtype(features.dtype, np.uint16):
            features = features.astype(np.float32) * 0.0390625
        else:
            features = features.astype(np.float32, copy=False)

        limits = np.iinfo(self.input["dtype"])
        predictions: list[float] = []
        for last_index in range(self.input_feature_slices, len(features) + 1, self.stride):
            chunk = features[last_index - self.input_feature_slices : last_index]
            quantized = np.rint(chunk / self.input_scale + self.input_zero_point)
            quantized = np.clip(quantized, limits.min, limits.max).astype(self.input["dtype"])
            self.interpreter.set_tensor(
                self.input["index"], quantized.reshape(self.input["shape"])
            )
            self.interpreter.invoke()
            raw = int(np.asarray(self.interpreter.get_tensor(self.output["index"])).reshape(-1)[0])
            predictions.append(float(self.output_scale) * (raw - int(self.output_zero_point)))
        return predictions


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
        self.model_path = model_path.resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.reset_stream()

    def reset_stream(self) -> None:
        self._audio = np.zeros(0, dtype=np.int16)
        self._feature_cursor = 0
        self._feature_remainder = np.empty((0, 40), dtype=np.float32)
        if self.model_path is not None:
            self._model = _TFLiteStreamingModel(self.model_path, stride=3)

    def stream_scores(self, pcm16: np.ndarray) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call load() first")
        frame = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        self._audio = np.concatenate((self._audio, frame))
        # Recompute the frontend track and emit only newly stable 10 ms feature rows.
        features = _generate_features_for_clip(self._audio)
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
