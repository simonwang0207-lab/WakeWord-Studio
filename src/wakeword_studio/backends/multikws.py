"""Dynamic N-class Full-INT8 Multi-KWS runtime backend."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import BackendEvaluation, ExportArtifact, WakeWordBackend
from ..dataset.manifest import DatasetManifest


@dataclass(frozen=True, slots=True)
class KeywordClass:
    class_id: int
    keyword_id: str
    display_name: str


@dataclass(slots=True)
class MultiKWSPrediction:
    predicted_class_id: int
    predicted_keyword_id: str
    predicted_display_name: str
    top1_score: float
    top2_class_id: int
    top2_keyword_id: str
    top2_display_name: str
    top2_score: float
    margin: float
    background_score: float
    accepted: bool
    rejection_reason: str
    per_class_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiKWSBackend(WakeWordBackend):
    """Rolling two-second backend for ``[1, 99, 40] -> [1, N]`` models."""

    def __init__(
        self,
        classes: Sequence[KeywordClass | dict[str, Any]],
        *,
        threshold: float,
        margin_threshold: float,
        hop_seconds: float = 0.20,
        sample_rate_hz: int = 16_000,
        window_seconds: float = 2.0,
    ):
        parsed = [item if isinstance(item, KeywordClass) else KeywordClass(**item) for item in classes]
        if len(parsed) < 2:
            raise ValueError("Multi-KWS vocabulary must contain background and at least one keyword")
        if [item.class_id for item in parsed] != list(range(len(parsed))):
            raise ValueError("Multi-KWS class IDs must be contiguous and start at zero")
        if parsed[0].keyword_id != "background":
            raise ValueError("Multi-KWS class 0 must be background")
        if not 0.20 <= float(hop_seconds) <= 1.0:
            raise ValueError("Multi-KWS hop_seconds must be between 0.20 and 1.0")
        if abs(float(window_seconds) - 2.0) > 1e-9:
            raise ValueError("Multi-KWS deployment window must remain exactly 2.0 seconds")
        self.classes = tuple(parsed)
        self.threshold = float(threshold)
        self.margin_threshold = float(margin_threshold)
        self.hop_seconds = float(hop_seconds)
        self.sample_rate_hz = int(sample_rate_hz)
        self.window_seconds = float(window_seconds)
        self.model_path: Path | None = None
        self.last_prediction: MultiKWSPrediction | None = None
        self.last_raw_score = 0.0
        self._frontend = None
        self._interpreter = None
        self._input = None
        self._output = None
        self._audio = np.zeros(0, dtype=np.int16)
        self._samples_since_inference = 0

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def train(self, manifest_path: Path, config_path: Path) -> Path:
        manifest = DatasetManifest.load(manifest_path)
        errors = manifest.validate(manifest_path)
        if errors:
            raise ValueError("Invalid DatasetManifest: " + "; ".join(errors))
        return config_path

    def evaluate(self, manifest_path: Path) -> BackendEvaluation:
        if self._interpreter is None:
            raise RuntimeError("Call load() first")
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
        return ExportArtifact(path, path.stat().st_size, True)

    def load(self, model_path: Path) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        import tensorflow as tf
        from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend

        self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self._interpreter.allocate_tensors()
        inputs = self._interpreter.get_input_details()
        outputs = self._interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Multi-KWS deployment must have exactly one input and output")
        self._input, self._output = inputs[0], outputs[0]
        if self._input["shape"].tolist() != [1, 99, 40]:
            raise RuntimeError(f"Unexpected Multi-KWS input shape: {self._input['shape'].tolist()}")
        if self._output["shape"].tolist() != [1, self.num_classes]:
            raise RuntimeError(
                f"Vocabulary/model mismatch: expected [1, {self.num_classes}], got {self._output['shape'].tolist()}"
            )
        if np.dtype(self._input["dtype"]) != np.dtype(np.int8):
            raise RuntimeError("Multi-KWS input must be Full INT8")
        if np.dtype(self._output["dtype"]) not in (np.dtype(np.int8), np.dtype(np.uint8)):
            raise RuntimeError("Multi-KWS output must be quantized int8/uint8")
        self._frontend = MicroFrontend(
            sample_rate=self.sample_rate_hz,
            window_size_ms=30,
            window_step_ms=20,
            num_channels=40,
        )
        self.reset_stream()

    def reset_stream(self) -> None:
        self._audio = np.zeros(0, dtype=np.int16)
        self._samples_since_inference = 0
        self.last_prediction = None
        self.last_raw_score = 0.0

    def prediction_from_scores(self, values: Sequence[float]) -> MultiKWSPrediction:
        scores = np.asarray(values, dtype=np.float32).reshape(-1)
        if scores.shape != (self.num_classes,):
            raise ValueError(f"Expected {self.num_classes} class scores, got {scores.shape}")
        order = np.argsort(-scores, kind="stable")
        top1, top2 = int(order[0]), int(order[1])
        first, second = self.classes[top1], self.classes[top2]
        top1_score, top2_score = float(scores[top1]), float(scores[top2])
        margin = top1_score - top2_score
        if top1 == 0:
            accepted, reason = False, "BACKGROUND_TOP1"
        elif top1_score < self.threshold:
            accepted, reason = False, "LOW_TOP1_SCORE"
        elif margin < self.margin_threshold:
            accepted, reason = False, "LOW_MARGIN"
        else:
            accepted, reason = True, "ACCEPTED"
        return MultiKWSPrediction(
            predicted_class_id=top1,
            predicted_keyword_id=first.keyword_id,
            predicted_display_name=first.display_name,
            top1_score=top1_score,
            top2_class_id=top2,
            top2_keyword_id=second.keyword_id,
            top2_display_name=second.display_name,
            top2_score=top2_score,
            margin=margin,
            background_score=float(scores[0]),
            accepted=accepted,
            rejection_reason=reason,
            per_class_scores={item.keyword_id: float(scores[item.class_id]) for item in self.classes},
        )

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
        self._samples_since_inference = 0
        feature = np.asarray(self._frontend(self._audio.astype(np.float32) / 32768.0)[0], dtype=np.float32)
        if feature.shape != (99, 40):
            raise RuntimeError(f"Multi-KWS frontend produced unexpected shape: {feature.shape}")
        input_scale, input_zero = self._input["quantization"]
        output_scale, output_zero = self._output["quantization"]
        if input_scale <= 0 or output_scale <= 0:
            raise RuntimeError("Multi-KWS INT8 model lacks scalar quantization metadata")
        quantized = np.clip(
            np.rint(feature / float(input_scale) + int(input_zero)), -128, 127
        ).astype(np.int8)[np.newaxis, ...]
        self._interpreter.set_tensor(self._input["index"], quantized)
        self._interpreter.invoke()
        raw = np.asarray(self._interpreter.get_tensor(self._output["index"])).reshape(-1)
        values = float(output_scale) * (raw.astype(np.float32) - int(output_zero))
        self.last_prediction = self.prediction_from_scores(values)
        self.last_raw_score = self.last_prediction.top1_score
        return {item.keyword_id: float(values[item.class_id]) for item in self.classes[1:]}

    def score_state(self) -> dict[str, Any]:
        return self.last_prediction.to_dict() if self.last_prediction else {}
