"""Product-facing model registry and portable project-path resolution."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.base import WakeWordBackend


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    id: str
    config_key: str
    display_name: str
    backend_id: str
    model_path: Path
    threshold: float
    model_size_kib: float
    runtime_mode: str
    window_seconds: float | None
    hop_seconds: float | None
    smoothing: str
    supported_platforms: tuple[str, ...]
    description: str
    trainer: dict[str, str] = field(default_factory=dict)
    deployment: dict[str, Any] = field(default_factory=dict)
    architecture: str = "unknown"
    task_type: str = "binary"
    version: str = "unknown"
    full_int8: bool = False
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    num_classes: int = 1
    vocabulary_id: str | None = None
    margin_threshold: float = 0.0
    sha256: str = ""
    validation_summary: dict[str, Any] = field(default_factory=dict)
    test_summary: dict[str, Any] = field(default_factory=dict)
    hardware_runtime_verified: bool = False
    status: str = "HISTORICAL"
    created_at: str = ""
    classes: tuple[dict[str, Any], ...] = ()

    def create_backend(self, keyword: str) -> WakeWordBackend:
        if self.backend_id == "microwakeword":
            from .backends.microwakeword import MicroWakeWordBackend

            return MicroWakeWordBackend(keyword=keyword)
        if self.backend_id == "repcnn":
            from .backends.repcnn import RepCNNBackend

            return RepCNNBackend(
                keyword=keyword,
                window_seconds=float(self.window_seconds or 2.0),
                hop_seconds=float(self.hop_seconds or 0.20),
                smoothing_mode=self.smoothing,
            )
        if self.backend_id == "multikws":
            from .backends.multikws import MultiKWSBackend

            return MultiKWSBackend(
                self.classes,
                threshold=self.threshold,
                margin_threshold=self.margin_threshold,
                window_seconds=float(self.window_seconds or 2.0),
                hop_seconds=float(self.hop_seconds or 0.20),
            )
        raise ValueError(f"未注册的模型后端：{self.backend_id}")

    def training_python(self, project_root: Path) -> Path:
        configured = self.trainer.get("python")
        if configured:
            candidate = resolve_project_path(project_root, configured)
            if candidate.is_file():
                return candidate
        return Path(sys.executable).resolve()

    def verify_artifact(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        expected_bytes = int(self.deployment.get("bytes", 0) or 0)
        if expected_bytes and self.model_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Model byte-size mismatch: {self.model_path}")
        if self.sha256:
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            if digest != self.sha256:
                raise RuntimeError(f"Model SHA256 mismatch: {self.model_path}")


class ModelRegistry:
    def __init__(self, registrations: list[ModelRegistration]):
        if not registrations:
            raise ValueError("模型注册表不能为空")
        self._by_id = {item.id: item for item in registrations}
        self._by_display = {item.display_name: item for item in registrations}
        if len(self._by_id) != len(registrations) or len(self._by_display) != len(registrations):
            raise ValueError("模型 id 和展示名称必须唯一")

    @classmethod
    def from_config(cls, project_root: Path, config: dict[str, Any]) -> "ModelRegistry":
        registrations: list[ModelRegistration] = []
        for index, (config_key, raw) in enumerate(config.get("models", {}).items(), start=1):
            deployment = dict(raw.get("deployment", {}))
            registrations.append(
                ModelRegistration(
                    id=str(raw.get("id") or f"model_{index}"),
                    config_key=str(config_key),
                    display_name=str(raw.get("display_name") or config_key),
                    backend_id=str(raw["backend"]),
                    model_path=resolve_project_path(project_root, raw["path"]),
                    threshold=float(raw["threshold"]),
                    model_size_kib=float(deployment.get("kib", 0.0)),
                    runtime_mode=str(raw.get("runtime_mode", "native_streaming")),
                    window_seconds=(
                        float(raw["window_seconds"]) if raw.get("window_seconds") is not None else None
                    ),
                    hop_seconds=(
                        float(raw["hop_seconds"]) if raw.get("hop_seconds") is not None else None
                    ),
                    smoothing=str(raw.get("smoothing", "raw")),
                    supported_platforms=tuple(str(x) for x in raw.get("supported_platforms", ())),
                    description=str(raw.get("description", "")),
                    trainer={str(k): str(v) for k, v in raw.get("training", {}).items()},
                    deployment=deployment,
                    architecture=str(raw.get("architecture", raw.get("backend", "unknown"))),
                    task_type=str(raw.get("task_type", "binary")),
                    version=str(raw.get("version", "unknown")),
                    full_int8=bool(raw.get("full_int8", deployment.get("format") == "Full INT8")),
                    input_shape=tuple(int(x) for x in raw.get("input_shape", deployment.get("input_shape", ()))),
                    output_shape=tuple(int(x) for x in raw.get("output_shape", deployment.get("output_shape", ()))),
                    num_classes=int(raw.get("num_classes", 1)),
                    vocabulary_id=(str(raw["vocabulary_id"]) if raw.get("vocabulary_id") else None),
                    margin_threshold=float(raw.get("margin_threshold", 0.0)),
                    sha256=str(raw.get("sha256", deployment.get("sha256", ""))),
                    validation_summary=dict(raw.get("validation_summary", {})),
                    test_summary=dict(raw.get("test_summary", {})),
                    hardware_runtime_verified=bool(raw.get("hardware_runtime_verified", False)),
                    status=str(raw.get("status", "HISTORICAL")),
                    created_at=str(raw.get("created_at", "")),
                    classes=tuple(dict(x) for x in raw.get("classes", ())),
                )
            )
        return cls(registrations)

    def all(self) -> tuple[ModelRegistration, ...]:
        return tuple(self._by_id.values())

    def by_id(self, model_id: str) -> ModelRegistration:
        return self._by_id[model_id]

    def by_display_name(self, display_name: str) -> ModelRegistration:
        return self._by_display[display_name]

    def resolve(self, value: str) -> ModelRegistration:
        if value in self._by_id:
            return self._by_id[value]
        if value in self._by_display:
            return self._by_display[value]
        for item in self._by_id.values():
            if item.config_key == value:
                return item
        raise KeyError(value)

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(self._by_display)


class ActiveModelStore:
    """Explicit activation/rollback state, isolated from all training jobs."""

    def __init__(self, path: Path, registry: ModelRegistry, default_model_id: str):
        self.path = Path(path)
        self.registry = registry
        self.default_model_id = default_model_id
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        state = {"schema": "wakeword-studio.active-model/v1", "active_model_id": self.default_model_id, "history": []}
        if self.path.is_file():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("active_model_id") in {item.id for item in self.registry.all()}:
                state.update(loaded)
        self.registry.by_id(str(state["active_model_id"]))
        return state

    @property
    def active_model_id(self) -> str:
        return str(self._state["active_model_id"])

    @property
    def rollback_target_model_id(self) -> str:
        history = list(self._state.get("history", []))
        if not history:
            raise RuntimeError("No previous active model is available for rollback")
        target = str(history[-1])
        self.registry.by_id(target)
        return target

    def activate(self, model_id: str) -> dict[str, Any]:
        self.registry.by_id(model_id)
        previous = self.active_model_id
        if model_id == previous:
            return dict(self._state)
        history = list(self._state.get("history", []))
        history.append(previous)
        next_state = {
            "schema": "wakeword-studio.active-model/v1",
            "active_model_id": model_id,
            "previous_model_id": previous,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "history": history[-20:],
        }
        self._write(next_state)
        self._state = next_state
        return dict(self._state)

    def rollback(self) -> dict[str, Any]:
        target = self.rollback_target_model_id
        history = list(self._state.get("history", []))
        history.pop()
        current = self.active_model_id
        next_state = {
            "schema": "wakeword-studio.active-model/v1",
            "active_model_id": target,
            "previous_model_id": current,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "history": history,
        }
        self._write(next_state)
        self._state = next_state
        return dict(self._state)

    def _write(self, state: dict[str, Any]) -> None:
        from .json_utils import atomic_write_json

        atomic_write_json(self.path, state)


def teacher_six_model_configs(project_root: Path) -> dict[str, dict[str, Any]]:
    """Build Teacher-Six registrations only from immutable Phase 9 artifacts."""

    selection_path = project_root / "reports/multikws/MODEL_SELECTION_VALIDATION.json"
    vocabulary_path = project_root / "configs/multikws/teacher_six_keywords.json"
    if not selection_path.is_file() or not vocabulary_path.is_file():
        return {}
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    classes = [{
        "class_id": int(vocabulary["background"]["class_index"]),
        "keyword_id": str(vocabulary["background"]["keyword_id"]),
        "display_name": str(vocabulary["background"]["display_name"]),
    }]
    classes.extend({
        "class_id": int(item["class_index"]),
        "keyword_id": str(item["keyword_id"]),
        "display_name": str(item["display_name"]),
    } for item in vocabulary["keywords"] if item.get("enabled", True))
    if [item["keyword_id"] for item in classes] != list(selection["class_names"]):
        raise ValueError("Teacher-Six vocabulary does not match frozen model-selection classes")

    result: dict[str, dict[str, Any]] = {}
    for architecture, model in selection["models"].items():
        test_path = project_root / f"reports/multikws/test/{architecture}/TEST_REPORT.json"
        test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else {}
        artifact = model["tflite"]
        source_artifact_path = resolve_project_path(project_root, artifact["path"])
        release_artifact_path = (
            project_root
            / "artifacts"
            / "models"
            / "teacher_six"
            / f"teacher_six_{architecture}_full_int8.tflite"
        )
        artifact_path = release_artifact_path if release_artifact_path.is_file() else source_artifact_path
        configured_artifact_path = (
            artifact_path.relative_to(project_root).as_posix()
            if artifact_path.is_relative_to(project_root)
            else str(artifact_path)
        )
        display_arch = "BC-ResNet" if architecture == "bcresnet" else "ConvMixer"
        status = "BASELINE" if model["role"] == "COMPUTE_LIGHT_BASELINE" else "CANDIDATE"
        result[f"Teacher-Six {display_arch}"] = {
            "id": f"teacher_six_{architecture}",
            "display_name": f"{display_arch} Teacher-Six",
            "backend": "multikws",
            "path": configured_artifact_path,
            "threshold": float(model["frozen_int8_threshold"]),
            "margin_threshold": float(model["frozen_int8_margin_threshold"]),
            "runtime_mode": "rolling_window",
            "window_seconds": 2.0,
            "hop_seconds": 0.20,
            "smoothing": "raw",
            "supported_platforms": ["Windows/WSL GPU", "TFLite runtime"],
            "description": str(model["role"]),
            "architecture": display_arch,
            "task_type": "multi_kws",
            "version": str(selection["dataset"]["id"]),
            "full_int8": bool(artifact["full_int8"]),
            "input_shape": [1, 99, 40],
            "output_shape": [1, len(classes)],
            "num_classes": len(classes),
            "vocabulary_id": str(vocabulary["vocabulary_id"]),
            "sha256": str(artifact["sha256"]),
            "validation_summary": dict(model.get("validation_summary", {})),
            "test_summary": dict(test.get("overall_metrics", {})),
            "hardware_runtime_verified": False,
            "status": status,
            "created_at": datetime.fromtimestamp(artifact_path.stat().st_mtime, timezone.utc).isoformat() if artifact_path.is_file() else "",
            "classes": classes,
            "deployment": {
                "format": "Full INT8", "bytes": int(artifact["bytes"]),
                "kib": float(artifact["bytes"]) / 1024.0, "sha256": str(artifact["sha256"]),
                "input_shape": [1, 99, 40], "input_dtype": "int8",
                "output_shape": [1, len(classes)], "output_dtype": "int8",
            },
            # Frozen Teacher-Six artifacts are inference candidates. Vocabulary
            # expansion uses the explicit Phase 10 job preflight, never this
            # legacy binary training launcher.
            "training": {},
        }
    return result
