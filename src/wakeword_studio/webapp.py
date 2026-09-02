"""Local-only HTTP bridge for the modern WakeWord Studio dashboard."""

from __future__ import annotations

import json
import base64
import hashlib
import re
import threading
import time
import webbrowser
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml

from wakeword_studio.dataset.product_plan import SCALE_PRESETS, build_product_plan
from wakeword_studio.delivery import inspect_tflite_model, prepare_esp32s3_package
from wakeword_studio.launchers import GenerationRequest, ManagedProcess, TrainingRequest, build_generation_command, build_training_command
from wakeword_studio.providers import ProviderRegistry
from wakeword_studio.registry import (
    ActiveModelStore,
    ModelRegistry,
    resolve_project_path,
    teacher_six_model_configs,
)
from wakeword_studio.phase10 import (
    FalseWakeSession,
    JobState,
    MicAcceptanceSession,
    MultiKWSJob,
    RuntimeFeedbackStore,
    build_keyword_expansion_plan,
    materialize_job_preflight,
)
from wakeword_studio.runtime.detection_logic import DetectionConfig, DetectionLogic
from wakeword_studio.runtime.engine import StreamingWakeWordEngine
from wakeword_studio.runtime.episode import DetectionEpisodeTracker, request_final_wake_playback
from wakeword_studio.runtime.playback import WakePlaybackQueue


FRAME_SAMPLES = 480


class StudioController:
    def __init__(
        self,
        project_root: Path,
        config_path: Path,
        *,
        preload_runtime_backend: bool = False,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.user_models_path = self.project_root / "configs/demo/user_models.json"
        self._merge_user_models()
        self.config.setdefault("models", {}).update(teacher_six_model_configs(self.project_root))
        self.models = ModelRegistry.from_config(self.project_root, self.config)
        configured_default = str(self.config.get("active_model_id", "model_b"))
        if configured_default not in {item.id for item in self.models.all()}:
            configured_default = "model_b"
        self.active_models = ActiveModelStore(
            self.project_root / "runtime/active_model.json", self.models, configured_default,
        )
        self.providers = ProviderRegistry.from_config(self.project_root, self.config)
        self.lock = threading.RLock()
        self.engine: StreamingWakeWordEngine | None = None
        self.loaded_backend = None
        self.loaded_backend_model_id: str | None = None
        self.loaded_backend_keyword: str | None = None
        self.preload_runtime_backend = bool(preload_runtime_backend)
        self.tracker = DetectionEpisodeTracker(max_history=10)
        self.model = self.models.by_id(self.active_models.active_model_id)
        self.keyword = str(self.config["wake_phrase"])
        self.running = False
        self.pcm_remainder = np.empty(0, dtype=np.int16)
        self.last_runtime: dict[str, Any] = self._empty_runtime()
        self.wake_count = 0
        self.reject_count = 0
        self.playback = WakePlaybackQueue(lambda line: self._log(line))
        self.logs: list[str] = []
        self.generation_process, self.training_process = ManagedProcess(), ManagedProcess()
        self.jobs = {
            "generation": {"status": "idle", "exit_code": None, "logs": []},
            "training": {"status": "idle", "exit_code": None, "logs": []},
        }
        self.multikws_jobs: dict[str, MultiKWSJob] = {}
        self.feedback = RuntimeFeedbackStore(
            self.project_root / "runtime/runtime_feedback.jsonl", save_audio=False,
        )
        self.mic_acceptance: MicAcceptanceSession | None = None
        self.false_wake: FalseWakeSession | None = None
        if self.preload_runtime_backend:
            self._load_runtime_backend(self.model)

    def _prepare_runtime_backend(self, model):  # noqa: ANN001
        """Validate and load a candidate without disturbing the active backend."""

        model.verify_artifact()
        backend = model.create_backend(self.keyword)
        backend.load(model.model_path)
        return backend

    def _load_runtime_backend(self, model) -> None:  # noqa: ANN001
        backend = self._prepare_runtime_backend(model)
        self.loaded_backend = backend
        self.loaded_backend_model_id = model.id
        self.loaded_backend_keyword = self.keyword

    def _ensure_runtime_backend(self):  # noqa: ANN001
        keyword_changed = (
            self.model.task_type == "binary"
            and self.loaded_backend_keyword != self.keyword
        )
        if (
            self.loaded_backend_model_id != self.model.id
            or self.loaded_backend is None
            or keyword_changed
        ):
            self._load_runtime_backend(self.model)
        return self.loaded_backend

    def _merge_user_models(self) -> None:
        if not self.user_models_path.is_file():
            return
        raw = json.loads(self.user_models_path.read_text(encoding="utf-8"))
        for key, value in dict(raw.get("models", {})).items():
            self.config.setdefault("models", {})[key] = value

    def _log(self, line: str) -> None:
        with self.lock:
            self.logs.append(f"{datetime.now().strftime('%H:%M:%S')}  {line}")
            self.logs[:] = self.logs[-120:]

    def _empty_runtime(self) -> dict[str, Any]:
        return {
            "energy": 0.0, "energy_dbfs": -120.0, "adaptive_threshold": 0.0,
            "vad": False, "speech_frame_count": 0, "kws_active": False,
            "raw_wake_score": 0.0, "decision_wake_score": 0.0,
            "wake_threshold": float(self.model.threshold), "l1_status": "0/2:False",
            "l2_ratio": 0.0, "l2_status": "False", "cooldown": 0.0,
            "l4_status": "True", "l5_status": "waiting",
            "rejection_reason": "NO_NEW_SCORE", "tail_silence_frames": 0,
            "tail_required_frames": 27, "final_wake_event": False, "keyword": None,
            "energy_gate_passed": False, "predicted_class_id": None,
            "predicted_keyword_id": None, "predicted_display_name": None,
            "top1_score": 0.0, "top2_class_id": None, "top2_keyword_id": None,
            "top2_display_name": None, "top2_score": 0.0, "margin": 0.0,
            "margin_threshold": float(self.model.margin_threshold),
            "background_score": 0.0, "accepted": False,
        }

    def bootstrap(self) -> dict[str, Any]:
        models = []
        for model in self.models.all():
            deployment = model.deployment
            models.append({
                "id": model.id, "name": model.display_name, "description": model.description,
                "backend_id": model.backend_id,
                "threshold": model.threshold, "size_kib": model.model_size_kib,
                "runtime_mode": model.runtime_mode, "window_seconds": model.window_seconds,
                "hop_seconds": model.hop_seconds, "smoothing": model.smoothing,
                "platforms": list(model.supported_platforms), "format": deployment.get("format", "INT8"),
                "sha256": deployment.get("sha256", ""), "validation_recall": deployment.get("validation_recall"),
                "input_shape": list(model.input_shape), "output_shape": list(model.output_shape),
                "trainable": bool(model.trainer), "user_imported": model.id.startswith("user_"),
                "architecture": model.architecture, "task_type": model.task_type,
                "version": model.version, "full_int8": model.full_int8,
                "num_classes": model.num_classes, "keyword_count": max(0, model.num_classes - 1) if model.task_type == "multi_kws" else 1,
                "vocabulary_id": model.vocabulary_id, "margin_threshold": model.margin_threshold,
                "status": "ACTIVE" if model.id == self.active_models.active_model_id else model.status,
                "active": model.id == self.active_models.active_model_id,
                "validation_summary": model.validation_summary, "test_summary": model.test_summary,
                "validation_available": bool(model.validation_summary or deployment.get("validation_recall") is not None),
                "test_available": bool(model.test_summary),
                "hardware_runtime_verified": model.hardware_runtime_verified,
                "created_at": model.created_at, "classes": list(model.classes),
                "training_compatible": bool(model.trainer),
                "inference_compatible": True,
            })
        providers = [provider.metadata() for provider in self.providers.available(generation_only=True)]
        plans = {
            mode: build_product_plan(mode).as_dict()
            for mode in SCALE_PRESETS
        }
        return {
            "wake_phrase": self.keyword, "models": models, "providers": providers,
            "plans": plans, "state": self.state(),
            "active_model_id": self.active_models.active_model_id,
            "ADD_KEYWORD_REQUIRES_RETRAIN": True,
            "audio_retention_default": False,
            "formal_test_rerun": False,
        }

    def start_live(self, model_id: str, keyword: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.running:
                return self.state()
            if model_id != self.active_models.active_model_id:
                raise RuntimeError("该模型尚未激活；请先在“模型与部署”中点击“设为当前模型”")
            self.model = self.models.by_id(model_id)
            self.keyword = (keyword or self.config["wake_phrase"]).strip()
            backend = self._ensure_runtime_backend()
            raw = self.config["detection"]
            detection = DetectionLogic(DetectionConfig(
                wake_threshold=float(self.model.threshold),
                consecutive_wake_frames=int(raw["consecutive_wake_frames"]),
                peak_background_ratio=float(raw["peak_background_ratio"]),
                background_alpha=float(raw["background_alpha"]),
                cooldown_seconds=float(raw["cooldown_seconds"]),
                arbitration_margin=float(raw["arbitration_margin"]),
                pre_silence_frames=int(raw["pre_silence_frames"]),
                post_silence_frames=int(raw["post_silence_frames"]),
            ))
            self.engine = StreamingWakeWordEngine(backend, detection=detection)
            self.tracker.cancel_active()
            self.pcm_remainder = np.empty(0, dtype=np.int16)
            self.last_runtime = self._empty_runtime()
            self.last_runtime["wake_threshold"] = float(self.model.threshold)
            self.running = True
            self._log(f"LISTENING_STARTED model={self.model.id} threshold={self.model.threshold}")
            return self.state()

    def activate_model(self, model_id: str) -> dict[str, Any]:
        with self.lock:
            if self.running:
                raise RuntimeError("请先停止监听，再切换当前模型")
            candidate = self.models.by_id(model_id)
            candidate_backend = None
            if self.preload_runtime_backend and model_id != self.loaded_backend_model_id:
                candidate_backend = self._prepare_runtime_backend(candidate)
            state = self.active_models.activate(model_id)
            self.model = self.models.by_id(self.active_models.active_model_id)
            if candidate_backend is not None:
                self.loaded_backend = candidate_backend
                self.loaded_backend_model_id = model_id
                self.loaded_backend_keyword = self.keyword
            self.tracker.clear_latest()
            self.last_runtime = self._empty_runtime()
            self._log(f"MODEL_ACTIVATED model={model_id}")
            return {"ok": True, **state, "runtime_backend_model_id": self.loaded_backend_model_id}

    def rollback_model(self) -> dict[str, Any]:
        with self.lock:
            if self.running:
                raise RuntimeError("请先停止监听，再回滚模型")
            target = self.models.by_id(self.active_models.rollback_target_model_id)
            candidate_backend = None
            if self.preload_runtime_backend and target.id != self.loaded_backend_model_id:
                candidate_backend = self._prepare_runtime_backend(target)
            state = self.active_models.rollback()
            self.model = self.models.by_id(self.active_models.active_model_id)
            if candidate_backend is not None:
                self.loaded_backend = candidate_backend
                self.loaded_backend_model_id = target.id
                self.loaded_backend_keyword = self.keyword
            self.tracker.clear_latest()
            self.last_runtime = self._empty_runtime()
            self._log(f"MODEL_ROLLBACK model={self.model.id}")
            return {"ok": True, **state, "runtime_backend_model_id": self.loaded_backend_model_id}

    def stop_live(self) -> dict[str, Any]:
        with self.lock:
            self.running = False
            self.engine = None
            self.pcm_remainder = np.empty(0, dtype=np.int16)
            self.tracker.cancel_active()
            self._log("LISTENING_STOPPED")
            return self.state()

    def feed_audio(self, payload: bytes) -> None:
        with self.lock:
            if not self.running or self.engine is None:
                return
            pcm = np.frombuffer(payload, dtype="<i2").astype(np.int16, copy=False)
            if self.pcm_remainder.size:
                pcm = np.concatenate((self.pcm_remainder, pcm))
            offset = 0
            while offset + FRAME_SAMPLES <= pcm.size:
                runtime = self.engine.process_frame(pcm[offset:offset + FRAME_SAMPLES], time.monotonic())
                self.last_runtime = runtime.to_dict()
                runtime_keyword = runtime.predicted_display_name or runtime.keyword or self.keyword
                snapshot = self.tracker.update(runtime, keyword=runtime_keyword)
                if snapshot is not None:
                    if snapshot.result == "WAKE":
                        self.wake_count += 1
                        if self.false_wake is not None:
                            self.false_wake.false_wake_count += 1
                        request_final_wake_playback(
                            self.playback, snapshot,
                            resolve_project_path(self.project_root, self.config["awake_wav"]),
                        )
                    else:
                        self.reject_count += 1
                    self._log(f"EPISODE_{snapshot.result} score={snapshot.decision_max_score:.4f}")
                offset += FRAME_SAMPLES
            self.pcm_remainder = pcm[offset:].copy()

    def state(self) -> dict[str, Any]:
        with self.lock:
            runtime = dict(self.last_runtime)
            snapshot = self.tracker.latest
            history = [self._snapshot_dict(item) for item in reversed(self.tracker.history)]
            return {
                "running": self.running, "model_id": self.model.id, "keyword": self.keyword,
                "active_model_id": self.active_models.active_model_id,
                "runtime_backend_model_id": self.loaded_backend_model_id,
                "model": {
                    "id": self.model.id, "name": self.model.display_name,
                    "backend_id": self.model.backend_id,
                    "architecture": self.model.architecture, "task_type": self.model.task_type,
                    "version": self.model.version, "sha256": self.model.sha256,
                    "full_int8": self.model.full_int8, "num_classes": self.model.num_classes,
                    "vocabulary_id": self.model.vocabulary_id, "classes": list(self.model.classes),
                    "margin_threshold": self.model.margin_threshold,
                    "threshold": self.model.threshold,
                    "input_shape": list(self.model.input_shape),
                    "output_shape": list(self.model.output_shape),
                },
                "threshold": float(self.model.threshold), "runtime": runtime,
                "status": self._status(runtime), "latest": None if snapshot is None else self._snapshot_dict(snapshot),
                "history": history, "wake_count": self.wake_count, "reject_count": self.reject_count,
                "playback_count": self.playback.playback_count, "logs": self.logs[-30:],
                "false_wake": None if self.false_wake is None else self.false_wake.report(),
            }

    def keyword_preflight(self, data: dict[str, Any]) -> dict[str, Any]:
        return build_keyword_expansion_plan(
            self.project_root / "configs/multikws/teacher_six_keywords.json",
            str(data.get("display_name", "")),
            positive_samples=int(data.get("positive_samples", 600)),
            hard_negative_samples=int(data.get("hard_negative_samples", 300)),
            augmentation=str(data.get("augmentation", "standard")),
            speech_sources=tuple(data.get("speech_sources", ("kokoro", "voxcpm15"))),
            model_architecture=str(data.get("model_architecture", "convmixer")),
            input_mode=str(data.get("input_mode", "auto_generate")),
            user_wav_directory=data.get("user_wav_directory"),
        )

    def create_multikws_job(self, data: dict[str, Any]) -> dict[str, Any]:
        preflight = self.keyword_preflight(data)
        keyword_id = str(preflight["plan"]["keyword_id"])
        job = MultiKWSJob.pending(self.project_root / "runs/multikws/user_expansions", keyword_id)
        self.multikws_jobs[job.job_id] = job
        artifacts = materialize_job_preflight(self.project_root, job, preflight)
        return {
            "ok": True, "job": job.to_dict(), "preflight": preflight,
            "artifacts": artifacts, "long_job_started": False,
        }

    def multikws_job_state(self, job_id: str) -> dict[str, Any]:
        return self.multikws_jobs[job_id].to_dict()

    def cancel_multikws_job(self, job_id: str) -> dict[str, Any]:
        job = self.multikws_jobs[job_id]
        job.cancel()
        return {"ok": True, "job": job.to_dict(), "checkpoint_deleted": False}

    def resume_multikws_job(self, job_id: str) -> dict[str, Any]:
        job = self.multikws_jobs[job_id]
        if job.state not in {JobState.CANCELLED.value, JobState.FAILED.value}:
            raise RuntimeError("只有 CANCELLED/FAILED job 可以恢复")
        job.state = JobState.PENDING.value
        job.error_message = None
        return {"ok": True, "job": job.to_dict(), "long_job_started": False}

    def save_feedback(self, data: dict[str, Any]) -> dict[str, Any]:
        self.feedback.save_audio = bool(data.get("save_audio", False))
        event = {
            "model_id": self.model.id, "model_sha256": self.model.sha256,
            "top1": self.last_runtime.get("predicted_keyword_id"),
            "top1_score": self.last_runtime.get("top1_score", 0.0),
            "top2": self.last_runtime.get("top2_keyword_id"),
            "top2_score": self.last_runtime.get("top2_score", 0.0),
            "margin": self.last_runtime.get("margin", 0.0), "threshold": self.model.threshold,
            "vad": self.last_runtime.get("vad", False), "energy": self.last_runtime.get("energy", 0.0),
            "accepted": self.last_runtime.get("accepted", False),
            "rejection_reason": self.last_runtime.get("rejection_reason"),
            "audio_segment_path": data.get("audio_segment_path"),
        }
        return {"ok": True, "feedback": self.feedback.append(event, str(data["verdict"]), data.get("ground_truth"))}

    def start_mic_acceptance(self, data: dict[str, Any]) -> dict[str, Any]:
        self.mic_acceptance = MicAcceptanceSession(
            self.model.id, self.model.vocabulary_id or "binary",
            int(data.get("target_attempts_per_keyword", 10)),
        )
        return {"ok": True, "report_type": "REAL_MIC_ACCEPTANCE", "session": self.mic_acceptance.report()}

    def record_mic_acceptance(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.mic_acceptance is None:
            raise RuntimeError("请先开始真人麦克风验收")
        self.mic_acceptance.record(str(data["expected_keyword_id"]), str(data["result"]))
        return {"ok": True, "session": self.mic_acceptance.report()}

    def finish_mic_acceptance(self) -> dict[str, Any]:
        if self.mic_acceptance is None:
            raise RuntimeError("没有正在进行的真人麦克风验收")
        path = self.project_root / "reports/phase10/MIC_ACCEPTANCE_REPORT.json"
        self.mic_acceptance.save(path)
        report = self.mic_acceptance.report()
        self.mic_acceptance = None
        return {"ok": True, "path": str(path.relative_to(self.project_root)), "report": report}

    def start_false_wake(self) -> dict[str, Any]:
        self.false_wake = FalseWakeSession()
        return {"ok": True, "session": self.false_wake.report()}

    def stop_false_wake(self) -> dict[str, Any]:
        if self.false_wake is None:
            raise RuntimeError("没有正在进行的背景误唤醒测试")
        report = self.false_wake.report()
        self.false_wake = None
        return {"ok": True, "report": report}

    @staticmethod
    def _snapshot_dict(snapshot: Any) -> dict[str, Any]:
        value = asdict(snapshot)
        value["detected_at"] = snapshot.detected_at.isoformat(timespec="seconds")
        return value

    def _status(self, runtime: dict[str, Any]) -> str:
        if not self.running: return "STOPPED"
        if runtime.get("final_wake_event"): return "WAKE"
        if float(runtime.get("cooldown", 0.0)) > 0: return "COOLDOWN"
        if runtime.get("kws_active") and int(runtime.get("tail_silence_frames", 0)) > 0: return "TAIL"
        if runtime.get("kws_active"): return "EVALUATING"
        if runtime.get("vad"): return "SPEECH"
        energy = float(runtime.get("energy", 0.0))
        if energy > 0.0 and energy >= float(runtime.get("adaptive_threshold", 1.0)): return "SOUND"
        return "IDLE"

    def generation_preflight(self, data: dict[str, Any]) -> dict[str, Any]:
        custom_targets = None
        if data.get("mode") == "自定义" and isinstance(data.get("targets"), dict):
            custom_targets = {str(key): int(value) for key, value in data["targets"].items()}
        request = GenerationRequest(
            str(data.get("wake_phrase", self.keyword)), int(data.get("total", 12)),
            str(data["provider"]), bool(data.get("augmentation", True)),
            Path(str(data.get("output", "outputs/teacher_generated"))),
            scale_mode=str(data.get("mode", "快速测试")),
            custom_total=int(data["total"]) if data.get("mode") == "自定义" else None,
            input_folder=Path(str(data["input_folder"])) if data.get("input_folder") else None,
            custom_targets=custom_targets,
        )
        command = build_generation_command(self.project_root, request)
        return {"ok": True, "command": command}

    def training_preflight(self, data: dict[str, Any]) -> dict[str, Any]:
        request = TrainingRequest(
            Path(str(data["dataset"])), str(data["model_id"]),
            str(data.get("wake_phrase", self.keyword)), Path(str(data["output"])),
        )
        return {"ok": True, "command": build_training_command(self.project_root, request)}

    def start_job(self, kind: str, command: list[str]) -> dict[str, Any]:
        process = self.generation_process if kind == "generation" else self.training_process
        with self.lock:
            if process.running or self.jobs[kind]["status"] == "running":
                raise RuntimeError(f"{kind} 任务已经在运行")
            self.jobs[kind] = {"status": "running", "exit_code": None, "logs": []}

        def on_line(line: str) -> None:
            with self.lock:
                self.jobs[kind]["logs"].append(line)
                self.jobs[kind]["logs"] = self.jobs[kind]["logs"][-120:]

        def run() -> None:
            try:
                code = process.start(command, on_line, cwd=self.project_root)
                status = "completed" if code == 0 else "failed"
            except Exception as exc:
                code, status = -1, "failed"
                on_line(f"{type(exc).__name__}: {exc}")
            with self.lock:
                self.jobs[kind]["status"] = status
                self.jobs[kind]["exit_code"] = code

        threading.Thread(target=run, daemon=True).start()
        return self.job_state(kind)

    def stop_job(self, kind: str) -> dict[str, Any]:
        process = self.generation_process if kind == "generation" else self.training_process
        process.stop()
        with self.lock:
            self.jobs[kind]["status"] = "stopping"
        return self.job_state(kind)

    def job_state(self, kind: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.jobs[kind])

    def inspect_model(self, model_id: str) -> dict[str, Any]:
        model = self.models.by_id(model_id)
        info = inspect_tflite_model(model.model_path)
        return {"model_id": model_id, "bytes": info.bytes, "kib": info.kib, "sha256": info.sha256,
                "full_int8": info.full_int8, "input_shape": info.input_shape, "output_shape": info.output_shape,
                "input_dtype": info.input_dtype, "output_dtype": info.output_dtype}

    def deploy_model(self, model_id: str, output: str) -> dict[str, Any]:
        model = self.models.by_id(model_id)
        report = prepare_esp32s3_package(model.model_path, resolve_project_path(self.project_root, output))
        return {"ok": True, "report": report}

    def import_model(self, data: dict[str, Any]) -> dict[str, Any]:
        name = str(data.get("name", "")).strip()
        backend = str(data.get("backend", "repcnn"))
        threshold = float(data.get("threshold", 0.5))
        encoded = str(data.get("data_base64", ""))
        if not name or backend not in {"repcnn", "microwakeword"}:
            raise ValueError("必须填写模型名称，并选择受支持的 backend")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold 必须位于 (0, 1]")
        payload = base64.b64decode(encoded, validate=True)
        if len(payload) < 1024 or len(payload) > 20 * 1024 * 1024:
            raise ValueError("TFLite 文件大小必须介于 1 KiB 与 20 MiB")
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "custom_model"
        digest = hashlib.sha256(payload).hexdigest()
        model_id = f"user_{slug}_{digest[:8]}"
        relative = Path("models/imported") / model_id / "model.tflite"
        destination = self.project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != payload:
            raise RuntimeError("目标模型 ID 已存在但内容不同")
        created_here = not destination.exists()
        if created_here:
            destination.write_bytes(payload)
        try:
            info = inspect_tflite_model(destination)
            if backend == "repcnn" and (list(info.input_shape) != [1, 99, 40] or list(info.output_shape) != [1, 1]):
                raise ValueError("RepCNN imported model must expose [1,99,40] -> [1,1]")
            if backend == "microwakeword" and (list(info.input_shape) != [1, 3, 40] or list(info.output_shape) != [1, 1]):
                raise ValueError("microWakeWord imported model must expose [1,3,40] -> [1,1]")
            if not info.full_int8:
                raise ValueError("实时 imported backend 当前要求 Full INT8 interface")
        except Exception:
            if created_here:
                destination.unlink(missing_ok=True)
            raise
        config_key = f"用户模型 — {name}"
        registration = {
            "id": model_id, "display_name": name, "backend": backend,
            "path": relative.as_posix(), "threshold": threshold,
            "runtime_mode": "rolling_window" if backend == "repcnn" else "native_streaming",
            "window_seconds": 2.0 if backend == "repcnn" else None,
            "hop_seconds": 0.20 if backend == "repcnn" else None,
            "smoothing": "raw", "supported_platforms": ["用户指定"],
            "description": "用户导入的部署模型",
            "architecture": "Imported TFLite",
            "task_type": "binary",
            "version": digest[:12],
            "full_int8": bool(info.full_int8),
            "num_classes": 1,
            "status": "IMPORTED",
            "hardware_runtime_verified": False,
            "deployment": {
                "format": "Full INT8" if info.full_int8 else "TFLite",
                "bytes": info.bytes, "kib": info.kib, "sha256": info.sha256,
                "input_shape": info.input_shape, "input_dtype": info.input_dtype,
                "output_shape": info.output_shape, "output_dtype": info.output_dtype,
            },
        }
        stored = {"schema": "wakeword-studio.user-models/v1", "models": {}}
        if self.user_models_path.is_file():
            stored = json.loads(self.user_models_path.read_text(encoding="utf-8"))
        stored.setdefault("models", {})[config_key] = registration
        self.user_models_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_models_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
        self.config.setdefault("models", {})[config_key] = registration
        self.models = ModelRegistry.from_config(self.project_root, self.config)
        self.active_models.registry = self.models
        return {
            "ok": True, "model_id": model_id, "sha256": info.sha256,
            "full_int8": info.full_int8, "inference_compatible": True,
            "training_compatible": False,
        }


class StudioRequestHandler(BaseHTTPRequestHandler):
    controller: StudioController
    static_root: Path

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(payload)

    def _body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/bootstrap": return self._json(self.controller.bootstrap())
        if path == "/api/live/state": return self._json(self.controller.state())
        if path.startswith("/api/multikws/job/"):
            return self._json(self.controller.multikws_job_state(path.rsplit("/", 1)[-1]))
        if path.startswith("/api/job/"):
            return self._json(self.controller.job_state(path.rsplit("/", 1)[-1]))
        filenames = {"/": "index.html", "/app.css": "app.css", "/overrides.css": "overrides.css", "/app.js": "app.js"}
        filename = filenames.get(path)
        if filename is None: return self.send_error(HTTPStatus.NOT_FOUND)
        payload = (self.static_root / filename).read_bytes()
        content_type = "text/html; charset=utf-8" if filename.endswith(".html") else "text/css; charset=utf-8" if filename.endswith(".css") else "text/javascript; charset=utf-8"
        self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if path == "/api/live/audio":
                length = int(self.headers.get("Content-Length", "0")); self.controller.feed_audio(self.rfile.read(length))
                return self._json({"ok": True})
            data = self._body_json()
            if path == "/api/live/start": return self._json(self.controller.start_live(str(data["model_id"]), data.get("keyword")))
            if path == "/api/live/stop": return self._json(self.controller.stop_live())
            if path == "/api/generation/preflight": return self._json(self.controller.generation_preflight(data))
            if path == "/api/training/preflight": return self._json(self.controller.training_preflight(data))
            if path == "/api/generation/start":
                result = self.controller.generation_preflight(data)
                return self._json(self.controller.start_job("generation", result["command"]))
            if path == "/api/training/start":
                result = self.controller.training_preflight(data)
                return self._json(self.controller.start_job("training", result["command"]))
            if path == "/api/generation/stop": return self._json(self.controller.stop_job("generation"))
            if path == "/api/training/stop": return self._json(self.controller.stop_job("training"))
            if path == "/api/model/inspect": return self._json(self.controller.inspect_model(str(data["model_id"])))
            if path == "/api/model/deploy": return self._json(self.controller.deploy_model(str(data["model_id"]), str(data["output"])))
            if path == "/api/model/import": return self._json(self.controller.import_model(data))
            if path == "/api/model/activate": return self._json(self.controller.activate_model(str(data["model_id"])))
            if path == "/api/model/rollback": return self._json(self.controller.rollback_model())
            if path == "/api/keyword/preflight": return self._json(self.controller.keyword_preflight(data))
            if path == "/api/keyword/job/create": return self._json(self.controller.create_multikws_job(data))
            if path == "/api/keyword/job/cancel": return self._json(self.controller.cancel_multikws_job(str(data["job_id"])))
            if path == "/api/keyword/job/resume": return self._json(self.controller.resume_multikws_job(str(data["job_id"])))
            if path == "/api/runtime/feedback": return self._json(self.controller.save_feedback(data))
            if path == "/api/mic-acceptance/start": return self._json(self.controller.start_mic_acceptance(data))
            if path == "/api/mic-acceptance/record": return self._json(self.controller.record_mic_acceptance(data))
            if path == "/api/mic-acceptance/finish": return self._json(self.controller.finish_mic_acceptance())
            if path == "/api/false-wake/start": return self._json(self.controller.start_false_wake())
            if path == "/api/false-wake/stop": return self._json(self.controller.stop_false_wake())
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)


def serve(project_root: Path, config_path: Path, host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    controller = StudioController(project_root, config_path, preload_runtime_backend=True)
    handler = type("BoundStudioHandler", (StudioRequestHandler,), {
        "controller": controller,
        "static_root": project_root / "phase7/webui",
    })
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"WAKEWORD_STUDIO_WEB_READY {url}", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        controller.stop_live(); controller.generation_process.stop(); controller.training_process.stop()
        controller.playback.close(wait=False); server.server_close()
