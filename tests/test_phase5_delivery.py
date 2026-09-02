from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from wakeword_studio.backends.base import BackendEvaluation, ExportArtifact, WakeWordBackend
from wakeword_studio.launchers import (
    GenerationRequest,
    TrainingRequest,
    build_generation_command,
    build_training_command,
)
from wakeword_studio.runtime.detection_logic import DetectionConfig, DetectionLogic
from wakeword_studio.runtime.engine import StreamingWakeWordEngine
from wakeword_studio.runtime.ring_buffer import PreRollRingBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeBackend(WakeWordBackend):
    def __init__(self) -> None:
        self.reset_count = 0
        self.calls: list[int] = []

    def train(self, manifest_path: Path, config_path: Path) -> Path:
        return config_path

    def evaluate(self, manifest_path: Path) -> BackendEvaluation:
        return BackendEvaluation(0, [], {})

    def export(self, destination: Path | None = None) -> ExportArtifact:
        raise NotImplementedError

    def load(self, model_path: Path) -> None:
        return None

    def stream_scores(self, pcm16: np.ndarray) -> dict[str, float]:
        self.calls.append(len(pcm16))
        return {"你好，青小甲": 0.9}

    def reset_stream(self) -> None:
        self.reset_count += 1


class AlwaysSpeechVad:
    def process(self, pcm16: np.ndarray) -> bool:
        return True


def test_two_second_preroll_contract() -> None:
    ring = PreRollRingBuffer(frame_ms=30)
    for index in range(100):
        ring.append(np.full(480, index, np.int16))
    assert len(ring.audio()) == 67 * 480
    assert 2.0 <= len(ring.audio()) / 16000 < 2.02


def test_synthetic_audio_pipeline_activates_after_three_speech_frames() -> None:
    backend = FakeBackend()
    logic = DetectionLogic(
        DetectionConfig(
            wake_threshold=0.5,
            consecutive_wake_frames=1,
            peak_background_ratio=1.0,
            pre_silence_frames=0,
            post_silence_frames=0,
        )
    )
    engine = StreamingWakeWordEngine(backend, detection=logic)
    engine.vad_gate = AlwaysSpeechVad()
    frame = np.full(480, 5000, np.int16)
    assert not engine.process_frame(frame, 0.0).kws_active
    assert not engine.process_frame(frame, 0.03).kws_active
    third = engine.process_frame(frame, 0.06)
    assert third.kws_active
    assert third.final_wake_event
    assert backend.reset_count == 1
    assert backend.calls == [3 * 480]


def test_generation_launcher_dry_run_uses_existing_pipeline() -> None:
    command = build_generation_command(
        PROJECT_ROOT,
        GenerationRequest("你好，青小甲", 4, "Kokoro", False, PROJECT_ROOT / "outputs/test"),
    )
    assert command[0].endswith(".envs\\kokoro\\Scripts\\python.exe")
    assert command[1].endswith("phase1\\scripts\\generate_dataset.py")
    assert command[-2:] == ["--noise-augmentation", "none"]
    with pytest.raises(ValueError, match="between 2 and 12"):
        build_generation_command(
            PROJECT_ROOT,
            GenerationRequest("你好，青小甲", 100, "Kokoro", True, PROJECT_ROOT / "x", scale_mode="legacy"),
        )


def test_training_launcher_is_dry_run_only_and_model_specific() -> None:
    command = build_training_command(
        PROJECT_ROOT,
        TrainingRequest(
            PROJECT_ROOT / "datasets/projects/qingxiaojia_v3_fasttrack",
            "Legacy RepCNN Binary",
            "你好，青小甲",
            PROJECT_ROOT / "runs/teacher_ui/dry_run",
        ),
    )
    assert command[0].endswith(".envs\\livekit\\Scripts\\python.exe")
    assert command[1].endswith("phase4\\scripts\\run_repcnn_v2_fasttrack_training.py")
    assert "--allow-formal-training" in command
    with pytest.raises(ValueError, match="requires qingxiaojia_v3_fasttrack"):
        build_training_command(
            PROJECT_ROOT,
            TrainingRequest(
                PROJECT_ROOT / "datasets/projects/qingxiaojia_v2",
                "Legacy RepCNN Binary",
                "你好，青小甲",
                PROJECT_ROOT / "runs/teacher_ui/invalid",
            ),
        )


def test_teacher_demo_defaults_to_final_model_b_v2() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/demo/teacher_demo.yaml").read_text(encoding="utf-8")
    )
    model = config["models"]["Legacy RepCNN Binary"]
    assert model["backend"] == "repcnn"
    assert model["threshold"] == 0.21875
    assert model["window_seconds"] == pytest.approx(2.0)
    assert model["hop_seconds"] == pytest.approx(0.20)
    assert model["smoothing"] == "raw"
    # A public clone cannot depend on the ignored local runs/ tree.  The demo
    # must bind to the checksum-verified release copy while retaining the
    # frozen training version as provenance.
    assert model["path"] == "artifacts/models/binary/repcnn_full_int8.tflite"
    assert model["version"] == "phase6-finalization-v2"
    assert model["deployment"]["bytes"] == 112816
    assert model["deployment"]["kib"] == pytest.approx(110.171875)
    assert model["deployment"]["sha256"] == (
        "6acfecf7cc8651c1fba52809eee1d89abbcffa0a48bd46662b2e58ac23ce064f"
    )
    assert model["deployment"]["validation_recall"] == pytest.approx(0.810126582278481)
    assert model["deployment"]["validation_target_recall"] == pytest.approx(0.98)
    assert model["deployment"]["validation_target_met"] is False
    assert config["audio"] == {
        "sample_rate_hz": 16000,
        "channels": 1,
        "dtype": "int16",
        "frame_ms": 30,
        "pre_roll_seconds": 2.0,
    }
