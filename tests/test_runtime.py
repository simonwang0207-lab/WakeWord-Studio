from __future__ import annotations

from pathlib import Path

import numpy as np

from wakeword_studio.backends.base import BackendEvaluation, ExportArtifact, WakeWordBackend
from wakeword_studio.runtime.detection_logic import DetectionConfig, DetectionLogic
from wakeword_studio.runtime.gates import ConsecutiveSpeechGate
from wakeword_studio.runtime.ring_buffer import PreRollRingBuffer


class FakeBackend(WakeWordBackend):
    def train(self, manifest_path: Path, config_path: Path) -> Path: return config_path
    def evaluate(self, manifest_path: Path) -> BackendEvaluation: return BackendEvaluation(0, [], {})
    def export(self, destination: Path | None = None) -> ExportArtifact: raise NotImplementedError
    def load(self, model_path: Path) -> None: pass
    def stream_scores(self, pcm16: np.ndarray) -> dict[str, float]: return {"你好，青小甲": 0.9}
    def reset_stream(self) -> None: pass


def test_vad_three_frame_gate_is_independent() -> None:
    gate = ConsecutiveSpeechGate(3)
    assert gate.update(True) == (1, False)
    assert gate.update(True) == (2, False)
    assert gate.update(False) == (0, False)
    assert gate.update(True) == (1, False)
    assert gate.update(True) == (2, False)
    assert gate.update(True) == (3, True)


def test_detection_l1_to_l5_and_cooldown() -> None:
    logic = DetectionLogic(DetectionConfig(wake_threshold=0.5, consecutive_wake_frames=2, peak_background_ratio=1.2, cooldown_seconds=5, arbitration_margin=0.05, pre_silence_frames=2, post_silence_frames=2))
    logic.update({}, False, 0.0)
    first = logic.update({"你好，青小甲": 0.8, "other": 0.1}, True, 1.0)
    assert not first.wake_event and not first.l1_passed
    second = logic.update({"你好，青小甲": 0.9, "other": 0.1}, True, 1.1)
    assert second.l1_passed and second.l2_passed and second.l4_passed
    # A stateful streaming model can keep returning scores in silence; L5 must
    # still advance from VAD/energy transition alone.
    assert not logic.update({"你好，青小甲": 0.8}, False, 1.2).wake_event
    event = logic.update({"你好，青小甲": 0.8}, False, 1.3)
    assert event.wake_event and event.l5_state == "passed"
    assert logic.update({"你好，青小甲": 0.99}, True, 1.4).cooldown_remaining > 0


def test_preroll_keeps_onset_and_three_vad_frames() -> None:
    ring = PreRollRingBuffer(frame_ms=30, seconds=1.5)
    for index in range(60):
        ring.append(np.full(480, index, np.int16))
    audio = ring.audio()
    assert len(audio) == 50 * 480
    assert audio[0] == 10
    assert audio[-1] == 59
