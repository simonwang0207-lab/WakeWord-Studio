"""End-to-end streaming coordinator with detailed observable state."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..backends.base import WakeWordBackend
from .detection_logic import DetectionDecision, DetectionLogic
from .gates import AdaptiveEnergyGate, ConsecutiveSpeechGate, WebRTCVadGate
from .ring_buffer import PreRollRingBuffer


@dataclass(slots=True)
class RuntimeLog:
    energy: float
    energy_dbfs: float
    adaptive_threshold: float
    vad: bool
    speech_frame_count: int
    kws_active: bool
    raw_wake_score: float
    l1_status: str
    l2_ratio: float
    cooldown: float
    l5_status: str
    final_wake_event: bool
    keyword: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StreamingWakeWordEngine:
    def __init__(self, backend: WakeWordBackend, frame_ms: int = 30, pre_roll_seconds: float = 1.5, detection: DetectionLogic | None = None):
        self.backend = backend
        self.energy_gate = AdaptiveEnergyGate()
        self.vad_gate = WebRTCVadGate(frame_ms=frame_ms)
        self.speech_gate = ConsecutiveSpeechGate(required_frames=3)
        self.pre_roll = PreRollRingBuffer(frame_ms=frame_ms, seconds=pre_roll_seconds)
        self.detection = detection or DetectionLogic()
        self.kws_active = False

    def process_frame(self, pcm16: np.ndarray, now: float | None = None) -> RuntimeLog:
        frame = np.asarray(pcm16, dtype=np.int16)
        self.pre_roll.append(frame)
        energy = self.energy_gate.process(frame)
        vad = self.vad_gate.process(frame) if energy.passed else False
        speech_count, gate_active = self.speech_gate.update(vad)
        scores: dict[str, float] = {}
        if gate_active and not self.kws_active:
            self.kws_active = True
            self.backend.reset_stream()
            # Replay the buffer including all three VAD frames, so onset is never dropped.
            scores = self.backend.stream_scores(self.pre_roll.audio())
        elif self.kws_active:
            scores = self.backend.stream_scores(frame)
        if self.kws_active and not vad and speech_count == 0:
            # Keep the detector alive for post-silence validation; backend state remains intact.
            pass
        decision: DetectionDecision = self.detection.update(scores, vad, now)
        raw = max(scores.values(), default=0.0)
        return RuntimeLog(
            energy=energy.rms,
            energy_dbfs=energy.dbfs,
            adaptive_threshold=energy.adaptive_threshold,
            vad=vad,
            speech_frame_count=speech_count,
            kws_active=self.kws_active,
            raw_wake_score=raw,
            l1_status=f"{decision.l1_streak}/{self.detection.config.consecutive_wake_frames}:{decision.l1_passed}",
            l2_ratio=decision.l2_ratio,
            cooldown=decision.cooldown_remaining,
            l5_status=decision.l5_state,
            final_wake_event=decision.wake_event,
            keyword=decision.keyword,
        )

