"""End-to-end streaming coordinator with detailed observable state."""

from __future__ import annotations

import math
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
    decision_wake_score: float
    wake_threshold: float
    l1_status: str
    l2_ratio: float
    l2_status: str
    cooldown: float
    l4_status: str
    l5_status: str
    rejection_reason: str
    tail_silence_frames: int
    tail_required_frames: int
    final_wake_event: bool
    keyword: str | None
    energy_gate_passed: bool = False
    predicted_class_id: int | None = None
    predicted_keyword_id: str | None = None
    predicted_display_name: str | None = None
    top1_score: float = 0.0
    top2_class_id: int | None = None
    top2_keyword_id: str | None = None
    top2_display_name: str | None = None
    top2_score: float = 0.0
    margin: float = 0.0
    margin_threshold: float = 0.0
    background_score: float = 0.0
    accepted: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StreamingWakeWordEngine:
    def __init__(
        self,
        backend: WakeWordBackend,
        frame_ms: int = 30,
        pre_roll_seconds: float = 2.0,
        detection: DetectionLogic | None = None,
        tail_inference_seconds: float = 0.8,
        release_after_silence_frames: int | None = None,
    ):
        if pre_roll_seconds < 1.0:
            raise ValueError("Streaming pre-roll must be at least 1.0 second")
        if tail_inference_seconds < 0.8:
            raise ValueError("Tail inference must be at least 0.8 seconds")
        self.backend = backend
        self.energy_gate = AdaptiveEnergyGate()
        self.vad_gate = WebRTCVadGate(frame_ms=frame_ms)
        self.speech_gate = ConsecutiveSpeechGate(required_frames=3)
        self.pre_roll = PreRollRingBuffer(frame_ms=frame_ms, seconds=pre_roll_seconds)
        self.detection = detection or DetectionLogic()
        self.kws_active = False
        minimum_tail_frames = int(math.ceil(tail_inference_seconds * 1000.0 / frame_ms))
        requested_frames = (
            minimum_tail_frames
            if release_after_silence_frames is None
            else int(release_after_silence_frames)
        )
        self.release_after_silence_frames = max(minimum_tail_frames, requested_frames)
        self._active_silence_frames = 0

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
        inference = getattr(self.backend, "last_prediction", None) if scores else None
        decision: DetectionDecision = self.detection.update(scores, vad, now, inference=inference)
        if self.kws_active:
            self._active_silence_frames = 0 if vad else self._active_silence_frames + 1
            if (
                self._active_silence_frames >= self.release_after_silence_frames
                and not decision.wake_event
            ):
                self.kws_active = False
                self._active_silence_frames = 0
        decision_score = max(scores.values(), default=0.0)
        raw = (
            float(getattr(self.backend, "last_raw_score", decision_score))
            if scores
            else 0.0
        )
        if decision.wake_event:
            rejection = "ACCEPTED"
        elif not scores:
            rejection = "NO_NEW_SCORE"
        elif decision.rejection_reason in {"BACKGROUND_TOP1", "LOW_TOP1_SCORE", "LOW_MARGIN"}:
            rejection = decision.rejection_reason
        elif decision_score < self.detection.config.wake_threshold:
            rejection = "RAW_OR_SMOOTHED_SCORE_BELOW_THRESHOLD"
        elif not decision.l1_passed:
            rejection = "L1_CONSECUTIVE_SCORE_PENDING"
        elif not decision.l2_passed:
            rejection = "L2_BACKGROUND_RATIO_FAILED"
        elif decision.cooldown_remaining > 0.0:
            rejection = "L3_COOLDOWN_ACTIVE"
        elif not decision.l4_passed:
            rejection = "L4_ARBITRATION_FAILED"
        else:
            rejection = "L5_TRANSITION_PENDING"
        return RuntimeLog(
            energy=energy.rms,
            energy_dbfs=energy.dbfs,
            adaptive_threshold=energy.adaptive_threshold,
            vad=vad,
            speech_frame_count=speech_count,
            kws_active=self.kws_active,
            raw_wake_score=raw,
            decision_wake_score=decision_score,
            wake_threshold=self.detection.config.wake_threshold,
            l1_status=f"{decision.l1_streak}/{self.detection.config.consecutive_wake_frames}:{decision.l1_passed}",
            l2_ratio=decision.l2_ratio,
            l2_status=str(decision.l2_passed),
            cooldown=decision.cooldown_remaining,
            l4_status=str(decision.l4_passed),
            l5_status=decision.l5_state,
            rejection_reason=rejection,
            tail_silence_frames=self._active_silence_frames,
            tail_required_frames=self.release_after_silence_frames,
            final_wake_event=decision.wake_event,
            keyword=decision.keyword,
            energy_gate_passed=energy.passed,
            predicted_class_id=getattr(inference, "predicted_class_id", None),
            predicted_keyword_id=getattr(inference, "predicted_keyword_id", decision.keyword),
            predicted_display_name=getattr(inference, "predicted_display_name", decision.keyword),
            top1_score=float(getattr(inference, "top1_score", decision_score)),
            top2_class_id=getattr(inference, "top2_class_id", None),
            top2_keyword_id=getattr(inference, "top2_keyword_id", decision.top2_keyword),
            top2_display_name=getattr(inference, "top2_display_name", decision.top2_keyword),
            top2_score=float(getattr(inference, "top2_score", decision.top2_score)),
            margin=float(getattr(inference, "margin", decision.margin)),
            margin_threshold=float(getattr(self.backend, "margin_threshold", self.detection.config.arbitration_margin)),
            background_score=float(getattr(inference, "background_score", decision.background_score)),
            accepted=decision.wake_event,
        )


# Public delivery name; retain the original name for existing callers/tests.
WakeWordEngine = StreamingWakeWordEngine
