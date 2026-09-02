"""Detection episode latching shared by the packaged and legacy UIs."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .playback import WakePlaybackQueue


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    """One completed KWS episode, detached from mutable backend state."""

    detected_at: datetime
    keyword: str
    result: str
    raw_max_score: float
    decision_max_score: float
    threshold: float
    energy_result: str
    vad_result: str
    speech_result: str
    l1_result: str
    l2_result: str
    l3_result: str
    l4_result: str
    l5_result: str
    rejection_reason: str
    duration_seconds: float
    inference_windows: int
    peak_window: int | None


def request_final_wake_playback(
    playback: WakePlaybackQueue,
    snapshot: DetectionSnapshot,
    awake_wav: Path,
) -> bool:
    """Bind playback only to the one latched edge of a completed WAKE episode."""

    if snapshot.result != "WAKE":
        return False
    return playback.request(
        awake_wav,
        episode_id=snapshot.detected_at.isoformat(timespec="milliseconds"),
    )


class DetectionEpisodeTracker:
    """Latch WAKE/REJECT results while discarding background IGNORE traffic."""

    NO_SCORE_REASONS = {"", "NO_NEW_SCORE"}

    def __init__(self, *, max_history: int = 10, timeout_seconds: float = 5.0) -> None:
        self.max_history = int(max_history)
        self.timeout_seconds = float(timeout_seconds)
        self.latest: DetectionSnapshot | None = None
        self.history: deque[DetectionSnapshot] = deque(maxlen=self.max_history)
        self._active = False
        self._suppressed_until_inactive = False
        self._reset_episode_fields()

    def _reset_episode_fields(self) -> None:
        self._started_at = 0.0
        self._raw_max = 0.0
        self._decision_max = 0.0
        self._threshold = 0.0
        self._energy_passed = False
        self._vad_passed = False
        self._speech_passed = False
        self._l1_executed = False
        self._l1_passed = False
        self._l2_executed = False
        self._l2_passed = False
        self._cooldown_blocked = False
        self._l3_executed = False
        self._l5_executed = False
        self._l5_passed = False
        self._last_rejection_reason = ""
        self._inference_windows = 0
        self._peak_window: int | None = None

    def cancel_active(self) -> None:
        """Cancel an unfinished episode on manual stop without recording REJECT."""

        self._active = False
        self._suppressed_until_inactive = False
        self._reset_episode_fields()

    def clear_latest(self) -> None:
        self.latest = None

    def clear_history(self) -> None:
        self.history.clear()

    def update(
        self,
        state,  # noqa: ANN001
        *,
        keyword: str,
        now: float | None = None,
        wall_time: datetime | None = None,
    ) -> DetectionSnapshot | None:
        """Consume one RuntimeLog and return exactly one completed snapshot."""

        now = time.monotonic() if now is None else float(now)
        kws_active = bool(getattr(state, "kws_active", False))
        if self._suppressed_until_inactive:
            if not kws_active:
                self._suppressed_until_inactive = False
            return None
        if not self._active:
            if not kws_active:
                return None
            self._active = True
            self._reset_episode_fields()
            self._started_at = now

        self._accumulate(state)
        if bool(getattr(state, "final_wake_event", False)):
            return self._finalize("WAKE", keyword, now, wall_time, suppress=kws_active)
        if now - self._started_at >= self.timeout_seconds:
            self._last_rejection_reason = "REJECT_TIMEOUT"
            return self._finalize("REJECT", keyword, now, wall_time, suppress=kws_active)
        if not kws_active:
            if not self._last_rejection_reason:
                self._last_rejection_reason = "NO_VALID_MODEL_SCORE"
            return self._finalize("REJECT", keyword, now, wall_time, suppress=False)
        return None

    def _accumulate(self, state) -> None:  # noqa: ANN001
        energy = float(getattr(state, "energy", 0.0) or 0.0)
        adaptive_threshold = float(getattr(state, "adaptive_threshold", 0.0) or 0.0)
        self._energy_passed = self._energy_passed or energy >= adaptive_threshold
        self._vad_passed = self._vad_passed or bool(getattr(state, "vad", False))
        self._speech_passed = self._speech_passed or int(
            getattr(state, "speech_frame_count", 0) or 0
        ) >= 3
        self._threshold = float(
            getattr(state, "wake_threshold", self._threshold) or self._threshold
        )

        reason = str(getattr(state, "rejection_reason", "") or "")
        new_score = reason not in self.NO_SCORE_REASONS
        raw = float(getattr(state, "raw_wake_score", 0.0) or 0.0)
        decision = float(getattr(state, "decision_wake_score", raw) or 0.0)
        if new_score:
            self._inference_windows += 1
            self._l1_executed = True
            self._l2_executed = True
            self._l3_executed = True
            if decision > self._decision_max or self._peak_window is None:
                self._peak_window = self._inference_windows
            self._raw_max = max(self._raw_max, raw)
            self._decision_max = max(self._decision_max, decision)
        l1_text = str(getattr(state, "l1_status", ""))
        self._l1_passed = self._l1_passed or l1_text.endswith(":True")
        self._l2_passed = self._l2_passed or str(
            getattr(state, "l2_status", "False")
        ).lower() == "true"
        cooldown = float(getattr(state, "cooldown", 0.0) or 0.0)
        self._cooldown_blocked = (
            self._cooldown_blocked
            or cooldown > 0.0
            or reason in {"COOLDOWN", "L3_COOLDOWN_ACTIVE"}
        )
        l5 = str(getattr(state, "l5_status", "waiting") or "waiting")
        self._l5_executed = self._l5_executed or l5 != "waiting"
        self._l5_passed = self._l5_passed or l5 == "passed"
        if reason not in self.NO_SCORE_REASONS | {"FINAL_WAKE_EVENT"}:
            self._last_rejection_reason = reason

    def _finalize(
        self,
        result: str,
        keyword: str,
        now: float,
        wall_time: datetime | None,
        *,
        suppress: bool,
    ) -> DetectionSnapshot:
        snapshot = DetectionSnapshot(
            detected_at=wall_time or datetime.now(),
            keyword=keyword,
            result=result,
            raw_max_score=self._raw_max,
            decision_max_score=self._decision_max,
            threshold=self._threshold,
            energy_result="通过" if self._energy_passed else "未通过",
            vad_result="通过" if self._vad_passed else "未通过",
            speech_result="3/3" if self._speech_passed else "未达到",
            l1_result="通过" if self._l1_passed else "未通过" if self._l1_executed else "未执行",
            l2_result="通过" if self._l2_passed else "未通过" if self._l2_executed else "未执行",
            l3_result=(
                "冷却阻止"
                if self._cooldown_blocked
                else "通过"
                if self._l3_executed
                else "未执行"
            ),
            l4_result="单关键词无需竞争",
            l5_result=(
                "通过"
                if self._l5_passed or result == "WAKE"
                else "未通过"
                if self._l5_executed
                else "未执行"
            ),
            rejection_reason="" if result == "WAKE" else self._last_rejection_reason,
            duration_seconds=max(0.0, now - self._started_at),
            inference_windows=self._inference_windows,
            peak_window=self._peak_window,
        )
        self.latest = snapshot
        self.history.append(snapshot)
        self._active = False
        self._suppressed_until_inactive = suppress
        self._reset_episode_fields()
        return snapshot
