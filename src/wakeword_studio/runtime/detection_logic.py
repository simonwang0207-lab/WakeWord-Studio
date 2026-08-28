"""Five-layer wake-word decision state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class DetectionConfig:
    wake_threshold: float = 0.55
    consecutive_wake_frames: int = 3
    peak_background_ratio: float = 1.35
    background_alpha: float = 0.97
    cooldown_seconds: float = 2.0
    arbitration_margin: float = 0.05
    pre_silence_frames: int = 3
    post_silence_frames: int = 2


@dataclass(slots=True)
class DetectionDecision:
    keyword: str | None
    raw_score: float
    l1_streak: int
    l1_passed: bool
    l2_ratio: float
    l2_passed: bool
    cooldown_remaining: float
    l4_passed: bool
    l5_state: str
    wake_event: bool


class DetectionLogic:
    """L1 score streak, L2 ratio, L3 cooldown, L4 arbitration, L5 transitions."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self.streaks: dict[str, int] = {}
        self.background: dict[str, float] = {}
        self.cooldown_until = 0.0
        self.silence_frames = self.config.pre_silence_frames
        self.had_pre_silence = True
        self.pending_keyword: str | None = None
        self.pending_post_silence = 0

    def update(self, scores: dict[str, float], speech: bool, now: float | None = None) -> DetectionDecision:
        now = time.monotonic() if now is None else now
        if speech:
            if self.silence_frames >= self.config.pre_silence_frames:
                self.had_pre_silence = True
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        # L5 is driven by the acoustic transition, not by whether a streaming
        # model happens to keep producing scores during trailing silence.
        if self.pending_keyword and not speech:
            return self._maybe_finalize_pending(speech, now, max(0.0, self.cooldown_until - now))
        if self.pending_keyword and speech:
            self.pending_post_silence = 0

        keyword, score, second = self._arbitrate(scores)
        cooldown = max(0.0, self.cooldown_until - now)
        if keyword is None:
            return self._maybe_finalize_pending(speech, now, cooldown)

        # A first high peak must not define its own background and force ratio=1.
        # Until low-score frames establish an EMA, use a conservative fraction
        # of the wake threshold as the baseline.
        previous_bg = self.background.get(
            keyword, max(self.config.wake_threshold * 0.25, 1e-4)
        )
        ratio = score / max(previous_bg, 1e-6)
        if score < self.config.wake_threshold:
            self.background[keyword] = self.config.background_alpha * previous_bg + (1.0 - self.config.background_alpha) * score
        streak = self.streaks.get(keyword, 0) + 1 if score >= self.config.wake_threshold else 0
        self.streaks = {name: (streak if name == keyword else 0) for name in set(self.streaks) | {keyword}}
        l1 = streak >= self.config.consecutive_wake_frames
        l2 = ratio >= self.config.peak_background_ratio
        l4 = score - second >= self.config.arbitration_margin

        if l1 and l2 and cooldown == 0.0 and l4 and self.had_pre_silence:
            self.pending_keyword = keyword
            self.pending_post_silence = 0
            self.had_pre_silence = False
        decision = DetectionDecision(keyword, score, streak, l1, ratio, l2, cooldown, l4, "pending_post_silence" if self.pending_keyword else "waiting", False)
        if self.config.post_silence_frames == 0 and self.pending_keyword:
            decision.wake_event = True
            decision.l5_state = "passed"
            self._enter_cooldown(now)
        return decision

    def _maybe_finalize_pending(self, speech: bool, now: float, cooldown: float) -> DetectionDecision:
        event = False
        l5 = "waiting"
        keyword = self.pending_keyword
        if keyword:
            self.pending_post_silence = 0 if speech else self.pending_post_silence + 1
            l5 = f"post_silence_{self.pending_post_silence}/{self.config.post_silence_frames}"
            if self.pending_post_silence >= self.config.post_silence_frames:
                event = True
                l5 = "passed"
                self._enter_cooldown(now)
        return DetectionDecision(keyword, 0.0, 0, False, 0.0, False, cooldown, True, l5, event)

    def _enter_cooldown(self, now: float) -> None:
        self.cooldown_until = now + self.config.cooldown_seconds
        self.pending_keyword = None
        self.pending_post_silence = 0
        self.streaks.clear()

    @staticmethod
    def _arbitrate(scores: dict[str, float]) -> tuple[str | None, float, float]:
        if not scores:
            return None, 0.0, 0.0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[0][0], float(ranked[0][1]), float(ranked[1][1]) if len(ranked) > 1 else 0.0
