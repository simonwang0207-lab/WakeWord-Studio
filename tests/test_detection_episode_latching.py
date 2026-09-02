from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from phase5.scripts.wakeword_studio_demo import DetectionEpisodeTracker


def runtime_state(
    *,
    active: bool,
    raw: float = 0.0,
    decision: float = 0.0,
    reason: str = "NO_NEW_SCORE",
    wake: bool = False,
    vad: bool = False,
    speech_frames: int = 0,
    l1: str = "0/2:False",
    l2: str = "False",
    l5: str = "waiting",
    cooldown: float = 0.0,
):
    return SimpleNamespace(
        kws_active=active,
        raw_wake_score=raw,
        decision_wake_score=decision,
        wake_threshold=0.84375,
        energy=500.0 if active or vad else 0.0,
        adaptive_threshold=100.0,
        vad=vad,
        speech_frame_count=speech_frames,
        l1_status=l1,
        l2_status=l2,
        l2_ratio=2.0 if l2 == "True" else 0.5,
        cooldown=cooldown,
        l4_status="True",
        l5_status=l5,
        rejection_reason=reason,
        tail_silence_frames=0,
        tail_required_frames=27,
        final_wake_event=wake,
        keyword="你好，青小甲" if wake else None,
    )


def test_ignore_traffic_never_creates_episode_or_history() -> None:
    tracker = DetectionEpisodeTracker()
    # 40,000 x 30 ms = 20 minutes of background/silence.
    for index in range(40_000):
        assert tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=index * 0.03) is None
    assert tracker.latest is None
    assert not tracker.history


def test_one_success_latches_one_snapshot_with_episode_maxima() -> None:
    tracker = DetectionEpisodeTracker()
    assert tracker.update(
        runtime_state(active=True, raw=0.55, decision=0.50, reason="L1_CONSECUTIVE_SCORE_PENDING", vad=True, speech_frames=3),
        keyword="你好，青小甲", now=10.0,
    ) is None
    assert tracker.update(
        runtime_state(active=True, raw=0.95, decision=0.90, reason="L5_TRANSITION_PENDING", vad=True, speech_frames=4, l1="2/2:True", l2="True", l5="pending_post_silence"),
        keyword="你好，青小甲", now=10.2,
    ) is None
    snapshot = tracker.update(
        runtime_state(active=True, raw=0.80, decision=0.75, reason="FINAL_WAKE_EVENT", wake=True, speech_frames=0, l1="0/2:False", l2="False", l5="passed"),
        keyword="你好，青小甲", now=10.4, wall_time=datetime(2026, 8, 31, 1, 58, 42),
    )
    assert snapshot is not None and snapshot.result == "WAKE"
    assert snapshot.raw_max_score == pytest.approx(0.95)
    assert snapshot.decision_max_score == pytest.approx(0.90)
    assert snapshot.inference_windows == 3
    assert snapshot.peak_window == 2
    assert snapshot.l1_result == snapshot.l2_result == snapshot.l3_result == snapshot.l5_result == "通过"
    assert len(tracker.history) == 1
    # KWS can remain active after WAKE; suppression prevents duplicate snapshots.
    for offset in range(20):
        assert tracker.update(runtime_state(active=True, reason="NO_NEW_SCORE"), keyword="你好，青小甲", now=10.5 + offset * 0.03) is None
    assert len(tracker.history) == 1


def test_one_tail_end_reject_latches_once_and_survives_backend_zero_state() -> None:
    tracker = DetectionEpisodeTracker()
    tracker.update(
        runtime_state(active=True, raw=0.52, decision=0.51, reason="RAW_OR_SMOOTHED_SCORE_BELOW_THRESHOLD", vad=True, speech_frames=3),
        keyword="你好，青小甲", now=20.0,
    )
    tracker.update(runtime_state(active=True), keyword="你好，青小甲", now=20.5)
    snapshot = tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=20.8)
    assert snapshot is not None and snapshot.result == "REJECT"
    assert snapshot.raw_max_score == pytest.approx(0.52)
    assert snapshot.decision_max_score == pytest.approx(0.51)
    assert snapshot.rejection_reason == "RAW_OR_SMOOTHED_SCORE_BELOW_THRESHOLD"
    # Inactive zero-score frames model backend reset; the immutable snapshot is unchanged.
    for index in range(5):
        assert tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=21.0 + index) is None
    assert tracker.latest is snapshot
    assert tracker.latest.raw_max_score == pytest.approx(0.52)
    assert len(tracker.history) == 1


def test_next_episode_replaces_latest_and_history_is_bounded() -> None:
    tracker = DetectionEpisodeTracker(max_history=10)
    for episode in range(12):
        start = episode * 10.0
        tracker.update(
            runtime_state(active=True, raw=episode / 20, decision=episode / 20, reason="RAW_OR_SMOOTHED_SCORE_BELOW_THRESHOLD", vad=True, speech_frames=3),
            keyword="你好，青小甲", now=start,
        )
        result = tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=start + 1.0)
        assert result is not None
    assert tracker.latest is not None and tracker.latest.decision_max_score == pytest.approx(11 / 20)
    assert len(tracker.history) == 10
    assert tracker.history[0].decision_max_score == pytest.approx(2 / 20)


def test_timeout_rejects_once_and_suppresses_until_kws_inactive() -> None:
    tracker = DetectionEpisodeTracker(timeout_seconds=5.0)
    tracker.update(runtime_state(active=True, vad=True, speech_frames=3), keyword="你好，青小甲", now=0.0)
    snapshot = tracker.update(runtime_state(active=True), keyword="你好，青小甲", now=5.0)
    assert snapshot is not None and snapshot.rejection_reason == "REJECT_TIMEOUT"
    for second in range(6, 12):
        assert tracker.update(runtime_state(active=True), keyword="你好，青小甲", now=float(second)) is None
    assert len(tracker.history) == 1
    assert tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=12.0) is None
    tracker.update(runtime_state(active=True, vad=True, speech_frames=3), keyword="你好，青小甲", now=13.0)
    assert tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=14.0) is not None
    assert len(tracker.history) == 2


def test_clear_latest_and_history_are_independent() -> None:
    tracker = DetectionEpisodeTracker()
    tracker.update(runtime_state(active=True, vad=True, speech_frames=3), keyword="你好，青小甲", now=0.0)
    tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=1.0)
    assert tracker.latest is not None and len(tracker.history) == 1
    tracker.clear_latest()
    assert tracker.latest is None and len(tracker.history) == 1
    tracker.clear_history()
    assert tracker.latest is None and len(tracker.history) == 0


def test_cancel_active_preserves_completed_latest_and_history() -> None:
    tracker = DetectionEpisodeTracker()
    tracker.update(runtime_state(active=True, vad=True, speech_frames=3), keyword="你好，青小甲", now=0.0)
    completed = tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=1.0)
    tracker.update(runtime_state(active=True, vad=True, speech_frames=3), keyword="你好，青小甲", now=2.0)
    tracker.cancel_active()  # Equivalent to the Stop Listening UI action.
    assert tracker.latest is completed
    assert list(tracker.history) == [completed]
