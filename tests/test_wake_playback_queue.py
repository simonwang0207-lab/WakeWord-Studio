from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from phase5.scripts.wakeword_studio_demo import (
    DetectionEpisodeTracker,
    request_final_wake_playback,
)
from tests.test_detection_episode_latching import runtime_state
from wakeword_studio.runtime.playback import WakePlaybackQueue


def test_five_independent_wake_episodes_play_five_times(tmp_path: Path) -> None:
    awake_wav = tmp_path / "awake.wav"
    awake_wav.write_bytes(b"test-audio-placeholder")
    logs: list[str] = []
    played: list[Path] = []
    playback = WakePlaybackQueue(logs.append, player=played.append)
    tracker = DetectionEpisodeTracker()
    wake_count = 0

    for episode in range(5):
        start = float(episode * 10)
        tracker.update(
            runtime_state(active=True, raw=0.5, decision=0.5, vad=True, speech_frames=3),
            keyword="你好，青小甲",
            now=start,
        )
        snapshot = tracker.update(
            runtime_state(
                active=True,
                raw=0.7,
                decision=0.7,
                reason="FINAL_WAKE_EVENT",
                wake=True,
                l1="2/2:True",
                l2="True",
                l5="passed",
            ),
            keyword="你好，青小甲",
            now=start + 0.2,
            wall_time=datetime(2026, 8, 31) + timedelta(seconds=episode),
        )
        assert snapshot is not None and snapshot.result == "WAKE"
        wake_count += 1
        assert request_final_wake_playback(
            playback,
            snapshot,
            awake_wav,
        )
        # End the episode so the next WAKE is a genuinely independent edge.
        tracker.update(runtime_state(active=False), keyword="你好，青小甲", now=start + 1.0)

    assert playback.wait_until_idle(timeout=2.0)
    playback.close(wait=True)

    assert wake_count == 5
    assert playback.request_count == 5
    assert playback.started_count == 5
    assert playback.playback_count == 5
    assert playback.skipped_count == 0
    assert len(played) == 5
    assert sum(line.startswith("PLAYBACK_REQUESTED") for line in logs) == 5
    assert sum(line.startswith("PLAYBACK_STARTED") for line in logs) == 5
    assert sum(line.startswith("PLAYBACK_FINISHED") for line in logs) == 5
    assert not any(line.startswith("PLAYBACK_SKIPPED") for line in logs)


def test_overlapping_requests_are_queued_in_fifo_order(tmp_path: Path) -> None:
    awake_wav = tmp_path / "awake.wav"
    awake_wav.write_bytes(b"test-audio-placeholder")
    played: list[str] = []
    playback = WakePlaybackQueue(lambda _line: None, player=lambda _path: played.append("played"))

    for episode in range(5):
        assert playback.request(awake_wav, episode_id=str(episode))

    assert playback.wait_until_idle(timeout=2.0)
    playback.close(wait=True)
    assert played == ["played"] * 5
    assert playback.playback_count == 5


def test_missing_audio_is_reported_as_skipped(tmp_path: Path) -> None:
    logs: list[str] = []
    playback = WakePlaybackQueue(logs.append, player=lambda _path: None)

    assert not playback.request(tmp_path / "missing.wav", episode_id="missing")
    playback.close(wait=True)

    assert playback.skipped_count == 1
    assert logs == ["PLAYBACK_SKIPPED episode=missing reason=file_not_found"]
