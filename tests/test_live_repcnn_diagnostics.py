from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from wakeword_studio.diagnostics.live_repcnn import (
    amplitude_statistics,
    plan_fixed_windows,
    read_pcm16_wav,
    safe_rms_normalize,
    score_windows,
    write_pcm16_wav,
    write_score_artifacts,
)


def test_four_second_window_count_and_coordinates_are_fixed() -> None:
    windows = plan_fixed_windows(4 * 16_000)
    assert len(windows) == 11
    assert [(row.index, row.start_ms, row.end_ms) for row in windows[:3]] == [
        (0, 0.0, 2000.0),
        (1, 200.0, 2200.0),
        (2, 400.0, 2400.0),
    ]
    assert windows[-1].start_ms == 2000.0
    assert windows[-1].end_ms == 4000.0
    assert all(row.trailing_padding_samples == 0 for row in windows)


def test_tail_anchor_and_short_wav_padding_are_deterministic() -> None:
    non_aligned = plan_fixed_windows(int(4.1 * 16_000))
    assert non_aligned[-2].start_ms == 2000.0
    assert non_aligned[-1].start_ms == 2100.0
    assert non_aligned[-1].end_ms == 4100.0
    short = plan_fixed_windows(16_000)
    assert len(short) == 1
    assert short[0].start_ms == 0.0
    assert short[0].end_ms == 2000.0
    assert short[0].trailing_padding_samples == 16_000


def test_recording_wav_is_16k_mono_pcm16(tmp_path: Path) -> None:
    source = np.arange(-1000, 1000, dtype=np.int16)
    path = tmp_path / "recording.wav"
    write_pcm16_wav(path, source)
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == len(source)
    loaded, rate = read_pcm16_wav(path)
    assert rate == 16_000
    np.testing.assert_array_equal(loaded, source)


def test_score_log_contains_every_window_and_live_offline_delta(tmp_path: Path) -> None:
    audio = np.arange(64_000, dtype=np.int16)

    def score(value: np.ndarray) -> float:
        return float(np.mean(value.astype(np.float64)) / 32768.0)

    report = score_windows(audio, live_score=score, offline_score=score)
    assert report["window_count"] == 11
    assert report["summary"]["max_abs_delta"] == 0.0
    assert len(report["summary"]["top5"]) == 5
    wrapper = {
        "window_scores": report,
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }
    write_score_artifacts(tmp_path, wrapper)
    saved = json.loads((tmp_path / "diagnostic_report.json").read_text(encoding="utf-8"))
    assert saved["v2_test_loaded"] is False
    assert saved["v1_external_test_loaded"] is False
    csv_lines = (tmp_path / "window_scores.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(csv_lines) == 12
    assert "live_score" in csv_lines[0] and "offline_score" in csv_lines[0]


def test_amplitude_and_safe_normalization_are_diagnostic_only() -> None:
    audio = np.tile(np.asarray([-1000, 1000], dtype=np.int16), 100)
    stats = amplitude_statistics(audio)
    assert stats["rms_pcm16"] == pytest.approx(1000.0)
    assert stats["peak_pcm16"] == 1000.0
    normalized, metadata = safe_rms_normalize(audio, target_rms_pcm16=2000.0)
    assert amplitude_statistics(normalized)["rms_pcm16"] == pytest.approx(2000.0)
    assert metadata["gain"] == pytest.approx(2.0)
    assert metadata["clipped_fraction"] == 0.0
    np.testing.assert_array_equal(audio[:2], [-1000, 1000])
