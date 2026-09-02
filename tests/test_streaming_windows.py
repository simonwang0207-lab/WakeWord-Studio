from __future__ import annotations

import numpy as np
import pytest

from wakeword_studio.training.streaming_windows import (
    extract_streaming_window,
    plan_streaming_window,
)


def test_positive_window_contains_phrase_and_is_deterministic() -> None:
    kwargs = dict(
        record_id="positive-1",
        label="positive",
        duration_seconds=5.0,
        phrase_start_ms=3063.25,
        phrase_end_ms=4299.0,
        phrase_placement="back",
        window_ms=4000.0,
        seed=20260829,
    )
    first = plan_streaming_window(**kwargs)
    second = plan_streaming_window(**kwargs)
    assert first == second
    assert first.alignment_ok
    assert first.window_start_ms <= 3063.25
    assert first.window_end_ms >= 4299.0


def test_negative_phrase_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="contains a target phrase"):
        plan_streaming_window(
            record_id="negative-1",
            label="negative",
            duration_seconds=2.0,
            phrase_start_ms=100.0,
            phrase_end_ms=800.0,
            phrase_placement=None,
            window_ms=4000.0,
            seed=1,
        )


def test_extract_window_has_exact_size_and_padding() -> None:
    plan = plan_streaming_window(
        record_id="ambient-1",
        label="ambient",
        duration_seconds=1.0,
        phrase_start_ms=None,
        phrase_end_ms=None,
        phrase_placement=None,
        window_ms=4000.0,
        seed=7,
    )
    source = np.ones(16000, dtype=np.float32)
    result = extract_streaming_window(source, plan, sample_rate_hz=16000, window_ms=4000.0)
    assert result.shape == (64000,)
    assert np.count_nonzero(result) == 16000


def test_overlength_positive_uses_terminal_decision_window() -> None:
    plan = plan_streaming_window(
        record_id="slow-positive",
        label="positive",
        duration_seconds=3.907,
        phrase_start_ms=0.0,
        phrase_end_ms=3827.0,
        phrase_placement="middle",
        window_ms=3000.0,
        seed=20260829,
    )
    assert plan.alignment_ok
    assert plan.overlength_terminal_decision_window
    assert not plan.full_phrase_contained
    assert plan.window_start_ms == pytest.approx(827.0)
    assert plan.window_end_ms == pytest.approx(3827.0)
    assert plan.effective_phrase_start_ms == pytest.approx(827.0)
