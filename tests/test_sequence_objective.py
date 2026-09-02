from __future__ import annotations

import numpy as np
import pytest

from wakeword_studio.training.sequence_objective import (
    build_sequence_target,
    consecutive_trigger,
    consecutive_trigger_score,
    decision_timestamps_ms,
    false_accept_count,
)


COMMON = {
    "original_feature_frames": 297,
    "tail_padding_feature_frames": 9,
    "frontend_window_ms": 30.0,
    "frontend_step_ms": 10.0,
    "model_stride": 3,
}


def test_decision_step_is_30_ms() -> None:
    timestamps = decision_timestamps_ms(**COMMON)
    assert timestamps[0] == pytest.approx(30.0)
    assert np.diff(timestamps).tolist() == pytest.approx([30.0] * (len(timestamps) - 1))
    assert timestamps[-1] == pytest.approx(3060.0)


def test_positive_is_causal_and_exactly_three_consecutive_frames() -> None:
    target = build_sequence_target(
        label="positive",
        phrase_start_ms=500.0,
        phrase_end_ms=1816.0,
        window_start_ms=-4.0,
        positive_frames=3,
        **COMMON,
    )
    positive_indices = np.flatnonzero(target.targets)
    assert positive_indices.tolist() == list(range(positive_indices[0], positive_indices[0] + 3))
    assert target.decision_timestamps_ms[positive_indices[0]] >= target.phrase_end_relative_ms
    assert not np.any(
        target.targets[target.decision_timestamps_ms < target.phrase_end_relative_ms]
    )


def test_overlength_terminal_window_uses_effective_phrase_start() -> None:
    target = build_sequence_target(
        label="positive",
        phrase_start_ms=900.0,
        phrase_end_ms=3850.0,
        window_start_ms=900.0,
        positive_frames=3,
        **COMMON,
    )
    assert target.phrase_start_relative_ms == pytest.approx(0.0)
    assert target.phrase_end_relative_ms == pytest.approx(2950.0)
    assert np.flatnonzero(target.targets).tolist() == [98, 99, 100]


@pytest.mark.parametrize("label", ["negative", "hard_negative", "ambient"])
def test_all_negative_classes_are_zero_for_entire_sequence(label: str) -> None:
    target = build_sequence_target(
        label=label,
        phrase_start_ms=None,
        phrase_end_ms=None,
        window_start_ms=0.0,
        positive_frames=3,
        **COMMON,
    )
    assert not np.any(target.targets)


def test_negative_phrase_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        build_sequence_target(
            label="hard_negative",
            phrase_start_ms=100.0,
            phrase_end_ms=800.0,
            window_start_ms=0.0,
            positive_frames=3,
            **COMMON,
        )


def test_consecutive_trigger_score_matches_three_frame_logic() -> None:
    scores = np.asarray([0.9, 0.8, 0.2, 0.7, 0.6, 0.5, 0.1])
    assert consecutive_trigger_score(scores, 3) == pytest.approx(0.5)
    assert consecutive_trigger(scores, threshold=0.5, consecutive_frames=3)
    assert not consecutive_trigger(scores, threshold=0.5001, consecutive_frames=3)


def test_false_accept_count_is_json_safe_native_int() -> None:
    value = false_accept_count(
        np.asarray([0.8, 0.2, 0.9], dtype=np.float32),
        ["hard_negative", "hard_negative", "negative"],
        group="hard_negative",
        threshold=0.5,
    )
    assert value == 1
    assert type(value) is int
