"""Sequence targets and deployment-aligned trigger helpers for streaming KWS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceTarget:
    """Frame timestamps and binary targets for one fixed streaming feature clip."""

    label: str
    decision_timestamps_ms: np.ndarray
    targets: np.ndarray
    phrase_start_relative_ms: float | None
    phrase_end_relative_ms: float | None
    first_positive_frame: int | None
    positive_frame_count: int


def decision_timestamps_ms(
    *,
    original_feature_frames: int,
    tail_padding_feature_frames: int,
    frontend_window_ms: float,
    frontend_step_ms: float,
    model_stride: int,
) -> np.ndarray:
    """Return causal decision times for a left-padded feature sequence.

    The first decision consumes the first real frontend frame. Subsequent
    decisions advance by ``model_stride`` frontend frames. Tail padding lets a
    positive phrase at the end of a clip receive the complete confirmation
    region without using future speech.
    """

    if original_feature_frames <= 0:
        raise ValueError("original_feature_frames must be positive")
    if tail_padding_feature_frames < 0:
        raise ValueError("tail_padding_feature_frames must be non-negative")
    if frontend_window_ms <= 0 or frontend_step_ms <= 0 or model_stride <= 0:
        raise ValueError("frontend timing and model_stride must be positive")

    final_feature_index = original_feature_frames + tail_padding_feature_frames - 1
    feature_indices = np.arange(0, final_feature_index + 1, model_stride, dtype=np.int32)
    return frontend_window_ms + feature_indices.astype(np.float64) * frontend_step_ms


def build_sequence_target(
    *,
    label: str,
    phrase_start_ms: float | None,
    phrase_end_ms: float | None,
    window_start_ms: float,
    original_feature_frames: int,
    tail_padding_feature_frames: int,
    frontend_window_ms: float,
    frontend_step_ms: float,
    model_stride: int,
    positive_frames: int,
) -> SequenceTarget:
    """Construct a causal frame target.

    Positive records receive exactly ``positive_frames`` consecutive ones,
    beginning at the first decision at or after the complete phrase end. Every
    other frame, including every frame of all negative classes, is zero.
    """

    if positive_frames <= 0:
        raise ValueError("positive_frames must be positive")
    timestamps = decision_timestamps_ms(
        original_feature_frames=original_feature_frames,
        tail_padding_feature_frames=tail_padding_feature_frames,
        frontend_window_ms=frontend_window_ms,
        frontend_step_ms=frontend_step_ms,
        model_stride=model_stride,
    )
    targets = np.zeros(len(timestamps), dtype=np.float32)
    negative_labels = {"negative", "hard_negative", "ambient"}

    if label in negative_labels:
        if phrase_start_ms is not None or phrase_end_ms is not None:
            raise ValueError(f"{label} sequence must not contain target phrase metadata")
        return SequenceTarget(label, timestamps, targets, None, None, None, 0)
    if label != "positive":
        raise ValueError(f"Unsupported label: {label}")
    if phrase_start_ms is None or phrase_end_ms is None:
        raise ValueError("positive sequence requires phrase_start_ms and phrase_end_ms")

    phrase_start_relative = float(phrase_start_ms) - float(window_start_ms)
    phrase_end_relative = float(phrase_end_ms) - float(window_start_ms)
    if not 0 <= phrase_start_relative < phrase_end_relative:
        raise ValueError("invalid positive phrase interval relative to streaming window")

    candidates = np.flatnonzero(timestamps + 1e-6 >= phrase_end_relative)
    if not len(candidates):
        raise ValueError("no causal decision frame exists at or after phrase end")
    first = int(candidates[0])
    stop = first + positive_frames
    if stop > len(targets):
        raise ValueError("tail padding is insufficient for the positive confirmation region")
    targets[first:stop] = 1.0
    if np.any(targets[timestamps + 1e-6 < phrase_end_relative] != 0):
        raise AssertionError("positive target leaked before complete phrase end")
    return SequenceTarget(
        label,
        timestamps,
        targets,
        phrase_start_relative,
        phrase_end_relative,
        first,
        positive_frames,
    )


def consecutive_trigger_score(scores: np.ndarray, consecutive_frames: int) -> float:
    """Maximum floor of a consecutive run.

    Thresholding this scalar at ``q`` is exactly equivalent to asking whether
    any run of ``consecutive_frames`` scores is entirely at least ``q``.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if consecutive_frames <= 0:
        raise ValueError("consecutive_frames must be positive")
    if len(values) < consecutive_frames:
        return 0.0
    return float(
        max(
            np.min(values[offset : offset + consecutive_frames])
            for offset in range(len(values) - consecutive_frames + 1)
        )
    )


def consecutive_trigger(
    scores: np.ndarray, *, threshold: float, consecutive_frames: int
) -> bool:
    return consecutive_trigger_score(scores, consecutive_frames) >= float(threshold)


def false_accept_count(
    scores: np.ndarray,
    groups: list[str],
    *,
    group: str,
    threshold: float,
) -> int:
    """Return a JSON-safe native integer for one negative group."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(groups):
        raise ValueError("scores and groups must have the same length")
    return int(
        sum(
            bool(score >= float(threshold))
            for score, item_group in zip(values, groups)
            if item_group == group
        )
    )
