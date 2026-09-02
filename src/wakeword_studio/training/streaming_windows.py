"""Deterministic, manifest-aligned fixed windows for streaming KWS training."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StreamingWindow:
    record_id: str
    label: str
    window_start_ms: float
    window_end_ms: float
    phrase_start_ms: float | None
    phrase_end_ms: float | None
    effective_phrase_start_ms: float | None
    effective_phrase_end_ms: float | None
    phrase_placement: str | None
    target_phrase_center_fraction: float | None
    leading_padding_ms: float
    trailing_padding_ms: float
    full_phrase_contained: bool
    overlength_terminal_decision_window: bool
    alignment_ok: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _unit_random(seed: int, record_id: str, purpose: str) -> float:
    payload = f"{seed}:{record_id}:{purpose}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64 - 1)


def _placement_fraction(seed: int, record_id: str, placement: str | None) -> float:
    bands = {
        "front": (0.22, 0.36),
        "middle": (0.42, 0.58),
        "back": (0.64, 0.78),
    }
    low, high = bands.get(str(placement), (0.42, 0.58))
    return low + (high - low) * _unit_random(seed, record_id, "positive-placement")


def plan_streaming_window(
    *,
    record_id: str,
    label: str,
    duration_seconds: float,
    phrase_start_ms: float | None,
    phrase_end_ms: float | None,
    phrase_placement: str | None,
    window_ms: float,
    seed: int,
) -> StreamingWindow:
    """Plan one reproducible fixed window without changing the source WAV.

    Positive windows must contain the complete annotated phrase interval. Negative,
    hard-negative, and ambient records must not carry a target phrase interval.
    Window coordinates are relative to the source WAV; negative starts and ends
    beyond the WAV duration explicitly represent zero padding.
    """

    duration_ms = float(duration_seconds) * 1000.0
    if duration_ms <= 0 or window_ms <= 0:
        raise ValueError(f"{record_id}: duration and window must be positive")

    target_fraction: float | None = None
    if label == "positive":
        if phrase_start_ms is None or phrase_end_ms is None:
            raise ValueError(f"{record_id}: positive record lacks phrase interval")
        phrase_start_ms = float(phrase_start_ms)
        phrase_end_ms = float(phrase_end_ms)
        if not 0 <= phrase_start_ms < phrase_end_ms <= duration_ms + 1.0:
            raise ValueError(f"{record_id}: invalid phrase interval")
        phrase_span_ms = phrase_end_ms - phrase_start_ms
        overlength = phrase_span_ms > window_ms + 1e-6
        if overlength:
            # A causal streaming detector makes its decision at the phrase end. For
            # the rare phrase interval longer than the fixed model context, retain
            # the complete terminal receptive field instead of changing architecture.
            window_start_ms = phrase_end_ms - window_ms
            effective_phrase_start_ms = window_start_ms
            effective_phrase_end_ms = phrase_end_ms
            full_phrase_contained = False
        else:
            target_fraction = _placement_fraction(seed, record_id, phrase_placement)
            phrase_center_ms = (phrase_start_ms + phrase_end_ms) / 2.0
            desired_start_ms = phrase_center_ms - target_fraction * window_ms
            minimum_start_ms = phrase_end_ms - window_ms
            maximum_start_ms = phrase_start_ms
            window_start_ms = min(max(desired_start_ms, minimum_start_ms), maximum_start_ms)
            effective_phrase_start_ms = phrase_start_ms
            effective_phrase_end_ms = phrase_end_ms
            full_phrase_contained = True
        alignment_ok = (
            effective_phrase_start_ms >= window_start_ms - 1e-6
            and effective_phrase_end_ms <= window_start_ms + window_ms + 1e-6
        )
    else:
        if phrase_start_ms is not None or phrase_end_ms is not None:
            raise ValueError(f"{record_id}: negative-class record contains a target phrase interval")
        if duration_ms >= window_ms:
            window_start_ms = _unit_random(seed, record_id, "negative-crop") * (
                duration_ms - window_ms
            )
        else:
            padding_ms = window_ms - duration_ms
            window_start_ms = -_unit_random(seed, record_id, "negative-padding") * padding_ms
        alignment_ok = True
        effective_phrase_start_ms = None
        effective_phrase_end_ms = None
        full_phrase_contained = False
        overlength = False

    window_end_ms = window_start_ms + window_ms
    return StreamingWindow(
        record_id=record_id,
        label=label,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        phrase_start_ms=phrase_start_ms,
        phrase_end_ms=phrase_end_ms,
        effective_phrase_start_ms=effective_phrase_start_ms,
        effective_phrase_end_ms=effective_phrase_end_ms,
        phrase_placement=phrase_placement,
        target_phrase_center_fraction=target_fraction,
        leading_padding_ms=max(0.0, -window_start_ms),
        trailing_padding_ms=max(0.0, window_end_ms - duration_ms),
        full_phrase_contained=full_phrase_contained,
        overlength_terminal_decision_window=overlength,
        alignment_ok=alignment_ok,
    )


def extract_streaming_window(
    audio: np.ndarray,
    plan: StreamingWindow,
    *,
    sample_rate_hz: int,
    window_ms: float,
) -> np.ndarray:
    """Apply a planned crop/pad operation and return an exact-length float32 clip."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    window_samples = int(round(window_ms * sample_rate_hz / 1000.0))
    start_sample = int(round(plan.window_start_ms * sample_rate_hz / 1000.0))
    source_start = max(0, start_sample)
    source_end = min(len(audio), start_sample + window_samples)
    destination_start = source_start - start_sample
    result = np.zeros(window_samples, dtype=np.float32)
    if source_end > source_start:
        count = source_end - source_start
        result[destination_start : destination_start + count] = audio[source_start:source_end]
    return result
