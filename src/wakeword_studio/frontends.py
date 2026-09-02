"""Shared audio entry points for training and inference frontends."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_audio_float32


def generate_pymicro_features(
    audio_samples: np.ndarray,
    *,
    window_step_ms: int = 10,
) -> np.ndarray:
    """Generate the deployment-compatible 40-channel microfrontend features."""

    from pymicro_features import MicroFrontend

    if window_step_ms < 10 or window_step_ms % 10:
        raise ValueError("pymicro-features requires a window step divisible by 10 ms")
    samples = np.asarray(audio_samples).reshape(-1)
    if samples.dtype in (np.float32, np.float64):
        samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    elif samples.dtype != np.int16:
        samples = samples.astype(np.int16)

    frontend = MicroFrontend()
    process_samples = getattr(
        frontend,
        "ProcessSamples",
        getattr(frontend, "process_samples", None),
    )
    if process_samples is None:
        raise RuntimeError("Unsupported pymicro-features MicroFrontend API")

    audio_bytes = samples.tobytes()
    audio_index = 0
    packet_bytes = 160 * 2
    features: list[object] = []
    while audio_index + packet_bytes < len(audio_bytes):
        result = process_samples(audio_bytes[audio_index : audio_index + packet_bytes])
        consumed = int(result.samples_read) * 2
        if consumed <= 0:
            raise RuntimeError("pymicro-features did not consume input audio")
        audio_index += consumed
        if result.features:
            features.append(result.features)
    if not features:
        return np.zeros((0, 40), dtype=np.float32)
    values = np.asarray(features, dtype=np.float32).reshape(-1, 40)
    return values[:: window_step_ms // 10]


class PymicroFrontend:
    """Runtime frontend that never imports TensorFlow."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        window_size_ms: int = 30,
        window_step_ms: int = 20,
        num_channels: int = 40,
    ) -> None:
        if sample_rate != 16_000 or window_size_ms != 30 or num_channels != 40:
            raise ValueError("Published models require 16 kHz, 30 ms windows and 40 channels")
        if window_step_ms < 10 or window_step_ms % 10:
            raise ValueError("window_step_ms must be divisible by 10")
        self.window_step_ms = int(window_step_ms)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        values = np.asarray(audio)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2:
            raise ValueError("Audio must have shape [samples] or [batch, samples]")
        rows = [
            generate_pymicro_features(row, window_step_ms=self.window_step_ms)
            for row in values
        ]
        if not rows:
            return np.zeros((0, 0, 40), dtype=np.float32)
        max_frames = max(row.shape[0] for row in rows)
        padded = [
            np.pad(row, ((max_frames - row.shape[0], 0), (0, 0)))
            if row.shape[0] < max_frames
            else row
            for row in rows
        ]
        return np.stack(padded).astype(np.float32, copy=False)


def load_training_audio(path: Path) -> np.ndarray:
    """Return the canonical float32 waveform used before training features."""
    audio, _ = load_audio_float32(path)
    return audio


def load_inference_audio(path: Path) -> np.ndarray:
    """Return the same canonical float32 waveform used during inference."""
    audio, _ = load_audio_float32(path)
    return audio
