"""Shared audio entry points for training and inference frontends."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_audio_float32


def load_training_audio(path: Path) -> np.ndarray:
    """Return the canonical float32 waveform used before training features."""
    audio, _ = load_audio_float32(path)
    return audio


def load_inference_audio(path: Path) -> np.ndarray:
    """Return the same canonical float32 waveform used during inference."""
    audio, _ = load_audio_float32(path)
    return audio
