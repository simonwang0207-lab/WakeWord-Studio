"""Canonical audio contract and conversion helpers for WakeWord Studio."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2
TARGET_PCM_SUBTYPE = "PCM_16"
KOKORO_OUTPUT_SAMPLE_RATE_HZ = 24_000


@dataclass(frozen=True, slots=True)
class AudioFileInfo:
    original_sample_rate_hz: int
    original_channels: int
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    number_of_samples: int
    duration_seconds: float


def resample_audio(
    audio: np.ndarray,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Convert mono float audio to the requested rate with polyphase filtering."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_sample_rate_hz == target_sample_rate_hz:
        return samples
    divisor = math.gcd(source_sample_rate_hz, target_sample_rate_hz)
    converted = resample_poly(
        samples,
        target_sample_rate_hz // divisor,
        source_sample_rate_hz // divisor,
    )
    return np.asarray(converted, dtype=np.float32)


def load_audio_float32(path: Path) -> tuple[np.ndarray, int]:
    """Load any SoundFile-supported WAV as standardized mono 16 kHz float32."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    standardized = resample_audio(mono, int(source_rate))
    return np.clip(standardized, -1.0, 1.0), TARGET_SAMPLE_RATE_HZ


def standardize_wav(source: Path, destination: Path) -> AudioFileInfo:
    """Write a new 16 kHz mono PCM16 WAV without modifying the source file."""
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite standardized audio: {destination}")
    source_info = sf.info(source)
    audio, _ = load_audio_float32(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE, format="WAV")

    with wave.open(str(destination), "rb") as handle:
        sample_rate_hz = handle.getframerate()
        channels = handle.getnchannels()
        sample_width_bytes = handle.getsampwidth()
        number_of_samples = handle.getnframes()
    if (
        sample_rate_hz != TARGET_SAMPLE_RATE_HZ
        or channels != TARGET_CHANNELS
        or sample_width_bytes != TARGET_SAMPLE_WIDTH_BYTES
    ):
        raise RuntimeError(f"Standardized WAV failed validation: {destination}")
    return AudioFileInfo(
        original_sample_rate_hz=int(source_info.samplerate),
        original_channels=int(source_info.channels),
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        number_of_samples=number_of_samples,
        duration_seconds=number_of_samples / sample_rate_hz,
    )
