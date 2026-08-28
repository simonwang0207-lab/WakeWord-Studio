"""P1 energy gate, P2 real WebRTC VAD, and independent P3 speech gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio import TARGET_SAMPLE_RATE_HZ


@dataclass(slots=True)
class EnergyState:
    rms: float
    dbfs: float
    adaptive_threshold: float
    passed: bool


class AdaptiveEnergyGate:
    def __init__(self, initial_noise_rms: float = 120.0, ratio: float = 2.5, alpha: float = 0.97, minimum_rms: float = 80.0):
        self.noise_rms = initial_noise_rms
        self.ratio = ratio
        self.alpha = alpha
        self.minimum_rms = minimum_rms

    def process(self, pcm16: np.ndarray) -> EnergyState:
        samples = np.asarray(pcm16, dtype=np.float32)
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
        threshold = max(self.minimum_rms, self.noise_rms * self.ratio)
        passed = rms >= threshold
        if not passed:
            self.noise_rms = self.alpha * self.noise_rms + (1.0 - self.alpha) * rms
        dbfs = 20.0 * np.log10(max(rms, 1e-9) / 32768.0)
        return EnergyState(rms, float(dbfs), threshold, passed)


class WebRTCVadGate:
    VALID_FRAME_MS = {10, 20, 30}

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE_HZ, frame_ms: int = 30, aggressiveness: int = 2):
        if frame_ms not in self.VALID_FRAME_MS:
            raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30")
        import webrtcvad

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.expected_samples = sample_rate * frame_ms // 1000
        self.vad = webrtcvad.Vad(aggressiveness)

    def process(self, pcm16: np.ndarray) -> bool:
        frame = np.asarray(pcm16, dtype="<i2").reshape(-1)
        if len(frame) != self.expected_samples:
            raise ValueError(f"Expected {self.expected_samples} samples, got {len(frame)}")
        return bool(self.vad.is_speech(frame.tobytes(), self.sample_rate))


class ConsecutiveSpeechGate:
    def __init__(self, required_frames: int = 3):
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        self.required_frames = required_frames
        self.count = 0

    def update(self, speech: bool) -> tuple[int, bool]:
        self.count = self.count + 1 if speech else 0
        return self.count, self.count >= self.required_frames

    def reset(self) -> None:
        self.count = 0
