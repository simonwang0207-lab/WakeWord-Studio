from __future__ import annotations

from collections import deque

import numpy as np

from ..audio import TARGET_SAMPLE_RATE_HZ


class PreRollRingBuffer:
    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE_HZ, frame_ms: int = 30, seconds: float = 1.5):
        if not 1.0 <= seconds <= 2.0:
            raise ValueError("pre-roll must be between 1 and 2 seconds")
        self.max_frames = int(np.ceil(seconds * 1000 / frame_ms))
        self.frames: deque[np.ndarray] = deque(maxlen=self.max_frames)

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame, dtype=np.int16).copy())

    def audio(self) -> np.ndarray:
        return np.concatenate(tuple(self.frames)) if self.frames else np.zeros(0, np.int16)

    def clear(self) -> None:
        self.frames.clear()
