"""Small causal score smoothers for Validation-controlled experiments."""

from __future__ import annotations

from collections import deque

import numpy as np


SMOOTHING_MODES = ("raw", "mean", "max_mean_hybrid")


class RollingScoreSmoother:
    """Causal N-score smoother; ``raw`` is the deployment-safe default."""

    def __init__(
        self,
        mode: str = "raw",
        *,
        window_size: int = 3,
        hybrid_max_weight: float = 0.5,
    ) -> None:
        if mode not in SMOOTHING_MODES:
            raise ValueError(f"Unknown smoothing mode: {mode}")
        if not 1 <= int(window_size) <= 8:
            raise ValueError("smoothing window_size must be between 1 and 8")
        if not 0.0 <= float(hybrid_max_weight) <= 1.0:
            raise ValueError("hybrid_max_weight must be between zero and one")
        self.mode = mode
        self.window_size = int(window_size)
        self.hybrid_max_weight = float(hybrid_max_weight)
        self._history: deque[float] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._history.clear()

    def update(self, score: float) -> float:
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("wake score must be finite")
        self._history.append(value)
        if self.mode == "raw":
            return value
        mean = float(np.mean(self._history))
        if self.mode == "mean":
            return mean
        maximum = max(self._history)
        return self.hybrid_max_weight * maximum + (1.0 - self.hybrid_max_weight) * mean

    def state(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "window_size": self.window_size,
            "hybrid_max_weight": self.hybrid_max_weight,
            "history": list(self._history),
        }
