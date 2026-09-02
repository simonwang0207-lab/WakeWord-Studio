"""Deterministic, resumable epoch sampler for fair Multi-KWS training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeterministicEpochSampler:
    sample_count: int
    batch_size: int
    seed: int
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.sample_count < 1 or self.batch_size < 1:
            raise ValueError("sample_count and batch_size must be positive")
        if self.drop_last and self.sample_count < self.batch_size:
            raise ValueError("drop_last would produce an empty epoch")

    @property
    def steps_per_epoch(self) -> int:
        if self.drop_last:
            return self.sample_count // self.batch_size
        return (self.sample_count + self.batch_size - 1) // self.batch_size

    def permutation(self, epoch: int) -> np.ndarray:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        return generator.permutation(self.sample_count).astype(np.int64, copy=False)

    def batch_indices(self, absolute_step: int) -> np.ndarray:
        if absolute_step < 0:
            raise ValueError("absolute_step must be non-negative")
        epoch, batch = divmod(absolute_step, self.steps_per_epoch)
        start = batch * self.batch_size
        stop = start + self.batch_size
        return self.permutation(epoch)[start:stop]

    def first_epoch_audit(self) -> dict[str, int]:
        seen = np.concatenate(
            [self.batch_indices(step) for step in range(self.steps_per_epoch)]
        )
        unique = int(len(np.unique(seen)))
        return {
            "unique_samples": unique,
            "missing_samples": int(self.sample_count - unique),
            "duplicate_samples": int(len(seen) - unique),
        }
