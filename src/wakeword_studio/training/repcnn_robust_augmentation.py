"""Low-risk feature-domain augmentation hooks for optional B2.1 fine-tuning."""

from __future__ import annotations

import numpy as np


def zero_padded_temporal_shift(features: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift each [time, frequency] feature without circular wrap-around."""

    values = np.asarray(features, dtype=np.float32)
    shift_values = np.asarray(shifts, dtype=np.int32).reshape(-1)
    if values.ndim != 3 or len(values) != len(shift_values):
        raise ValueError("features must be [batch,time,freq] with one shift per sample")
    result = np.zeros_like(values)
    for index, shift in enumerate(shift_values):
        amount = int(shift)
        if amount == 0:
            result[index] = values[index]
        elif amount > 0 and amount < values.shape[1]:
            result[index, amount:] = values[index, :-amount]
        elif amount < 0 and -amount < values.shape[1]:
            result[index, :amount] = values[index, -amount:]
    return result


def mild_spec_augment(
    features: np.ndarray,
    rng: np.random.Generator,
    *,
    maximum_time_frames: int = 3,
    maximum_frequency_bins: int = 2,
    probability: float = 0.5,
) -> np.ndarray:
    """Apply at most one tiny time mask and one tiny frequency mask per sample."""

    result = np.asarray(features, dtype=np.float32).copy()
    if result.ndim != 3:
        raise ValueError("features must be [batch,time,freq]")
    for sample in result:
        if rng.random() > probability:
            continue
        time_width = int(rng.integers(0, maximum_time_frames + 1))
        frequency_width = int(rng.integers(0, maximum_frequency_bins + 1))
        if time_width:
            start = int(rng.integers(0, sample.shape[0] - time_width + 1))
            sample[start : start + time_width, :] = 0.0
        if frequency_width:
            start = int(rng.integers(0, sample.shape[1] - frequency_width + 1))
            sample[:, start : start + frequency_width] = 0.0
    return result


def microphone_frequency_tilt(
    features: np.ndarray,
    rng: np.random.Generator,
    *,
    maximum_edge_gain_db: float = 0.0,
) -> np.ndarray:
    """Optional bounded frequency tilt hook; zero dB is an exact no-op.

    The hook is intentionally disabled in the B2.1 default config.  It exists so
    microphone bandwidth/EQ can later be tested as a controlled ablation rather
    than silently added to the formal recipe.
    """

    values = np.asarray(features, dtype=np.float32)
    if maximum_edge_gain_db <= 0.0:
        return values.copy()
    result = values.copy()
    axis = np.linspace(-1.0, 1.0, values.shape[-1], dtype=np.float32)
    for sample in result:
        edge_gain = float(rng.uniform(-maximum_edge_gain_db, maximum_edge_gain_db))
        multiplier = np.power(10.0, edge_gain * axis / 20.0).astype(np.float32)
        sample *= multiplier[np.newaxis, :]
    return result


def augment_training_features(
    features: np.ndarray,
    *,
    seed: int,
    step: int,
    maximum_shift_frames: int = 3,
    shift_probability: float = 0.5,
    spec_augment_probability: float = 0.5,
    maximum_time_mask_frames: int = 3,
    maximum_frequency_mask_bins: int = 2,
    microphone_eq_maximum_edge_gain_db: float = 0.0,
) -> np.ndarray:
    """Deterministic Train-only B2.1 augmentation composition."""

    rng = np.random.default_rng(int(seed) + int(step) * 1_000_003)
    values = np.asarray(features, dtype=np.float32)
    shifts = rng.integers(-maximum_shift_frames, maximum_shift_frames + 1, size=len(values))
    shifts = np.where(rng.random(len(values)) <= shift_probability, shifts, 0)
    result = zero_padded_temporal_shift(values, shifts)
    result = mild_spec_augment(
        result,
        rng,
        maximum_time_frames=maximum_time_mask_frames,
        maximum_frequency_bins=maximum_frequency_mask_bins,
        probability=spec_augment_probability,
    )
    return microphone_frequency_tilt(
        result,
        rng,
        maximum_edge_gain_db=microphone_eq_maximum_edge_gain_db,
    )
