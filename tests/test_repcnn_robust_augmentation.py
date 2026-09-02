import numpy as np

from wakeword_studio.training.repcnn_robust_augmentation import (
    augment_training_features,
    microphone_frequency_tilt,
    zero_padded_temporal_shift,
)


def test_temporal_shift_zero_pads_without_wraparound():
    values = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2)
    result = zero_padded_temporal_shift(values, np.asarray([1, -1]))
    np.testing.assert_array_equal(result[0, 0], 0.0)
    np.testing.assert_array_equal(result[0, 1:], values[0, :-1])
    np.testing.assert_array_equal(result[1, :-1], values[1, 1:])
    np.testing.assert_array_equal(result[1, -1], 0.0)


def test_robust_augmentation_is_deterministic_and_shape_preserving():
    values = np.ones((4, 99, 40), np.float32)
    first = augment_training_features(values, seed=7, step=11)
    second = augment_training_features(values, seed=7, step=11)
    assert first.shape == values.shape
    np.testing.assert_array_equal(first, second)


def test_microphone_eq_hook_is_disabled_as_exact_copy_by_default():
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    result = microphone_frequency_tilt(values, np.random.default_rng(1))
    np.testing.assert_array_equal(result, values)
    assert result is not values
