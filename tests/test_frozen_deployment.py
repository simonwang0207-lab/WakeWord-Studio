from __future__ import annotations

from copy import deepcopy

import pytest

from wakeword_studio.training.frozen_deployment import (
    assert_frozen_deployment_contract,
)


BASELINE = {
    "frontend": {
        "name": "TensorFlow Lite Micro microfrontend",
        "implementation": "pymicro-features",
        "implementation_version": "2.0.2",
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_width_bytes": 2,
        "feature_bins": 40,
        "window_size_ms": 30,
        "window_step_ms": 10,
        "clip_duration_ms": 3000,
    },
    "architecture": {
        "family": "microWakeWord",
        "model": "MixedNet",
        "size": "Tiny",
        "parameter_count": 19697,
        "pointwise_filters": "48,48,48,48",
        "repeat_in_block": "1,1,1,1",
        "mixconv_kernel_sizes": "[5],[7,11],[9,15],[23]",
        "residual_connection": "0,0,0,0",
        "first_conv_filters": 32,
        "first_conv_kernel_size": 5,
        "stride": 3,
        "pooled": 0,
        "max_pool": 0,
        "spatial_attention": 0,
    },
    "quantization": {
        "format": "full_int8_streaming_tflite",
        "input_type": "int8",
        "output_type": "uint8",
        "output_dequantization": "scale_times_raw_minus_zero_point",
    },
}


def candidate() -> dict[str, object]:
    value = deepcopy(BASELINE)
    # v3 records this training-store fact; it is not a frontend semantic input.
    value["frontend"]["stored_feature_frames"] = 297
    # The v3 config relies on the frozen exporter's explicit full-INT8 defaults.
    del value["quantization"]["input_type"]
    del value["quantization"]["output_type"]
    return value


def test_stored_feature_frames_only_difference_passes() -> None:
    assert_frozen_deployment_contract(BASELINE, candidate())


def test_sample_rate_difference_fails() -> None:
    changed = candidate()
    changed["frontend"]["sample_rate_hz"] = 8000
    with pytest.raises(RuntimeError, match=r"frontend\.sample_rate_hz"):
        assert_frozen_deployment_contract(BASELINE, changed)


def test_frame_step_difference_fails() -> None:
    changed = candidate()
    changed["frontend"]["window_step_ms"] = 20
    with pytest.raises(RuntimeError, match=r"frontend\.window_step_ms"):
        assert_frozen_deployment_contract(BASELINE, changed)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("frontend", "feature_bins", 32),
        ("frontend", "implementation_version", "3.0.0"),
        ("architecture", "stride", 2),
        ("quantization", "input_type", "float32"),
        ("quantization", "output_dequantization", "raw_uint8"),
    ],
)
def test_feature_frontend_and_quantization_changes_fail(
    section: str, field: str, value: object
) -> None:
    changed = candidate()
    changed[section][field] = value
    with pytest.raises(RuntimeError, match=rf"{section}\.{field}"):
        assert_frozen_deployment_contract(BASELINE, changed)


def test_normalization_change_fails() -> None:
    changed = candidate()
    changed["frontend"]["normalization"] = "per_feature_standardization"
    with pytest.raises(RuntimeError, match=r"frontend\.normalization"):
        assert_frozen_deployment_contract(BASELINE, changed)
