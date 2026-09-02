"""Explicit deployment-contract checks shared by frozen model evaluations."""

from __future__ import annotations

from typing import Any, Mapping


# These fields determine the audio-to-feature semantics or the feature tensor
# consumed by the deployed model.  ``stored_feature_frames`` is intentionally
# absent: it records the length of the v3 training feature store and is not a
# microfrontend setting.
FROZEN_FRONTEND_FIELDS = (
    "name",
    "implementation",
    "implementation_version",
    "sample_rate_hz",
    "channels",
    "sample_width_bytes",
    "feature_bins",
    "window_size_ms",
    "window_step_ms",
    "clip_duration_ms",
)

# Optional frontend metadata is still semantic when supplied.  Listing it here
# prevents a future normalization/shape setting from silently bypassing the
# frozen contract while permitting the current configs, which predate it.
OPTIONAL_FROZEN_FRONTEND_FIELDS = (
    "feature_shape",
    "input_shape",
    "normalization",
    "feature_normalization",
    "normalization_mean",
    "normalization_std",
)

FROZEN_ARCHITECTURE_FIELDS = (
    "family",
    "model",
    "size",
    "parameter_count",
    "pointwise_filters",
    "repeat_in_block",
    "mixconv_kernel_sizes",
    "residual_connection",
    "first_conv_filters",
    "first_conv_kernel_size",
    "stride",
    "pooled",
    "max_pool",
    "spatial_attention",
)

FROZEN_QUANTIZATION_FIELDS = (
    "format",
    "output_dequantization",
)

# The v2 config states these explicitly; v3 relies on the frozen full-INT8
# exporter defaults.  Resolve both representations to the same deployment
# contract, while still rejecting an explicit incompatible v3 value.
QUANTIZATION_DEFAULTS = {
    "input_type": "int8",
    "output_type": "uint8",
}

_MISSING = object()


def _assert_equal(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    section: str,
    fields: tuple[str, ...],
    *,
    optional: bool = False,
) -> None:
    baseline_section = baseline.get(section, {})
    candidate_section = candidate.get(section, {})
    for field in fields:
        baseline_value = baseline_section.get(field, _MISSING)
        candidate_value = candidate_section.get(field, _MISSING)
        if optional and baseline_value is _MISSING and candidate_value is _MISSING:
            continue
        if baseline_value != candidate_value:
            raise RuntimeError(f"V2/V3 frozen deployment field differs: {section}.{field}")


def assert_frozen_deployment_contract(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Reject changes to deployment semantics, ignoring v3-only bookkeeping."""

    _assert_equal(baseline, candidate, "frontend", FROZEN_FRONTEND_FIELDS)
    _assert_equal(
        baseline,
        candidate,
        "frontend",
        OPTIONAL_FROZEN_FRONTEND_FIELDS,
        optional=True,
    )
    _assert_equal(baseline, candidate, "architecture", FROZEN_ARCHITECTURE_FIELDS)
    _assert_equal(baseline, candidate, "quantization", FROZEN_QUANTIZATION_FIELDS)

    baseline_quantization = baseline.get("quantization", {})
    candidate_quantization = candidate.get("quantization", {})
    for field, default in QUANTIZATION_DEFAULTS.items():
        baseline_value = baseline_quantization.get(field, default)
        candidate_value = candidate_quantization.get(field, default)
        if baseline_value != candidate_value:
            raise RuntimeError(
                f"V2/V3 frozen deployment field differs: quantization.{field}"
            )
