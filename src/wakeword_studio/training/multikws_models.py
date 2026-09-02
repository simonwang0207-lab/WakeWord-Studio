"""Dynamic softmax model construction for Multi-KWS."""

from __future__ import annotations

from typing import Any, Mapping

from .binary_kws_models import build_binary_kws_model


def build_multikws_model(
    model_name: str,
    input_shape: tuple[int, int],
    num_classes: int,
    config: Mapping[str, Any],
) -> Any:
    if num_classes < 2:
        raise ValueError("Multi-KWS requires background plus at least one keyword")
    if model_name not in {"bcresnet", "convmixer"}:
        raise ValueError("Multi-KWS supports bcresnet and convmixer")
    # Reuse the audited real architecture while leaving its frozen binary
    # constructor untouched. The tensor entering the sigmoid head is the
    # backbone representation (including the configured classifier dropout).
    base = build_binary_kws_model(model_name, input_shape, config)
    import tensorflow as tf

    representation = base.get_layer("class_scores").input
    outputs = tf.keras.layers.Dense(
        int(num_classes), activation="softmax", name="multikws_class_scores"
    )(representation)
    model = tf.keras.Model(base.input, outputs, name=f"{model_name}_multikws")
    if tuple(model.output_shape) != (None, num_classes):
        raise RuntimeError(f"Dynamic classifier output mismatch: {model.output_shape}")
    return model


def estimate_macs(model_name: str, input_shape: tuple[int, int], num_classes: int, config: Mapping[str, Any]) -> int:
    time_steps, mel_bins = input_shape
    if model_name == "bcresnet":
        channels = max(4, round(int(config.get("channels", 40)) * float(config.get("width_multiplier", 1.0))))
        depth = int(config.get("depth", 8))
        stem = time_steps * mel_bins * channels * 9
        block = time_steps * mel_bins * channels * 3 + time_steps * channels * 3 + time_steps * channels * channels
        return int(stem + depth * block + channels * num_classes)
    if model_name == "convmixer":
        hidden = int(config.get("hidden_dim", 48)); depth = int(config.get("depth", 6))
        patch = tuple(int(v) for v in config.get("patch_size", (3, 2)))
        stride = tuple(int(v) for v in config.get("stride", (2, 2)))
        kernel = tuple(int(v) for v in config.get("kernel_size", (7, 5)))
        out_t = (time_steps + stride[0] - 1) // stride[0]; out_f = (mel_bins + stride[1] - 1) // stride[1]
        projection = out_t * out_f * hidden * patch[0] * patch[1]
        block = out_t * out_f * (hidden * kernel[0] * kernel[1] + hidden * hidden)
        return int(projection + depth * block + hidden * num_classes)
    raise ValueError(model_name)
