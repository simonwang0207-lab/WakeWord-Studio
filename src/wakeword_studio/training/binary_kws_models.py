"""TensorFlow model registry for the fair binary-KWS comparison.

All registered models consume the same ``[batch, time, mel]`` tensor and emit
one sigmoid score per clip.  TensorFlow is imported lazily so the rest of WakeWord
Studio (including the WebUI) does not acquire a TensorFlow runtime dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


MODEL_NAMES = ("repcnn", "bcresnet", "convmixer")


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "TensorFlow is required for model construction; activate the WSL GPU environment"
        ) from error
    return tf


def count_trainable_parameters(model: Any) -> int:
    """Return the number of trainable scalar parameters without NumPy."""

    total = 0
    for variable in model.trainable_variables:
        size = 1
        for dimension in variable.shape:
            size *= int(dimension)
        total += size
    return total


def _activation_layer(tf: Any, name: str, *, layer_name: str) -> Any:
    normalized = name.strip().lower()
    if normalized == "relu":
        return tf.keras.layers.ReLU(name=layer_name)
    if normalized in {"swish", "silu"}:
        return tf.keras.layers.Activation("swish", name=layer_name)
    if normalized == "gelu":
        return tf.keras.layers.Activation("gelu", name=layer_name)
    raise ValueError(f"Unsupported activation: {name}")


def build_bcresnet(input_shape: tuple[int, int], config: Mapping[str, Any]) -> Any:
    """Build a compact Broadcasted Residual Network for acoustic feature maps.

    The block is intentionally not a renamed image ResNet.  Each block performs
    frequency depthwise processing with sub-spectral normalization, collapses the
    frequency axis, performs temporal depthwise processing, and broadcasts that
    temporal context back across frequency before the residual addition.
    """

    tf = _tensorflow()
    channels = max(4, round(int(config.get("channels", 40)) * float(config.get("width_multiplier", 1.0))))
    depth = int(config.get("depth", 8))
    dropout = float(config.get("dropout", 0.1))
    activation = str(config.get("activation", "relu"))
    subbands = int(config.get("subbands", 4))
    dilations = tuple(int(value) for value in config.get("temporal_dilations", (1, 2, 4)))
    if depth < 1 or channels < 1 or subbands < 1 or not dilations:
        raise ValueError("BC-ResNet depth/channels/subbands/dilations must be positive")
    if input_shape[1] % subbands:
        raise ValueError("BC-ResNet mel bins must be divisible by subbands")

    inputs = tf.keras.Input(shape=input_shape, name="microfrontend")
    x = tf.keras.layers.Reshape((*input_shape, 1), name="add_channel")(inputs)
    x = tf.keras.layers.Conv2D(
        channels, (3, 3), padding="same", use_bias=False, name="stem_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = _activation_layer(tf, activation, layer_name="stem_activation")(x)

    for index in range(depth):
        residual = x
        frequency = tf.keras.layers.DepthwiseConv2D(
            (1, 3), padding="same", use_bias=False, name=f"block_{index}_frequency_dw"
        )(x)
        chunks = tf.keras.layers.Lambda(
            lambda value, n=subbands: tf.split(value, n, axis=2),
            name=f"block_{index}_subband_split",
        )(frequency)
        normalized_chunks = [
            tf.keras.layers.BatchNormalization(name=f"block_{index}_ssn_{band}")(chunk)
            for band, chunk in enumerate(chunks)
        ]
        frequency = tf.keras.layers.Concatenate(axis=2, name=f"block_{index}_ssn_concat")(
            normalized_chunks
        )
        frequency = _activation_layer(
            tf, activation, layer_name=f"block_{index}_frequency_activation"
        )(frequency)

        temporal = tf.keras.layers.Lambda(
            lambda value: tf.reduce_mean(value, axis=2, keepdims=True),
            name=f"block_{index}_frequency_average",
        )(frequency)
        temporal = tf.keras.layers.DepthwiseConv2D(
            (3, 1),
            padding="same",
            dilation_rate=(dilations[index % len(dilations)], 1),
            use_bias=False,
            name=f"block_{index}_temporal_dw",
        )(temporal)
        temporal = tf.keras.layers.BatchNormalization(name=f"block_{index}_temporal_bn")(
            temporal
        )
        temporal = _activation_layer(
            tf, activation, layer_name=f"block_{index}_temporal_activation"
        )(temporal)
        temporal = tf.keras.layers.Conv2D(
            channels, (1, 1), use_bias=False, name=f"block_{index}_pointwise"
        )(temporal)
        temporal = tf.keras.layers.BatchNormalization(name=f"block_{index}_pointwise_bn")(
            temporal
        )
        if dropout:
            temporal = tf.keras.layers.Dropout(dropout, name=f"block_{index}_dropout")(
                temporal
            )

        # TensorFlow broadcasts [B,T,1,C] temporal context over the mel axis.
        broadcast = tf.keras.layers.Lambda(
            lambda values: values[0] + values[1],
            name=f"block_{index}_broadcast",
        )([frequency, temporal])
        x = tf.keras.layers.Add(name=f"block_{index}_residual")([residual, broadcast])
        x = _activation_layer(tf, activation, layer_name=f"block_{index}_output")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="class_scores")(x)
    return tf.keras.Model(inputs, outputs, name="bcresnet_binary_kws")


def build_convmixer(input_shape: tuple[int, int], config: Mapping[str, Any]) -> Any:
    """Build a compact ConvMixer adapted to time-by-mel acoustic features."""

    tf = _tensorflow()
    hidden_dim = int(config.get("hidden_dim", 48))
    depth = int(config.get("depth", 6))
    kernel_size = config.get("kernel_size", (7, 5))
    patch_size = config.get("patch_size", (3, 2))
    stride = config.get("stride", (2, 2))
    dropout = float(config.get("dropout", 0.1))
    activation = str(config.get("activation", "relu"))
    kernel = tuple(int(value) for value in kernel_size)
    patch = tuple(int(value) for value in patch_size)
    strides = tuple(int(value) for value in stride)
    if hidden_dim < 1 or depth < 1 or any(value < 1 for value in (*kernel, *patch, *strides)):
        raise ValueError("ConvMixer dimensions must be positive")

    inputs = tf.keras.Input(shape=input_shape, name="microfrontend")
    x = tf.keras.layers.Reshape((*input_shape, 1), name="add_channel")(inputs)
    x = tf.keras.layers.Conv2D(
        hidden_dim,
        patch,
        strides=strides,
        padding="same",
        use_bias=False,
        name="patch_embedding",
    )(x)
    x = _activation_layer(tf, activation, layer_name="patch_activation")(x)
    x = tf.keras.layers.BatchNormalization(name="patch_bn")(x)

    for index in range(depth):
        residual = x
        mixed = tf.keras.layers.DepthwiseConv2D(
            kernel, padding="same", use_bias=False, name=f"block_{index}_depthwise"
        )(x)
        mixed = _activation_layer(tf, activation, layer_name=f"block_{index}_dw_activation")(
            mixed
        )
        mixed = tf.keras.layers.BatchNormalization(name=f"block_{index}_dw_bn")(mixed)
        x = tf.keras.layers.Add(name=f"block_{index}_residual")([residual, mixed])
        x = tf.keras.layers.Conv2D(
            hidden_dim, (1, 1), use_bias=False, name=f"block_{index}_pointwise"
        )(x)
        x = _activation_layer(tf, activation, layer_name=f"block_{index}_pw_activation")(x)
        x = tf.keras.layers.BatchNormalization(name=f"block_{index}_pw_bn")(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout, name=f"block_{index}_dropout")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="class_scores")(x)
    return tf.keras.Model(inputs, outputs, name="convmixer_binary_kws")


def build_repcnn(input_shape: tuple[int, int], config: Mapping[str, Any]) -> Any:
    """Build the pinned LiveKit RepCNN through the same registry interface."""

    tf = _tensorflow()
    try:
        from livekit.embedded_wakeword.models.classifier import RepCNNClassifier
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("The pinned LiveKit Embedded Wakeword package is required") from error
    model = RepCNNClassifier(
        n_frames=int(input_shape[0]),
        n_features=int(input_shape[1]),
        filters=int(config.get("filters", 64)),
        n_blocks=int(config.get("n_blocks", 11)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model(tf.zeros((1, *input_shape), tf.float32), training=False)
    if config.get("sigmoid_head_initializer") == "zeros_for_stable_unsaturated_start":
        kernel, bias = model.dense_out.get_weights()
        import numpy as np

        model.dense_out.set_weights([np.zeros_like(kernel), np.zeros_like(bias)])
    return model


_BUILDERS: dict[str, Callable[[tuple[int, int], Mapping[str, Any]], Any]] = {
    "repcnn": build_repcnn,
    "bcresnet": build_bcresnet,
    "convmixer": build_convmixer,
}


def build_binary_kws_model(
    model_name: str, input_shape: tuple[int, int], config: Mapping[str, Any]
) -> Any:
    """Construct one registered binary-KWS model and materialize its variables."""

    normalized = model_name.strip().lower()
    try:
        builder = _BUILDERS[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown model_name={model_name!r}; choices={MODEL_NAMES}") from error
    model = builder(tuple(int(value) for value in input_shape), config)
    tf = _tensorflow()
    model(tf.zeros((1, *input_shape), tf.float32), training=False)
    return model
