"""Export a fully integer RepCNN using the official fused inference graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from livekit.embedded_wakeword.config import load_config
from livekit.embedded_wakeword.models.classifier import reparameterize_model
from livekit.embedded_wakeword.models.pipeline import WakeWordClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model = WakeWordClassifier(config)
    shape = (1, config.n_frames, config.augmentation.num_channels)
    model(tf.zeros(shape), training=False)
    model.load_weights(str(config.model_output_dir / f"{config.model_name}.weights.h5"))
    fused = reparameterize_model(model.classifier)

    @tf.function(input_signature=[tf.TensorSpec(shape=shape, dtype=tf.float32)])
    def serving_fn(x: tf.Tensor) -> tf.Tensor:
        return fused(x, training=False)

    calibration = np.load(config.model_output_dir / "positive_features_train.npy")
    calibration_neg = np.load(config.model_output_dir / "negative_features_train.npy")
    calibration = np.concatenate([calibration[:20], calibration_neg[:20]], axis=0).astype(np.float32)

    def representative_dataset():
        for sample in calibration:
            yield [sample[np.newaxis]]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(f"Wrote {args.output} ({len(content)} bytes, {len(content) / 1024:.3f} KiB)")


if __name__ == "__main__":
    main()
