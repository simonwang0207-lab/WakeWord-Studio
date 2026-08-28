"""Measure fused full-INT8 RepCNN sizes before choosing the Phase 0.5 preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from livekit.embedded_wakeword.models.classifier import RepCNNClassifier, reparameterize_model


def export_one(filters: int, blocks: int, output: Path) -> dict[str, object]:
    tf.keras.utils.set_random_seed(20260828 + filters + blocks)
    model = RepCNNClassifier(n_frames=99, n_features=40, filters=filters, n_blocks=blocks)
    model(tf.zeros((1, 99, 40)), training=False)
    fused = reparameterize_model(model)

    @tf.function(input_signature=[tf.TensorSpec((1, 99, 40), tf.float32)])
    def serving(x: tf.Tensor) -> tf.Tensor:
        return fused(x, training=False)

    rng = np.random.default_rng(20260828)

    def representative_dataset():
        for _ in range(16):
            yield [rng.uniform(0.0, 6.0, (1, 99, 40)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return {
        "filters": filters,
        "blocks": blocks,
        "training_parameters": int(model.count_params()),
        "fused_parameters": int(fused.count_params()),
        "bytes": len(content),
        "kib": round(len(content) / 1024, 3),
        "path": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = [(32, 7), (48, 9), (64, 11), (80, 13), (96, 13), (112, 15)]
    results = [
        export_one(filters, blocks, args.output_dir / f"repcnn_f{filters}_b{blocks}.tflite")
        for filters, blocks in candidates
    ]
    result_path = args.output_dir / "capacity_probe.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
