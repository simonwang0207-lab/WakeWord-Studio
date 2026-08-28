"""Export the same trained DS-TC-ResNet classifier without its MFCC frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from kws_smoke_common import make_flags
from kws_streaming.models import ds_tc_resnet


def audio(path: Path, size: int) -> np.ndarray:
    value, sr = sf.read(path, always_2d=False)
    if sr != 16_000:
        raise ValueError(sr)
    value = np.asarray(value, np.float32)[:size]
    return np.pad(value, (0, max(0, size - len(value))))[np.newaxis]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ops", type=Path, required=True)
    args = parser.parse_args()

    raw_flags = make_flags(args.data_dir, args.train_dir)
    raw_flags.batch_size = 1
    raw_model = ds_tc_resnet.model(raw_flags)
    raw_model.load_weights(str(args.train_dir / "best_weights"))
    feature_model = tf.keras.Model(raw_model.input, raw_model.get_layer("speech_features").output)

    classifier_flags = make_flags(args.data_dir, args.train_dir)
    classifier_flags.batch_size = 1
    classifier_flags.preprocess = "mfcc"
    classifier = ds_tc_resnet.model(classifier_flags)
    classifier.load_weights(str(args.train_dir / "best_weights"))

    pos_paths = sorted((args.data_dir / "training" / "qingxiaojia").glob("*.wav"))
    neg_paths = sorted((args.data_dir / "training" / "other").glob("*.wav"))
    feature_samples = [
        feature_model(audio(path, raw_flags.desired_samples), training=False).numpy()
        for path in pos_paths[:20] + neg_paths[:20]
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    float_converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
    float_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    float_path = args.output_dir / "ds_tc_resnet_classifier_float.tflite"
    float_path.write_bytes(float_converter.convert())

    def representative():
        for value in feature_samples:
            yield [value.astype(np.float32)]

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = representative
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_path = args.output_dir / "ds_tc_resnet_classifier_full_int8.tflite"
    int8_path.write_bytes(int8_converter.convert())

    result = {}
    lines = []
    pos_feature = feature_model(audio(args.data_dir / "testing" / "qingxiaojia" / "clip_000000.wav", raw_flags.desired_samples)).numpy()
    neg_feature = feature_model(audio(args.data_dir / "testing" / "other" / "clip_000000.wav", raw_flags.desired_samples)).numpy()
    for name, path in (("float", float_path), ("full_int8", int8_path)):
        interpreter = tf.lite.Interpreter(
            model_path=str(path),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        ops = [row["op_name"] for row in interpreter._get_ops_details()]
        item = {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "kib": path.stat().st_size / 1024,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_shape": inp["shape"].tolist(), "input_dtype": np.dtype(inp["dtype"]).name,
            "input_quantization": list(inp["quantization"]),
            "output_dtype": np.dtype(out["dtype"]).name,
            "output_quantization": list(out["quantization"]),
            "unique_operators": sorted(set(ops)),
        }
        predictions = {}
        for label, value in (("positive", pos_feature), ("negative", neg_feature)):
            scale, zero = inp["quantization"]
            if inp["dtype"] != np.float32:
                info = np.iinfo(inp["dtype"])
                value = np.clip(np.rint(value / scale + zero), info.min, info.max).astype(inp["dtype"])
            interpreter.set_tensor(inp["index"], value)
            interpreter.invoke()
            logits = interpreter.get_tensor(out["index"])
            if out["dtype"] != np.float32:
                out_scale, out_zero = out["quantization"]
                logits = (logits.astype(np.float32) - out_zero) * out_scale
            logits = logits.reshape(-1)
            exp = np.exp(logits - logits.max())
            predictions[label] = {"logits": logits.tolist(), "softmax": (exp / exp.sum()).tolist()}
        item["predictions"] = predictions
        result[name] = item
        lines += [f"[{name}]", f"bytes={item['bytes']}", f"sha256={item['sha256']}",
                  "unique_operators=" + ", ".join(item["unique_operators"]), ""]
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.ops.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
