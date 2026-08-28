"""Export non-streaming float/full-INT8 DS-TC-ResNet and inspect real outputs/ops."""

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


def load_audio(path: Path, size: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if sr != 16_000:
        raise ValueError(f"Expected 16 kHz: {path}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio[:size]
    return np.pad(audio, (0, max(0, size - len(audio))))[np.newaxis]


def describe(path: Path) -> tuple[dict, tf.lite.Interpreter]:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    ops = [row["op_name"] for row in interpreter._get_ops_details()]
    return ({
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "kib": path.stat().st_size / 1024,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_shape": inp["shape"].tolist(),
        "input_dtype": np.dtype(inp["dtype"]).name,
        "input_quantization": list(inp["quantization"]),
        "output_shape": out["shape"].tolist(),
        "output_dtype": np.dtype(out["dtype"]).name,
        "output_quantization": list(out["quantization"]),
        "operators_in_order": ops,
        "unique_operators": sorted(set(ops)),
    }, interpreter)


def predict(interpreter: tf.lite.Interpreter, audio: np.ndarray) -> dict:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    if inp["dtype"] == np.float32:
        value = audio.astype(np.float32)
    else:
        info = np.iinfo(inp["dtype"])
        value = np.clip(np.rint(audio / scale + zero), info.min, info.max).astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], value)
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])
    out_scale, out_zero = out["quantization"]
    if out["dtype"] != np.float32:
        raw = (raw.astype(np.float32) - out_zero) * out_scale
    logits = raw.reshape(-1).astype(float)
    exp = np.exp(logits - logits.max())
    return {"logits": logits.tolist(), "softmax": (exp / exp.sum()).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--ops-txt", type=Path, required=True)
    args = parser.parse_args()

    flags = make_flags(args.data_dir, args.train_dir)
    # Training uses batch=4; deployment and representative calibration use a
    # fixed batch of one. Batch size does not change any learned weight shape.
    flags.batch_size = 1
    model = ds_tc_resnet.model(flags)
    model.load_weights(str(args.train_dir / "best_weights"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positive_files = sorted((args.data_dir / "training" / "qingxiaojia").glob("*.wav"))
    negative_files = sorted((args.data_dir / "training" / "other").glob("*.wav"))
    calibration = [load_audio(path, flags.desired_samples) for path in positive_files[:20] + negative_files[:20]]

    # The upstream model_to_tflite path hangs in SavedModel/private-clone code
    # on this Windows/TF 2.13 environment. This bounded fallback keeps the exact
    # trained official Keras graph and uses the public direct converter.
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    float_converter.allow_custom_ops = False
    float_content = float_converter.convert()
    float_path = args.output_dir / "ds_tc_resnet_nonstream_float.tflite"
    float_path.write_bytes(float_content)

    def representative_dataset():
        for sample in calibration:
            yield [sample.astype(np.float32)]

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = representative_dataset
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_converter.allow_custom_ops = False
    int8_content = int8_converter.convert()
    int8_path = args.output_dir / "ds_tc_resnet_nonstream_full_int8.tflite"
    int8_path.write_bytes(int8_content)

    pos = load_audio(args.data_dir / "testing" / "qingxiaojia" / "clip_000000.wav", flags.desired_samples)
    neg = load_audio(args.data_dir / "testing" / "other" / "clip_000000.wav", flags.desired_samples)
    result = {"parameter_count": int(model.count_params()), "models": {}}
    lines = []
    for name, path in (("float", float_path), ("full_int8", int8_path)):
        item, interpreter = describe(path)
        item["positive"] = predict(interpreter, pos)
        item["negative"] = predict(interpreter, neg)
        result["models"][name] = item
        lines += [f"[{name}]", f"path={item['path']}", f"bytes={item['bytes']}",
                  f"sha256={item['sha256']}", "operators_in_order:"]
        lines += [f"  {i:02d}: {op}" for i, op in enumerate(item["operators_in_order"])]
        lines += ["unique_operators=" + ", ".join(item["unique_operators"]), ""]

    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.ops_txt.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
