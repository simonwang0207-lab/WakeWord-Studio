"""Run positive/negative PC inference and inspect actual TFLite operators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly

from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.inference import Model


def load_features(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16_000:
        divisor = np.gcd(sample_rate, 16_000)
        audio = resample_poly(audio, 16_000 // divisor, sample_rate // divisor)
    return generate_features_for_clip(np.clip(audio, -1, 1), step_ms=10)


def infer(model_path: Path, wav_path: Path) -> dict:
    features = load_features(wav_path)
    model = Model(str(model_path), stride=3)
    probabilities = np.asarray(model.predict_spectrogram(features), dtype=np.float32)
    return {
        "wav": str(wav_path.resolve()),
        "feature_shape": list(features.shape),
        "prediction_count": int(probabilities.size),
        "minimum": float(probabilities.min()),
        "maximum": float(probabilities.max()),
        "mean": float(probabilities.mean()),
        "last": float(probabilities[-1]),
        "raw_probabilities": probabilities.tolist(),
    }


def describe_model(path: Path) -> dict:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    operators = [item["op_name"] for item in interpreter._get_ops_details()]
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "kib": path.stat().st_size / 1024,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": list(input_detail["quantization"]),
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": list(output_detail["quantization"]),
        "operators_in_order": operators,
        "unique_operators": sorted(set(operators)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--float-model", type=Path, required=True)
    parser.add_argument("--int8-model", type=Path, required=True)
    parser.add_argument("--positive-wav", type=Path, required=True)
    parser.add_argument("--negative-wav", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--ops-txt", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "float_model": describe_model(args.float_model),
        "int8_model": describe_model(args.int8_model),
        "int8_inference": {
            "positive": infer(args.int8_model, args.positive_wav),
            "negative": infer(args.int8_model, args.negative_wav),
        },
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = []
    for key in ("float_model", "int8_model"):
        item = result[key]
        lines.append(f"[{key}]")
        lines.append(f"path={item['path']}")
        lines.append(f"bytes={item['bytes']}")
        lines.append(f"sha256={item['sha256']}")
        lines.append("operators_in_order:")
        lines.extend(f"  {index:02d}: {op}" for index, op in enumerate(item["operators_in_order"]))
        lines.append("unique_operators=" + ", ".join(item["unique_operators"]))
        lines.append("")
    args.ops_txt.write_text("\n".join(lines), encoding="utf-8")

    compact = {
        "float_bytes": result["float_model"]["bytes"],
        "int8_bytes": result["int8_model"]["bytes"],
        "int8_unique_ops": result["int8_model"]["unique_operators"],
        "positive": {k: v for k, v in result["int8_inference"]["positive"].items() if k != "raw_probabilities"},
        "negative": {k: v for k, v in result["int8_inference"]["negative"].items() if k != "raw_probabilities"},
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
