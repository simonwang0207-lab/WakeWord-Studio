"""Inspect RepCNN TFLite artifacts and run real positive/negative WAV inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from livekit.embedded_wakeword.config import load_config
from livekit.embedded_wakeword.models.classifier import reparameterize_model
from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend
from livekit.embedded_wakeword.models.pipeline import WakeWordClassifier


def describe(path: Path) -> dict:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    ops = [row["op_name"] for row in interpreter._get_ops_details()]
    return {
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
    }


def wav_features(path: Path, frontend: MicroFrontend, n_frames: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if sr != 16_000:
        raise ValueError(f"Expected 16 kHz WAV, got {sr}: {path}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    features = frontend(audio)[0]
    if features.shape[0] >= n_frames:
        features = features[-n_frames:]
    else:
        features = np.pad(features, ((n_frames - features.shape[0], 0), (0, 0)))
    return features[np.newaxis].astype(np.float32)


def infer(model: Path, features: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(
        model_path=str(model),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    if inp["dtype"] == np.float32:
        value = features
    else:
        info = np.iinfo(inp["dtype"])
        value = np.clip(np.rint(features / scale + zero), info.min, info.max).astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], value)
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])
    out_scale, out_zero = out["quantization"]
    if out["dtype"] != np.float32:
        raw = (raw.astype(np.float32) - out_zero) * out_scale
    return float(raw.reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--float-model", type=Path, required=True)
    parser.add_argument("--official-quant-model", type=Path, required=True)
    parser.add_argument("--full-int8-model", type=Path)
    parser.add_argument("--positive-wav", type=Path, required=True)
    parser.add_argument("--negative-wav", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--ops-txt", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    network = WakeWordClassifier(config)
    network(tf.zeros((1, config.n_frames, config.augmentation.num_channels)), training=False)
    network.load_weights(str(config.model_output_dir / f"{config.model_name}.weights.h5"))
    fused = reparameterize_model(network.classifier)

    frontend = MicroFrontend(
        window_size_ms=config.augmentation.window_size_ms,
        window_step_ms=config.augmentation.window_step_ms,
        num_channels=config.augmentation.num_channels,
    )
    pos = wav_features(args.positive_wav, frontend, config.n_frames)
    neg = wav_features(args.negative_wav, frontend, config.n_frames)

    models = {
        "float_model": args.float_model,
        "official_quant_model": args.official_quant_model,
    }
    if args.full_int8_model:
        models["full_int8_model"] = args.full_int8_model

    result = {
        "training_parameter_count": int(network.count_params()),
        "fused_classifier_parameter_count": int(fused.count_params()),
        "models": {},
    }
    for name, path in models.items():
        item = describe(path)
        item["positive_probability"] = infer(path, pos)
        item["negative_probability"] = infer(path, neg)
        result["models"][name] = item

    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines: list[str] = []
    for name, item in result["models"].items():
        lines += [
            f"[{name}]",
            f"path={item['path']}",
            f"bytes={item['bytes']}",
            f"sha256={item['sha256']}",
            f"input={item['input_dtype']} {item['input_shape']} quant={item['input_quantization']}",
            f"output={item['output_dtype']} {item['output_shape']} quant={item['output_quantization']}",
            "operators_in_order:",
        ]
        lines += [f"  {index:02d}: {op}" for index, op in enumerate(item["operators_in_order"])]
        lines += ["unique_operators=" + ", ".join(item["unique_operators"]), ""]
    args.ops_txt.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
