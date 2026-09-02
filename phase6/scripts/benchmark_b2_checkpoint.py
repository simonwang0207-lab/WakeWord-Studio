"""Read-only B1/B2 checkpoint replay on the five frozen live diagnostic WAVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit.embedded_wakeword.models.classifier import reparameterize_model  # noqa: E402
from phase3.scripts.evaluate_repcnn_model_b_frozen import Int8Scorer  # noqa: E402
from phase3.scripts.preflight_repcnn_model_b import (  # noqa: E402
    build_model,
    indexed_model_state,
    quantization_metadata,
)
from wakeword_studio.diagnostics.live_repcnn import (  # noqa: E402
    extract_fixed_window,
    plan_fixed_windows,
    read_pcm16_wav,
)


FROZEN_RECORDS = (
    "20260830T224435+0800",
    "20260830T224901+0800",
    "20260830T224942+0800",
    "20260830T225023+0800",
    "20260830T225102+0800",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def representative_features(config: dict[str, object]) -> np.ndarray:
    root = Path(str(config["source_feature_cache"])).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_loaded") is not False:
        raise RuntimeError("Representative feature cache is not Train/Validation-only")
    rows: list[np.ndarray] = []
    for split in ("train", "validation"):
        for label in ("positive", "negative", "hard_negative", "ambient"):
            values = np.load(root / f"{split}_{label}.npy", mmap_mode="r")
            rows.extend(np.asarray(values[:4], dtype=np.float32))
    return np.asarray(rows, dtype=np.float32)


def export_checkpoint(
    config: dict[str, object], checkpoint: Path, destination: Path
) -> dict[str, object]:
    model = build_model(config)
    restore = tf.train.Checkpoint(model_state=indexed_model_state(model)).restore(
        str(checkpoint)
    )
    restore.expect_partial()
    restore.assert_existing_objects_matched()
    fused = reparameterize_model(model)
    calibration = representative_features(config)
    shape = tuple(int(value) for value in config["frontend"]["input_shape"])  # type: ignore[index]

    @tf.function(input_signature=[tf.TensorSpec((1, *shape), tf.float32)])
    def serving(value: tf.Tensor) -> tf.Tensor:
        return fused(value, training=False)

    def representatives() -> Iterator[list[np.ndarray]]:
        for feature in calibration:
            yield [feature[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representatives
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    interpreter = tf.lite.Interpreter(model_path=str(destination))
    interpreter.allocate_tensors()
    return {
        "path": str(destination.resolve()),
        "bytes": len(content),
        "kib": len(content) / 1024.0,
        "sha256": sha256_bytes(content),
        "quantization": quantization_metadata(interpreter),
        "training_parameters": int(model.count_params()),
        "deployment_parameters": int(fused.count_params()),
        "test_loaded": False,
    }


def max_scores(
    wav_path: Path,
    frontend,
    b1: Int8Scorer,
    b2: Int8Scorer,
) -> tuple[float, float, list[dict[str, object]]]:
    audio, _ = read_pcm16_wav(wav_path)
    plans = plan_fixed_windows(len(audio), hop_seconds=0.20)
    rows = []
    for plan in plans:
        clip = extract_fixed_window(audio, plan)
        feature = np.asarray(frontend(clip.astype(np.float32) / 32768.0)[0], np.float32)
        b1_score = float(b1.score(feature)["score"])
        b2_score = float(b2.score(feature)["score"])
        rows.append(
            {
                "window_index": plan.index,
                "start_ms": plan.start_ms,
                "end_ms": plan.end_ms,
                "b1_score": b1_score,
                "b2_score": b2_score,
            }
        )
    return (
        max(float(row["b1_score"]) for row in rows),
        max(float(row["b2_score"]) for row in rows),
        rows,
    )


def aggregate(values: list[float], threshold: float) -> dict[str, object]:
    return {
        "mean_max_score": float(np.mean(values)),
        "median_max_score": float(np.median(values)),
        "passed_count": int(sum(value >= threshold for value in values)),
        "record_count": len(values),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/models/repcnn_performance_v2_fasttrack.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--b1-model",
        type=Path,
        default=PROJECT_ROOT
        / "runs/qingxiaojia/repcnn_performance_v1/formal/user_run_01/phase3c_model_b_frozen"
        / "final_model/qingxiaojia_repcnn_performance_v1_best1750_full_int8.tflite",
    )
    parser.add_argument("--b1-threshold", type=float, default=0.84375)
    parser.add_argument("--b2-threshold", type=float, required=True)
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=PROJECT_ROOT / "phase5/artifacts/live_diagnostics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "phase6/artifacts/b2_checkpoint_benchmark",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.with_suffix(".index").is_file():
        raise FileNotFoundError(checkpoint.with_suffix(".index"))
    output_dir = args.output_dir.resolve()
    b2_model_path = output_dir / f"{checkpoint.name}_temporary_full_int8.tflite"
    export = export_checkpoint(config, checkpoint, b2_model_path)

    from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend

    frontend = MicroFrontend(
        sample_rate=16000, window_size_ms=30, window_step_ms=20, num_channels=40
    )
    b1 = Int8Scorer(args.b1_model.resolve())
    b2 = Int8Scorer(b2_model_path)
    records = []
    b1_values: list[float] = []
    b2_values: list[float] = []
    for record in FROZEN_RECORDS:
        wav = args.diagnostics_root / record / "recording.wav"
        b1_max, b2_max, windows = max_scores(wav, frontend, b1, b2)
        b1_values.append(b1_max)
        b2_values.append(b2_max)
        records.append(
            {
                "record": record,
                "wav": str(wav.resolve()),
                "window_count": len(windows),
                "b1_max_score": b1_max,
                "b2_max_score": b2_max,
                "delta_b2_minus_b1": b2_max - b1_max,
                "b1_pass_at_frozen_threshold": b1_max >= args.b1_threshold,
                "b2_pass_at_validation_threshold": b2_max >= args.b2_threshold,
                "windows": windows,
            }
        )
    report = {
        "schema": "wakeword-studio.b2-live-checkpoint-benchmark/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": [
            "LIVE_DIAGNOSTIC_ONLY",
            "NOT_FOR_THRESHOLD_SELECTION",
            "NOT_A_TEST_SET",
        ],
        "checkpoint": str(checkpoint),
        "checkpoint_access": "read_only",
        "b2_export": export,
        "window_policy": {
            "window_seconds": 2.0,
            "hop_seconds": 0.20,
            "planned_before_inference": True,
            "tail_anchor": True,
        },
        "records": records,
        "b1_aggregate": aggregate(b1_values, args.b1_threshold),
        "b2_aggregate": aggregate(b2_values, args.b2_threshold),
        "mean_delta_b2_minus_b1": float(np.mean(np.asarray(b2_values) - b1_values)),
        "test_loaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{checkpoint.name}_b1_vs_b2_live_diagnostic.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
