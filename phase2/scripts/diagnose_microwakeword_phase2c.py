"""Phase 2C: frozen-model quantization, streaming, and split diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from microwakeword import mixednet
from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.data import fixed_length_spectrogram
from microwakeword.layers import modes
from microwakeword.utils import model_to_saved
from run_microwakeword_training import build_runtime_config, sha256_file
from wakeword_studio.dataset.manifest import DatasetManifest
from wakeword_studio.frontends import load_inference_audio, load_training_audio


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None, "std": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "min": float(data.min()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "max": float(data.max()),
        "std": float(data.std()),
    }


def binary_metrics(rows: list[dict[str, object]], score_key: str, threshold: float) -> dict[str, object]:
    labels = np.asarray([row["label"] == "positive" for row in rows], dtype=bool)
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    tn = int(np.sum(~predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else None
    result: dict[str, object] = {
        "threshold": float(threshold),
        "recall_tpr": recall,
        "precision": precision,
        "f1": f1,
        "false_rejection_rate": 1.0 - recall if recall is not None else None,
        "false_positive_rate": fpr,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    if len(set(labels.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def select_threshold(rows: list[dict[str, object]], score_key: str) -> tuple[float, dict[str, object]]:
    candidates = sorted({float(row[score_key]) for row in rows})
    sweep = [binary_metrics(rows, score_key, threshold) for threshold in candidates]
    feasible = [row for row in sweep if float(row["recall_tpr"] or 0.0) >= 0.98]
    pool = feasible or sweep
    selected = max(
        pool,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -float(row["false_positive_rate"] or 0.0),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), selected


def correlation(rows: list[dict[str, object]], left: str, right: str) -> dict[str, float]:
    x = np.asarray([float(row[left]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[right]) for row in rows], dtype=np.float64)
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(spearmanr(x, y).statistic),
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean(np.square(x - y)))),
        "max_absolute_error": float(np.max(np.abs(x - y))),
    }


def stratified_limit(records: list[object], limit: int) -> list[object]:
    """Select at most limit records while representing every label deterministically."""
    labels = ("positive", "negative", "hard_negative", "ambient")
    grouped = {label: [row for row in records if row.label == label] for label in labels}
    selected: list[object] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for label in labels:
            if offset < len(grouped[label]) and len(selected) < limit:
                selected.append(grouped[label][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def record_metadata(record: object, dataset_root: Path) -> dict[str, object]:
    return {
        "path": str((dataset_root / record.audio_path).resolve()),
        "record_id": record.record_id,
        "split": record.split,
        "label": record.label,
        "source": record.speaker.source,
        "speaker_id": record.speaker.speaker_id,
        "text": record.text,
        "noise": record.acoustic.noise_id,
        "snr_db": record.acoustic.snr_db,
        "duration_seconds": record.duration_seconds,
        "age_group": record.speaker.age_group,
        "age_proxy": record.acoustic.acoustic_age_proxy,
        "augmentation_id": record.augmentation_id,
        "hard_negative_tier": record.hard_negative_tier,
    }


class Int8StreamingRunner:
    def __init__(self, model_path: Path, stride: int):
        self.model_path = model_path
        self.interpreter = self._new_interpreter()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.stride = stride
        self.input_scale, self.input_zero_point = self.input["quantization"]
        self.output_scale, self.output_zero_point = self.output["quantization"]

    def _new_interpreter(self):
        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        return interpreter

    def score(self, features: np.ndarray) -> tuple[int, int, float, float]:
        # reset_all_variables() does not reliably reset this model's resource-backed
        # streaming states. A fresh interpreter provides deterministic clip isolation.
        self.interpreter = self._new_interpreter()
        input_detail = self.interpreter.get_input_details()[0]
        output_detail = self.interpreter.get_output_details()[0]
        raw_outputs: list[int] = []
        usable = len(features) - len(features) % self.stride
        for offset in range(0, usable, self.stride):
            chunk = features[offset : offset + self.stride]
            # Preserve the exact upstream baseline behavior: truncating astype, not rounding.
            quantized = (chunk / self.input_scale + self.input_zero_point).astype(input_detail["dtype"])
            self.interpreter.set_tensor(input_detail["index"], quantized.reshape(input_detail["shape"]))
            self.interpreter.invoke()
            raw_outputs.append(int(self.interpreter.get_tensor(output_detail["index"])[0][0]))
        raw_max = max(raw_outputs, default=int(self.output_zero_point))
        raw_final = raw_outputs[-1] if raw_outputs else int(self.output_zero_point)
        dequantized_max = self.output_scale * (raw_max - self.output_zero_point)
        dequantized_final = self.output_scale * (raw_final - self.output_zero_point)
        return raw_max, raw_final, float(dequantized_max), float(dequantized_final)

    def metadata(self) -> dict[str, object]:
        return {
            "input_dtype": np.dtype(self.input["dtype"]).name,
            "input_shape": self.input["shape"].tolist(),
            "input_scale": float(self.input_scale),
            "input_zero_point": int(self.input_zero_point),
            "input_quantization_behavior": "upstream truncating astype(value / scale + zero_point)",
            "stream_state_reset_strategy": "fresh TFLite interpreter per audio clip",
            "output_dtype": np.dtype(self.output["dtype"]).name,
            "output_shape": self.output["shape"].tolist(),
            "output_scale": float(self.output_scale),
            "output_zero_point": int(self.output_zero_point),
            "dequantization_formula": "real = output_scale * (raw - output_zero_point)",
            "raw_max": int(np.iinfo(self.output["dtype"]).max),
            "real_value_at_raw_max": float(
                self.output_scale * (np.iinfo(self.output["dtype"]).max - self.output_zero_point)
            ),
            "legacy_upstream_formula": "legacy_score = raw / 255",
        }


def build_models(raw: dict[str, object], run_dir: Path, weights: Path):
    runtime, flags = build_runtime_config(raw, run_dir)
    source = mixednet.model(flags, runtime["training_input_shape"], batch_size=1)
    source.load_weights(weights)
    float_streaming = model_to_saved(source, runtime, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    float_non_streaming = mixednet.model(flags, runtime["training_input_shape"], batch_size=1)
    float_non_streaming.load_weights(weights)
    state_variables = [variable for variable in float_streaming.variables if variable.name == "states"]

    @tf.function(input_signature=[tf.TensorSpec([1, runtime["spectrogram_length"], 40], tf.float32)])
    def score_non_streaming(features):
        return tf.reshape(float_non_streaming(features, training=False), [])

    @tf.function(input_signature=[tf.TensorSpec([None, int(runtime["stride"]), 40], tf.float32)])
    def score_streaming(chunks):
        for variable in state_variables:
            variable.assign(tf.zeros_like(variable))
        maximum = tf.constant(0.0, dtype=tf.float32)
        final = tf.constant(0.0, dtype=tf.float32)
        for index in tf.range(tf.shape(chunks)[0]):
            final = tf.reshape(float_streaming(chunks[index : index + 1], training=False), [])
            maximum = tf.maximum(maximum, final)
        return maximum, final

    return runtime, score_non_streaming, score_streaming, state_variables


def score_records(
    records: list[object],
    dataset_root: Path,
    runtime: dict[str, object],
    float_non_streaming,
    float_streaming,
    int8_runner: Int8StreamingRunner,
    split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stride = int(runtime["stride"])
    for index, record in enumerate(records, start=1):
        audio = load_inference_audio(dataset_root / record.audio_path)
        features = generate_features_for_clip(audio, step_ms=int(runtime["window_step_ms"]))
        if index == 1:
            print("FIRST SAMPLE LOADED", flush=True)
        fixed = fixed_length_spectrogram(
            features, int(runtime["spectrogram_length"]), "truncate_start"
        ).astype(np.float32)
        float_score = float(float_non_streaming(fixed[np.newaxis, ...]).numpy())
        usable = len(features) - len(features) % stride
        chunks = features[:usable].reshape(-1, stride, features.shape[1]).astype(np.float32)
        float_stream_max, float_stream_final = float_streaming(chunks)
        raw_max, raw_final, int8_score, int8_final = int8_runner.score(features)
        row = {
            **record_metadata(record, dataset_root),
            "float_score": float_score,
            "float_streaming_score": float(float_stream_max.numpy()),
            "float_streaming_final_score": float(float_stream_final.numpy()),
            "int8_raw_output": raw_max,
            "int8_raw_final_output": raw_final,
            "int8_dequantized_score": int8_score,
            "int8_dequantized_final_score": int8_final,
            "int8_legacy_score": raw_max / 255.0,
            "difference": int8_score - float_score,
            "difference_int8_vs_float_streaming": int8_score - float(float_stream_max.numpy()),
            "feature_frames": int(len(features)),
            "stream_packets": int(len(chunks)),
            "discarded_feature_frames": int(len(features) - usable),
        }
        rows.append(row)
        heartbeat_interval = 10 if len(records) <= 20 else 50
        if index % heartbeat_interval == 0 or index == len(records):
            print(f"DIAG_HEARTBEAT split={split} records={index}/{len(records)}", flush=True)
    return rows


def noise_family(noise_id: str | None) -> str:
    if not noise_id or noise_id == "clean":
        return "clean"
    return noise_id.split(":", 1)[0]


def split_audit(manifest: DatasetManifest) -> dict[str, object]:
    result: dict[str, object] = {}
    split_records: dict[str, list[object]] = {
        split: [row for row in manifest.records if row.split == split]
        for split in ("train", "validation", "test")
    }
    for split, rows in split_records.items():
        result[split] = {
            "count": len(rows),
            "label": dict(sorted(Counter(row.label for row in rows).items())),
            "source_tts_family": dict(sorted(Counter(row.speaker.source for row in rows).items())),
            "speaker": dict(sorted(Counter(row.speaker.speaker_id for row in rows).items())),
            "unique_speakers": len({row.speaker.speaker_id for row in rows}),
            "text": dict(sorted(Counter(str(row.text) for row in rows).items())),
            "noise_type": dict(sorted(Counter(noise_family(row.acoustic.noise_id) for row in rows).items())),
            "snr_db": dict(sorted(Counter(str(row.acoustic.snr_db) for row in rows).items())),
            "duration_seconds": describe([float(row.duration_seconds) for row in rows]),
            "positive_duration_seconds": describe(
                [float(row.duration_seconds) for row in rows if row.label == "positive"]
            ),
            "age_group": dict(sorted(Counter(str(row.speaker.age_group) for row in rows).items())),
            "age_proxy": dict(sorted(Counter(str(row.acoustic.acoustic_age_proxy) for row in rows).items())),
            "augmentation_present": sum(row.augmentation_id is not None for row in rows),
        }
    result["speaker_overlap"] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_speakers = {row.speaker.speaker_id for row in split_records[left]}
        right_speakers = {row.speaker.speaker_id for row in split_records[right]}
        result["speaker_overlap"][f"{left}_vs_{right}"] = sorted(left_speakers & right_speakers)
    return result


def audio_statistics(audio: np.ndarray) -> dict[str, float]:
    active = np.flatnonzero(np.abs(audio) >= 0.01)
    leading = float(active[0] / 16000.0) if active.size else float(len(audio) / 16000.0)
    trailing = float((len(audio) - 1 - active[-1]) / 16000.0) if active.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return {
        "actual_duration_seconds": float(len(audio) / 16000.0),
        "leading_silence_seconds_at_-40dbfs": leading,
        "trailing_silence_seconds_at_-40dbfs": trailing,
        "rms": rms,
        "rms_dbfs": float(20.0 * math.log10(max(rms, 1e-12))),
        "peak": float(np.max(np.abs(audio))),
    }


def positive_sample_audit(manifest: DatasetManifest, dataset_root: Path, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        candidates = [row for row in manifest.records if row.split == split and row.label == "positive"]
        for record in rng.sample(candidates, 10):
            audio = load_inference_audio(dataset_root / record.audio_path)
            rows.append({**record_metadata(record, dataset_root), **audio_statistics(audio)})
    return rows


def feature_audit(
    manifest: DatasetManifest,
    dataset_root: Path,
    runtime: dict[str, object],
    int8_runner: Int8StreamingRunner,
) -> dict[str, object]:
    records = [
        next(row for row in manifest.records if row.split == split and row.label == "positive")
        for split in ("validation", "test")
    ]
    output: dict[str, object] = {}
    for record in records:
        path = dataset_root / record.audio_path
        training_audio = load_training_audio(path)
        inference_audio = load_inference_audio(path)
        training_features = generate_features_for_clip(training_audio, step_ms=int(runtime["window_step_ms"]))
        inference_features = generate_features_for_clip(inference_audio, step_ms=int(runtime["window_step_ms"]))
        quantized = (
            inference_features / int8_runner.input_scale + int8_runner.input_zero_point
        ).astype(int8_runner.input["dtype"])
        dequantized = int8_runner.input_scale * (
            quantized.astype(np.float32) - int8_runner.input_zero_point
        )
        output[record.record_id] = {
            "path": str(path.resolve()),
            "training_feature_shape": list(training_features.shape),
            "streaming_feature_shape": list(inference_features.shape),
            "training_range": [float(training_features.min()), float(training_features.max())],
            "training_mean_std": [float(training_features.mean()), float(training_features.std())],
            "streaming_range": [float(inference_features.min()), float(inference_features.max())],
            "streaming_mean_std": [float(inference_features.mean()), float(inference_features.std())],
            "frontend_max_abs_difference": float(np.max(np.abs(training_features - inference_features))),
            "frontend_exact_equal": bool(np.array_equal(training_features, inference_features)),
            "quantized_input_range": [int(quantized.min()), int(quantized.max())],
            "dequantized_input_range": [float(dequantized.min()), float(dequantized.max())],
            "input_quantization_mae": float(np.mean(np.abs(dequantized - inference_features))),
            "feature_frames": int(len(inference_features)),
            "complete_stream_packets": int(len(inference_features) // int(runtime["stride"])),
            "discarded_tail_frames": int(len(inference_features) % int(runtime["stride"])),
        }
    return output


def saturation_audit(rows: list[dict[str, object]], raw_max: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    saturated = [row for row in rows if int(row["int8_raw_output"]) == raw_max]
    totals = Counter((str(row["split"]), str(row["label"])) for row in rows)
    grouped: dict[str, object] = {}
    for (split, label), total in sorted(totals.items()):
        selected = [row for row in saturated if row["split"] == split and row["label"] == label]
        grouped[f"{split}:{label}"] = {
            "count": len(selected),
            "total": total,
            "proportion": len(selected) / total,
            "source": dict(sorted(Counter(str(row["source"]) for row in selected).items())),
            "text": dict(sorted(Counter(str(row["text"]) for row in selected).items())),
            "noise": dict(sorted(Counter(str(row["noise"]) for row in selected).items())),
            "snr_db": dict(sorted(Counter(str(row["snr_db"]) for row in selected).items())),
            "corresponding_float_score": describe([float(row["float_score"]) for row in selected]),
            "corresponding_float_streaming_score": describe(
                [float(row["float_streaming_score"]) for row in selected]
            ),
            "float_streaming_below_0_8": sum(float(row["float_streaming_score"]) < 0.8 for row in selected),
            "float_streaming_at_least_0_95": sum(
                float(row["float_streaming_score"]) >= 0.95 for row in selected
            ),
        }
    return {"raw_saturation_value": raw_max, "total_saturated": len(saturated), "groups": grouped}, saturated


def main() -> None:
    print("PHASE2C START", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    print("CONFIG LOADED", flush=True)
    run_dir = args.run_dir.resolve()
    requested_output_dir = args.output_dir.resolve()
    output_dir = (
        requested_output_dir / f"limit_{args.limit}"
        if args.limit is not None
        else requested_output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("OUTPUT DIRECTORY READY", flush=True)
    status_path = run_dir / "TRAINING_STATUS.json"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    print("MODEL FILE FOUND", flush=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise RuntimeError("Frozen diagnosis requires COMPLETED training status")
    manifest_path = Path(config["dataset_manifest"]).resolve()
    weights = Path(status["best_checkpoint"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    print("CHECKPOINT FOUND", flush=True)
    tflite_path = run_dir / "final_model/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    if not tflite_path.is_file():
        raise FileNotFoundError(tflite_path)
    print("TFLITE FOUND", flush=True)
    if sha256_file(manifest_path) != status["dataset_manifest_sha256"]:
        raise RuntimeError("DatasetManifest changed since formal training")
    hashes_before = {
        "manifest": sha256_file(manifest_path),
        "best_checkpoint": sha256_file(weights),
        "tflite": hashlib.sha256(tflite_path.read_bytes()).hexdigest(),
    }
    print("HASH CHECK PASSED", flush=True)
    manifest = DatasetManifest.load(manifest_path)
    dataset_root = Path(manifest.root).resolve()

    print("DIAG_STAGE build_frozen_models", flush=True)
    runtime, float_non_streaming, float_streaming, state_variables = build_models(config, run_dir, weights)
    print("FLOAT MODEL LOADED", flush=True)
    int8_runner = Int8StreamingRunner(tflite_path, int(runtime["stride"]))
    print("TFLITE INTERPRETER LOADED", flush=True)
    if not state_variables:
        raise RuntimeError("Float streaming conversion did not expose internal state variables")

    validation_records = [row for row in manifest.records if row.split == "validation"]
    test_records = [row for row in manifest.records if row.split == "test"]
    if args.limit is not None:
        validation_records = stratified_limit(validation_records, args.limit)
        test_records = stratified_limit(test_records, args.limit)
        print(
            "LIMIT_SELECTION "
            f"validation={dict(Counter(row.label for row in validation_records))} "
            f"test={dict(Counter(row.label for row in test_records))}",
            flush=True,
        )
    print("DIAG_STAGE score_validation", flush=True)
    validation_rows = score_records(
        validation_records, dataset_root, runtime, float_non_streaming, float_streaming, int8_runner, "validation"
    )
    print("DIAG_STAGE score_test", flush=True)
    test_rows = score_records(
        test_records, dataset_root, runtime, float_non_streaming, float_streaming, int8_runner, "test"
    )

    score_keys = (
        "float_score",
        "float_streaming_score",
        "int8_dequantized_score",
        "int8_legacy_score",
    )
    matrix: dict[str, object] = {}
    thresholds: dict[str, object] = {}
    for key in score_keys:
        threshold, validation_metrics = select_threshold(validation_rows, key)
        thresholds[key] = {
            "selection_split": "validation",
            "rule": "maximize F1 subject to Recall >= 0.98; tie-break Precision/FPR/threshold",
            "threshold": threshold,
        }
        matrix[key] = {
            "validation": validation_metrics,
            "test": binary_metrics(test_rows, key, threshold),
        }

    all_rows = validation_rows + test_rows
    quantization = int8_runner.metadata()
    saturation, saturated_rows = saturation_audit(all_rows, int(quantization["raw_max"]))
    correlations = {
        "test_float_nonstream_vs_int8_streaming": correlation(
            test_rows, "float_score", "int8_dequantized_score"
        ),
        "test_float_streaming_vs_int8_streaming": correlation(
            test_rows, "float_streaming_score", "int8_dequantized_score"
        ),
        "test_float_nonstream_vs_float_streaming_final": correlation(
            test_rows, "float_score", "float_streaming_final_score"
        ),
        "test_float_streaming_final_vs_int8_streaming_final": correlation(
            test_rows, "float_streaming_final_score", "int8_dequantized_final_score"
        ),
    }

    legacy_comparison: dict[str, object] = {
        "correct_formula": "output_scale * (raw - output_zero_point)",
        "legacy_formula": "raw / 255",
        "correct_threshold": thresholds["int8_dequantized_score"]["threshold"],
        "legacy_threshold": thresholds["int8_legacy_score"]["threshold"],
        "same_raw_operating_point": (
            matrix["int8_dequantized_score"]["validation"]["confusion_matrix"]
            == matrix["int8_legacy_score"]["validation"]["confusion_matrix"]
            and matrix["int8_dequantized_score"]["test"]["confusion_matrix"]
            == matrix["int8_legacy_score"]["test"]["confusion_matrix"]
        ),
        "metric_delta_correct_minus_legacy": {},
    }
    for split in ("validation", "test"):
        legacy_comparison["metric_delta_correct_minus_legacy"][split] = {
            metric: float(matrix["int8_dequantized_score"][split][metric])
            - float(matrix["int8_legacy_score"][split][metric])
            for metric in (
                "recall_tpr",
                "precision",
                "f1",
                "false_positive_rate",
                "false_rejection_rate",
                "roc_auc",
                "pr_auc",
            )
            if matrix["int8_dequantized_score"][split][metric] is not None
            and matrix["int8_legacy_score"][split][metric] is not None
        }

    label_score_summary: dict[str, object] = {}
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        label_score_summary[split] = {}
        for label in ("positive", "negative", "hard_negative", "ambient"):
            selected = [row for row in rows if row["label"] == label]
            label_score_summary[split][label] = {
                key: float(np.mean([float(row[key]) for row in selected])) if selected else None
                for key in ("float_score", "float_streaming_score", "int8_dequantized_score")
            }
            print(
                f"SCORE_MEAN split={split} label={label} "
                + " ".join(
                    f"{key}={label_score_summary[split][label][key]:.8f}"
                    for key in ("float_score", "float_streaming_score", "int8_dequantized_score")
                    if label_score_summary[split][label][key] is not None
                ),
                flush=True,
            )
    print(
        "CORRELATION float_nonstream_vs_float_stream="
        f"{correlations['test_float_nonstream_vs_float_streaming_final']['pearson']:.8f} "
        "float_stream_vs_int8_stream="
        f"{correlations['test_float_streaming_vs_int8_streaming']['pearson']:.8f}",
        flush=True,
    )

    audit_split = split_audit(manifest)
    sampled_positives = positive_sample_audit(manifest, dataset_root, int(config["seed"]))
    features = feature_audit(manifest, dataset_root, runtime, int8_runner)

    required_test_columns = [
        "path", "label", "source", "float_score", "int8_raw_output",
        "int8_dequantized_score", "difference",
    ]
    extra_columns = [key for key in test_rows[0] if key not in required_test_columns]
    required_test_rows = [
        {key: row[key] for key in required_test_columns + extra_columns} for row in test_rows
    ]
    write_csv(output_dir / "float_vs_int8_scores.csv", required_test_rows)
    write_csv(output_dir / "validation_model_scores.csv", validation_rows)
    write_csv(output_dir / "int8_saturated_samples.csv", saturated_rows)
    write_csv(output_dir / "positive_sample_audit.csv", sampled_positives)

    sample_20: list[dict[str, object]] = []
    for label in ("positive", "negative", "hard_negative", "ambient"):
        sample_20.extend([row for row in test_rows if row["label"] == label][:5])
    sample_20 = [
        {
            "path": row["path"],
            "label": row["label"],
            "raw_uint8": row["int8_raw_output"],
            "dequantized_float": row["int8_dequantized_score"],
            "legacy_raw_div_255": row["int8_legacy_score"],
            "float_nonstream": row["float_score"],
            "float_streaming": row["float_streaming_score"],
        }
        for row in sample_20
    ]
    write_csv(output_dir / "quantization_samples_20.csv", sample_20)
    for row in sample_20:
        print(
            "QUANT_SAMPLE "
            f"label={row['label']} raw={row['raw_uint8']} "
            f"dequant={float(row['dequantized_float']):.8f} "
            f"float={float(row['float_nonstream']):.8f} "
            f"float_stream={float(row['float_streaming']):.8f}",
            flush=True,
        )

    hashes_after = {
        "manifest": sha256_file(manifest_path),
        "best_checkpoint": sha256_file(weights),
        "tflite": hashlib.sha256(tflite_path.read_bytes()).hexdigest(),
    }
    if hashes_before != hashes_after:
        raise RuntimeError("A frozen input artifact changed during diagnosis")

    atomic_json(output_dir / "diagnostic_matrix.json", matrix)
    atomic_json(output_dir / "thresholds.json", thresholds)
    atomic_json(output_dir / "quantization_metadata.json", quantization)
    atomic_json(output_dir / "score_correlations.json", correlations)
    atomic_json(output_dir / "legacy_raw_div_255_comparison.json", legacy_comparison)
    atomic_json(output_dir / "label_score_summary.json", label_score_summary)
    atomic_json(output_dir / "saturation_audit.json", saturation)
    atomic_json(output_dir / "dataset_split_audit.json", audit_split)
    atomic_json(output_dir / "feature_consistency_audit.json", features)
    atomic_json(
        output_dir / "diagnostic_provenance.json",
        {
            "schema": "wakeword-studio.phase2c-diagnosis/v1",
            "training_status": status["status"],
            "training_steps": status["current_step"],
            "manifest_path": str(manifest_path),
            "best_checkpoint": str(weights),
            "tflite_path": str(tflite_path),
            "frozen_hashes_before": hashes_before,
            "frozen_hashes_after": hashes_after,
            "manifest_unchanged": hashes_before["manifest"] == hashes_after["manifest"],
            "no_training_api_used": True,
        },
    )
    print(json.dumps({"matrix": matrix, "correlations": correlations, "quantization": quantization}, indent=2), flush=True)
    print(f"DIAG_COMPLETE output_dir={output_dir}", flush=True)
    print("DIAG COMPLETE", flush=True)


if __name__ == "__main__":
    main()
