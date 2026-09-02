"""Phase 3C frozen RepCNN export and strictly gated evaluation.

The normal short path (``export`` and ``smoke``) reads Train/Validation feature
caches only.  Held-out audio is reachable only through the explicit Test stages,
after a Validation threshold freeze has been written and hash-verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit.embedded_wakeword.models.classifier import reparameterize_model  # noqa: E402
from phase3.scripts.preflight_repcnn_model_b import (  # noqa: E402
    FeatureLoader,
    build_model,
    quantization_metadata,
)
from wakeword_studio.dataset.manifest import DatasetManifest  # noqa: E402
from wakeword_studio.dataset.repcnn_adapter import REPCNN_LABELS, RepCNNSample  # noqa: E402
from wakeword_studio.frontends import load_inference_audio, load_training_audio  # noqa: E402
from wakeword_studio.training.streaming_windows import (  # noqa: E402
    extract_streaming_window,
    plan_streaming_window,
)


EXPECTED_BEST_STEP = 1750
EXPECTED_BEST_F1 = 0.7813953488372092
EXPECTED_FROZEN_THRESHOLD = 0.84375
V1_EXTERNAL_PROTOCOL_ID = "repcnn_v1_fullwav_sliding2s_max_v1"
V1_MANIFEST_SHA256 = "70b089652a7f8eb407c9d23ccc0efe7e33ce241fad2309f87f35702dc4752391"
MACS_PER_INVOCATION = 210_102_000
REASONABLE_FPR_LIMIT = 0.01
MODEL_A_V3_BASELINE = {
    "model_size_bytes": 52_840,
    "model_size_kib": 51.6015625,
    "v2_test": {
        "recall_tpr": 0.59, "precision": 0.7444795, "f1": 0.6582985,
        "false_positive_rate": 0.0675, "hard_negative_fpr": 0.136111,
        "ordinary_negative_fpr": 0.05, "ambient_fpr": 0.0,
        "roc_auc": 0.82870208, "pr_auc": 0.69874107,
    },
    "v1_external_test": {
        "recall_tpr": 0.45, "false_positive_rate": 0.0714286,
        "roc_auc": 0.76654643,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_log_text(path: Path) -> str:
    content = path.read_bytes()
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    return content.decode("utf-8-sig")


def quantize_input(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Input quantization scale must be positive")
    return np.clip(np.rint(value / scale + zero_point), -128, 127).astype(np.int8)


def dequantize_output(raw: np.ndarray | int, scale: float, zero_point: int) -> np.ndarray:
    """Apply the TFLite affine formula; raw/255 is intentionally forbidden."""

    if scale <= 0:
        raise ValueError("Output quantization scale must be positive")
    return scale * (np.asarray(raw, dtype=np.float64) - zero_point)


def fullwav_window_starts(num_samples: int, sample_rate_hz: int) -> list[int]:
    """Return all approved 2-second window starts before any inference.

    A negative start represents deterministic symmetric zero padding for a WAV
    shorter than two seconds.  When the missing sample count is odd, integer
    division leaves the extra sample on the trailing side.
    """

    if num_samples <= 0 or sample_rate_hz <= 0:
        raise ValueError("Audio length and sample rate must be positive")
    window_samples = 2 * sample_rate_hz
    if num_samples <= window_samples:
        leading_padding = (window_samples - num_samples) // 2
        return [-leading_padding]
    starts = list(range(0, num_samples - window_samples + 1, sample_rate_hz))
    tail_start = num_samples - window_samples
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def fullwav_windows(audio: np.ndarray, sample_rate_hz: int) -> list[tuple[int, np.ndarray]]:
    """Materialize the approved, label/source/score-independent clip list."""

    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    window_samples = 2 * sample_rate_hz
    starts = fullwav_window_starts(len(waveform), sample_rate_hz)
    windows: list[tuple[int, np.ndarray]] = []
    for start in starts:
        clip = np.zeros(window_samples, dtype=np.float32)
        source_start = max(0, start)
        source_end = min(len(waveform), start + window_samples)
        destination_start = source_start - start
        if source_end > source_start:
            clip[destination_start : destination_start + source_end - source_start] = waveform[
                source_start:source_end
            ]
        windows.append((start, clip))
    return windows


def score_fullwav_clips(
    windows: list[tuple[int, np.ndarray]], score_clip: Any
) -> dict[str, object]:
    """Score every predetermined clip once, then take the record-level maximum."""

    if not windows:
        raise ValueError("At least one predetermined window is required")
    scores = [float(score_clip(clip)) for _, clip in windows]
    winning_index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return {
        "window_scores": scores,
        "record_score": scores[winning_index],
        "winning_window_index": winning_index,
        "winning_window_start_sample": int(windows[winning_index][0]),
    }


def binary_metrics(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    labels = np.asarray([row["label"] == "positive" for row in rows], dtype=bool)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
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
        "count": len(rows), "threshold": float(threshold), "recall_tpr": recall,
        "precision": precision, "f1": f1,
        "false_rejection_rate": None if recall is None else 1.0 - recall,
        "false_positive_rate": fpr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    if len(set(labels.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        result["roc_auc"] = result["pr_auc"] = None
    return result


def threshold_sweep(
    rows: list[dict[str, object]], output_scale: float, output_zero_point: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    thresholds = [output_scale * (raw - output_zero_point) for raw in range(-128, 128)]
    thresholds.append(float(np.nextafter(max(thresholds), np.inf)))
    sweep = [binary_metrics(rows, float(threshold)) for threshold in thresholds]
    best = max(
        sweep,
        key=lambda row: (
            float(row["f1"]), float(row["recall_tpr"] or 0), float(row["precision"]),
            -float(row["false_positive_rate"] or 0), float(row["threshold"]),
        ),
    )
    targets: dict[str, object] = {}
    for target in (0.90, 0.95, 0.98):
        feasible = [row for row in sweep if float(row["recall_tpr"] or 0) >= target]
        if not feasible:
            targets[f"recall_at_least_{int(target * 100)}"] = {
                "feasible": False, "verdict": f"NO {int(target * 100)}% OPERATING POINT"
            }
            continue
        selected = max(
            feasible,
            key=lambda row: (
                float(row["precision"]), -float(row["false_positive_rate"] or 0),
                float(row["f1"]), float(row["threshold"]),
            ),
        )
        reasonable = float(selected["false_positive_rate"] or 0) <= REASONABLE_FPR_LIMIT
        targets[f"recall_at_least_{int(target * 100)}"] = {
            **selected,
            "reasonable_policy": "Validation FPR <= 0.01",
            "reasonable": reasonable,
            "verdict": "REASONABLE OPERATING POINT" if reasonable else f"NO REASONABLE {int(target * 100)}% OPERATING POINT",
        }
    return sweep, {"best_f1": best, "recall_targets": targets}


def load_context(config_path: Path, run_dir: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status = json.loads((run_dir / "TRAINING_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED" or not status.get("early_stopped"):
        raise RuntimeError("Formal Model B training is not completed with early stopping")
    if int(status.get("final_step", -1)) != 4250 or bool(status.get("test_loaded")):
        raise RuntimeError("Training status final-step/Test-access contract changed")
    if abs(float(status.get("best_validation_f1", -1)) - EXPECTED_BEST_F1) > 1e-12:
        raise RuntimeError("Frozen best Validation F1 changed")

    best_lines = [
        line for line in read_log_text(run_dir / "training.log").splitlines()
        if "VALIDATION step=" in line and "best=True" in line
    ]
    if not best_lines or f"VALIDATION step={EXPECTED_BEST_STEP} " not in best_lines[-1]:
        raise RuntimeError("Training log does not prove that the final best event was step 1750")
    if "f1=0.781395" not in best_lines[-1]:
        raise RuntimeError("Step-1750 training-log F1 changed")

    best_weights = (run_dir / "best_weights.weights.h5").resolve()
    if not best_weights.is_file():
        raise FileNotFoundError(best_weights)
    manifest_path = Path(config["dataset_manifest"]).resolve()
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(config["dataset_manifest_sha256"]).lower():
        raise RuntimeError("Dataset manifest differs from frozen config")
    if manifest_sha != str(status["dataset_manifest_sha256"]).lower():
        raise RuntimeError("Dataset manifest differs from the completed run")
    cache_root = run_dir / "feature_cache_train_validation_only"
    cache_summary = json.loads((cache_root / "summary.json").read_text(encoding="utf-8"))
    if cache_summary.get("test_loaded") is not False or cache_summary.get("dataset_manifest_sha256") != manifest_sha:
        raise RuntimeError("Train/Validation feature cache integrity check failed")
    return {
        "config": config, "config_path": config_path, "config_sha256": sha256_file(config_path),
        "status": status, "best_weights": best_weights,
        "best_weights_sha256": sha256_file(best_weights), "best_log_line": best_lines[-1],
        "manifest_path": manifest_path, "manifest_sha256": manifest_sha, "cache_root": cache_root,
    }


class Int8Scorer:
    """Stateless, clip-level, one-input/one-sigmoid-output TFLite scorer."""

    def __init__(self, model_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("RepCNN deployment must have exactly one input and one output")
        self.input = inputs[0]
        self.output = outputs[0]
        self.metadata = quantization_metadata(self.interpreter)
        if self.metadata["input_dtype"] != "int8" or self.metadata["output_dtype"] != "int8":
            raise RuntimeError("Deployment model is not full INT8")

    def score(self, feature: np.ndarray) -> dict[str, object]:
        input_scale, input_zp = self.input["quantization"]
        output_scale, output_zp = self.output["quantization"]
        quantized = quantize_input(feature[np.newaxis, ...], float(input_scale), int(input_zp))
        self.interpreter.set_tensor(self.input["index"], quantized)
        self.interpreter.invoke()
        raw = int(np.asarray(self.interpreter.get_tensor(self.output["index"])).reshape(-1)[0])
        score = float(dequantize_output(raw, float(output_scale), int(output_zp)))
        return {"raw_int8": raw, "score": score}


def _representatives(context: dict[str, object], count: int = 4) -> np.ndarray:
    values: list[np.ndarray] = []
    root = context["cache_root"]
    for split in ("train", "validation"):
        for label in REPCNN_LABELS:
            array = np.load(root / f"{split}_{label}.npy", mmap_mode="r")
            values.extend(np.asarray(array[:count], dtype=np.float32))
    return np.stack(values)


def export_frozen(context: dict[str, object], output_root: Path) -> dict[str, object]:
    frozen_root = output_root / "frozen_checkpoint"
    frozen_weights = frozen_root / "best_step_1750.weights.h5"
    freeze_path = frozen_root / "checkpoint_freeze.json"
    frozen_root.mkdir(parents=True, exist_ok=True)
    if not frozen_weights.exists():
        shutil.copy2(context["best_weights"], frozen_weights)
    if sha256_file(frozen_weights) != context["best_weights_sha256"]:
        raise RuntimeError("Frozen weight copy differs from formal best_weights.weights.h5")
    freeze = {
        "schema": "wakeword-studio.repcnn-checkpoint-freeze/v1", "frozen_at": utc_now(),
        "best_step": EXPECTED_BEST_STEP, "best_validation_f1": EXPECTED_BEST_F1,
        "source": str(context["best_weights"]), "frozen_copy": str(frozen_weights),
        "best_weights_sha256": context["best_weights_sha256"],
        "config_path": str(context["config_path"]), "config_sha256": context["config_sha256"],
        "dataset_manifest_path": str(context["manifest_path"]),
        "dataset_manifest_sha256": context["manifest_sha256"],
        "training_log_evidence": context["best_log_line"],
        "last_weights_used": False, "checkpoint_4250_used": False, "test_loaded": False,
    }
    if freeze_path.exists():
        existing_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        immutable = (
            "best_step", "best_validation_f1", "best_weights_sha256",
            "config_sha256", "dataset_manifest_sha256",
        )
        if any(existing_freeze.get(key) != freeze.get(key) for key in immutable):
            raise RuntimeError("Existing checkpoint freeze no longer matches formal artifacts")
        freeze = existing_freeze
    else:
        atomic_json(freeze_path, freeze)

    model_path = output_root / "final_model" / "qingxiaojia_repcnn_performance_v1_best1750_full_int8.tflite"
    model_info_path = output_root / "model_info.json"
    if model_path.is_file() and model_info_path.is_file():
        existing_model = json.loads(model_info_path.read_text(encoding="utf-8"))
        if existing_model.get("sha256") != sha256_file(model_path):
            raise RuntimeError("Existing frozen TFLite hash mismatch")
        if existing_model.get("test_loaded") is not False:
            raise RuntimeError("Existing export provenance indicates Test access")
        inspected = Int8Scorer(model_path).metadata
        if inspected != existing_model.get("quantization"):
            raise RuntimeError("Existing TFLite quantization metadata changed")
        print("PHASE3C FROZEN EXPORT REUSED hash_verified=true", flush=True)
        return existing_model

    started = time.perf_counter()
    model = build_model(context["config"])
    model.load_weights(frozen_weights)
    fused = reparameterize_model(model)
    representatives = _representatives(context)
    probe = representatives[:4]
    fusion_error = float(np.max(np.abs(np.asarray(model(probe, training=False)) - np.asarray(fused(probe, training=False)))))
    if fusion_error > 1e-4:
        raise RuntimeError(f"RepCNN fusion error too large: {fusion_error}")
    shape = tuple(int(value) for value in context["config"]["frontend"]["input_shape"])

    @tf.function(input_signature=[tf.TensorSpec((1, *shape), tf.float32)])
    def serving(value: tf.Tensor) -> tf.Tensor:
        return fused(value, training=False)

    def representative_dataset():
        for feature in representatives:
            yield [feature[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serving.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(content)
    scorer = Int8Scorer(model_path)
    operators = sorted({row["op_name"] for row in scorer.interpreter._get_ops_details() if row["op_name"] != "DELEGATE"})
    report = {
        "status": "PASS", "path": str(model_path.resolve()), "bytes": len(content),
        "kib": len(content) / 1024.0, "sha256": sha256_file(model_path),
        "training_parameter_count": int(model.count_params()),
        "deployment_parameter_count": int(fused.count_params()),
        "fusion_max_abs_error": fusion_error, "operators": operators,
        "macs_per_99_frame_invocation": MACS_PER_INVOCATION,
        "mac_source": "Phase 3A TFLite converter measured 420,204,000 arithmetic ops / 2",
        "quantization": scorer.metadata,
        "semantics": "one clip-level sigmoid probability for one [99,40] feature clip",
        "stateful": False, "state_location": "none inside model; any rolling clip buffer is external",
        "test_loaded": False, "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(model_info_path, report)
    atomic_json(output_root / "export_provenance.json", {"checkpoint": freeze, "model": report, "training_api_used": False})
    return report


def cached_rows(context: dict[str, object], split: str, scorer: Int8Scorer, limit_per_label: int | None = None) -> list[dict[str, object]]:
    if split != "validation":
        raise ValueError("Cached evaluation path permits Validation only")
    rows: list[dict[str, object]] = []
    for label in REPCNN_LABELS:
        array = np.load(context["cache_root"] / f"validation_{label}.npy", mmap_mode="r")
        metadata = load_jsonl(context["cache_root"] / f"validation_{label}.jsonl")
        count = len(array) if limit_per_label is None else min(limit_per_label, len(array))
        for index in range(count):
            rows.append({**metadata[index], **scorer.score(np.asarray(array[index], np.float32))})
            if len(rows) % 50 == 0:
                print(f"PHASE3C HEARTBEAT split=validation scored={len(rows)}", flush=True)
    return rows


def verify_threshold_gate(context: dict[str, object], output_root: Path, model_info: dict[str, object]) -> dict[str, object]:
    path = output_root / "threshold_freeze.json"
    if not path.is_file():
        raise RuntimeError("Validation threshold must be frozen before any Test access")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint_sha256": context["best_weights_sha256"],
        "config_sha256": context["config_sha256"],
        "v2_manifest_sha256": context["manifest_sha256"],
        "tflite_sha256": model_info["sha256"],
    }
    if any(freeze.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Frozen threshold provenance no longer matches frozen artifacts")
    if freeze.get("selection_split") != "v2_validation_only":
        raise RuntimeError("Frozen threshold was not selected exclusively on v2 Validation")
    if float(freeze.get("selected_threshold", -1)) != EXPECTED_FROZEN_THRESHOLD:
        raise RuntimeError(
            f"Frozen threshold must remain exactly {EXPECTED_FROZEN_THRESHOLD}"
        )
    if freeze.get("v2_test_loaded") is not False or freeze.get("v1_external_test_loaded") is not False:
        raise RuntimeError("Threshold freeze must predate all held-out Test access")
    return freeze


def adapt_records(manifest_path: Path, split: str, config: dict[str, object]) -> list[RepCNNSample]:
    manifest = DatasetManifest.load(manifest_path)
    root = Path(manifest.root).resolve()
    result: list[RepCNNSample] = []
    for row in manifest.records:
        if row.split != split:
            continue
        window = plan_streaming_window(
            record_id=row.record_id, label=row.label, duration_seconds=float(row.duration_seconds or 0),
            phrase_start_ms=row.acoustic.phrase_start_ms, phrase_end_ms=row.acoustic.phrase_end_ms,
            phrase_placement=row.acoustic.phrase_placement,
            window_ms=float(config["augmentation"]["clip_duration"]) * 1000, seed=int(config["seed"]),
        )
        result.append(RepCNNSample(
            row.record_id, (root / row.audio_path).resolve(), row.label,
            1.0 if row.label == "positive" else 0.0, row.split, row.speaker.speaker_id,
            row.speaker.source, row.text, row.acoustic.snr_db, row.acoustic.phrase_start_ms,
            row.acoustic.phrase_end_ms, window,
        ))
    return result


def score_audio_records(samples: list[RepCNNSample], config: dict[str, object], scorer: Int8Scorer, name: str) -> list[dict[str, object]]:
    # This loader has no manifest access; records are admitted only after the stage gate.
    # FeatureLoader.audio_and_feature deliberately rejects Test, so the explicit
    # gated Test path repeats only its frontend/window calculation here.
    class _Adapter:
        dataset_root = PROJECT_ROOT
    loader = FeatureLoader(_Adapter(), config)
    sample_rate = int(config["frontend"]["sample_rate_hz"])
    clip_ms = float(config["augmentation"]["clip_duration"]) * 1000.0
    expected_shape = tuple(int(value) for value in config["frontend"]["input_shape"])
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples, 1):
        audio = load_training_audio(sample.audio_path)
        clip = extract_streaming_window(
            audio, sample.window, sample_rate_hz=sample_rate, window_ms=clip_ms
        )
        feature = np.asarray(loader.frontend(clip)[0], dtype=np.float32)
        if feature.shape != expected_shape or not np.all(np.isfinite(feature)):
            raise RuntimeError(f"Invalid Test feature for {sample.record_id}: {feature.shape}")
        rows.append({**sample.metadata(), **scorer.score(feature)})
        if index % 25 == 0 or index == len(samples):
            print(f"PHASE3C HEARTBEAT dataset={name} records={index}/{len(samples)}", flush=True)
    return rows


def v1_protocol_metadata() -> dict[str, object]:
    return {
        "protocol_id": V1_EXTERNAL_PROTOCOL_ID,
        "scope": "external-only compatibility protocol",
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "short_wav_policy": "symmetric zero padding; odd extra sample at end",
        "long_wav_policy": "integer-second starts plus one deduplicated tail-anchored window",
        "window_generation": "complete before inference and independent of all record metadata",
        "window_score": "frozen stateless RepCNN native 2-second clip sigmoid",
        "record_score": "maximum of all predetermined window scores",
        "threshold": EXPECTED_FROZEN_THRESHOLD,
        "threshold_selection": "frozen on qingxiaojia_v2 Validation; no v1 tuning",
        "fairness_note": (
            "Model A uses full-WAV streaming plus sequence max; Model B uses full-WAV "
            "deterministic sliding 2-second clips plus clip max. Both are record-level "
            "max-over-time evaluations while retaining native model semantics. v1 sliding-window "
            "FPR and v2 single-window FPR are not the same decision process."
        ),
    }


def load_v1_external_records() -> tuple[list[object], Path, Path]:
    manifest_path = PROJECT_ROOT / "datasets/projects/qingxiaojia_v1/DatasetManifest.json"
    if sha256_file(manifest_path) != V1_MANIFEST_SHA256:
        raise RuntimeError("Immutable v1 external manifest changed")
    manifest = DatasetManifest.load(manifest_path)
    root = Path(manifest.root).resolve()
    records = [row for row in manifest.records if row.split == "test"]
    if not records or any(row.split != "test" for row in records):
        raise RuntimeError("v1 external manifest has no valid Test records")
    return records, root, manifest_path


def score_v1_fullwav_records(
    records: list[object],
    dataset_root: Path,
    config: dict[str, object],
    scorer: Int8Scorer,
    threshold: float,
    name: str,
) -> list[dict[str, object]]:
    if threshold != EXPECTED_FROZEN_THRESHOLD:
        raise RuntimeError("v1 external protocol requires frozen threshold 0.84375")

    class _Adapter:
        dataset_root = PROJECT_ROOT

    frontend = FeatureLoader(_Adapter(), config).frontend
    sample_rate = int(config["frontend"]["sample_rate_hz"])
    expected_shape = tuple(int(value) for value in config["frontend"]["input_shape"])
    rows: list[dict[str, object]] = []
    for record_index, record in enumerate(records, 1):
        started = time.perf_counter()
        audio = load_inference_audio(dataset_root / record.audio_path)
        # The complete list is fixed before score_clip is called.
        windows = fullwav_windows(audio, sample_rate)

        def score_clip(clip: np.ndarray) -> float:
            feature = np.asarray(frontend(clip)[0], dtype=np.float32)
            if feature.shape != expected_shape or not np.all(np.isfinite(feature)):
                raise RuntimeError(
                    f"Invalid v1 external feature for {record.record_id}: {feature.shape}"
                )
            return float(scorer.score(feature)["score"])

        aggregated = score_fullwav_clips(windows, score_clip)
        winning_index = int(aggregated["winning_window_index"])
        winning_start = int(aggregated["winning_window_start_sample"])
        window_starts = [int(start) for start, _ in windows]
        record_score = float(aggregated["record_score"])
        rows.append(
            {
                "protocol_id": V1_EXTERNAL_PROTOCOL_ID,
                "record_id": record.record_id,
                "path": str((dataset_root / record.audio_path).resolve()),
                "split": record.split,
                "label": record.label,
                "source": record.speaker.source,
                "speaker_id": record.speaker.speaker_id,
                "text": record.text,
                "duration_ms": len(audio) * 1000.0 / sample_rate,
                "num_windows": len(windows),
                "score": record_score,
                "record_score": record_score,
                "winning_window_index": winning_index,
                "winning_window_start_ms": winning_start * 1000.0 / sample_rate,
                "winning_window_end_ms": (winning_start + 2 * sample_rate) * 1000.0 / sample_rate,
                "accepted": record_score >= threshold,
                "window_starts_ms": json.dumps(
                    [start * 1000.0 / sample_rate for start in window_starts]
                ),
                "window_scores": json.dumps(aggregated["window_scores"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if record_index % 25 == 0 or record_index == len(records):
            print(
                f"PHASE3C HEARTBEAT dataset={name} records={record_index}/{len(records)}",
                flush=True,
            )
    return rows


def v1_smoke_report(rows: list[dict[str, object]]) -> dict[str, object]:
    scores = [float(row["record_score"]) for row in rows]
    windows = [int(row["num_windows"]) for row in rows]
    timings = [float(row["elapsed_seconds"]) for row in rows]
    if len(set(scores)) < 2:
        raise RuntimeError("v1 external smoke produced constant record scores")
    if any(
        bool(row["accepted"])
        != (float(row["record_score"]) >= EXPECTED_FROZEN_THRESHOLD)
        for row in rows
    ):
        raise RuntimeError("v1 external smoke acceptance differs from frozen threshold")
    if any(row["protocol_id"] != V1_EXTERNAL_PROTOCOL_ID for row in rows):
        raise RuntimeError("v1 external smoke protocol metadata changed")
    mean_seconds = float(np.mean(timings))
    return {
        "status": "PASS",
        "protocol": v1_protocol_metadata(),
        "records": len(rows),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "mean_windows_per_record": float(np.mean(windows)),
        "max_windows_per_record": max(windows),
        "mean_seconds_per_record": mean_seconds,
        "estimated_full_900_seconds": mean_seconds * 900,
        "score_min": min(scores),
        "score_max": max(scores),
        "scores_nonconstant": True,
        "frozen_threshold": EXPECTED_FROZEN_THRESHOLD,
        "utf8_json": True,
        "formal_v1_report_created": False,
        "formal_v1_scores_created": False,
    }


def detailed_report(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    categories: dict[str, object] = {}
    category_names = {
        "positive": "positive", "negative": "ordinary_negative",
        "hard_negative": "hard_negative", "ambient": "ambient",
    }
    for label, report_name in category_names.items():
        selected = [row for row in rows if row["label"] == label]
        accepted = sum(float(row["score"]) >= threshold for row in selected)
        rate = accepted / len(selected) if selected else None
        categories[report_name] = {
            "count": len(selected), "accepted": accepted, "rejected": len(selected) - accepted,
            "recall_tpr": rate if label == "positive" else None,
            "false_rejection_rate": 1.0 - rate if label == "positive" and rate is not None else None,
            "false_positive_rate": rate if label != "positive" else None,
        }
    source_names = {"kokoro": "Kokoro", "voxcpm15": "VoxCPM1.5"}
    sources = {}
    for source in sorted({str(row["source"]) for row in rows}):
        report_name = source_names.get(source, source)
        sources[report_name] = binary_metrics(
            [row for row in rows if row["source"] == source], threshold
        )
    speakers = {
        speaker: binary_metrics([row for row in rows if row["speaker_id"] == speaker], threshold)
        for speaker in sorted({str(row["speaker_id"]) for row in rows})
    }
    special_texts: dict[str, object] = {}
    for text in ("你好，小甲", "你好，青甲"):
        selected = [row for row in rows if row.get("text") == text]
        accepted = sum(float(row["score"]) >= threshold for row in selected)
        special_texts[text] = {
            "count": len(selected), "accepted": accepted,
            "false_positive_rate": accepted / len(selected) if selected else None,
        }
    errors = [row for row in rows if (row["label"] == "positive") != (float(row["score"]) >= threshold)]
    return {
        "frozen_threshold": threshold,
        "metrics": binary_metrics(rows, threshold), "categories": categories, "sources": sources,
        "speakers": speakers, "special_hard_negative_texts": special_texts,
        "errors": {"count": len(errors), "by_label": dict(Counter(str(row["label"]) for row in errors)), "top": sorted(errors, key=lambda row: abs(float(row["score"]) - threshold))[:50]},
    }


def external_source_breakdown(
    rows: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    required = {
        "Kokoro zm_053": ("kokoro", "zm_053"),
        "Kokoro zm_056": ("kokoro", "zm_056"),
        "MeloTTS ZH": ("melotts", "ZH"),
    }
    result: dict[str, object] = {}
    for name, (source, speaker) in required.items():
        selected = [
            row for row in rows
            if str(row["source"]).lower() == source and row["speaker_id"] == speaker
        ]
        if not selected:
            raise RuntimeError(f"Required v1 external source group is empty: {name}")
        result[name] = binary_metrics(selected, threshold)
    return result


def frozen_provenance(
    context: dict[str, object], model_info: dict[str, object], output_root: Path
) -> dict[str, object]:
    return {
        "threshold_source": str((output_root / "threshold_freeze.json").resolve()),
        "threshold_reselected_on_test": False,
        "test_directed_tuning": False,
        "checkpoint_sha256": context["best_weights_sha256"],
        "config_sha256": context["config_sha256"],
        "v2_manifest_sha256": context["manifest_sha256"],
        "tflite_sha256": model_info["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "export", "smoke", "validation-freeze", "v2-test",
            "v1-external-smoke", "v1-external-test",
        ),
        required=True,
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_root = run_dir / "phase3c_model_b_frozen"
    context = load_context(args.config.resolve(), run_dir)
    print(f"PHASE3C START stage={args.stage} best_step={EXPECTED_BEST_STEP} test_loaded=false", flush=True)
    model_info = export_frozen(context, output_root)
    model_path = Path(model_info["path"])
    scorer = Int8Scorer(model_path)

    if args.stage == "export":
        print(json.dumps(model_info, ensure_ascii=False, indent=2), flush=True)
        return
    if args.stage == "smoke":
        rows = cached_rows(context, "validation", scorer, limit_per_label=4)
        if len({row["raw_int8"] for row in rows}) < 2:
            raise RuntimeError("Validation smoke produced a constant quantized output")
        if any(not 0.0 <= float(row["score"]) <= 1.0 for row in rows):
            raise RuntimeError("Dequantized sigmoid score is outside [0,1]")
        write_csv(output_root / "validation_smoke_scores.csv", rows)
        atomic_json(output_root / "validation_smoke.json", {"status": "PASS", "samples": len(rows), "per_label": 4, "test_loaded": False, "score_min": min(float(row["score"]) for row in rows), "score_max": max(float(row["score"]) for row in rows)})
        print("PHASE3C VALIDATION SMOKE PASS samples=16 test_loaded=false", flush=True)
        return
    if args.stage == "validation-freeze":
        freeze_path = output_root / "threshold_freeze.json"
        if freeze_path.exists():
            raise RuntimeError("Validation threshold is already frozen; refusing reselection")
        rows = cached_rows(context, "validation", scorer)
        write_csv(output_root / "v2_validation_scores.csv", rows)
        quant = model_info["quantization"]
        sweep, operating = threshold_sweep(rows, float(quant["output_scale"]), int(quant["output_zero_point"]))
        write_csv(output_root / "v2_validation_threshold_sweep.csv", sweep)
        freeze = {
            "schema": "wakeword-studio.repcnn-threshold-freeze/v1", "frozen_at": utc_now(),
            "selection_split": "v2_validation_only", "selection_rule": "best F1; tie recall, precision, FPR, threshold",
            "selected_threshold": operating["best_f1"]["threshold"], "operating_points": operating,
            "checkpoint_sha256": context["best_weights_sha256"], "config_sha256": context["config_sha256"],
            "v2_manifest_sha256": context["manifest_sha256"], "tflite_sha256": model_info["sha256"],
            "quantization": quant, "v2_test_loaded": False, "v1_external_test_loaded": False,
        }
        atomic_json(freeze_path, freeze)
        atomic_json(output_root / "v2_validation_report.json", detailed_report(rows, float(freeze["selected_threshold"])))
        print("PHASE3C VALIDATION THRESHOLD FROZEN; Test remains unopened", flush=True)
        return

    freeze = verify_threshold_gate(context, output_root, model_info)
    threshold = float(freeze["selected_threshold"])
    if args.stage == "v2-test":
        result_path = output_root / "v2_test_report.json"
        if result_path.exists():
            raise RuntimeError("v2 Test was already evaluated; refusing a second pass")
        samples = adapt_records(context["manifest_path"], "test", context["config"])
        rows = score_audio_records(samples, context["config"], scorer, "v2_test")
        write_csv(output_root / "v2_test_scores.csv", rows)
        atomic_json(result_path, {
            "frozen_threshold": threshold,
            "threshold_integrity": frozen_provenance(context, model_info, output_root),
            "v2_test_loaded": True,
            **detailed_report(rows, threshold),
            "model_a_v3_frozen_baseline": MODEL_A_V3_BASELINE["v2_test"],
            "comparison_scope": "same v2 Test and frozen-threshold metric definitions",
        })
        print("PHASE3C V2 TEST COMPLETE threshold unchanged", flush=True)
        return

    v2_result = output_root / "v2_test_report.json"
    if not v2_result.is_file():
        raise RuntimeError("v2 Test must complete before v1 external Test")
    completed_v2 = json.loads(v2_result.read_text(encoding="utf-8"))
    if float(completed_v2.get("frozen_threshold", -1)) != threshold:
        raise RuntimeError("Completed v2 Test did not use the current frozen threshold")
    if completed_v2.get("threshold_integrity") != frozen_provenance(
        context, model_info, output_root
    ):
        raise RuntimeError("Completed v2 Test provenance does not match frozen artifacts")
    records, v1_root, v1_manifest = load_v1_external_records()

    if args.stage == "v1-external-smoke":
        selected: list[object] = []
        for label in ("positive", "negative", "hard_negative"):
            candidates = sorted(
                (record for record in records if record.label == label),
                key=lambda record: record.record_id,
            )
            if len(candidates) < 4:
                raise RuntimeError(f"Not enough v1 external smoke records for {label}")
            selected.extend(candidates[:4])
        rows = score_v1_fullwav_records(
            selected, v1_root, context["config"], scorer, threshold, "v1_external_smoke"
        )
        smoke = v1_smoke_report(rows)
        write_csv(output_root / "v1_external_smoke_scores.csv", rows)
        atomic_json(output_root / "v1_external_smoke.json", smoke)
        atomic_json(output_root / "v1_external_protocol.json", v1_protocol_metadata())
        print(
            "PHASE3C V1 EXTERNAL SMOKE PASS records=12 "
            f"mean_windows={smoke['mean_windows_per_record']:.3f} "
            f"max_windows={smoke['max_windows_per_record']} "
            f"mean_seconds_per_record={smoke['mean_seconds_per_record']:.3f}",
            flush=True,
        )
        return

    result_path = output_root / "v1_external_test_report.json"
    if result_path.exists():
        raise RuntimeError("v1 external Test was already evaluated; refusing a second pass")
    rows = score_v1_fullwav_records(
        records, v1_root, context["config"], scorer, threshold, "v1_external_test"
    )
    write_csv(output_root / "v1_external_test_scores.csv", rows)
    atomic_json(result_path, {
        "frozen_threshold": threshold,
        "protocol": v1_protocol_metadata(),
        "threshold_integrity": frozen_provenance(context, model_info, output_root),
        "v2_test_completed_first": True,
        "v1_external_test_loaded": True,
        "manifest_sha256": V1_MANIFEST_SHA256,
        **detailed_report(rows, threshold),
        "external_source_breakdown": external_source_breakdown(rows, threshold),
        "model_a_v3_frozen_baseline": MODEL_A_V3_BASELINE["v1_external_test"],
        "comparison_scope": "same v1 external Test and frozen-threshold metric definitions",
    })
    print("PHASE3C V1 EXTERNAL TEST COMPLETE threshold unchanged", flush=True)


if __name__ == "__main__":
    main()
