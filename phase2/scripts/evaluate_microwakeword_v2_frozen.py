"""Phase 2G frozen export, validation threshold freeze, and one-shot held-out tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microwakeword import mixednet
from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.data import FeatureHandler
from microwakeword.layers import modes
from microwakeword.utils import convert_model_saved, convert_saved_model_to_tflite
from run_microwakeword_training import build_runtime_config
from wakeword_studio.dataset.manifest import DatasetManifest
from wakeword_studio.frontends import load_inference_audio


EXPECTED_BEST_STEP = 3500
EXPECTED_BEST_F1 = 0.49149922720247297
V1_MANIFEST_SHA256 = "70b089652a7f8eb407c9d23ccc0efe7e33ce241fad2309f87f35702dc4752391"


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


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, float | int | None] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for percentile in (1, 5, 10, 25, 75, 90, 95, 98, 99):
        result[f"p{percentile:02d}"] = float(np.percentile(array, percentile))
    return result


def binary_metrics(
    rows: list[dict[str, object]], threshold: float, score_key: str = "score"
) -> dict[str, object]:
    labels = np.asarray([str(row["label"]) == "positive" for row in rows], dtype=bool)
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    tn = int(np.sum(~predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else None
    result: dict[str, object] = {
        "count": len(rows),
        "threshold": float(threshold),
        "recall_tpr": recall,
        "precision": precision,
        "f1": f1,
        "false_rejection_rate": None if recall is None else 1.0 - recall,
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


def category_metrics(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    result: dict[str, object] = {}
    for label in ("positive", "negative", "hard_negative", "ambient"):
        selected = [row for row in rows if row["label"] == label]
        accepted = sum(float(row["score"]) >= threshold for row in selected)
        entry: dict[str, object] = {
            "count": len(selected),
            "accepted": accepted,
            "rejected": len(selected) - accepted,
            "score_distribution": describe([float(row["score"]) for row in selected]),
        }
        if label == "positive":
            entry["recall_tpr"] = accepted / len(selected) if selected else None
            entry["false_rejection_rate"] = (
                1.0 - accepted / len(selected) if selected else None
            )
        else:
            entry["false_accepts"] = accepted
            entry["false_positive_rate"] = accepted / len(selected) if selected else None
        result[label] = entry
    return result


def score_distributions(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        label: describe([float(row["score"]) for row in rows if row["label"] == label])
        for label in ("positive", "negative", "hard_negative", "ambient")
    }


def threshold_sweep(
    rows: list[dict[str, object]], output_detail: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    scale, zero_point = output_detail["quantization"]
    dtype = np.dtype(output_detail["dtype"])
    limits = np.iinfo(dtype)
    thresholds = [
        float(scale * (raw - zero_point)) for raw in range(int(limits.min), int(limits.max) + 1)
    ]
    thresholds.append(float(np.nextafter(max(thresholds), np.inf)))
    sweep = [binary_metrics(rows, threshold) for threshold in thresholds]
    best_f1 = max(
        sweep,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall_tpr"] or 0.0),
            float(row["precision"]),
            -float(row["false_positive_rate"] or 0.0),
            float(row["threshold"]),
        ),
    )
    target_points: dict[str, object] = {}
    for target in (0.90, 0.95, 0.98):
        feasible = [row for row in sweep if float(row["recall_tpr"] or 0.0) >= target]
        if feasible:
            selected = max(
                feasible,
                key=lambda row: (
                    float(row["precision"]),
                    -float(row["false_positive_rate"] or 0.0),
                    float(row["f1"]),
                    float(row["threshold"]),
                ),
            )
            reasonable = float(selected["false_positive_rate"] or 0.0) <= 0.01
            target_points[f"recall_at_least_{int(target * 100)}"] = {
                **selected,
                "reasonable_policy": "validation_fpr <= 0.01",
                "reasonable": reasonable,
                "verdict": (
                    "REASONABLE OPERATING POINT"
                    if reasonable
                    else f"NO REASONABLE {int(target * 100)}% OPERATING POINT"
                ),
            }
        else:
            target_points[f"recall_at_least_{int(target * 100)}"] = {
                "feasible": False,
                "verdict": f"NO {int(target * 100)}% OPERATING POINT",
            }
    return sweep, {"best_f1": best_f1, "recall_targets": target_points}


def record_row(record: object, dataset_root: Path, score: dict[str, object]) -> dict[str, object]:
    return {
        "path": str((dataset_root / record.audio_path).resolve()),
        "record_id": record.record_id,
        "split": record.split,
        "label": record.label,
        "speaker_id": record.speaker.speaker_id,
        "source": record.speaker.source,
        "text": record.text,
        "noise": record.acoustic.noise_id,
        "snr_db": record.acoustic.snr_db,
        "duration_seconds": record.duration_seconds,
        "phrase_position": record.acoustic.phrase_placement,
        "phrase_start_ms": record.acoustic.phrase_start_ms,
        "phrase_end_ms": record.acoustic.phrase_end_ms,
        "hard_negative_tier": record.hard_negative_tier,
        **score,
    }


class FreshInterpreterStreamingScorer:
    """A new TFLite interpreter for every WAV prevents resource-state leakage."""

    def __init__(self, model_path: Path, stride: int, step_ms: int):
        self.model_path = model_path
        self.stride = stride
        self.step_ms = step_ms
        probe = self._new_interpreter()
        self.input_detail = probe.get_input_details()[0]
        self.output_detail = probe.get_output_details()[0]
        self.input_scale, self.input_zero_point = self.input_detail["quantization"]
        self.output_scale, self.output_zero_point = self.output_detail["quantization"]
        if not self.input_scale or not self.output_scale:
            raise RuntimeError("Full-INT8 model lacks scalar input/output quantization metadata")

    def _new_interpreter(self) -> tf.lite.Interpreter:
        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        return interpreter

    def metadata(self) -> dict[str, object]:
        input_q = self.input_detail["quantization_parameters"]
        output_q = self.output_detail["quantization_parameters"]
        return {
            "input_dtype": np.dtype(self.input_detail["dtype"]).name,
            "input_shape": self.input_detail["shape"].tolist(),
            "input_scale": float(self.input_scale),
            "input_zero_point": int(self.input_zero_point),
            "input_scales": np.asarray(input_q["scales"]).astype(float).tolist(),
            "input_zero_points": np.asarray(input_q["zero_points"]).astype(int).tolist(),
            "input_quantization_formula": "q = clip(round(real / scale + zero_point))",
            "output_dtype": np.dtype(self.output_detail["dtype"]).name,
            "output_shape": self.output_detail["shape"].tolist(),
            "output_scale": float(self.output_scale),
            "output_zero_point": int(self.output_zero_point),
            "output_scales": np.asarray(output_q["scales"]).astype(float).tolist(),
            "output_zero_points": np.asarray(output_q["zero_points"]).astype(int).tolist(),
            "output_dequantization_formula": "real_score = scale * (raw - zero_point)",
            "legacy_raw_div_255_used": False,
            "stream_state_reset": "fresh TFLite interpreter per WAV",
        }

    def score_audio(self, audio: np.ndarray) -> dict[str, object]:
        interpreter = self._new_interpreter()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        features = generate_features_for_clip(audio.astype(np.float32), step_ms=self.step_ms)
        usable = len(features) - len(features) % self.stride
        raw_values: list[int] = []
        limits = np.iinfo(input_detail["dtype"])
        for offset in range(0, usable, self.stride):
            chunk = features[offset : offset + self.stride]
            quantized = np.rint(chunk / self.input_scale + self.input_zero_point)
            quantized = np.clip(quantized, limits.min, limits.max).astype(input_detail["dtype"])
            interpreter.set_tensor(input_detail["index"], quantized.reshape(input_detail["shape"]))
            interpreter.invoke()
            raw_values.append(int(interpreter.get_tensor(output_detail["index"])[0][0]))
        raw_max = max(raw_values, default=int(self.output_zero_point))
        raw_final = raw_values[-1] if raw_values else int(self.output_zero_point)
        return {
            "int8_raw_max": raw_max,
            "int8_raw_final": raw_final,
            "score": float(self.output_scale * (raw_max - self.output_zero_point)),
            "final_score": float(self.output_scale * (raw_final - self.output_zero_point)),
            "feature_frames": int(len(features)),
            "stream_packets": int(usable // self.stride),
            "discarded_feature_frames": int(len(features) - usable),
        }


def score_records(
    records: list[object],
    dataset_root: Path,
    scorer: FreshInterpreterStreamingScorer,
    dataset_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        audio = load_inference_audio(dataset_root / record.audio_path)
        rows.append(record_row(record, dataset_root, scorer.score_audio(audio)))
        if index % 50 == 0 or index == len(records):
            print(
                f"PHASE2G HEARTBEAT dataset={dataset_name} records={index}/{len(records)}",
                flush=True,
            )
    return rows


def load_context(config_path: Path, run_dir: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status_path = run_dir / "TRAINING_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"Training status must be COMPLETED, got {status['status']}")
    if int(status["best_metric"]["step"]) != EXPECTED_BEST_STEP:
        raise RuntimeError("Frozen best step is not 3500")
    if abs(float(status["best_metric"]["f1"]) - EXPECTED_BEST_F1) > 1e-12:
        raise RuntimeError("Frozen best validation F1 changed")
    manifest_path = Path(raw["dataset_manifest"]).resolve()
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != raw["dataset_manifest_sha256"]:
        raise RuntimeError("v2 DatasetManifest differs from frozen config")
    if manifest_hash != status["dataset_manifest_sha256"]:
        raise RuntimeError("v2 DatasetManifest differs from completed training run")
    best_weights = Path(status["best_checkpoint"]).resolve()
    if not best_weights.is_file():
        raise FileNotFoundError(best_weights)
    best_line = next(
        (
            line
            for line in (run_dir / "training.log").read_text(encoding="utf-8").splitlines()
            if "VALIDATION step=3500 " in line
        ),
        None,
    )
    if best_line is None or "f1=0.491499" not in best_line:
        raise RuntimeError("Training log does not prove the frozen step-3500 checkpoint")
    return {
        "raw": raw,
        "status": status,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "best_weights": best_weights,
        "best_weights_sha256": sha256_file(best_weights),
        "best_log_line": best_line,
    }


def export_frozen(context: dict[str, object], run_dir: Path, output_root: Path) -> dict[str, object]:
    raw = context["raw"]
    frozen_root = output_root / "frozen_checkpoint"
    frozen_weights = frozen_root / "best_step_3500.weights.h5"
    frozen_root.mkdir(parents=True, exist_ok=True)
    if not frozen_weights.exists():
        shutil.copy2(context["best_weights"], frozen_weights)
    if sha256_file(frozen_weights) != context["best_weights_sha256"]:
        raise RuntimeError("Frozen checkpoint copy differs from training best weights")
    checkpoint_freeze = {
        "frozen_at": utc_now(),
        "source": str(context["best_weights"]),
        "frozen_copy": str(frozen_weights),
        "sha256": context["best_weights_sha256"],
        "best_step": EXPECTED_BEST_STEP,
        "best_validation_f1_at_0_5": EXPECTED_BEST_F1,
        "training_log_evidence": context["best_log_line"],
        "last_step_7500_checkpoint_used": False,
    }
    atomic_json(frozen_root / "checkpoint_freeze.json", checkpoint_freeze)

    runtime, flags = build_runtime_config(raw, run_dir)
    final_root = output_root / "final_model"
    saved_name = "stream_state_internal"
    tflite_root = final_root / "tflite_stream_state_internal_quant"
    tflite_path = tflite_root / "stream_state_internal_quant.tflite"
    if not tflite_path.exists():
        model = mixednet.model(flags, runtime["training_input_shape"], batch_size=1)
        model.load_weights(frozen_weights)
        if int(model.count_params()) != int(raw["architecture"]["parameter_count"]):
            raise RuntimeError("Parameter count changed before export")
        export_config = {**runtime, "train_dir": str(final_root)}
        print("PHASE2G EXPORT saved_model", flush=True)
        convert_model_saved(
            model, export_config, saved_name, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
        )
        print("PHASE2G EXPORT full_int8_streaming_tflite", flush=True)
        convert_saved_model_to_tflite(
            config=export_config,
            audio_processor=FeatureHandler(runtime),
            path_to_model=str(final_root / saved_name),
            folder=str(tflite_root),
            fname=tflite_path.name,
            quantize=True,
        )

    scorer = FreshInterpreterStreamingScorer(
        tflite_path, int(runtime["stride"]), int(runtime["window_step_ms"])
    )
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    operators = sorted({row["op_name"] for row in interpreter._get_ops_details()})
    model_info = {
        "path": str(tflite_path),
        "bytes": tflite_path.stat().st_size,
        "kib": tflite_path.stat().st_size / 1024.0,
        "sha256": sha256_file(tflite_path),
        "parameter_count": int(raw["architecture"]["parameter_count"]),
        "within_50_100_kib": 50 <= tflite_path.stat().st_size / 1024.0 <= 100,
        "operators": operators,
        "quantization": scorer.metadata(),
    }
    atomic_json(output_root / "model_info.json", model_info)
    atomic_json(
        output_root / "export_provenance.json",
        {
            "created_at": utc_now(),
            "checkpoint": checkpoint_freeze,
            "dataset_manifest_sha256": context["manifest_sha256"],
            "config_sha256": sha256_file(Path(raw["config_path"]))
            if raw.get("config_path")
            else None,
            "model": model_info,
            "training_api_used": False,
        },
    )
    return {"runtime": runtime, "tflite_path": tflite_path, "model_info": model_info}


def speaker_breakdown(
    rows: list[dict[str, object]], threshold: float, speakers: list[str]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for speaker in speakers:
        selected = [row for row in rows if row["speaker_id"] == speaker]
        by_label = Counter(str(row["label"]) for row in selected)
        positive = [row for row in selected if row["label"] == "positive"]
        negative = [row for row in selected if row["label"] == "negative"]
        hard = [row for row in selected if row["label"] == "hard_negative"]
        result[speaker] = {
            "source": selected[0]["source"] if selected else None,
            "samples": len(selected),
            "label_counts": dict(sorted(by_label.items())),
            "positive_recall": (
                sum(float(row["score"]) >= threshold for row in positive) / len(positive)
                if positive
                else None
            ),
            "ordinary_negative_fpr": (
                sum(float(row["score"]) >= threshold for row in negative) / len(negative)
                if negative
                else None
            ),
            "hard_negative_fpr": (
                sum(float(row["score"]) >= threshold for row in hard) / len(hard)
                if hard
                else None
            ),
        }
    return result


def source_breakdown(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    return {
        source: binary_metrics([row for row in rows if row["source"] == source], threshold)
        for source in sorted({str(row["source"]) for row in rows})
        if source != "procedural_ambient"
    }


def external_speaker_breakdown(
    rows: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    return speaker_breakdown(rows, threshold, ["zm_053", "zm_056", "ZH"])


def error_analysis(
    rows: list[dict[str, object]], threshold: float, output_root: Path
) -> dict[str, object]:
    false_negatives = sorted(
        (
            row
            for row in rows
            if row["label"] == "positive" and float(row["score"]) < threshold
        ),
        key=lambda row: float(row["score"]),
    )
    false_positives = sorted(
        (
            row
            for row in rows
            if row["label"] != "positive" and float(row["score"]) >= threshold
        ),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    hard_false_accepts = [row for row in false_positives if row["label"] == "hard_negative"]
    write_csv(output_root / "false_negatives.csv", false_negatives)
    write_csv(output_root / "false_positives.csv", false_positives)
    write_csv(output_root / "hard_negative_false_accepts.csv", hard_false_accepts)
    return {
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "hard_negative_false_accepts": len(hard_false_accepts),
        "false_negative_by_speaker": dict(
            sorted(Counter(str(row["speaker_id"]) for row in false_negatives).items())
        ),
        "false_positive_by_label": dict(
            sorted(Counter(str(row["label"]) for row in false_positives).items())
        ),
        "hard_false_accept_by_text": dict(
            sorted(Counter(str(row["text"]) for row in hard_false_accepts).items())
        ),
        "top_false_negatives": false_negatives[:20],
        "top_false_positives": false_positives[:20],
        "top_hard_negative_false_accepts": hard_false_accepts[:20],
    }


def recompute_v1_baseline(project_root: Path) -> dict[str, object]:
    diagnostic_root = project_root / "outputs/diagnostics/phase2c_full"
    validation_rows = read_csv(diagnostic_root / "validation_model_scores.csv")
    test_rows = read_csv(diagnostic_root / "float_vs_int8_scores.csv")
    for row in validation_rows + test_rows:
        row["score"] = float(row["int8_dequantized_score"])
    fake_output = {"quantization": (1.0 / 256.0, 0), "dtype": np.dtype(np.uint8)}
    _, operating = threshold_sweep(validation_rows, fake_output)
    threshold = float(operating["best_f1"]["threshold"])
    test = binary_metrics(test_rows, threshold)
    categories = category_metrics(test_rows, threshold)
    old_metrics = json.loads(
        (
            project_root
            / "runs/qingxiaojia/microwakeword_tiny_v1/formal/20260828T190330Z/final_evaluation/metrics.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "provenance": "Phase 2C fresh-interpreter scores, re-thresholded on v1 Validation by best F1",
        "model_size_bytes": old_metrics["model"]["bytes"],
        "model_size_kib": old_metrics["model"]["kib"],
        "threshold": threshold,
        "validation": operating["best_f1"],
        "test": test,
        "hard_negative_fpr": categories["hard_negative"]["false_positive_rate"],
    }


def write_final_report(path: Path, result: dict[str, object]) -> None:
    v2 = result["v2_test"]["metrics"]
    ext = result["v1_external_test_with_v2_model"]["metrics"]
    comparison = result["comparison"]
    lines = [
        "# Phase 2G — Frozen Final Evaluation",
        "",
        f"- Frozen best checkpoint: step **{EXPECTED_BEST_STEP}**",
        f"- Final INT8 model: **{result['model']['bytes']:,} bytes / {result['model']['kib']:.3f} KiB**",
        f"- Frozen Validation threshold: **{result['threshold']['selected_threshold']:.8f}**",
        f"- v2 Test Recall / FPR / ROC AUC: **{v2['recall_tpr']:.4f} / {v2['false_positive_rate']:.4f} / {v2['roc_auc']:.4f}**",
        f"- v1 external Test Recall / FPR / ROC AUC: **{ext['recall_tpr']:.4f} / {ext['false_positive_rate']:.4f} / {ext['roc_auc']:.4f}**",
        f"- 98% verdict: **{result['threshold']['operating_points']['recall_targets']['recall_at_least_98']['verdict']}**",
        f"- Engineering conclusion: **{comparison['engineering_conclusion']}**",
        f"- Main bottleneck: **{comparison['main_bottleneck']}**",
        f"- One recommended action: **{comparison['one_recommended_action']}**",
        "",
        "All Test scores used the Validation-frozen threshold. Every WAV used a fresh TFLite interpreter, and uint8 outputs were dequantized from TFLite metadata.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_export(context: dict[str, object], run_dir: Path, output_root: Path) -> None:
    exported = export_frozen(context, run_dir, output_root)
    print(json.dumps(exported["model_info"], ensure_ascii=False, indent=2), flush=True)
    print("PHASE2G EXPORT COMPLETE", flush=True)


def stage_smoke(context: dict[str, object], run_dir: Path, output_root: Path) -> None:
    exported = export_frozen(context, run_dir, output_root)
    manifest = DatasetManifest.load(context["manifest_path"])
    dataset_root = Path(manifest.root).resolve()
    selected = [
        next(row for row in manifest.records if row.split == "validation" and row.label == label)
        for label in ("positive", "negative", "hard_negative", "ambient")
    ]
    scorer = FreshInterpreterStreamingScorer(
        exported["tflite_path"],
        int(exported["runtime"]["stride"]),
        int(exported["runtime"]["window_step_ms"]),
    )
    rows = score_records(selected, dataset_root, scorer, "v2_validation_smoke")
    write_csv(output_root / "smoke_scores.csv", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print("PHASE2G SMOKE COMPLETE", flush=True)


def stage_validation_freeze(
    context: dict[str, object], run_dir: Path, output_root: Path, script_path: Path
) -> None:
    freeze_path = output_root / "threshold_freeze.json"
    if freeze_path.exists():
        raise RuntimeError("Threshold is already frozen; refusing to select it again")
    exported = export_frozen(context, run_dir, output_root)
    manifest = DatasetManifest.load(context["manifest_path"])
    dataset_root = Path(manifest.root).resolve()
    records = [row for row in manifest.records if row.split == "validation"]
    scorer = FreshInterpreterStreamingScorer(
        exported["tflite_path"],
        int(exported["runtime"]["stride"]),
        int(exported["runtime"]["window_step_ms"]),
    )
    rows = score_records(records, dataset_root, scorer, "v2_validation")
    write_csv(output_root / "v2_validation_scores.csv", rows)
    sweep, operating_points = threshold_sweep(rows, scorer.output_detail)
    write_csv(output_root / "v2_validation_threshold_sweep.csv", sweep)
    selected_threshold = float(operating_points["best_f1"]["threshold"])
    freeze = {
        "schema": "wakeword-studio.phase2g-threshold-freeze/v1",
        "frozen_at": utc_now(),
        "selection_split": "v2_validation",
        "selection_rule": "global best Validation F1; tie recall, precision, FPR, threshold",
        "selected_threshold": selected_threshold,
        "selected_validation_metrics": operating_points["best_f1"],
        "operating_points": operating_points,
        "reasonable_policy": "Validation FPR <= 0.01",
        "checkpoint_step": EXPECTED_BEST_STEP,
        "checkpoint_sha256": context["best_weights_sha256"],
        "tflite_sha256": exported["model_info"]["sha256"],
        "v2_manifest_sha256": context["manifest_sha256"],
        "evaluation_script_sha256": sha256_file(script_path),
        "quantization": scorer.metadata(),
        "v2_test_audio_accessed": False,
        "v1_external_test_audio_accessed": False,
    }
    atomic_json(output_root / "v2_validation_score_distributions.json", score_distributions(rows))
    atomic_json(freeze_path, freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2), flush=True)
    print("PHASE2G THRESHOLD FROZEN", flush=True)


def stage_heldout_tests(
    context: dict[str, object], run_dir: Path, output_root: Path, script_path: Path
) -> None:
    result_path = output_root / "final_evaluation.json"
    if result_path.exists():
        raise RuntimeError("Held-out final evaluation already exists; refusing to run Test twice")
    freeze_path = output_root / "threshold_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["evaluation_script_sha256"] != sha256_file(script_path):
        raise RuntimeError("Evaluation logic changed after threshold freeze")
    if freeze["checkpoint_sha256"] != context["best_weights_sha256"]:
        raise RuntimeError("Checkpoint changed after threshold freeze")
    exported = export_frozen(context, run_dir, output_root)
    if freeze["tflite_sha256"] != exported["model_info"]["sha256"]:
        raise RuntimeError("TFLite changed after threshold freeze")
    threshold = float(freeze["selected_threshold"])
    scorer = FreshInterpreterStreamingScorer(
        exported["tflite_path"],
        int(exported["runtime"]["stride"]),
        int(exported["runtime"]["window_step_ms"]),
    )

    access_log = {
        "threshold_frozen_at": freeze["frozen_at"],
        "heldout_access_started_at": utc_now(),
        "threshold_at_access": threshold,
        "checkpoint_sha256": context["best_weights_sha256"],
        "tflite_sha256": exported["model_info"]["sha256"],
        "evaluation_script_sha256": sha256_file(script_path),
        "threshold_reselection_after_access_allowed": False,
    }
    atomic_json(output_root / "heldout_access_log.json", access_log)

    v2_manifest = DatasetManifest.load(context["manifest_path"])
    v2_root = Path(v2_manifest.root).resolve()
    v2_records = [row for row in v2_manifest.records if row.split == "test"]
    v2_rows = score_records(v2_records, v2_root, scorer, "v2_heldout_test")
    write_csv(output_root / "v2_test_scores.csv", v2_rows)

    v1_manifest_path = PROJECT_ROOT / "datasets/projects/qingxiaojia_v1/DatasetManifest.json"
    if sha256_file(v1_manifest_path) != V1_MANIFEST_SHA256:
        raise RuntimeError("Immutable v1 external DatasetManifest changed")
    v1_manifest = DatasetManifest.load(v1_manifest_path)
    v1_root = Path(v1_manifest.root).resolve()
    v1_records = [row for row in v1_manifest.records if row.split == "test"]
    v1_rows = score_records(v1_records, v1_root, scorer, "v1_external_test")
    write_csv(output_root / "v1_external_test_scores.csv", v1_rows)

    v2_metrics = binary_metrics(v2_rows, threshold)
    v2_categories = category_metrics(v2_rows, threshold)
    v1_external_metrics = binary_metrics(v1_rows, threshold)
    v1_external_categories = category_metrics(v1_rows, threshold)
    v2_speakers = speaker_breakdown(v2_rows, threshold, ["zf_021", "zm_041", "SSB0737"])
    v1_external_speakers = external_speaker_breakdown(v1_rows, threshold)
    v2_errors = error_analysis(v2_rows, threshold, output_root / "errors/v2_test")
    v1_errors = error_analysis(v1_rows, threshold, output_root / "errors/v1_external_test")

    baseline = recompute_v1_baseline(PROJECT_ROOT)
    direct_recall_delta = float(v1_external_metrics["recall_tpr"]) - float(
        baseline["test"]["recall_tpr"]
    )
    direct_fpr_delta = float(v1_external_metrics["false_positive_rate"]) - float(
        baseline["test"]["false_positive_rate"]
    )
    direct_auc_delta = float(v1_external_metrics["roc_auc"]) - float(
        baseline["test"]["roc_auc"]
    )
    reasonable_98 = bool(
        freeze["operating_points"]["recall_targets"]["recall_at_least_98"].get(
            "reasonable", False
        )
    )
    if (
        float(v2_metrics["recall_tpr"]) >= 0.98
        and float(v2_metrics["false_positive_rate"]) <= 0.01
        and reasonable_98
    ):
        engineering_conclusion = "A. 52KB Tiny 已基本可用"
    elif direct_recall_delta > 0.10 and float(v2_metrics["roc_auc"]) > 0.80:
        engineering_conclusion = "B. 泛化明显提升但仍未达98%"
    else:
        engineering_conclusion = "C. 仍然失败，需要改变训练目标/模型容量"

    positive_dist = score_distributions(v2_rows)["positive"]
    hard_dist = score_distributions(v2_rows)["hard_negative"]
    if float(v2_metrics["roc_auc"]) < 0.80 or (
        positive_dist["p25"] is not None
        and hard_dist["p75"] is not None
        and float(positive_dist["p25"]) <= float(hard_dist["p75"])
    ):
        main_bottleneck = (
            "streaming max-score下 positive 与 hard-negative 分布重叠；当前 clip-level BCE "
            "没有直接优化连续帧触发与近音拒绝"
        )
        one_action = (
            "下一轮只引入 sequence-level streaming objective：对 phrase 末端连续正帧和 "
            "hard-negative 全序列负帧直接监督，保持数据与 Tiny 架构不变"
        )
    else:
        main_bottleneck = "未见 speaker/source 上 positive 分数仍偏低，speaker 泛化不足"
        one_action = "下一轮只增加真实未见说话人的正样本覆盖，保持模型与训练目标不变"

    result = {
        "schema": "wakeword-studio.phase2g-frozen-final-evaluation/v1",
        "completed_at": utc_now(),
        "frozen_checkpoint": {
            "step": EXPECTED_BEST_STEP,
            "path": str(context["best_weights"]),
            "sha256": context["best_weights_sha256"],
            "last_step_7500_used": False,
        },
        "model": exported["model_info"],
        "threshold": {
            "selected_threshold": threshold,
            "selected_on": "v2_validation_only",
            "selected_validation_metrics": freeze["selected_validation_metrics"],
            "operating_points": freeze["operating_points"],
            "changed_after_test": False,
        },
        "v2_validation_score_distributions": json.loads(
            (output_root / "v2_validation_score_distributions.json").read_text(encoding="utf-8")
        ),
        "v2_test": {
            "metrics": v2_metrics,
            "categories": v2_categories,
            "sources": source_breakdown(v2_rows, threshold),
            "speakers": v2_speakers,
            "score_distributions": score_distributions(v2_rows),
            "error_analysis": v2_errors,
        },
        "v1_external_test_with_v2_model": {
            "manifest_sha256": V1_MANIFEST_SHA256,
            "metrics": v1_external_metrics,
            "categories": v1_external_categories,
            "sources": source_breakdown(v1_rows, threshold),
            "speakers": v1_external_speakers,
            "score_distributions": score_distributions(v1_rows),
            "error_analysis": v1_errors,
        },
        "comparison": {
            "v1_model_corrected_fresh_interpreter_baseline": baseline,
            "v2_model_on_same_v1_external_test": v1_external_metrics,
            "direct_v1_external_delta_v2_minus_v1": {
                "recall": direct_recall_delta,
                "false_positive_rate": direct_fpr_delta,
                "roc_auc": direct_auc_delta,
            },
            "v2_model_on_v2_heldout_test": v2_metrics,
            "engineering_conclusion": engineering_conclusion,
            "main_bottleneck": main_bottleneck,
            "one_recommended_action": one_action,
        },
        "evaluation_integrity": {
            "output_formula": "scale * (raw - zero_point)",
            "raw_div_255_used": False,
            "fresh_interpreter_per_wav": True,
            "v2_test_first_access_after_threshold_freeze": True,
            "v1_external_first_access_after_threshold_freeze": True,
            "threshold_changed_after_test": False,
            "training_or_model_adjustment_performed": False,
        },
        "artifacts": {
            "v2_validation_scores": str(output_root / "v2_validation_scores.csv"),
            "threshold_sweep": str(output_root / "v2_validation_threshold_sweep.csv"),
            "threshold_freeze": str(output_root / "threshold_freeze.json"),
            "v2_test_scores": str(output_root / "v2_test_scores.csv"),
            "v1_external_scores": str(output_root / "v1_external_test_scores.csv"),
            "error_root": str(output_root / "errors"),
        },
    }
    hashes_after = {
        "v2_manifest": sha256_file(context["manifest_path"]),
        "v1_manifest": sha256_file(v1_manifest_path),
        "checkpoint": sha256_file(context["best_weights"]),
        "tflite": sha256_file(exported["tflite_path"]),
        "script": sha256_file(script_path),
    }
    expected_hashes = {
        "v2_manifest": context["manifest_sha256"],
        "v1_manifest": V1_MANIFEST_SHA256,
        "checkpoint": freeze["checkpoint_sha256"],
        "tflite": freeze["tflite_sha256"],
        "script": freeze["evaluation_script_sha256"],
    }
    if hashes_after != expected_hashes:
        raise RuntimeError("Frozen artifact changed during held-out evaluation")
    result["evaluation_integrity"]["frozen_hashes_after"] = hashes_after
    atomic_json(result_path, result)
    write_final_report(output_root / "FINAL_REPORT.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("PHASE2G FINAL EVALUATION COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("export", "smoke", "validation-freeze", "heldout-tests"),
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    script_path = Path(__file__).resolve()
    output_root = run_dir / "phase2g_frozen_final"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"PHASE2G START stage={args.stage}", flush=True)
    context = load_context(config_path, run_dir)
    context["raw"]["config_path"] = str(config_path)
    print(
        f"FROZEN INPUTS VERIFIED best_step={EXPECTED_BEST_STEP} "
        f"checkpoint_sha256={context['best_weights_sha256']} "
        f"manifest_sha256={context['manifest_sha256']}",
        flush=True,
    )
    if args.stage == "export":
        stage_export(context, run_dir, output_root)
    elif args.stage == "smoke":
        stage_smoke(context, run_dir, output_root)
    elif args.stage == "validation-freeze":
        stage_validation_freeze(context, run_dir, output_root, script_path)
    else:
        stage_heldout_tests(context, run_dir, output_root, script_path)


if __name__ == "__main__":
    main()
