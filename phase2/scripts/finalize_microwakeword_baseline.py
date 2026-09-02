"""Export the frozen best model, tune on validation, and evaluate held-out test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from microwakeword import mixednet
from microwakeword.data import FeatureHandler
from microwakeword.inference import Model as StreamingModel
from microwakeword.layers import modes
from microwakeword.utils import convert_model_saved, convert_saved_model_to_tflite
from run_microwakeword_training import build_runtime_config, sha256_file
from wakeword_studio.dataset.manifest import DatasetManifest
from wakeword_studio.frontends import load_inference_audio


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def metrics(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    labels = np.asarray([row["label"] == "positive" for row in rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else None
    result: dict[str, object] = {
        "count": len(rows),
        "threshold": threshold,
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
        scores = [float(row["score"]) for row in selected]
        accepted = sum(score >= threshold for score in scores)
        entry: dict[str, object] = {
            "count": len(selected),
            "accepted": accepted,
            "rejected": len(selected) - accepted,
            "score_distribution": describe(scores),
        }
        if label == "positive":
            entry["recall_tpr"] = accepted / len(selected)
            entry["false_rejection_rate"] = 1.0 - accepted / len(selected)
        else:
            entry["false_accepts"] = accepted
            entry["false_positive_rate"] = accepted / len(selected)
        result[label] = entry
    return result


def metadata_row(record: object, score: float) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "audio_path": record.audio_path,
        "label": record.label,
        "split": record.split,
        "score": score,
        "text": record.text,
        "source": record.speaker.source,
        "speaker_id": record.speaker.speaker_id,
        "gender": record.speaker.gender,
        "age_group": record.speaker.age_group,
        "age_source": record.speaker.age_source,
        "acoustic_age_proxy": record.acoustic.acoustic_age_proxy,
        "noise_id": record.acoustic.noise_id,
        "snr_db": record.acoustic.snr_db,
        "reverb_id": record.acoustic.reverb_id,
        "hard_negative_tier": record.hard_negative_tier,
    }


def score_split(
    model_path: Path,
    records: list[object],
    dataset_root: Path,
    split: str,
) -> list[dict[str, object]]:
    scorer = StreamingModel(str(model_path), stride=3)
    can_reset = hasattr(scorer.model, "reset_all_variables")
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        if can_reset:
            scorer.model.reset_all_variables()
        else:
            scorer = StreamingModel(str(model_path), stride=3)
        audio = load_inference_audio(dataset_root / record.audio_path)
        predictions = scorer.predict_clip(audio.astype(np.float32), step_ms=10)
        score = float(max(predictions, default=0.0))
        rows.append(metadata_row(record, score))
        if index % 50 == 0 or index == len(records):
            print(f"SCORE_HEARTBEAT split={split} records={index}/{len(records)}", flush=True)
    return rows


def choose_threshold(rows: list[dict[str, object]]) -> tuple[float, list[dict[str, object]], str]:
    sweep = [metrics(rows, float(value)) for value in np.linspace(0.0, 1.0, 256)]
    feasible = [row for row in sweep if float(row["recall_tpr"] or 0.0) >= 0.98]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row["f1"]),
                float(row["precision"]),
                -float(row["false_positive_rate"] or 0.0),
                float(row["threshold"]),
            ),
        )
        rule = "maximize validation F1 subject to Recall >= 0.98; tie-break precision/FPR/threshold"
    else:
        selected = max(sweep, key=lambda row: (float(row["f1"]), float(row["recall_tpr"] or 0.0)))
        rule = "Recall >= 0.98 infeasible; maximize validation F1"
    return float(selected["threshold"]), sweep, rule


def error_summary(rows: list[dict[str, object]], threshold: float) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    false_negatives = [
        row for row in rows if row["label"] == "positive" and float(row["score"]) < threshold
    ]
    false_positives = [
        row for row in rows if row["label"] != "positive" and float(row["score"]) >= threshold
    ]

    def counts(items: list[dict[str, object]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(key)) for row in items).items()))

    summary = {
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "false_negative_by_source": counts(false_negatives, "source"),
        "false_negative_by_noise": counts(false_negatives, "noise_id"),
        "false_negative_by_snr": counts(false_negatives, "snr_db"),
        "false_positive_by_label": counts(false_positives, "label"),
        "false_positive_by_source": counts(false_positives, "source"),
        "false_positive_by_text": counts(false_positives, "text"),
        "false_positive_by_tier": counts(false_positives, "hard_negative_tier"),
        "false_positive_by_noise": counts(false_positives, "noise_id"),
    }
    return false_negatives, false_positives, summary


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "TRAINING_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"Training is not complete: {status['status']}")
    best_weights = Path(status["best_checkpoint"]).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(raw["dataset_manifest"]).resolve()
    if sha256_file(manifest_path) != status["dataset_manifest_sha256"]:
        raise RuntimeError("DatasetManifest hash no longer matches the completed run")
    manifest = DatasetManifest.load(manifest_path)
    dataset_root = Path(manifest.root).resolve()
    runtime, flags = build_runtime_config(raw, run_dir)

    final_root = run_dir / "final_model"
    export_config = {**runtime, "train_dir": str(final_root)}
    saved_name = "stream_state_internal"
    tflite_root = final_root / "tflite_stream_state_internal_quant"
    tflite_path = tflite_root / "stream_state_internal_quant.tflite"
    if not tflite_path.exists():
        model = mixednet.model(flags, runtime["training_input_shape"], batch_size=1)
        model.load_weights(best_weights)
        if int(model.count_params()) != int(raw["architecture"]["parameter_count"]):
            raise RuntimeError("Parameter count changed before export")
        print("EXPORT_STAGE saved_model", flush=True)
        convert_model_saved(model, export_config, saved_name, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        print("EXPORT_STAGE full_int8_tflite", flush=True)
        feature_handler = FeatureHandler(runtime)
        convert_saved_model_to_tflite(
            config=export_config,
            audio_processor=feature_handler,
            path_to_model=str(final_root / saved_name),
            folder=str(tflite_root),
            fname=tflite_path.name,
            quantize=True,
        )

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    operators = sorted({row["op_name"] for row in interpreter._get_ops_details()})
    model_info = {
        "path": str(tflite_path),
        "bytes": tflite_path.stat().st_size,
        "kib": tflite_path.stat().st_size / 1024.0,
        "sha256": hashlib.sha256(tflite_path.read_bytes()).hexdigest(),
        "parameter_count": int(raw["architecture"]["parameter_count"]),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "operators": operators,
        "within_50_100_kib": 50.0 <= tflite_path.stat().st_size / 1024.0 <= 100.0,
    }

    validation_records = [row for row in manifest.records if row.split == "validation"]
    test_records = [row for row in manifest.records if row.split == "test"]
    validation_rows = score_split(tflite_path, validation_records, dataset_root, "validation")
    threshold, sweep, threshold_rule = choose_threshold(validation_rows)
    validation_metrics = metrics(validation_rows, threshold)
    test_rows = score_split(tflite_path, test_records, dataset_root, "test")
    test_metrics = metrics(test_rows, threshold)
    categories = category_metrics(test_rows, threshold)
    source_metrics = {
        source: metrics([row for row in test_rows if row["source"] == source], threshold)
        for source in ("kokoro", "melotts")
    }

    output_root = run_dir / "final_evaluation"
    scores_path = output_root / "scores.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list((validation_rows + test_rows)[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(validation_rows + test_rows)
    threshold_csv = output_root / "threshold_sweep.csv"
    with threshold_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "threshold",
            "recall_tpr",
            "precision",
            "f1",
            "false_rejection_rate",
            "false_positive_rate",
            "roc_auc",
            "pr_auc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep)
    threshold_report = {
        "selection_split": "validation",
        "selection_rule": threshold_rule,
        "selected_threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_was_not_used_for_selection": True,
        "sweep_csv": str(threshold_csv),
    }
    threshold_path = output_root / "threshold_report.json"
    atomic_json(threshold_path, threshold_report)

    false_negatives, false_positives, errors = error_summary(test_rows, threshold)
    error_root = output_root / "error_analysis"
    write_jsonl(error_root / "false_negatives.jsonl", false_negatives)
    write_jsonl(error_root / "false_positives.jsonl", false_positives)
    atomic_json(error_root / "summary.json", errors)

    best_log_line = next(
        (
            line
            for line in (run_dir / "training.log").read_text(encoding="utf-8").splitlines()
            if f"VALIDATION step={status['best_metric']['step']} " in line
        ),
        None,
    )
    best_precision = (
        float(best_log_line.split("precision=", 1)[1].split()[0])
        if best_log_line and "precision=" in best_log_line
        else None
    )
    final_metrics = {
        "schema": "wakeword-studio.microwakeword-final-evaluation/v1",
        "training": {
            "status": status["status"],
            "completed_steps": status["current_step"],
            "planned_steps": status["planned_steps"],
            "early_stopped": "planned_steps_after_early_stop" in status,
            "elapsed_seconds": status["total_elapsed_time_seconds"],
            "best_checkpoint": str(best_weights),
            "best_checkpoint_sha256": sha256_file(best_weights),
            "best_step": status["best_metric"]["step"],
            "best_validation_at_0_5": {
                "recall": status["best_metric"]["recall"],
                "precision": best_precision,
                "f1": status["best_metric"]["f1"],
                "log_line": best_log_line,
            },
        },
        "dataset_manifest_sha256": status["dataset_manifest_sha256"],
        "model": model_info,
        "threshold": threshold_report,
        "held_out_test": test_metrics,
        "test_categories": categories,
        "test_sources": source_metrics,
        "error_summary": errors,
        "scores_csv": str(scores_path),
        "error_analysis_root": str(error_root),
    }
    metrics_path = output_root / "metrics.json"
    atomic_json(metrics_path, final_metrics)

    report_lines = [
        "# microWakeWord Tiny v1 Formal Baseline Analysis",
        "",
        f"- Training status: **{status['status']}**",
        f"- Completed steps: **{status['current_step']:,}** (early stopped from {status['planned_steps']:,})",
        f"- Training elapsed: **{status['total_elapsed_time_seconds']:.3f} s**",
        f"- Best step: **{status['best_metric']['step']:,}**",
        f"- Best validation at threshold 0.5: Recall **{status['best_metric']['recall']:.4f}**, F1 **{status['best_metric']['f1']:.4f}**",
        f"- Frozen streaming INT8 threshold selected on validation: **{threshold:.6f}**",
        "",
        "## Held-out Test",
        "",
        f"- Recall / TPR: **{test_metrics['recall_tpr']:.4f}**",
        f"- Precision: **{test_metrics['precision']:.4f}**",
        f"- F1: **{test_metrics['f1']:.4f}**",
        f"- FRR: **{test_metrics['false_rejection_rate']:.4f}**",
        f"- FPR: **{test_metrics['false_positive_rate']:.4f}**",
        f"- ROC AUC: **{test_metrics['roc_auc']:.4f}**",
        f"- PR AUC: **{test_metrics['pr_auc']:.4f}**",
        f"- Confusion matrix: `{test_metrics['confusion_matrix']}`",
        "",
        "## Category false accepts / rejects",
        "",
    ]
    for label, item in categories.items():
        report_lines.append(f"- {label}: `{item}`")
    report_lines += [
        "",
        "## Held-out source comparison",
        "",
        f"- Kokoro held-out: `{source_metrics['kokoro']}`",
        f"- MeloTTS test-only: `{source_metrics['melotts']}`",
        "",
        "## Final model",
        "",
        f"- Path: `{tflite_path}`",
        f"- Size: **{model_info['bytes']:,} bytes / {model_info['kib']:.3f} KiB**",
        f"- Parameters: **{model_info['parameter_count']:,}**",
        f"- 50–100 KiB requirement: **{'PASS' if model_info['within_50_100_kib'] else 'FAIL'}**",
        f"- Input/output dtype: **{model_info['input_dtype']} / {model_info['output_dtype']}**",
        "",
        "## Error analysis",
        "",
        f"- `{json.dumps(errors, ensure_ascii=False)}`",
        "",
        f"Metrics: `{metrics_path}`",
        f"Threshold report: `{threshold_path}`",
        f"Error analysis: `{error_root}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"FINAL_REPORT={args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
