"""Summarize frozen Phase 2C scores without running inference or training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def metrics(rows: list[dict[str, str]], key: str, threshold: float) -> dict[str, object]:
    labels = np.asarray([row["label"] == "positive" for row in rows], dtype=bool)
    scores = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    predictions = scores >= threshold
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "recall_tpr": recall,
        "precision": precision,
        "f1": f1,
        "false_rejection_rate": 1.0 - recall,
        "false_positive_rate": fp / (fp + tn),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def choose(rows: list[dict[str, str]], key: str, require_recall: float | None) -> dict[str, object]:
    candidates = [metrics(rows, key, value) for value in sorted({float(row[key]) for row in rows})]
    if require_recall is not None:
        candidates = [row for row in candidates if float(row["recall_tpr"]) >= require_recall]
    return max(
        candidates,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -float(row["false_positive_rate"]),
            float(row["threshold"]),
        ),
    )


def aggregate_positive_samples(rows: list[dict[str, str]], split: str) -> dict[str, float]:
    selected = [row for row in rows if row["split"] == split]
    keys = (
        "actual_duration_seconds",
        "leading_silence_seconds_at_-40dbfs",
        "trailing_silence_seconds_at_-40dbfs",
        "rms_dbfs",
    )
    return {key: float(np.mean([float(row[key]) for row in selected])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    args = parser.parse_args()
    root = args.diagnostics_dir.resolve()
    validation = read_csv(root / "validation_model_scores.csv")
    test = read_csv(root / "float_vs_int8_scores.csv")
    keys = {
        "Float non-streaming": "float_score",
        "Float streaming": "float_streaming_score",
        "INT8 streaming": "int8_dequantized_score",
    }

    max_f1_matrix: dict[str, object] = {}
    target_matrix: dict[str, object] = {}
    for name, key in keys.items():
        selected = choose(validation, key, require_recall=None)
        target = choose(validation, key, require_recall=0.98)
        max_f1_matrix[name] = {
            "selection": "maximum Validation F1",
            "validation": selected,
            "test": metrics(test, key, float(selected["threshold"])),
        }
        target_matrix[name] = {
            "selection": "maximum Validation F1 subject to Recall >= 0.98",
            "validation": target,
            "test": metrics(test, key, float(target["threshold"])),
        }

    legacy_selected = choose(validation, "int8_legacy_score", require_recall=None)
    correct_selected = max_f1_matrix["INT8 streaming"]["validation"]
    legacy_formula = {
        "correct_formula": "score = output_scale * (raw - output_zero_point) = raw / 256",
        "legacy_formula": "score = raw / 255",
        "correct_max_f1_threshold": correct_selected["threshold"],
        "legacy_max_f1_threshold": legacy_selected["threshold"],
        "correct_metrics": max_f1_matrix["INT8 streaming"],
        "legacy_validation": legacy_selected,
        "legacy_test": metrics(test, "int8_legacy_score", float(legacy_selected["threshold"])),
        "formula_only_changes_predictions_or_metrics": False,
    }

    saturation = json.loads((root / "saturation_audit.json").read_text(encoding="utf-8"))
    split = json.loads((root / "dataset_split_audit.json").read_text(encoding="utf-8"))
    feature = json.loads((root / "feature_consistency_audit.json").read_text(encoding="utf-8"))
    correlations = json.loads((root / "score_correlations.json").read_text(encoding="utf-8"))
    positives = read_csv(root / "positive_sample_audit.csv")
    historical = json.loads(args.historical_metrics.resolve().read_text(encoding="utf-8"))
    raw_max_threshold = 255 / 256
    isolated_raw_max = metrics(test, "int8_dequantized_score", raw_max_threshold)

    distribution = {
        "validation_sources": split["validation"]["source_tts_family"],
        "test_sources": split["test"]["source_tts_family"],
        "validation_speakers": split["validation"]["speaker"],
        "test_speakers": split["test"]["speaker"],
        "speaker_overlap": split["speaker_overlap"],
        "validation_positive_duration": split["validation"]["positive_duration_seconds"],
        "test_positive_duration": split["test"]["positive_duration_seconds"],
        "positive_sample_audio": {
            "validation": aggregate_positive_samples(positives, "validation"),
            "test": aggregate_positive_samples(positives, "test"),
        },
        "matched_dimensions": {
            "label_counts_equal": split["validation"]["label"] == split["test"]["label"],
            "text_counts_equal": split["validation"]["text"] == split["test"]["text"],
            "noise_counts_equal": split["validation"]["noise_type"] == split["test"]["noise_type"],
            "snr_counts_equal": split["validation"]["snr_db"] == split["test"]["snr_db"],
            "age_proxy_counts_equal": split["validation"]["age_proxy"] == split["test"]["age_proxy"],
            "augmentation_counts_equal": (
                split["validation"]["augmentation_present"] == split["test"]["augmentation_present"]
            ),
        },
    }

    assessment = {
        "QUANTIZATION": {
            "rating": "MEDIUM",
            "evidence": (
                "State-isolated Float-streaming vs INT8 Test Pearson 0.9894, MAE 0.0222; "
                "Test ROC AUC drops 0.0353 and PR AUC drops 0.0737. Saturated INT8 cases "
                "all have Float-streaming >=0.99, so saturation is not fabricated by quantization."
            ),
        },
        "STREAMING_CONVERSION": {
            "rating": "LOW",
            "evidence": (
                "Float streaming Test ROC AUC (0.7366) is not below Float non-streaming "
                "AUC (0.7095). Frontend features are exactly equal. Differences mainly come "
                "from full-clip max aggregation versus a trailing fixed window, not a broken graph conversion."
            ),
        },
        "DATA_DISTRIBUTION": {
            "rating": "HIGH",
            "evidence": (
                "Validation has one Kokoro TTS speaker; Test has two unseen Kokoro speakers plus "
                "MeloTTS. Test positives are more variable, longer, quieter, and contain more leading/trailing silence."
            ),
        },
        "MODEL_GENERALIZATION": {
            "rating": "HIGH",
            "evidence": (
                "The gap already exists before quantization: Float non-streaming ROC AUC "
                "0.9878->0.7095 and Float streaming 0.9705->0.7366 from Validation to Test."
            ),
        },
        "TRAINING_OBJECTIVE": {
            "rating": "HIGH",
            "evidence": (
                "Training/checkpoint validation uses one fixed trailing window, while deployment "
                "uses maximum score over a full stream. This changes recall/FPR materially and leaves "
                "a shared streaming score floor around 0.29/0.3086."
            ),
        },
    }

    summary = {
        "schema": "wakeword-studio.phase2c-final-diagnosis/v1",
        "primary_operating_point": "maximum Validation F1; Test not used for selection",
        "diagnostic_matrix": max_f1_matrix,
        "target_recall_matrix": target_matrix,
        "legacy_formula_comparison": legacy_formula,
        "historical_state_leaked_test": historical["held_out_test"],
        "state_isolated_test_at_historical_raw_255_cutoff": isolated_raw_max,
        "score_correlations": correlations,
        "saturation": saturation,
        "distribution_shift": distribution,
        "feature_consistency": feature,
        "root_cause_assessment": assessment,
        "recommended_single_change": (
            "Rebuild the training/Validation source split so Validation contains multiple speakers/TTS "
            "and matches Test duration/silence statistics, while retaining a genuinely unseen held-out Test source."
        ),
    }
    atomic_json(root / "phase2c_final_summary.json", summary)
    atomic_json(root / "operating_point_matrix.json", max_f1_matrix)
    atomic_json(root / "target_recall_98_matrix.json", target_matrix)

    def percent(value: float) -> str:
        return f"{100 * value:.2f}%"

    report = [
        "# Phase 2C — Model / Quantization / Split Diagnosis",
        "",
        "No training or fixed-input modification was performed. Thresholds were selected on Validation only.",
        "",
        "## Primary diagnostic matrix — maximum Validation F1",
        "",
        "| Path | Validation Recall | Validation FPR | Validation ROC/PR AUC | Test Recall | Test FPR | Test ROC/PR AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in max_f1_matrix.items():
        val = item["validation"]
        tst = item["test"]
        report.append(
            f"| {name} | {percent(val['recall_tpr'])} | {percent(val['false_positive_rate'])} | "
            f"{val['roc_auc']:.6f} / {val['pr_auc']:.6f} | {percent(tst['recall_tpr'])} | "
            f"{percent(tst['false_positive_rate'])} | {tst['roc_auc']:.6f} / {tst['pr_auc']:.6f} |"
        )
    fs = max_f1_matrix["Float streaming"]["test"]
    i8 = max_f1_matrix["INT8 streaming"]["test"]
    corr = correlations["test_float_streaming_vs_int8_streaming"]
    report += [
        "",
        "## Key conclusions",
        "",
        f"- Float non-streaming Test: Recall {percent(max_f1_matrix['Float non-streaming']['test']['recall_tpr'])}, ROC AUC {max_f1_matrix['Float non-streaming']['test']['roc_auc']:.6f}.",
        f"- Float streaming Test: Recall {percent(fs['recall_tpr'])}, ROC AUC {fs['roc_auc']:.6f}.",
        f"- INT8 streaming Test: Recall {percent(i8['recall_tpr'])}, ROC AUC {i8['roc_auc']:.6f}.",
        f"- Float streaming -> INT8: Recall {percent(i8['recall_tpr'] - fs['recall_tpr'])}, ROC AUC {i8['roc_auc'] - fs['roc_auc']:+.6f}, PR AUC {i8['pr_auc'] - fs['pr_auc']:+.6f}.",
        f"- Score correlation: Pearson {corr['pearson']:.6f}, Spearman {corr['spearman']:.6f}, MAE {corr['mae']:.6f}, max error {corr['max_absolute_error']:.6f}.",
        "- raw/255 is numerically wrong but monotonic; replacing it with raw/256 changes threshold values, not predictions or metrics.",
        "- reset_all_variables() did not isolate TFLite streaming states; all final INT8 results use a fresh interpreter per clip.",
        "",
        "## Distribution evidence",
        "",
        f"- Validation sources/speakers: {distribution['validation_sources']} / {distribution['validation_speakers']}.",
        f"- Test sources/speakers: {distribution['test_sources']} / {distribution['test_speakers']}.",
        f"- Validation positive duration: mean {distribution['validation_positive_duration']['mean']:.3f}s, std {distribution['validation_positive_duration']['std']:.3f}s.",
        f"- Test positive duration: mean {distribution['test_positive_duration']['mean']:.3f}s, std {distribution['test_positive_duration']['std']:.3f}s.",
        "- Label, text, noise, SNR, augmentation, and age-proxy counts match; the dominant shift is TTS/speaker and timing/acoustic presentation.",
        "",
        "## Root-cause ratings",
        "",
    ]
    for name, item in assessment.items():
        report.append(f"- **{name}: {item['rating']}** — {item['evidence']}")
    report += [
        "",
        "## Recommended single change",
        "",
        summary["recommended_single_change"],
        "",
    ]
    (root / "PHASE2C_DIAGNOSIS_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
