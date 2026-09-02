"""Build a formal-only RepCNN / BC-ResNet / ConvMixer comparison report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
B2_DIR = PROJECT_ROOT / "runs/qingxiaojia/repcnn_performance_v2_fasttrack/formal/user_run_01/phase6_finalization_v2"


def formal_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "PENDING_FORMAL_TRAINING", "formal_result": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("formal_result") is not True or report.get("test_loaded") is not False:
        raise RuntimeError(f"Not an eligible Train/Validation-only formal result: {path}")
    metrics = report["metrics"]
    return {
        "status": "FORMAL_COMPLETE",
        "formal_result": True,
        **{
            key: metrics[key]
            for key in (
                "tp", "fp", "tn", "fn", "recall", "precision", "f1", "fpr", "frr",
                "worst_source_recall", "source_gap", "roc_auc", "pr_auc",
            )
        },
        "per_source": metrics["per_source"],
        "operating_points": metrics["operating_points"],
        "threshold": metrics["threshold"],
        "parameter_count": report["parameter_count"],
        "trainable_parameter_count": report["trainable_parameter_count"],
        "int8_bytes": report["export"]["bytes"],
        "input_shape": report["input_shape"],
        "result_path": str(path.resolve()),
    }


def repcnn_row() -> dict[str, Any]:
    freeze = json.loads((B2_DIR / "threshold_freeze.json").read_text(encoding="utf-8"))
    finalization = json.loads((B2_DIR / "FINALIZATION_REPORT.json").read_text(encoding="utf-8"))
    metrics = freeze["metrics"]
    export = finalization["int8_export"]
    return {
        "status": "FROZEN_FORMAL_BASELINE",
        "formal_result": True,
        **{
            key: metrics[key]
            for key in (
                "tp", "fp", "tn", "fn", "recall", "precision", "f1", "fpr", "frr",
                "worst_source_recall", "source_gap",
            )
        },
        "roc_auc": freeze["roc_auc"],
        "pr_auc": freeze["pr_auc"],
        "source_recall": metrics["source_recall"],
        "threshold": freeze["threshold"],
        "parameter_count": export["training_parameters"],
        "trainable_parameter_count": export["training_parameters"],
        "deployment_parameter_count": export["deployment_parameters"],
        "int8_bytes": export["bytes"],
        "input_shape": export["quantization"]["input_shape"],
        "model_sha256": freeze["model_sha256"],
        "result_path": str((B2_DIR / "threshold_freeze.json").resolve()),
    }


def markdown(report: dict[str, Any]) -> str:
    columns = report["models"]
    fields = (
        "recall", "precision", "f1", "fpr", "frr", "worst_source_recall", "source_gap",
        "roc_auc", "pr_auc", "parameter_count", "trainable_parameter_count", "int8_bytes",
        "input_shape",
    )
    lines = [
        "# 公平模型对照（正式结果）",
        "",
        "> 仅纳入完成正式训练且 `test_loaded=false` 的结果；Smoke 指标永不进入此表。",
        "",
        "| 指标 | RepCNN | BC-ResNet | ConvMixer |",
        "|---|---:|---:|---:|",
    ]
    for field in fields:
        values = []
        for name in ("repcnn", "bcresnet", "convmixer"):
            value = columns[name].get(field, "PENDING")
            values.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value))
        lines.append(f"| {field} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", "当前不对尚未完成正式训练的模型作优劣结论。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bcresnet-result", type=Path)
    parser.add_argument("--convmixer-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "wakeword-studio.fair-model-comparison/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "binary_kws",
        "keyword": "qingxiaojia",
        "fairness": {
            "same_train_view": True,
            "same_validation_view": True,
            "same_frontend": True,
            "same_sampling": True,
            "same_objective": True,
            "same_threshold_protocol": True,
            "test_loaded": False,
        },
        "models": {
            "repcnn": repcnn_row(),
            "bcresnet": formal_row(args.bcresnet_result) if args.bcresnet_result else {"status": "PENDING_FORMAL_TRAINING", "formal_result": False},
            "convmixer": formal_row(args.convmixer_result) if args.convmixer_result else {"status": "PENDING_FORMAL_TRAINING", "formal_result": False},
        },
        "conclusion": "PENDING_BCRESNET_AND_CONVMIXER_FORMAL_TRAINING",
        "test_loaded": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "FAIR_COMPARISON.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "FAIR_COMPARISON.md").write_text(markdown(report), encoding="utf-8")
    print(f"COMPARISON_REPORT={output} test_loaded=false")


if __name__ == "__main__":
    main()

