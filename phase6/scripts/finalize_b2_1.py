"""Validation-only final selection between Final B2 and B2.1 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase3.scripts.preflight_repcnn_model_b import (  # noqa: E402
    build_model,
    indexed_model_state,
)
from phase4.scripts.run_repcnn_v2_fasttrack_training import load_feature_groups  # noqa: E402
from phase6.scripts.finalize_b2 import (  # noqa: E402
    auc_metrics,
    atomic_json,
    export_full_int8,
    int8_scores,
    model_scores,
    representative_features,
    utc_now,
    validation_bundle,
)
from wakeword_studio.dataset.manifest import sha256_file  # noqa: E402
from wakeword_studio.training.repcnn_fasttrack import validation_rank  # noqa: E402
from wakeword_studio.training.repcnn_finalization import operating_points  # noqa: E402


def restore_candidate(config: dict[str, object], candidate: dict[str, object]):
    model = build_model(config)
    path = Path(str(candidate["path"]))
    if candidate["kind"] == "baseline_weights":
        model.load_weights(path)
    else:
        restored = tf.train.Checkpoint(model_state=indexed_model_state(model)).restore(
            str(path)
        )
        restored.expect_partial()
        restored.assert_existing_objects_matched()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/models/repcnn_performance_v2_1_robust_finetune.yaml",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/qingxiaojia/repcnn_performance_v2_1_robust_finetune/formal/user_run_01",
    )
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=PROJECT_ROOT
        / "runs/qingxiaojia/repcnn_performance_v2_fasttrack/formal/user_run_01"
        / "phase6_finalization_v2/final_b2.weights.h5",
    )
    parser.add_argument("--allow-finalize", action="store_true")
    args = parser.parse_args()
    if not args.allow_finalize:
        raise SystemExit("B2.1 finalization is gated; pass --allow-finalize")

    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    status = json.loads((run_dir / "TRAINING_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED" or int(status.get("current_step", -1)) != 750:
        raise RuntimeError("B2.1 run is not completely finished at step 750")
    if status.get("test_loaded") is not False:
        raise RuntimeError("B2.1 status does not prove test_loaded=false")

    baseline_weights = args.baseline_weights.resolve()
    if not baseline_weights.is_file():
        raise FileNotFoundError(baseline_weights)
    if Path(str(status["base_weights"])).resolve() != baseline_weights:
        raise RuntimeError("B2.1 was not trained from the requested Final B2 weights")

    baseline_finalization = baseline_weights.parent
    baseline_report = json.loads(
        (baseline_finalization / "FINALIZATION_REPORT.json").read_text(encoding="utf-8")
    )
    baseline_freeze = json.loads(
        (baseline_finalization / "threshold_freeze.json").read_text(encoding="utf-8")
    )
    if baseline_report.get("test_loaded") is not False or baseline_freeze.get("test_loaded") is not False:
        raise RuntimeError("Baseline finalization does not prove test_loaded=false")

    candidates: list[dict[str, object]] = [
        {
            "name": "b2_baseline",
            "kind": "baseline_weights",
            "step": None,
            "path": str(baseline_weights),
        }
    ]
    for step in (250, 500, 750):
        prefix = run_dir / "checkpoints" / f"ckpt-{step}"
        if not prefix.with_suffix(".index").is_file() or not list(
            prefix.parent.glob(prefix.name + ".data-*")
        ):
            raise FileNotFoundError(f"Incomplete B2.1 checkpoint: {prefix}")
        candidates.append(
            {
                "name": f"b2_1_step_{step}",
                "kind": "b2_1_checkpoint",
                "step": step,
                "path": str(prefix.resolve()),
            }
        )

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    groups = load_feature_groups(config)
    if any(sample.split == "test" for group in groups.values() for sample in group.samples):
        raise RuntimeError("Held-out Test entered B2.1 finalization")
    batches, targets, labels, sources = validation_bundle(groups)

    rows: list[dict[str, object]] = []
    candidate_models: dict[str, tf.keras.Model] = {}
    for candidate in candidates:
        model = restore_candidate(config, candidate)
        scores = model_scores(model, batches)
        points = operating_points(scores, targets, labels, sources)
        selection = points["fpr_caps"]["fpr_at_most_10pct"]
        row = {
            **candidate,
            "weights_sha256": (
                sha256_file(Path(str(candidate["path"])))
                if candidate["kind"] == "baseline_weights"
                else None
            ),
            "selection_metrics": selection,
            "operating_points": points,
            **auc_metrics(targets, scores),
        }
        rows.append(row)
        candidate_models[str(candidate["name"])] = model
        print(
            f"B2_1_CANDIDATE_EVALUATED name={candidate['name']} "
            f"recall={selection['recall']:.6f} fpr={selection['fpr']:.6f} "
            f"worst_source_recall={selection['worst_source_recall']:.6f}",
            flush=True,
        )

    eligible = [
        row
        for row in rows
        if row["selection_metrics"]["eligible_for_checkpoint_selection"]
    ]
    winner = max(eligible, key=lambda row: validation_rank(row["selection_metrics"]))
    baseline_wins = winner["kind"] == "baseline_weights"
    output_dir = run_dir / "phase6_finalization"
    report_path = output_dir / "FINAL_SELECTION_REPORT.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if baseline_wins:
        final_export = baseline_report["int8_export"]
        final_validation = baseline_report["final_int8_validation"]
        final_threshold = baseline_freeze["threshold"]
        final_weights = {
            "path": str(baseline_weights),
            "sha256": sha256_file(baseline_weights),
        }
        freeze_path = baseline_finalization / "threshold_freeze.json"
        selection_action = "retain_final_b2_baseline"
    else:
        chosen_model = candidate_models[str(winner["name"])]
        weights_path = output_dir / "final_b2_1.weights.h5"
        chosen_model.save_weights(weights_path)
        model_path = output_dir / "qingxiaojia_repcnn_performance_v2_1_full_int8.tflite"
        final_export = export_full_int8(
            chosen_model, config, representative_features(groups), model_path
        )
        scores = int8_scores(model_path, batches)
        points = operating_points(scores, targets, labels, sources)
        selection = points["fpr_caps"]["fpr_at_most_10pct"]
        if selection["operating_point_degenerate"]:
            raise RuntimeError("Winning B2.1 INT8 operating point is degenerate")
        final_validation = {
            "selection_metrics": selection,
            "operating_points": points,
            **auc_metrics(targets, scores),
        }
        final_threshold = selection["threshold"]
        final_weights = {
            "path": str(weights_path.resolve()),
            "sha256": sha256_file(weights_path),
        }
        freeze_path = output_dir / "threshold_freeze.json"
        atomic_json(
            freeze_path,
            {
                "schema": "wakeword-studio.repcnn-b2-1-threshold-freeze/v1",
                "frozen_at": utc_now(),
                "threshold": final_threshold,
                "threshold_source": "validation_only_final_int8_scores",
                "metrics": final_validation["selection_metrics"],
                "roc_auc": final_validation["roc_auc"],
                "pr_auc": final_validation["pr_auc"],
                "model_path": final_export["path"],
                "model_sha256": final_export["sha256"],
                "weights_sha256": final_weights["sha256"],
                "winner": winner["name"],
                "test_loaded": False,
            },
        )
        selection_action = "adopt_b2_1"

    report = {
        "schema": "wakeword-studio.repcnn-b2-1-final-selection/v1",
        "created_at": utc_now(),
        "run_status": status["status"],
        "candidate_evaluation": rows,
        "winner": winner,
        "selection_action": selection_action,
        "final_weights": final_weights,
        "final_int8_export": final_export,
        "final_int8_validation": final_validation,
        "final_threshold": final_threshold,
        "threshold_freeze_path": str(freeze_path.resolve()),
        "selection_dataset": "validation_only",
        "test_loaded": False,
        "live_diagnostic_wavs_used_for_selection": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(report_path, report)
    print(
        f"B2_1_FINAL_SELECTION winner={winner['name']} action={selection_action} "
        f"threshold={float(final_threshold):.8f} test_loaded=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
