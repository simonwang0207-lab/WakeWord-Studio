"""Finalize a completed Model B v2 run using Validation data only.

The script is intentionally inert while training is RUNNING.  It evaluates every
checkpoint still retained by the formal run, considers an upstream-compatible
checkpoint average, exports the chosen model to full INT8, and freezes a final
INT8 Validation threshold.  Held-out Test and live diagnostic WAVs are never
loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit.embedded_wakeword.models.classifier import reparameterize_model  # noqa: E402
from phase3.scripts.evaluate_repcnn_model_b_frozen import (  # noqa: E402
    dequantize_output,
    quantize_input,
)
from phase3.scripts.preflight_repcnn_model_b import (  # noqa: E402
    build_model,
    indexed_model_state,
    quantization_metadata,
)
from phase4.scripts.run_repcnn_v2_fasttrack_training import (  # noqa: E402
    FeatureGroup,
    load_feature_groups,
)
from wakeword_studio.dataset.manifest import sha256_file  # noqa: E402
from wakeword_studio.training.repcnn_fasttrack import (  # noqa: E402
    ALL_LABELS,
    validation_rank,
)
from wakeword_studio.training.repcnn_finalization import (  # noqa: E402
    FinalizationCandidate,
    average_weight_sets,
    discover_finalization_candidates,
    operating_points,
    resolve_finalization_v2_output_dir,
    select_average_candidates,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validation_bundle(
    groups: dict[tuple[str, str], FeatureGroup],
) -> tuple[list[np.ndarray], np.ndarray, list[str], list[str]]:
    batches: list[np.ndarray] = []
    targets: list[int] = []
    labels: list[str] = []
    sources: list[str] = []
    for label in ALL_LABELS:
        group = groups[("validation", label)]
        batches.extend(group.batches(128))
        targets.extend([1 if label == "positive" else 0] * len(group))
        labels.extend([label] * len(group))
        sources.extend(sample.source for sample in group.samples)
    return batches, np.asarray(targets, np.int32), labels, sources


def model_scores(model: tf.keras.Model, batches: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(model(values, training=False)).reshape(-1) for values in batches]
    ).astype(np.float64)


def auc_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(targets, scores)),
        "pr_auc": float(average_precision_score(targets, scores)),
    }


def restore_candidate(
    config: dict[str, Any], candidate: FinalizationCandidate
) -> tf.keras.Model:
    model = build_model(config)
    if candidate.candidate_kind == "best_single_weights":
        model.load_weights(candidate.path)
    else:
        restored = tf.train.Checkpoint(model_state=indexed_model_state(model)).restore(
            str(candidate.path)
        )
        restored.expect_partial()
        restored.assert_existing_objects_matched()
    return model


def averaging_strategies(
    eligible_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return bounded, Validation-only averaging strategies.

    The upstream-compatible strategy remains first.  Explicit top-2 and top-3
    Validation-rank averages are added without enumerating arbitrary subsets.
    Identical member sets are evaluated only once and retain all strategy names.
    """

    ranked = sorted(
        eligible_rows,
        key=lambda row: validation_rank(row["selection_metrics"]),
        reverse=True,
    )
    upstream = select_average_candidates(eligible_rows)
    proposals: list[tuple[str, list[str], dict[str, object] | None]] = [
        (
            "upstream_compatible",
            [str(value) for value in upstream["selected_checkpoints"]],
            upstream,
        ),
        ("top_2_by_validation_rank", [str(row["candidate_key"]) for row in ranked[:2]], None),
        ("top_3_by_validation_rank", [str(row["candidate_key"]) for row in ranked[:3]], None),
    ]
    unique: dict[tuple[str, ...], dict[str, object]] = {}
    for name, members, policy in proposals:
        if len(members) < 2:
            continue
        identity = tuple(sorted(members))
        if identity in unique:
            unique[identity]["strategy_names"].append(name)  # type: ignore[index]
            if policy is not None:
                unique[identity]["upstream_policy"] = policy
            continue
        unique[identity] = {
            "strategy_names": [name],
            "selected_candidates": members,
            "upstream_policy": policy,
        }
    return list(unique.values())


def validation_metric_deltas(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, float]:
    fields = (
        "recall",
        "precision",
        "f1",
        "frr",
        "fpr",
        "worst_source_recall",
        "source_gap",
    )
    return {name: float(after[name]) - float(before[name]) for name in fields}


def representative_features(
    groups: dict[tuple[str, str], FeatureGroup], count: int = 8
) -> np.ndarray:
    values: list[np.ndarray] = []
    for split in ("train", "validation"):
        for label in ALL_LABELS:
            group = groups[(split, label)]
            values.extend(group.take(np.arange(min(count, len(group)))))
    return np.asarray(values, np.float32)


def export_full_int8(
    model: tf.keras.Model,
    config: dict[str, Any],
    representatives: np.ndarray,
    destination: Path,
) -> dict[str, object]:
    fused = reparameterize_model(model)
    probe = representatives[:8]
    fusion_error = float(
        np.max(
            np.abs(
                np.asarray(model(probe, training=False))
                - np.asarray(fused(probe, training=False))
            )
        )
    )
    if fusion_error > 1e-4:
        raise RuntimeError(f"RepCNN fusion error exceeds tolerance: {fusion_error}")
    shape = tuple(int(value) for value in config["frontend"]["input_shape"])

    @tf.function(input_signature=[tf.TensorSpec((1, *shape), tf.float32)])
    def serving(value: tf.Tensor) -> tf.Tensor:
        return fused(value, training=False)

    def calibration() -> Iterator[list[np.ndarray]]:
        for feature in representatives:
            yield [feature[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    interpreter = tf.lite.Interpreter(model_path=str(destination))
    interpreter.allocate_tensors()
    metadata = quantization_metadata(interpreter)
    if metadata["input_dtype"] != "int8" or metadata["output_dtype"] != "int8":
        raise RuntimeError("Final deployment is not full INT8")
    if metadata["input_shape"] != config["quantization"]["expected_input_shape"]:
        raise RuntimeError("Final INT8 input shape changed")
    if metadata["output_shape"] != config["quantization"]["expected_output_shape"]:
        raise RuntimeError("Final INT8 output shape changed")
    return {
        "path": str(destination.resolve()),
        "bytes": len(content),
        "kib": len(content) / 1024.0,
        "sha256": sha256_bytes(content),
        "training_parameters": int(model.count_params()),
        "deployment_parameters": int(fused.count_params()),
        "fusion_max_abs_error": fusion_error,
        "quantization": metadata,
        "test_loaded": False,
    }


def int8_scores(model_path: Path, batches: list[np.ndarray]) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    scores: list[float] = []
    for batch in batches:
        for feature in batch:
            value = quantize_input(
                feature[np.newaxis, ...], float(input_scale), int(input_zero_point)
            )
            interpreter.set_tensor(input_detail["index"], value)
            interpreter.invoke()
            raw = interpreter.get_tensor(output_detail["index"])
            scores.append(
                float(dequantize_output(raw, float(output_scale), int(output_zero_point)).reshape(-1)[0])
            )
    return np.asarray(scores, np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only B2 checkpoint evaluation, averaging, INT8 export and freeze"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/models/repcnn_performance_v2_fasttrack.yaml",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/qingxiaojia/repcnn_performance_v2_fasttrack/formal/user_run_01",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Independent output directory (default: RUN_DIR/phase6_finalization_v2)",
    )
    parser.add_argument("--allow-finalize", action="store_true")
    args = parser.parse_args()
    if not args.allow_finalize:
        raise SystemExit("Finalization is gated; pass --allow-finalize after B2 training completes")

    started = time.perf_counter()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "TRAINING_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED":
        raise RuntimeError(
            f"B2 is not completed; finalizer refuses status={status.get('status')!r}"
        )
    if status.get("test_loaded") is not False:
        raise RuntimeError("Training status does not prove test_loaded=false")
    old_output_dir = (run_dir / "phase6_finalization").resolve()
    output_dir = resolve_finalization_v2_output_dir(run_dir, args.output_dir)
    freeze_path = output_dir / "threshold_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Finalization already frozen: {freeze_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    groups = load_feature_groups(config)
    batches, targets, labels, sources = validation_bundle(groups)
    candidates = discover_finalization_candidates(run_dir)
    if not candidates:
        raise FileNotFoundError(run_dir / "checkpoints")
    candidate_by_key = {candidate.key: candidate for candidate in candidates}

    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_weights: dict[str, list[np.ndarray]] = {}
    for candidate in candidates:
        model = restore_candidate(config, candidate)
        scores = model_scores(model, batches)
        points = operating_points(scores, targets, labels, sources)
        selection = points["fpr_caps"]["fpr_at_most_10pct"]
        checkpoint_rows.append(
            {
                "checkpoint": candidate.key,
                "candidate_key": candidate.key,
                **candidate.report_metadata(),
                "selection_metrics": selection,
                "operating_points": points,
                **auc_metrics(targets, scores),
            }
        )
        checkpoint_weights[candidate.key] = [
            np.array(value, copy=True) for value in model.get_weights()
        ]
        print(
            f"CANDIDATE_EVALUATED step={candidate.step} "
            f"kind={candidate.candidate_kind} "
            f"recall={selection['recall']:.6f} fpr={selection['fpr']:.6f}",
            flush=True,
        )

    eligible_rows = [
        row
        for row in checkpoint_rows
        if row["selection_metrics"]["eligible_for_checkpoint_selection"]
    ]
    if not eligible_rows:
        raise RuntimeError(
            "Every retained checkpoint has only a degenerate reject-all Validation point"
        )
    best_single = max(
        eligible_rows,
        key=lambda row: validation_rank(row["selection_metrics"]),
    )
    averaging_reports: list[dict[str, object]] = []
    averaged_models: dict[tuple[str, ...], tf.keras.Model] = {}
    for strategy in averaging_strategies(eligible_rows):
        selected_paths = [str(value) for value in strategy["selected_candidates"]]
        averaged_model = build_model(config)
        averaged_model.set_weights(
            average_weight_sets([checkpoint_weights[path] for path in selected_paths])
        )
        averaged_scores = model_scores(averaged_model, batches)
        averaged_points = operating_points(averaged_scores, targets, labels, sources)
        averaged_selection = averaged_points["fpr_caps"]["fpr_at_most_10pct"]
        identity = tuple(sorted(selected_paths))
        averaged_models[identity] = averaged_model
        averaging_reports.append(
            {
                **strategy,
                "selected_candidate_metadata": [
                    candidate_by_key[path].report_metadata() for path in selected_paths
                ],
                "validation": {
                    "selection_metrics": averaged_selection,
                    "operating_points": averaged_points,
                    **auc_metrics(targets, averaged_scores),
                },
            }
        )
        print(
            "AVERAGE_EVALUATED "
            f"strategies={','.join(strategy['strategy_names'])} "
            f"members={len(selected_paths)} "
            f"recall={averaged_selection['recall']:.6f} "
            f"fpr={averaged_selection['fpr']:.6f}",
            flush=True,
        )

    eligible_averages = [
        row
        for row in averaging_reports
        if row["validation"]["selection_metrics"]["eligible_for_checkpoint_selection"]
    ]
    best_average = (
        max(
            eligible_averages,
            key=lambda row: validation_rank(row["validation"]["selection_metrics"]),
        )
        if eligible_averages
        else None
    )
    average_wins = bool(
        best_average is not None
        and validation_rank(best_average["validation"]["selection_metrics"])
        > validation_rank(best_single["selection_metrics"])
    )
    if average_wins:
        chosen_paths = [str(value) for value in best_average["selected_candidates"]]
        chosen_model = averaged_models[tuple(sorted(chosen_paths))]
        chosen_kind = "checkpoint_average"
        chosen_source = chosen_paths
        chosen_float_validation = best_average["validation"]
        chosen_step = None
    else:
        chosen_key = str(best_single["candidate_key"])
        chosen_candidate = candidate_by_key[chosen_key]
        chosen_model = restore_candidate(config, chosen_candidate)
        chosen_kind = chosen_candidate.candidate_kind
        chosen_source = [chosen_key]
        chosen_float_validation = {
            "selection_metrics": best_single["selection_metrics"],
            "operating_points": best_single["operating_points"],
            "roc_auc": best_single["roc_auc"],
            "pr_auc": best_single["pr_auc"],
        }
        chosen_step = chosen_candidate.step

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "final_b2.weights.h5"
    chosen_model.save_weights(weights_path)
    model_path = output_dir / "qingxiaojia_repcnn_performance_v2_full_int8.tflite"
    export = export_full_int8(
        chosen_model, config, representative_features(groups), model_path
    )
    final_scores = int8_scores(model_path, batches)
    final_points = operating_points(final_scores, targets, labels, sources)
    final_selection = final_points["fpr_caps"]["fpr_at_most_10pct"]
    if final_selection["operating_point_degenerate"]:
        raise RuntimeError("Final INT8 model has only a degenerate reject-all operating point")
    final_validation = {
        "selection_metrics": final_selection,
        "operating_points": final_points,
        **auc_metrics(targets, final_scores),
    }

    report = {
        "schema": "wakeword-studio.repcnn-b2-finalization/v2",
        "created_at": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "run_dir": str(run_dir),
        "training_status": status.get("status"),
        "original_finalization_preserved": str(old_output_dir),
        "available_candidate_count": len(candidates),
        "candidate_inventory": [candidate.report_metadata() for candidate in candidates],
        "checkpoint_evaluation": checkpoint_rows,
        "best_single": best_single,
        "checkpoint_averaging": {
            "bounded_strategies": averaging_reports,
            "best_average": best_average,
        },
        "chosen_kind": chosen_kind,
        "chosen_step": chosen_step,
        "chosen_source_checkpoints": chosen_source,
        "chosen_source_candidates": [
            candidate_by_key[path].report_metadata() for path in chosen_source
        ],
        "average_selected_over_single": average_wins,
        "chosen_float_validation": chosen_float_validation,
        "final_weights": {
            "path": str(weights_path.resolve()),
            "sha256": sha256_file(weights_path),
        },
        "int8_export": export,
        "final_int8_validation": final_validation,
        "quantization_validation_delta": {
            "metrics": validation_metric_deltas(
                chosen_float_validation["selection_metrics"], final_selection
            ),
            "roc_auc": float(final_validation["roc_auc"])
            - float(chosen_float_validation["roc_auc"]),
            "pr_auc": float(final_validation["pr_auc"])
            - float(chosen_float_validation["pr_auc"]),
        },
        "threshold_source": "validation_only_final_int8_scores",
        "test_loaded": False,
        "live_diagnostic_wavs_used_for_selection": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "FINALIZATION_REPORT.json", report)
    freeze = {
        "schema": "wakeword-studio.repcnn-threshold-freeze/v2",
        "frozen_at": utc_now(),
        "threshold": final_selection["threshold"],
        "threshold_source": "validation_only_final_int8_scores",
        "selection_policy": "maximum Validation rank with overall FPR <= 0.10",
        "metrics": final_selection,
        "roc_auc": final_validation["roc_auc"],
        "pr_auc": final_validation["pr_auc"],
        "model_path": export["path"],
        "model_sha256": export["sha256"],
        "weights_sha256": report["final_weights"]["sha256"],
        "chosen_kind": chosen_kind,
        "chosen_step": chosen_step,
        "chosen_source_checkpoints": chosen_source,
        "chosen_source_candidates": [
            candidate_by_key[path].report_metadata() for path in chosen_source
        ],
        "test_loaded": False,
    }
    atomic_json(freeze_path, freeze)
    print(
        f"B2_FINALIZED kind={chosen_kind} threshold={final_selection['threshold']:.8f} "
        f"recall={final_selection['recall']:.6f} fpr={final_selection['fpr']:.6f} "
        "test_loaded=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
