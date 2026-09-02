"""Resumable formal trainer for the frozen Phase 2H sequence objective."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase2.scripts.preflight_microwakeword_v3_sequence import (
    FrozenFeatureStore,
    build_sequence_model,
    predict_records,
    prepare_batch,
    sample_training_records,
    score_diagnostics,
    train_step,
)
from phase2.scripts.run_microwakeword_training import (
    atomic_json,
    device_info,
    effective_stage,
    sha256_file,
    working_set_bytes,
)
from wakeword_studio.training.sequence_objective import (
    consecutive_trigger_score,
    false_accept_count,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def validation_sequence_metrics(
    model: tf.keras.Model,
    store: FrozenFeatureStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    records = sum(
        (
            store.records[("validation", label)]
            for label in ("positive", "negative", "hard_negative", "ambient")
        ),
        [],
    )
    frame_scores = predict_records(model, records, store, config)
    consecutive = int(config["sequence_objective"]["deployment_consecutive_frames"])
    sequence_scores = np.asarray(
        [consecutive_trigger_score(row, consecutive) for row in frame_scores], dtype=np.float64
    )
    labels = np.asarray([record.label == "positive" for record in records], dtype=np.int32)
    thresholds = np.unique(sequence_scores)
    candidates = [metrics_at_threshold(labels, sequence_scores, float(value)) for value in thresholds]
    candidates.append(
        metrics_at_threshold(labels, sequence_scores, float(np.nextafter(np.max(sequence_scores), np.inf)))
    )
    best = max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["precision"],
            -item["fpr"],
            item["threshold"],
        ),
    )
    group_false_accepts: dict[str, int] = {}
    record_groups = [record.label for record in records]
    for group in ("negative", "hard_negative", "ambient"):
        group_false_accepts[group] = false_accept_count(
            sequence_scores,
            record_groups,
            group=group,
            threshold=best["threshold"],
        )
    return {
        **best,
        "selection_split": "validation",
        "selection_rule": "best deployment-sequence F1; tie recall, precision, FPR, threshold",
        "sequence_score_formula": "max_t(min(p[t], p[t+1], p[t+2]))",
        "trigger_logic": "at least 3 consecutive decision frames >= threshold",
        "roc_auc": float(roc_auc_score(labels, sequence_scores)),
        "pr_auc": float(average_precision_score(labels, sequence_scores)),
        "false_accepts_by_group": group_false_accepts,
        "score_statistics": {
            "mean": float(np.mean(sequence_scores)),
            "std": float(np.std(sequence_scores)),
            "min": float(np.min(sequence_scores)),
            "max": float(np.max(sequence_scores)),
            "all_identical": bool(np.ptp(sequence_scores) <= 1e-12),
        },
        "frame_sequence_diagnostics": score_diagnostics(records, frame_scores, config),
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }


def latest_preflight_eta(run_root: Path) -> float:
    reports = sorted((run_root / "preflight").glob("*/benchmark_report.json"), reverse=True)
    for path in reports:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("status") == "PASSED":
            return float(report["estimated_formal_seconds"])
    raise RuntimeError("No passed Phase 2H benchmark report exists")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-formal-training", action="store_true")
    args = parser.parse_args()
    if not args.allow_formal_training:
        raise SystemExit("V3 formal training is gated; explicit user approval is required")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["dataset_manifest"]).resolve()
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != str(config["dataset_manifest_sha256"]).lower():
        raise RuntimeError("Frozen qingxiaojia_v2 DatasetManifest hash changed")
    feature_summary = json.loads(
        (Path(config["features_root"]) / "summary.json").read_text(encoding="utf-8")
    )
    if feature_summary["manifest_sha256"] != actual_manifest_hash:
        raise RuntimeError("Frozen feature store manifest hash mismatch")

    run_dir = args.run_dir.resolve()
    launcher_files = {"launcher.stdout.log", "launcher.stderr.log", "RESUME_COMMAND.txt"}
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError("Resume run directory does not exist")
    else:
        if not run_dir.is_dir():
            raise FileNotFoundError("Launcher did not create the formal run directory")
        unexpected = {item.name for item in run_dir.iterdir()} - launcher_files
        if unexpected:
            raise FileExistsError(f"Formal run directory is not an inert launcher scaffold: {sorted(unexpected)}")

    status_path = run_dir / "TRAINING_STATUS.json"
    log_path = run_dir / "training.log"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=args.resume)
    best_path = run_dir / "best_weights.weights.h5"
    last_path = run_dir / "last_weights.weights.h5"
    traceback_path = run_dir / "traceback.txt"
    previous = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if args.resume and status_path.exists()
        else {}
    )

    seed = int(config["seed"])
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    rng = random.Random(seed)
    batch_size = int(config["batch_size"])
    planned_steps = int(config["planned_steps"])
    eval_interval = int(config["eval_step_interval"])
    checkpoint_interval = int(config["checkpoint_interval"])
    estimated_seconds = latest_preflight_eta(Path(config["run_root"]))
    start_time = previous.get("start_time", utc_now())
    estimated_completion = (
        datetime.now(timezone.utc) + timedelta(seconds=estimated_seconds)
    ).isoformat()
    devices = device_info()
    command = subprocess.list2cmdline([sys.executable, *sys.argv])

    status: dict[str, Any] = {
        **previous,
        "status": "RUNNING",
        "mode": "formal_sequence",
        "pid": __import__("os").getpid(),
        "start_time": start_time,
        "resume_time": utc_now() if args.resume else None,
        "last_update": utc_now(),
        "current_step": int(previous.get("current_step", 0)) if args.resume else 0,
        "planned_steps": planned_steps,
        "estimated_training_seconds": estimated_seconds,
        "estimated_completion_time_utc": estimated_completion,
        "command": command,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": actual_manifest_hash,
        "sequence_objective": config["sequence_objective"],
        "checkpoint_selection": "Validation deployment-sequence best F1",
        "sequence_score_formula": "max_t(min(p[t], p[t+1], p[t+2]))",
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
        "best_validation_sequence_f1": previous.get("best_validation_sequence_f1"),
        "best_validation_threshold": previous.get("best_validation_threshold"),
        "best_step": previous.get("best_step"),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "last_checkpoint": previous.get("last_checkpoint"),
        "resume_command_file": str(run_dir / "RESUME_COMMAND.txt"),
        "device": devices,
    }
    atomic_json(status_path, status)

    def log(event: str, **values: Any) -> None:
        line = f"{utc_now()} {event} " + " ".join(f"{key}={value}" for key, value in values.items())
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")
        print(line.rstrip(), flush=True)

    started = time.perf_counter()
    losses: list[float] = list(previous.get("loss_history_tail", []))
    step_times: list[float] = []
    sampling_counts = Counter(previous.get("actual_sampler_counts", {}))
    max_working_set = int(previous.get("process_peak_working_set_bytes", 0))
    best_f1 = float(previous.get("best_validation_sequence_f1", -1.0) or -1.0)
    best_recall = float(previous.get("best_validation_recall", -1.0) or -1.0)
    early_reference = float(previous.get("early_stopping_reference_f1", -1.0) or -1.0)
    stale_evaluations = int(previous.get("stale_evaluations", 0))
    last_completed_step = 0

    try:
        store = FrozenFeatureStore(Path(config["features_root"]))
        pools = {
            label: store.records[("train", label)]
            for label in ("positive", "negative", "hard_negative", "ambient")
        }
        log(
            "DATA_READY",
            train=sum(len(value) for value in pools.values()),
            validation=sum(len(store.records[("validation", label)]) for label in pools),
            test_loaded=False,
        )
        model, _, model_details = build_sequence_model(config, batch_size=batch_size)
        if int(model_details["parameters"]) != 19697:
            raise RuntimeError("Frozen Tiny parameter count changed")
        optimizer = tf.keras.optimizers.Adam(float(config["learning_rates"][0]))
        optimizer.build(model.trainable_variables)
        step_variable = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        checkpoint = tf.train.Checkpoint(step=step_variable, optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=5)
        if args.resume:
            if not manager.latest_checkpoint:
                raise RuntimeError("Resume requested but no checkpoint exists")
            restore = checkpoint.restore(manager.latest_checkpoint)
            restore.assert_consumed()
            last_completed_step = int(step_variable.numpy())
            if int(optimizer.iterations.numpy()) != last_completed_step:
                raise RuntimeError("Optimizer/global-step strict resume mismatch")
            status["resume_verified_at"] = utc_now()
            log("RESUMED", step=last_completed_step, checkpoint=manager.latest_checkpoint)
        log(
            "TRAINING_STARTED",
            pid=status["pid"],
            step=last_completed_step,
            parameters=model_details["parameters"],
            decision_frames=model_details["sequence_decision_frames"],
        )

        stopped_early = False
        for step in range(last_completed_step + 1, planned_steps + 1):
            learning_rate = float(
                effective_stage(config["learning_rates"], config["training_steps"], step)
            )
            optimizer.learning_rate.assign(learning_rate)
            records = sample_training_records(pools, config, rng)
            sampling_counts.update(record.label for record in records)
            batch = prepare_batch(
                records, store, config, augment=True, required_batch_size=batch_size
            )
            step_started = time.perf_counter()
            total, frame, hard, gradient = train_step(model, optimizer, batch, config)
            step_times.append(time.perf_counter() - step_started)
            losses.append(total)
            losses = losses[-100:]
            step_variable.assign(step)
            max_working_set = max(max_working_set, working_set_bytes() or 0)

            if step % eval_interval == 0 or step == planned_steps:
                validation_started = time.perf_counter()
                metrics = validation_sequence_metrics(model, store, config)
                validation_elapsed = time.perf_counter() - validation_started
                f1 = float(metrics["f1"])
                recall = float(metrics["recall"])
                is_best = f1 > best_f1 or (math.isclose(f1, best_f1) and recall > best_recall)
                if is_best:
                    best_f1 = f1
                    best_recall = recall
                    model.save_weights(best_path)
                    status.update(
                        {
                            "best_validation_sequence_f1": best_f1,
                            "best_validation_recall": best_recall,
                            "best_validation_threshold": metrics["threshold"],
                            "best_step": step,
                            "best_checkpoint": str(best_path),
                        }
                    )
                early = config["early_stopping"]
                if step < int(early["warmup_steps"]):
                    early_reference = max(early_reference, f1)
                    stale_evaluations = 0
                elif f1 >= early_reference + float(early["min_delta"]):
                    early_reference = f1
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1
                status.update(
                    {
                        "last_validation": metrics,
                        "last_validation_step": step,
                        "last_validation_elapsed_seconds": validation_elapsed,
                        "early_stopping_reference_f1": early_reference,
                        "stale_evaluations": stale_evaluations,
                    }
                )
                log(
                    "VALIDATION",
                    step=step,
                    threshold=f"{metrics['threshold']:.8f}",
                    recall=f"{recall:.6f}",
                    precision=f"{metrics['precision']:.6f}",
                    f1=f"{f1:.6f}",
                    fpr=f"{metrics['fpr']:.6f}",
                    roc_auc=f"{metrics['roc_auc']:.6f}",
                    best=is_best,
                    stale=stale_evaluations,
                    elapsed=f"{validation_elapsed:.3f}",
                )
                if (
                    bool(early["enabled"])
                    and step >= int(early["warmup_steps"])
                    and stale_evaluations >= int(early["patience_evaluations"])
                ):
                    stopped_early = True

            should_checkpoint = step % checkpoint_interval == 0 or step == planned_steps or stopped_early
            if should_checkpoint:
                checkpoint_started = time.perf_counter()
                saved = manager.save(checkpoint_number=step)
                model.save_weights(last_path)
                status.update(
                    {
                        "last_checkpoint": saved,
                        "last_successful_checkpoint": saved,
                        "last_successful_step": step,
                    }
                )
                log("CHECKPOINT", step=step, path=saved, elapsed=f"{time.perf_counter()-checkpoint_started:.3f}")

            if step % int(config["heartbeat_interval_steps"]) == 0 or stopped_early or step == planned_steps:
                elapsed = time.perf_counter() - started
                mean_step = statistics.mean(step_times[-100:]) if step_times else 0.0
                remaining = max(0, planned_steps - step) * mean_step
                status.update(
                    {
                        "status": "RUNNING",
                        "current_step": step,
                        "last_update": utc_now(),
                        "last_loss": total,
                        "last_frame_loss": frame,
                        "last_hard_negative_max_loss": hard,
                        "last_gradient_norm": gradient,
                        "elapsed_seconds": elapsed,
                        "eta_seconds_training_only": remaining,
                        "actual_sampler_counts": dict(sampling_counts),
                        "loss_history_tail": losses,
                        "process_peak_working_set_bytes": max_working_set,
                        "v2_test_loaded": False,
                        "v1_external_test_loaded": False,
                    }
                )
                atomic_json(status_path, status)
                log(
                    "HEARTBEAT",
                    step=f"{step}/{planned_steps}",
                    loss=f"{total:.6f}",
                    best_f1=("none" if best_f1 < 0 else f"{best_f1:.6f}"),
                    eta_seconds=f"{remaining:.1f}",
                )
            if stopped_early:
                log("EARLY_STOPPING", step=step, best_step=status.get("best_step"), best_f1=best_f1)
                break

        final_step = int(step_variable.numpy())
        status.update(
            {
                "status": "COMPLETED",
                "current_step": final_step,
                "final_step": final_step,
                "early_stopped": stopped_early,
                "last_update": utc_now(),
                "end_time": utc_now(),
                "total_elapsed_time_seconds": time.perf_counter() - started,
                "v2_test_loaded": False,
                "v1_external_test_loaded": False,
            }
        )
        atomic_json(status_path, status)
        log("TRAINING_COMPLETED", step=final_step, early_stopped=stopped_early, best_step=status.get("best_step"), best_f1=best_f1)
    except BaseException as exc:
        status.update(
            {
                "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                "last_update": utc_now(),
                "error": repr(exc),
                "v2_test_loaded": False,
                "v1_external_test_loaded": False,
            }
        )
        atomic_json(status_path, status)
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
