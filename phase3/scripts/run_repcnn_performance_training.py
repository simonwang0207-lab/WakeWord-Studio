"""Formal Model B RepCNN training runner. User launch only; never loads Test."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit.embedded_wakeword.training.trainer import focal_loss  # noqa: E402
from phase3.scripts.preflight_repcnn_model_b import (  # noqa: E402
    FeatureLoader,
    build_adapter,
    build_model,
    device_info,
    indexed_model_state,
    working_set_bytes,
)


LABELS = ("positive", "negative", "hard_negative", "ambient")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def build_feature_cache(
    config: dict[str, Any], run_dir: Path
) -> dict[tuple[str, str], Path]:
    adapter = build_adapter(config)
    if adapter.test_loaded:
        raise RuntimeError("Held-out Test access is prohibited")
    loader = FeatureLoader(adapter, config)
    root = run_dir / "feature_cache_train_validation_only"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    expected: dict[tuple[str, str], Path] = {
        (split, label): root / f"{split}_{label}.npy"
        for split in ("train", "validation")
        for label in LABELS
    }
    if summary_path.is_file() and all(path.is_file() for path in expected.values()):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("dataset_manifest_sha256") == config["dataset_manifest_sha256"]
            and summary.get("test_loaded") is False
        ):
            print("FEATURE_CACHE REUSED train_validation_only=true", flush=True)
            return expected

    status_path = root / "FEATURE_STATUS.json"
    total = sum(len(adapter.samples(split, label)) for split, label in expected)
    completed = 0
    started = time.perf_counter()
    atomic_json(
        status_path,
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "started_at": utc_now(),
            "completed": 0,
            "total": total,
            "test_loaded": False,
        },
    )
    for (split, label), destination in expected.items():
        rows = list(adapter.samples(split, label))
        partial = destination.with_suffix(".partial.npy")
        metadata_path = destination.with_suffix(".jsonl")
        values = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), *tuple(config["frontend"]["input_shape"])),
        )
        metadata_lines: list[str] = []
        for index, sample in enumerate(rows):
            _, feature = loader.audio_and_feature(sample)
            values[index] = feature
            metadata_lines.append(
                json.dumps({"feature_index": index, **sample.metadata()}, ensure_ascii=False)
            )
            completed += 1
            if completed % 25 == 0 or completed == total:
                values.flush()
                atomic_json(
                    status_path,
                    {
                        "status": "RUNNING",
                        "pid": os.getpid(),
                        "started_at": utc_now(),
                        "completed": completed,
                        "total": total,
                        "current_split": split,
                        "current_label": label,
                        "elapsed_seconds": time.perf_counter() - started,
                        "test_loaded": False,
                    },
                )
                print(
                    f"FEATURE_CACHE completed={completed}/{total} split={split} "
                    f"label={label} elapsed_seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
        del values
        partial.replace(destination)
        metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    summary = {
        "schema": "wakeword-studio.repcnn-feature-cache/v1",
        "created_at": utc_now(),
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "counts": adapter.counts(),
        "feature_shape": config["frontend"]["input_shape"],
        "test_loaded": False,
        "wav_copy_policy": "read_manifest_paths_in_place",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(summary_path, summary)
    atomic_json(status_path, {**summary, "status": "COMPLETED", "pid": os.getpid()})
    return expected


def deterministic_batch(
    arrays: dict[str, np.ndarray], counts: dict[str, int], seed: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed + step * 104729)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label, count in counts.items():
        values = arrays[label]
        indices = rng.integers(0, len(values), size=count)
        features.append(np.asarray(values[indices], dtype=np.float32))
        targets.append(np.full(count, 1.0 if label == "positive" else 0.0, np.float32))
    x = np.concatenate(features)
    y = np.concatenate(targets)
    order = rng.permutation(len(y))
    mix_order = rng.permutation(len(y))
    lam = float(rng.beta(0.2, 0.2))
    return x[order], y[order], mix_order, lam


def deterministic_spec_augment(features: np.ndarray, seed: int, step: int) -> np.ndarray:
    rng = np.random.default_rng(seed ^ (step * 65537))
    value = features.copy()
    for row in value:
        time_length = int(rng.integers(1, 11))
        time_start = int(rng.integers(0, row.shape[0] - time_length + 1))
        row[time_start : time_start + time_length] = 0.0
        frequency_length = int(rng.integers(1, 6))
        frequency_start = int(rng.integers(0, row.shape[1] - frequency_length + 1))
        row[:, frequency_start : frequency_start + frequency_length] = 0.0
    return value


def learning_rate(config: dict[str, Any], step: int) -> float:
    formal = config["formal_training"]
    primary = int(formal["primary_steps"])
    refinement = int(formal["refinement_steps"])
    rates = [float(value) for value in formal["learning_rates"]]
    if step <= primary:
        return rates[0]
    if step <= primary + refinement:
        return rates[1]
    return rates[2]


def validation_metrics(
    model: tf.keras.Model,
    arrays: dict[str, np.ndarray],
    batch_size: int = 128,
) -> dict[str, object]:
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[str] = []
    for label in LABELS:
        values = arrays[label]
        for offset in range(0, len(values), batch_size):
            scores.append(
                np.asarray(model(np.asarray(values[offset : offset + batch_size]), training=False))
                .reshape(-1)
                .astype(np.float64)
            )
        targets.append(np.full(len(values), 1 if label == "positive" else 0, np.int32))
        groups.extend([label] * len(values))
    score = np.concatenate(scores)
    target = np.concatenate(targets)
    best: dict[str, float] | None = None
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = score >= threshold
        tp = int(np.sum(predicted & (target == 1)))
        fp = int(np.sum(predicted & (target == 0)))
        fn = int(np.sum(~predicted & (target == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = {
            "threshold": float(threshold),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fp / max(1, int(np.sum(target == 0))),
        }
        if best is None or (candidate["f1"], candidate["recall"]) > (
            best["f1"],
            best["recall"],
        ):
            best = candidate
    assert best is not None
    best["score_mean"] = float(np.mean(score))
    best["score_std"] = float(np.std(score))
    best["scores_by_label"] = {
        label: float(np.mean(score[np.asarray(groups) == label])) for label in LABELS
    }
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-formal-training", action="store_true")
    args = parser.parse_args()
    if not args.allow_formal_training:
        raise SystemExit("Formal Model B training is gated; user must pass --allow-formal-training")

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    feature_paths = build_feature_cache(config, run_dir)
    train_arrays = {
        label: np.load(feature_paths[("train", label)], mmap_mode="r") for label in LABELS
    }
    validation_arrays = {
        label: np.load(feature_paths[("validation", label)], mmap_mode="r") for label in LABELS
    }
    counts = {
        key: int(value) for key, value in config["sampling"]["batch_n_per_class"].items()
    }
    formal = config["formal_training"]
    planned = int(formal["planned_total_steps"])
    seed = int(config["seed"])
    tf.keras.utils.set_random_seed(seed)
    model = build_model(config)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(formal["learning_rates"][0]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    optimizer.build(model.trainable_variables)
    global_step = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        step=global_step,
        model_state=indexed_model_state(model),
        optimizer=optimizer,
    )
    checkpoint_dir = run_dir / "checkpoints"
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=5)
    status_path = run_dir / "TRAINING_STATUS.json"
    if args.resume:
        if not manager.latest_checkpoint:
            raise FileNotFoundError("No formal RepCNN checkpoint is available to resume")
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
        print(
            f"STRICT_RESUME restored_step={int(global_step.numpy())} "
            f"checkpoint={manager.latest_checkpoint}",
            flush=True,
        )
    elif manager.latest_checkpoint or status_path.exists():
        raise RuntimeError("Run directory already contains state; use --resume or a new run-dir")

    gamma = float(config["objective"]["gamma"])
    smoothing = float(config["objective"]["label_smoothing"])
    max_negative_weight = float(formal["max_negative_weight"])
    stabilization_steps = int(formal["stabilization_steps_without_mixup_or_specaugment"])

    @tf.function
    def train_step(
        features: tf.Tensor,
        targets: tf.Tensor,
        permutation: tf.Tensor,
        lam: tf.Tensor,
        negative_weight: tf.Tensor,
        use_mixup: tf.Tensor,
    ) -> tf.Tensor:
        mixed_features = tf.cond(
            use_mixup,
            lambda: lam * features + (1.0 - lam) * tf.gather(features, permutation),
            lambda: features,
        )
        mixed_targets = tf.cond(
            use_mixup,
            lambda: lam * targets + (1.0 - lam) * tf.gather(targets, permutation),
            lambda: targets,
        )
        mixed_targets = tf.cond(
            use_mixup,
            lambda: mixed_targets * (1.0 - smoothing) + 0.5 * smoothing,
            lambda: mixed_targets,
        )
        with tf.GradientTape() as tape:
            predictions = tf.squeeze(model(mixed_features, training=True), axis=-1)
            losses = focal_loss(predictions, mixed_targets, gamma=gamma)
            weights = tf.where(mixed_targets < 0.5, negative_weight, 1.0)
            loss = tf.reduce_mean(losses * weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if args.resume and status_path.is_file()
        else {}
    )
    best_f1 = float(status.get("best_validation_f1", -1.0))
    stale = int(status.get("stale_evaluations", 0))
    losses: list[float] = []
    started = time.perf_counter()
    start_step = int(global_step.numpy()) + 1
    for step in range(start_step, planned + 1):
        batch_x, batch_y, permutation, lam = deterministic_batch(train_arrays, counts, seed, step)
        use_augmentation = step > stabilization_steps
        if use_augmentation:
            batch_x = deterministic_spec_augment(batch_x, seed, step)
        rate = learning_rate(config, step)
        optimizer.learning_rate.assign(rate)
        negative_weight = 1.0 + (max_negative_weight - 1.0) * (step - 1) / max(1, planned - 1)
        loss = float(
            train_step(
                tf.constant(batch_x),
                tf.constant(batch_y),
                tf.constant(permutation, tf.int32),
                tf.constant(lam, tf.float32),
                tf.constant(negative_weight, tf.float32),
                tf.constant(use_augmentation),
            ).numpy()
        )
        if not math.isfinite(loss):
            raise RuntimeError(f"Formal RepCNN loss became NaN/Inf at step {step}")
        global_step.assign(step)
        losses.append(loss)

        validation: dict[str, object] | None = None
        if step % int(formal["validation_interval"]) == 0:
            validation = validation_metrics(model, validation_arrays)
            improved = float(validation["f1"]) > best_f1 + float(
                formal["early_stopping"]["min_delta"]
            )
            if improved:
                best_f1 = float(validation["f1"])
                stale = 0
                model.save_weights(run_dir / "best_weights.weights.h5")
            else:
                stale += 1
            print(
                f"VALIDATION step={step} f1={validation['f1']:.6f} "
                f"recall={validation['recall']:.6f} fpr={validation['fpr']:.6f} "
                f"best={improved}",
                flush=True,
            )

        checkpoint_path: str | None = None
        if step % int(formal["checkpoint_interval"]) == 0 or validation is not None:
            checkpoint_path = manager.save(checkpoint_number=step)
            model.save_weights(run_dir / "last_weights.weights.h5")
            print(f"CHECKPOINT step={step} path={checkpoint_path}", flush=True)

        status = {
            "status": "RUNNING",
            "pid": os.getpid(),
            "started_at": status.get("started_at", utc_now()),
            "last_update": utc_now(),
            "current_step": step,
            "planned_steps": planned,
            "last_loss": loss,
            "mean_recent_loss": float(np.mean(losses[-25:])),
            "learning_rate": rate,
            "negative_weight": negative_weight,
            "best_validation_f1": best_f1,
            "stale_evaluations": stale,
            "last_validation": validation or status.get("last_validation"),
            "last_checkpoint": checkpoint_path or status.get("last_checkpoint"),
            "dataset_manifest_sha256": config["dataset_manifest_sha256"],
            "test_loaded": False,
            "device": device_info() if step == start_step else status.get("device"),
            "working_set_bytes": working_set_bytes(),
            "elapsed_seconds_this_process": time.perf_counter() - started,
        }
        if step % 25 == 0 or validation is not None:
            atomic_json(status_path, status)
            print(
                f"HEARTBEAT step={step}/{planned} loss={loss:.6f} lr={rate:.2e} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )

        early = formal["early_stopping"]
        if (
            bool(early["enabled"])
            and step >= int(early["warmup_steps"])
            and validation is not None
            and stale >= int(early["patience_evaluations"])
        ):
            status["early_stopped"] = True
            break

    status.update(
        {
            "status": "COMPLETED",
            "last_update": utc_now(),
            "final_step": int(global_step.numpy()),
            "elapsed_seconds_this_process": time.perf_counter() - started,
            "test_loaded": False,
        }
    )
    atomic_json(status_path, status)
    print(
        f"TRAINING_COMPLETED step={int(global_step.numpy())} best_validation_f1={best_f1:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
