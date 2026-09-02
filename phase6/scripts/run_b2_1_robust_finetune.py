"""Explicitly gated 750-step Train-only/Validation-only B2.1 fine-tuner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit.embedded_wakeword.training.trainer import focal_loss  # noqa: E402
from phase3.scripts.preflight_repcnn_model_b import build_model, indexed_model_state  # noqa: E402
from phase4.scripts.run_repcnn_v2_fasttrack_training import (  # noqa: E402
    atomic_json,
    load_feature_groups,
    make_batch,
    sampling_records,
    utc_now,
    validation_metrics,
)
from wakeword_studio.training.repcnn_fasttrack import (  # noqa: E402
    CHECKPOINT_METRIC_FORMULA,
    HierarchicalBatchSampler,
    validation_improves_best,
    validation_rank,
)
from wakeword_studio.training.repcnn_robust_augmentation import (  # noqa: E402
    augment_training_features,
)
from wakeword_studio.dataset.manifest import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/run the separately gated B2.1 robust fine-tune")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/models/repcnn_performance_v2_1_robust_finetune.yaml",
    )
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-b2-1-finetune", action="store_true")
    args = parser.parse_args()
    if not args.allow_b2_1_finetune and not args.preflight_only:
        raise SystemExit("B2.1 is gated; pass --allow-b2-1-finetune only after user approval")

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("B2.1 run directory must be new and empty")
    base_weights = args.base_weights.resolve()
    if not base_weights.is_file():
        raise FileNotFoundError(base_weights)
    if base_weights == run_dir or run_dir in base_weights.parents:
        raise RuntimeError("B2.1 run directory overlaps the frozen base model")
    planned = int(config["fine_tune"]["planned_steps"])
    allowed_low, allowed_high = (int(value) for value in config["fine_tune"]["allowed_step_range"])
    if not allowed_low <= planned <= allowed_high:
        raise RuntimeError("B2.1 planned steps left the approved 500..1000 range")
    shift = config["augmentation"]["temporal_shift"]
    spec = config["augmentation"]["mild_spec_augment"]
    if not shift["enabled"] or int(shift["maximum_frames"]) != 3:
        raise RuntimeError("B2.1 temporal shift is not the approved +/-3-frame policy")
    if not spec["enabled"]:
        raise RuntimeError("B2.1 mild SpecAugment is not enabled")
    groups = load_feature_groups(config)
    if any(sample.split == "test" for group in groups.values() for sample in group.samples):
        raise RuntimeError("Held-out Test entered B2.1")

    seed = int(config["seed"])
    tf.keras.utils.set_random_seed(seed)
    model = build_model(config)
    model.load_weights(base_weights)
    probe = groups[("validation", "positive")].take(np.asarray([0], np.int64))
    probe_score = float(np.asarray(model(probe, training=False)).reshape(-1)[0])
    if not np.isfinite(probe_score):
        raise RuntimeError("Loaded B2 base produced a non-finite Validation probe score")
    base_sha256 = sha256_file(base_weights)
    finalization_report_path = base_weights.parent / "FINALIZATION_REPORT.json"
    if finalization_report_path.is_file():
        finalization_report = json.loads(
            finalization_report_path.read_text(encoding="utf-8")
        )
        expected_sha256 = str(finalization_report["final_weights"]["sha256"])
        if base_sha256 != expected_sha256:
            raise RuntimeError("Base weights SHA256 differs from Finalizer v2 report")
    preflight_report = {
        "status": "PASS",
        "planned_steps": planned,
        "base_weights": str(base_weights),
        "base_weights_sha256": base_sha256,
        "base_model_loaded": True,
        "validation_probe_score": probe_score,
        "validation_counts": {
            label: len(groups[("validation", label)])
            for label in ("positive", "negative", "hard_negative", "ambient")
        },
        "temporal_shift": {
            "enabled": bool(shift["enabled"]),
            "maximum_frames": int(shift["maximum_frames"]),
            "probability": float(shift["probability"]),
        },
        "mild_spec_augment": {
            "enabled": bool(spec["enabled"]),
            "maximum_time_mask_frames": int(spec["maximum_time_mask_frames"]),
            "maximum_frequency_mask_bins": int(spec["maximum_frequency_mask_bins"]),
            "probability": float(spec["probability"]),
        },
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
        "run_dir_empty_or_absent": not run_dir.exists() or not any(run_dir.iterdir()),
        "test_loaded": False,
        "training_started": False,
    }
    if args.preflight_only:
        print("B2_1_PREFLIGHT " + json.dumps(preflight_report, ensure_ascii=False), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(config["fine_tune"]["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    optimizer.build(model.trainable_variables)
    step_variable = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        step=step_variable,
        model_state=indexed_model_state(model),
        optimizer=optimizer,
    )
    manager = tf.train.CheckpointManager(
        checkpoint,
        str(run_dir / "checkpoints"),
        max_to_keep=int(config["fine_tune"]["checkpoint_max_to_keep"]),
    )
    counts = {key: int(value) for key, value in config["sampling"]["batch_n_per_class"].items()}
    sampler = HierarchicalBatchSampler(
        sampling_records(groups),
        counts,
        seed=seed,
        required_sources=config["sampling"]["required_speech_sources"],
        required_hard_phrases=config["sampling"]["hard_negative_phrases"],
    )
    gamma = float(config["objective"]["gamma"])
    negative_weight = float(config["objective"]["negative_weight"])
    mixup_alpha = float(config["objective"]["mixup_alpha"])
    augmentation = config["augmentation"]
    shift = augmentation["temporal_shift"]
    spec = augmentation["mild_spec_augment"]
    microphone_eq = augmentation["microphone_eq"]

    @tf.function
    def train_step(x, y, permutation, lam):
        mixed_x = lam * x + (1.0 - lam) * tf.gather(x, permutation)
        mixed_y = lam * y + (1.0 - lam) * tf.gather(y, permutation)
        with tf.GradientTape() as tape:
            prediction = tf.squeeze(model(mixed_x, training=True), axis=-1)
            losses = focal_loss(prediction, mixed_y, gamma=gamma)
            weights = tf.where(mixed_y < 0.5, negative_weight, 1.0)
            loss = tf.reduce_mean(losses * weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    started = time.perf_counter()
    best_rank: tuple[float, ...] = ()
    best_validation = None
    for step_number in range(1, planned + 1):
        selected = sampler.sample(step_number)
        x, y, permutation, lam = make_batch(
            groups,
            selected,
            seed=seed,
            step=step_number,
            mixup_alpha=mixup_alpha,
        )
        x = augment_training_features(
            x,
            seed=seed,
            step=step_number,
            maximum_shift_frames=int(shift["maximum_frames"]),
            shift_probability=float(shift["probability"]),
            spec_augment_probability=float(spec["probability"]),
            maximum_time_mask_frames=int(spec["maximum_time_mask_frames"]),
            maximum_frequency_mask_bins=int(spec["maximum_frequency_mask_bins"]),
            microphone_eq_maximum_edge_gain_db=(
                float(microphone_eq["maximum_edge_gain_db"])
                if microphone_eq["enabled"]
                else 0.0
            ),
        )
        loss = float(
            train_step(
                tf.constant(x),
                tf.constant(y),
                tf.constant(permutation, tf.int32),
                tf.constant(lam, tf.float32),
            ).numpy()
        )
        if not np.isfinite(loss):
            raise RuntimeError(f"B2.1 loss became non-finite at step {step_number}")
        step_variable.assign(step_number)
        should_validate = (
            step_number % int(config["fine_tune"]["validation_interval"]) == 0
            or step_number == planned
        )
        if should_validate:
            metrics = validation_metrics(
                model,
                groups,
                maximum_overall_fpr=float(config["fine_tune"]["maximum_overall_fpr"]),
            )
            improved = validation_improves_best(metrics, best_rank)
            if improved:
                best_rank = validation_rank(metrics)
                best_validation = metrics
                model.save_weights(run_dir / "best.weights.h5")
            checkpoint_path = manager.save(checkpoint_number=step_number)
            atomic_json(
                run_dir / "VALIDATION_LATEST.json",
                {
                    "step": step_number,
                    "metrics": metrics,
                    "improved": improved,
                    "metric_formula": CHECKPOINT_METRIC_FORMULA,
                    "test_loaded": False,
                },
            )
            print(
                f"B2_1_VALIDATION step={step_number} recall={metrics['recall']:.6f} "
                f"fpr={metrics['fpr']:.6f} best={improved} checkpoint={checkpoint_path}",
                flush=True,
            )
        if step_number == 1 or step_number % 25 == 0:
            atomic_json(
                run_dir / "TRAINING_STATUS.json",
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "current_step": step_number,
                    "planned_steps": planned,
                    "base_weights": str(base_weights),
                    "last_loss": loss,
                    "best_validation": best_validation,
                    "test_loaded": False,
                    "elapsed_seconds": time.perf_counter() - started,
                    "last_update": utc_now(),
                },
            )
            print(f"B2_1_HEARTBEAT step={step_number}/{planned} loss={loss:.6f}", flush=True)
    atomic_json(
        run_dir / "TRAINING_STATUS.json",
        {
            "status": "COMPLETED",
            "current_step": planned,
            "planned_steps": planned,
            "base_weights": str(base_weights),
            "best_validation": best_validation,
            "test_loaded": False,
            "elapsed_seconds": time.perf_counter() - started,
            "last_update": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
