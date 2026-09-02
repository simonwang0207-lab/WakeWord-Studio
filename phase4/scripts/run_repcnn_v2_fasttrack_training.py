"""Model B v2 fast-track trainer; Train/Validation only, formal launch is gated."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit.embedded_wakeword.models.classifier import reparameterize_model  # noqa: E402
from livekit.embedded_wakeword.training.trainer import focal_loss  # noqa: E402
from phase3.scripts.preflight_repcnn_model_b import (  # noqa: E402
    build_adapter,
    build_model,
    device_info,
    indexed_model_state,
    quantization_metadata,
    sha256_bytes,
    working_set_bytes,
)
from wakeword_studio.dataset.manifest import sha256_file  # noqa: E402
from wakeword_studio.dataset.repcnn_adapter import RepCNNSample  # noqa: E402
from wakeword_studio.json_utils import atomic_write_json  # noqa: E402
from wakeword_studio.training.repcnn_fasttrack import (  # noqa: E402
    ALL_LABELS,
    CHECKPOINT_METRIC_FORMULA,
    HierarchicalBatchSampler,
    SamplingRecord,
    negative_weight,
    phase_for_step,
    preserve_checkpoint_prefix,
    production_learning_rate,
    select_validation_threshold,
    validation_improves_best,
    validation_rank,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


@dataclass(slots=True)
class FeatureGroup:
    values: np.ndarray
    view_indices: np.ndarray
    samples: tuple[RepCNNSample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def take(self, indices: list[int] | np.ndarray) -> np.ndarray:
        local = np.asarray(indices, dtype=np.int64)
        return np.asarray(self.values[self.view_indices[local]], dtype=np.float32)

    def batches(self, batch_size: int) -> Iterator[np.ndarray]:
        for offset in range(0, len(self), batch_size):
            yield self.take(np.arange(offset, min(len(self), offset + batch_size)))


def load_feature_groups(config: dict[str, Any]) -> dict[tuple[str, str], FeatureGroup]:
    """Map the v3 metadata view onto the verified immutable v2 feature cache."""

    adapter = build_adapter(config)
    if adapter.test_loaded:
        raise RuntimeError("Held-out Test access is prohibited")
    cache_root = Path(config["source_feature_cache"]).resolve()
    summary = json.loads((cache_root / "summary.json").read_text(encoding="utf-8"))
    expected_source_hash = str(config["source_dataset_manifest_sha256"]).lower()
    if summary.get("dataset_manifest_sha256") != expected_source_hash:
        raise RuntimeError("Source feature cache manifest hash does not match the frozen v2 source")
    if summary.get("test_loaded") is not False:
        raise RuntimeError("Source feature cache is not Train/Validation-only")

    groups: dict[tuple[str, str], FeatureGroup] = {}
    for split in ("train", "validation"):
        for label in ALL_LABELS:
            values = np.load(cache_root / f"{split}_{label}.npy", mmap_mode="r")
            metadata_path = cache_root / f"{split}_{label}.jsonl"
            metadata = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(metadata) != len(values):
                raise RuntimeError(f"Feature metadata length mismatch for {split}/{label}")
            by_record = {str(row["record_id"]): int(row["feature_index"]) for row in metadata}
            samples = adapter.samples(split, label)
            missing = [sample.record_id for sample in samples if sample.record_id not in by_record]
            if missing:
                raise RuntimeError(
                    f"Fast-track records missing from source feature cache: {missing[:5]}"
                )
            view_indices = np.asarray([by_record[sample.record_id] for sample in samples], np.int64)
            if tuple(values.shape[1:]) != tuple(config["frontend"]["input_shape"]):
                raise RuntimeError(f"Frozen feature shape changed for {split}/{label}")
            groups[(split, label)] = FeatureGroup(values, view_indices, samples)
    return groups


def sampling_records(groups: dict[tuple[str, str], FeatureGroup]) -> dict[str, list[SamplingRecord]]:
    result: dict[str, list[SamplingRecord]] = {}
    for label in ALL_LABELS:
        result[label] = [
            SamplingRecord(
                index=index,
                record_id=sample.record_id,
                label=sample.label,
                source=sample.source,
                speaker_id=sample.speaker_id,
                text=sample.text,
            )
            for index, sample in enumerate(groups[("train", label)].samples)
        ]
    return result


def make_batch(
    groups: dict[tuple[str, str], FeatureGroup],
    selected: dict[str, list[int]],
    *,
    seed: int,
    step: int,
    mixup_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label in ALL_LABELS:
        indices = selected[label]
        features.append(groups[("train", label)].take(indices))
        targets.append(
            np.full(len(indices), 1.0 if label == "positive" else 0.0, np.float32)
        )
    x = np.concatenate(features)
    y = np.concatenate(targets)
    rng = np.random.default_rng(seed + step * 104729)
    order = rng.permutation(len(y))
    permutation = rng.permutation(len(y))
    lam = float(rng.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
    return x[order], y[order], permutation, lam


def validation_metrics(
    model: tf.keras.Model,
    groups: dict[tuple[str, str], FeatureGroup],
    *,
    maximum_overall_fpr: float,
    batch_size: int = 128,
) -> dict[str, object]:
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []
    for label in ALL_LABELS:
        group = groups[("validation", label)]
        for values in group.batches(batch_size):
            scores.append(
                np.asarray(model(values, training=False)).reshape(-1).astype(np.float64)
            )
        targets.append(np.full(len(group), 1 if label == "positive" else 0, np.int32))
        labels.extend([label] * len(group))
        sources.extend(sample.source for sample in group.samples)
    return select_validation_threshold(
        np.concatenate(scores),
        np.concatenate(targets),
        labels,
        sources,
        maximum_overall_fpr=maximum_overall_fpr,
    )


def export_smoke(
    model: tf.keras.Model,
    groups: dict[tuple[str, str], FeatureGroup],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, object]:
    fused = reparameterize_model(model)
    representatives: list[np.ndarray] = []
    for split in ("train", "validation"):
        for label in ALL_LABELS:
            group = groups[(split, label)]
            representatives.extend(group.take(np.arange(min(4, len(group)))))
    representative_features = np.asarray(representatives, dtype=np.float32)
    probe = representative_features[:8]
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

    def representative_dataset() -> Iterator[list[np.ndarray]]:
        for feature in representative_features:
            yield [feature[np.newaxis, ...].astype(np.float32)]

    started = time.perf_counter()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    model_path = output_dir / "qingxiaojia_repcnn_performance_v2_fasttrack_smoke_full_int8.tflite"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(content)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    quantization = quantization_metadata(interpreter)
    expected_input = config["quantization"]["expected_input_shape"]
    expected_output = config["quantization"]["expected_output_shape"]
    if quantization["input_shape"] != expected_input or quantization["output_shape"] != expected_output:
        raise RuntimeError(f"Deployment graph shape changed: {quantization}")
    if quantization["input_dtype"] != "int8" or quantization["output_dtype"] != "int8":
        raise RuntimeError("Smoke export is not full INT8")
    return {
        "status": "PASS",
        "path": str(model_path),
        "bytes": len(content),
        "kib": len(content) / 1024.0,
        "sha256": sha256_bytes(content),
        "training_parameters": int(model.count_params()),
        "deployment_parameters": int(fused.count_params()),
        "fusion_max_abs_error": fusion_error,
        "quantization": quantization,
        "elapsed_seconds": time.perf_counter() - started,
        "test_loaded": False,
    }


def consume_stop_request(run_dir: Path, *, resume: bool) -> None:
    request = run_dir / "STOP_REQUESTED"
    if not request.exists():
        return
    if not resume:
        raise RuntimeError(f"Stop request already exists: {request}")
    consumed = run_dir / f"STOP_REQUESTED.consumed.{int(time.time())}"
    request.replace(consumed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-formal-training", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_steps:
        if args.allow_formal_training:
            raise SystemExit("Smoke and formal training gates are mutually exclusive")
        if not 1 <= args.smoke_steps <= 20:
            raise SystemExit("Smoke is limited to 1..20 optimization steps")
    elif not args.allow_formal_training:
        raise SystemExit("Formal fast-track training is gated; pass --allow-formal-training")

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    consume_stop_request(run_dir, resume=args.resume)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if sha256_file(Path(config["dataset_manifest"])) != str(config["dataset_manifest_sha256"]):
        raise RuntimeError("Fast-track DatasetManifest hash mismatch")
    if sha256_file(Path(config["source_dataset_manifest"])) != str(
        config["source_dataset_manifest_sha256"]
    ):
        raise RuntimeError("Frozen qingxiaojia_v2 source manifest hash mismatch")
    groups = load_feature_groups(config)
    if any(sample.split == "test" for group in groups.values() for sample in group.samples):
        raise RuntimeError("Held-out Test entered the fast-track feature store")

    counts = {key: int(value) for key, value in config["sampling"]["batch_n_per_class"].items()}
    if sum(counts.values()) != int(config["augmentation"]["batch_size"]):
        raise RuntimeError("Batch composition does not match configured batch size")
    sampler = HierarchicalBatchSampler(
        sampling_records(groups),
        counts,
        seed=int(config["seed"]),
        required_sources=config["sampling"]["required_speech_sources"],
        required_hard_phrases=config["sampling"]["hard_negative_phrases"],
    )

    formal = config["formal_training"]
    configured_total = int(formal["planned_total_steps"])
    if sum(int(value) for value in formal["phase_steps"]) != configured_total:
        raise RuntimeError("Configured phase lengths do not sum to planned_total_steps")
    planned = args.smoke_steps or configured_total
    seed = int(config["seed"])
    tf.keras.utils.set_random_seed(seed)
    model = build_model(config)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=production_learning_rate(formal, 1),
        weight_decay=float(config["weight_decay"]),
    )
    optimizer.build(model.trainable_variables)
    global_step = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        step=global_step,
        model_state=indexed_model_state(model),
        optimizer=optimizer,
    )
    manager = tf.train.CheckpointManager(
        checkpoint,
        str(run_dir / "checkpoints"),
        max_to_keep=int(formal["checkpoint_max_to_keep"]),
    )
    status_path = run_dir / "TRAINING_STATUS.json"
    if args.resume:
        if args.smoke_steps:
            raise RuntimeError("Smoke resume is intentionally unsupported")
        if not manager.latest_checkpoint:
            raise FileNotFoundError("No fast-track checkpoint is available to resume")
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
        print(
            f"STRICT_RESUME restored_step={int(global_step.numpy())} "
            f"checkpoint={manager.latest_checkpoint}",
            flush=True,
        )
    elif manager.latest_checkpoint or status_path.exists():
        raise RuntimeError("Run directory already contains state; use --resume or a new run-dir")

    start_step = int(global_step.numpy()) + 1
    for replay_step in range(1, start_step):
        sampler.sample(replay_step)
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if args.resume and status_path.is_file()
        else {}
    )
    if args.resume:
        best_report_path = run_dir / "BEST_SINGLE_VALIDATION.json"
        if best_report_path.is_file():
            best_step = int(json.loads(best_report_path.read_text(encoding="utf-8"))["step"])
            best_prefix = run_dir / "checkpoints" / f"ckpt-{best_step}"
            preserved = preserve_checkpoint_prefix(
                best_prefix, run_dir / "preserved_best_checkpoint"
            )
            print(
                f"BEST_CHECKPOINT_PRESERVED step={best_step} files={len(preserved)} "
                f"destination={preserved[0].parent}",
                flush=True,
            )
    best_rank = tuple(float(value) for value in status.get("best_validation_rank", []))
    stale = int(status.get("stale_evaluations_after_minimum_step", 0))
    started = time.perf_counter()
    recent_losses: list[float] = []
    gamma = float(config["objective"]["gamma"])
    mixup_alpha = float(config["objective"]["mixup_alpha"])
    stabilization = int(config["objective"]["stabilization_steps_without_mixup"])

    @tf.function
    def train_step(
        features: tf.Tensor,
        targets: tf.Tensor,
        permutation: tf.Tensor,
        lam: tf.Tensor,
        class_negative_weight: tf.Tensor,
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
        with tf.GradientTape() as tape:
            predictions = tf.squeeze(model(mixed_features, training=True), axis=-1)
            losses = focal_loss(predictions, mixed_targets, gamma=gamma)
            weights = tf.where(mixed_targets < 0.5, class_negative_weight, 1.0)
            loss = tf.reduce_mean(losses * weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    stopped = False
    last_validation: dict[str, object] | None = status.get("last_validation")
    for step in range(start_step, planned + 1):
        selected = sampler.sample(step)
        batch_x, batch_y, permutation, lam = make_batch(
            groups, selected, seed=seed, step=step, mixup_alpha=mixup_alpha
        )
        rate = production_learning_rate(formal, step)
        phase, phase_step, phase_length = phase_for_step(formal, step)
        neg_weight = negative_weight(formal, step)
        optimizer.learning_rate.assign(rate)
        use_mixup = step > stabilization
        loss = float(
            train_step(
                tf.constant(batch_x),
                tf.constant(batch_y),
                tf.constant(permutation, tf.int32),
                tf.constant(lam, tf.float32),
                tf.constant(neg_weight, tf.float32),
                tf.constant(use_mixup),
            ).numpy()
        )
        if not math.isfinite(loss):
            raise RuntimeError(f"Fast-track loss became NaN/Inf at step {step}")
        global_step.assign(step)
        recent_losses.append(loss)

        should_validate = step % int(formal["validation_interval"]) == 0 or step == planned
        improved = False
        checkpoint_path: str | None = None
        if should_validate:
            last_validation = validation_metrics(
                model,
                groups,
                maximum_overall_fpr=float(
                    formal["checkpoint_selection"]["maximum_overall_fpr"]
                ),
            )
            rank = validation_rank(last_validation)
            improved = validation_improves_best(last_validation, best_rank)
            if improved:
                best_rank = rank
                stale = 0
                model.save_weights(run_dir / "best_single.weights.h5")
                atomic_json(
                    run_dir / "BEST_SINGLE_VALIDATION.json",
                    {
                        "step": step,
                        "rank": list(rank),
                        "metric_formula": CHECKPOINT_METRIC_FORMULA,
                        "metrics": last_validation,
                        "test_loaded": False,
                    },
                )
            elif step >= int(formal["early_stopping"]["minimum_step"]):
                stale += 1
            checkpoint_path = manager.save(checkpoint_number=step)
            model.save_weights(run_dir / "last.weights.h5")
            atomic_json(run_dir / "SAMPLING_EXPOSURE.json", sampler.exposure_report())
            print(
                f"VALIDATION step={step} threshold={last_validation['threshold']:.4f} "
                f"worst_source_recall={last_validation['worst_source_recall']:.6f} "
                f"recall={last_validation['recall']:.6f} "
                f"source_gap={last_validation['source_gap']:.6f} "
                f"fpr={last_validation['fpr']:.6f} best={improved}",
                flush=True,
            )
            if not bool(last_validation["fpr_cap_satisfied"]):
                print(
                    "VALIDATION_INFEASIBLE "
                    f"configured_fpr_cap={last_validation['maximum_overall_fpr']:.6f} "
                    f"best_available_fpr={last_validation['best_available_fpr']:.6f} "
                    f"fallback_threshold={last_validation['threshold']:.6f} "
                    "best=false training_continues=true",
                    flush=True,
                )
            if bool(last_validation["operating_point_degenerate"]):
                print(
                    "VALIDATION_DEGENERATE "
                    f"threshold={last_validation['threshold']:.6f} "
                    f"recall={last_validation['recall']:.6f} "
                    f"fpr={last_validation['fpr']:.6f} "
                    "reject_all=true eligible_for_best=false training_continues=true",
                    flush=True,
                )
            print(f"CHECKPOINT step={step} path={checkpoint_path}", flush=True)

        status = {
            "status": "SMOKE_RUNNING" if args.smoke_steps else "RUNNING",
            "pid": os.getpid(),
            "started_at": status.get("started_at", utc_now()),
            "last_update": utc_now(),
            "current_step": step,
            "planned_steps": planned,
            "formal_planned_steps": configured_total,
            "phase": phase,
            "phase_step": phase_step,
            "phase_length": phase_length,
            "last_loss": loss,
            "mean_recent_loss": float(np.mean(recent_losses[-25:])),
            "learning_rate": rate,
            "negative_weight": neg_weight,
            "mixup_enabled_this_step": use_mixup,
            "best_validation_rank": list(best_rank),
            "stale_evaluations_after_minimum_step": stale,
            "last_validation": last_validation,
            "last_checkpoint": checkpoint_path or status.get("last_checkpoint"),
            "dataset_manifest_sha256": config["dataset_manifest_sha256"],
            "source_dataset_manifest_sha256": config["source_dataset_manifest_sha256"],
            "checkpoint_metric_formula": CHECKPOINT_METRIC_FORMULA,
            "test_loaded": False,
            "resume_supported": not bool(args.smoke_steps),
            "device": device_info() if step == start_step else status.get("device"),
            "working_set_bytes": working_set_bytes(),
            "elapsed_seconds_this_process": time.perf_counter() - started,
        }
        if step % 25 == 0 or should_validate or step == start_step:
            atomic_json(status_path, status)
            print(
                f"HEARTBEAT step={step}/{planned} phase={phase} loss={loss:.6f} "
                f"lr={rate:.3e} elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )

        if (run_dir / "STOP_REQUESTED").exists():
            checkpoint_path = manager.save(checkpoint_number=step)
            model.save_weights(run_dir / "last.weights.h5")
            status.update(
                {
                    "status": "STOPPED_BY_USER",
                    "last_checkpoint": checkpoint_path,
                    "last_update": utc_now(),
                }
            )
            atomic_json(status_path, status)
            print(f"TRAINING_STOPPED_BY_USER step={step} checkpoint={checkpoint_path}", flush=True)
            stopped = True
            break

        early = formal["early_stopping"]
        if (
            not args.smoke_steps
            and bool(early["enabled"])
            and step >= int(early["minimum_step"])
            and should_validate
            and stale >= int(early["patience_evaluations"])
        ):
            status["early_stopped"] = True
            break

    if stopped:
        return

    smoke_export: dict[str, object] | None = None
    if args.smoke_steps:
        smoke_export = export_smoke(model, groups, config, run_dir / "export_smoke")
        atomic_json(
            run_dir / "SMOKE_REPORT.json",
            {
                "status": "PASS",
                "optimization_steps": args.smoke_steps,
                "validation": last_validation,
                "sampling_exposure": sampler.exposure_report(),
                "checkpoint_saved": bool(manager.latest_checkpoint),
                "export": smoke_export,
                "test_loaded": False,
            },
        )
    status.update(
        {
            "status": "SMOKE_COMPLETED" if args.smoke_steps else "COMPLETED",
            "last_update": utc_now(),
            "final_step": int(global_step.numpy()),
            "elapsed_seconds_this_process": time.perf_counter() - started,
            "smoke_export": smoke_export,
            "test_loaded": False,
        }
    )
    atomic_json(status_path, status)
    print(
        f"TRAINING_COMPLETED step={int(global_step.numpy())} "
        f"mode={'smoke' if args.smoke_steps else 'formal'} test_loaded=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
