"""Unified TensorFlow trainer for the fair binary-KWS architecture experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from wakeword_studio.json_utils import atomic_write_json
from wakeword_studio.training.binary_kws_models import (
    build_binary_kws_model,
    count_trainable_parameters,
)
from wakeword_studio.training.fair_evaluator import evaluate_validation_scores
from wakeword_studio.training.fair_feature_store import FrozenFeatureStore
from wakeword_studio.training.repcnn_fasttrack import (
    ALL_LABELS,
    CHECKPOINT_METRIC_FORMULA,
    HierarchicalBatchSampler,
    negative_weight,
    phase_for_step,
    production_learning_rate,
    validation_improves_best,
    validation_rank,
)
from wakeword_studio.training.repcnn_finalization import threshold_candidates


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def focal_binary_loss(tf: Any, predictions: Any, targets: Any, gamma: float) -> Any:
    predictions = tf.clip_by_value(predictions, 1e-7, 1.0 - 1e-7)
    cross_entropy = -(
        targets * tf.math.log(predictions)
        + (1.0 - targets) * tf.math.log(1.0 - predictions)
    )
    probability_true = targets * predictions + (1.0 - targets) * (1.0 - predictions)
    return tf.pow(1.0 - probability_true, gamma) * cross_entropy


def make_batch(
    store: FrozenFeatureStore,
    selected: Mapping[str, list[int]],
    *,
    seed: int,
    step: int,
    mixup_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label in ALL_LABELS:
        indices = selected[label]
        features.append(store.groups[("train", label)].take(indices))
        targets.append(np.full(len(indices), label == "positive", np.float32))
    x = np.concatenate(features)
    y = np.concatenate(targets)
    rng = np.random.default_rng(seed + step * 104729)
    order = rng.permutation(len(y))
    permutation = rng.permutation(len(y))
    lam = float(rng.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
    return x[order], y[order], permutation, lam


def validation_arrays(
    model: Any, store: FrozenFeatureStore, *, batch_size: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []
    for label in ALL_LABELS:
        group = store.groups[("validation", label)]
        for values in group.batches(batch_size):
            scores.append(np.asarray(model(values, training=False)).reshape(-1).astype(np.float64))
        targets.append(np.full(len(group), label == "positive", np.int32))
        labels.extend([label] * len(group))
        sources.extend(sample.source for sample in group.samples)
    return np.concatenate(scores), np.concatenate(targets), labels, sources


def compact_evaluation(metrics: Mapping[str, object]) -> dict[str, object]:
    """Keep reports useful without embedding multi-megabyte threshold sweeps."""

    result = dict(metrics)
    result.pop("threshold_sweep", None)
    points = dict(result.get("operating_points", {}))
    points.pop("threshold_sweep", None)
    result["operating_points"] = points
    return result


def evaluate_model(
    model: Any,
    store: FrozenFeatureStore,
    *,
    maximum_overall_fpr: float,
    batch_size: int = 128,
) -> dict[str, object]:
    scores, targets, labels, sources = validation_arrays(model, store, batch_size=batch_size)
    return compact_evaluation(
        evaluate_validation_scores(
            scores,
            targets,
            labels,
            sources,
            maximum_overall_fpr=maximum_overall_fpr,
        )
    )


def tensorflow_device_info(tf: Any) -> dict[str, object]:
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [
        str(tf.config.experimental.get_device_details(device).get("device_name", device.name))
        for device in gpus
    ]
    op_device = None
    if gpus:
        with tf.device("/GPU:0"):
            probe = tf.linalg.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
        _ = float(tf.reduce_sum(probe).numpy())
        op_device = str(probe.device)
    return {
        "framework": "TensorFlow",
        "framework_version": tf.__version__,
        "train_device": "GPU" if gpus else "CPU",
        "gpu_count": len(gpus),
        "gpu_names": gpu_names,
        "cuda_status": "AVAILABLE" if gpus else "UNAVAILABLE",
        "gpu_op_device": op_device,
        "gpu_op_executed": bool(gpus and op_device and "GPU:0" in op_device.upper()),
    }


def _representative_features(store: FrozenFeatureStore, count_per_group: int = 4) -> np.ndarray:
    rows: list[np.ndarray] = []
    for split in ("train", "validation"):
        for label in ALL_LABELS:
            group = store.groups[(split, label)]
            rows.extend(group.take(np.arange(min(count_per_group, len(group)))))
    return np.asarray(rows, dtype=np.float32)


def export_full_int8(
    tf: Any,
    model: Any,
    store: FrozenFeatureStore,
    destination: Path,
) -> dict[str, object]:
    shape = store.input_shape
    representatives = _representative_features(store)

    @tf.function(input_signature=[tf.TensorSpec((1, *shape), tf.float32)])
    def serving(value: Any) -> Any:
        return model(value, training=False)

    def representative_dataset() -> Iterator[list[np.ndarray]]:
        for feature in representatives:
            yield [feature[np.newaxis, ...].astype(np.float32)]

    started = time.perf_counter()
    # Match the frozen RepCNN export route.  Passing the Keras trackable as the
    # second argument can retain READ_VARIABLE ops in TF 2.21 calibration.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)

    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_shape = [int(value) for value in input_detail["shape"]]
    output_shape = [int(value) for value in output_detail["shape"]]
    if input_shape != [1, *shape] or output_shape != [1, 1]:
        raise RuntimeError(f"Unexpected TFLite tensor shape: input={input_shape} output={output_shape}")
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("Export is not full INT8")
    return {
        "path": str(destination.resolve()),
        "bytes": len(content),
        "kib": len(content) / 1024.0,
        "sha256": sha256_bytes(content),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "input_dtype": str(np.dtype(input_detail["dtype"])),
        "output_dtype": str(np.dtype(output_detail["dtype"])),
        "input_quantization": [float(input_detail["quantization"][0]), int(input_detail["quantization"][1])],
        "output_quantization": [float(output_detail["quantization"][0]), int(output_detail["quantization"][1])],
        "elapsed_seconds": time.perf_counter() - started,
        "test_loaded": False,
    }


def evaluate_int8(
    tf: Any,
    model_path: Path,
    store: FrozenFeatureStore,
    *,
    maximum_overall_fpr: float,
) -> dict[str, object]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    if not input_scale or not output_scale:
        raise RuntimeError("INT8 model lacks quantization scales")
    scores: list[float] = []
    targets: list[int] = []
    labels: list[str] = []
    sources: list[str] = []
    for label in ALL_LABELS:
        group = store.groups[("validation", label)]
        for index in range(len(group)):
            feature = group.take([index])
            quantized = np.clip(np.rint(feature / input_scale + input_zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_detail["index"], quantized)
            interpreter.invoke()
            raw = interpreter.get_tensor(output_detail["index"])[0, 0]
            scores.append(float((float(raw) - output_zero) * output_scale))
        targets.extend([1 if label == "positive" else 0] * len(group))
        labels.extend([label] * len(group))
        sources.extend(sample.source for sample in group.samples)
    return compact_evaluation(
        evaluate_validation_scores(
            scores,
            targets,
            labels,
            sources,
            maximum_overall_fpr=maximum_overall_fpr,
            thresholds=threshold_candidates(scores),
        )
    )


def checkpoint_roundtrip(
    tf: Any,
    *,
    checkpoint_path: str,
    model_name: str,
    model_config: Mapping[str, object],
    store: FrozenFeatureStore,
    original_model: Any,
) -> dict[str, object]:
    restored_model = build_binary_kws_model(model_name, store.input_shape, model_config)
    restore = tf.train.Checkpoint(model=restored_model).restore(checkpoint_path)
    restore.expect_partial()
    probe = store.groups[("validation", "positive")].take([0, 1])
    error = float(
        np.max(
            np.abs(
                np.asarray(original_model(probe, training=False))
                - np.asarray(restored_model(probe, training=False))
            )
        )
    )
    if error > 1e-6:
        raise RuntimeError(f"Checkpoint round-trip changed predictions: {error}")
    return {"status": "PASS", "max_abs_error": error, "checkpoint": checkpoint_path}


def run_training(
    *,
    tf: Any,
    config: Mapping[str, Any],
    store: FrozenFeatureStore,
    run_dir: Path,
    smoke_steps: int,
    resume: bool,
) -> dict[str, object]:
    """Run a bounded smoke or an explicitly authorized formal experiment."""

    run_dir.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    model_name = str(experiment["model_name"])
    model_config = config["model"]
    formal = config["formal_training"]
    objective = config["objective"]
    seed = int(config["seed"])
    device = tensorflow_device_info(tf)
    if not smoke_steps and bool(experiment.get("require_gpu_for_formal", True)) and not device["gpu_count"]:
        raise RuntimeError("Formal training requires a TensorFlow-visible GPU in WSL2")

    tf.keras.utils.set_random_seed(seed)
    model = build_binary_kws_model(model_name, store.input_shape, model_config)
    parameter_count = int(model.count_params())
    trainable_count = count_trainable_parameters(model)
    print("FRAMEWORK=TensorFlow", flush=True)
    print(f"TRAIN_DEVICE={device['train_device']}", flush=True)
    print(f"GPU_NAME={','.join(device['gpu_names']) or 'NONE'}", flush=True)
    print(f"GPU_COUNT={device['gpu_count']}", flush=True)
    print(f"CUDA_STATUS={device['cuda_status']}", flush=True)
    print(f"INPUT_SHAPE={[1, *store.input_shape]}", flush=True)
    print(f"MODEL_NAME={model_name}", flush=True)
    print(f"PARAM_COUNT={parameter_count}", flush=True)

    counts = {key: int(value) for key, value in config["sampling"]["batch_n_per_class"].items()}
    if sum(counts.values()) != int(config["sampling"]["batch_size"]):
        raise RuntimeError("Batch composition does not match batch_size")
    sampler = HierarchicalBatchSampler(
        store.sampling_records(),
        counts,
        seed=seed,
        required_sources=tuple(config["sampling"]["required_speech_sources"]),
        required_hard_phrases=tuple(config["sampling"].get("required_hard_phrases", ())),
    )
    configured_total = int(formal["planned_total_steps"])
    if sum(int(value) for value in formal["phase_steps"]) != configured_total:
        raise RuntimeError("phase_steps do not sum to planned_total_steps")
    planned = smoke_steps or configured_total
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=production_learning_rate(formal, 1),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    optimizer.build(model.trainable_variables)
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    checkpoint = tf.train.Checkpoint(step=global_step, model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, str(run_dir / "checkpoints"), max_to_keep=int(formal["checkpoint_max_to_keep"])
    )
    status_path = run_dir / "TRAINING_STATUS.json"
    status: dict[str, object] = {}
    if resume:
        if smoke_steps:
            raise RuntimeError("Smoke resume is intentionally unsupported")
        if not manager.latest_checkpoint:
            raise FileNotFoundError("No checkpoint is available to resume")
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
    elif manager.latest_checkpoint or status_path.exists():
        raise RuntimeError("Run directory already contains state; use --resume or a new run directory")

    start_step = int(global_step.numpy()) + 1
    for replay_step in range(1, start_step):
        sampler.sample(replay_step)
    best_rank = tuple(float(value) for value in status.get("best_validation_rank", []))
    stale = int(status.get("stale_evaluations", 0))
    started = time.perf_counter()
    gamma = float(objective["gamma"])
    mixup_alpha = float(objective["mixup_alpha"])
    stabilization = int(objective["stabilization_steps_without_mixup"])

    @tf.function
    def train_step(features: Any, targets: Any, permutation: Any, lam: Any, neg_weight: Any, use_mixup: Any) -> Any:
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
            losses = focal_binary_loss(tf, predictions, mixed_targets, gamma)
            weights = tf.where(mixed_targets < 0.5, neg_weight, 1.0)
            loss = tf.reduce_mean(losses * weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    latest_validation: dict[str, object] | None = None
    try:
        for step in range(start_step, planned + 1):
            selected = sampler.sample(step)
            batch_x, batch_y, permutation, lam = make_batch(
                store, selected, seed=seed, step=step, mixup_alpha=mixup_alpha
            )
            rate = production_learning_rate(formal, step)
            phase, phase_step, phase_length = phase_for_step(formal, step)
            neg_weight = negative_weight(formal, step)
            optimizer.learning_rate.assign(rate)
            loss = float(
                train_step(
                    tf.constant(batch_x),
                    tf.constant(batch_y),
                    tf.constant(permutation, tf.int32),
                    tf.constant(lam, tf.float32),
                    tf.constant(neg_weight, tf.float32),
                    tf.constant(step > stabilization),
                ).numpy()
            )
            if not math.isfinite(loss):
                raise RuntimeError(f"Loss became non-finite at step {step}")
            global_step.assign(step)
            validate = step % int(formal["validation_interval"]) == 0 or step == planned
            checkpoint_path = None
            if validate:
                latest_validation = evaluate_model(
                    model,
                    store,
                    maximum_overall_fpr=float(formal["checkpoint_selection"]["maximum_overall_fpr"]),
                )
                rank = validation_rank(latest_validation)
                improved = validation_improves_best(latest_validation, best_rank)
                if improved:
                    best_rank = rank
                    stale = 0
                    model.save_weights(run_dir / "best_single.weights.h5")
                    atomic_write_json(
                        run_dir / "BEST_SINGLE_VALIDATION.json",
                        {"step": step, "rank": list(rank), "metrics": latest_validation, "test_loaded": False},
                    )
                else:
                    stale += 1
                checkpoint_path = manager.save(checkpoint_number=step)
                model.save_weights(run_dir / "last.weights.h5")
                print(
                    f"VALIDATION step={step} threshold={latest_validation['threshold']:.8f} "
                    f"recall={latest_validation['recall']:.6f} fpr={latest_validation['fpr']:.6f} "
                    f"worst_source_recall={latest_validation['worst_source_recall']:.6f} best={improved}",
                    flush=True,
                )
            status = {
                "status": "SMOKE_RUNNING" if smoke_steps else "RUNNING",
                "model_name": model_name,
                "current_step": step,
                "planned_steps": planned,
                "formal_planned_steps": configured_total,
                "phase": phase,
                "phase_step": phase_step,
                "phase_length": phase_length,
                "last_loss": loss,
                "learning_rate": rate,
                "negative_weight": neg_weight,
                "best_validation_rank": list(best_rank),
                "stale_evaluations": stale,
                "last_validation": latest_validation,
                "last_checkpoint": checkpoint_path or status.get("last_checkpoint"),
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_count,
                "input_shape": [1, *store.input_shape],
                "device": device,
                "dataset_counts": store.counts(),
                "source_manifest_sha256": store.source_manifest_sha256,
                "view_manifest_sha256": store.view_manifest_sha256,
                "checkpoint_metric_formula": CHECKPOINT_METRIC_FORMULA,
                "test_loaded": False,
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at": utc_now(),
            }
            atomic_write_json(status_path, status)
            print(f"HEARTBEAT step={step}/{planned} loss={loss:.6f}", flush=True)
            early = formal["early_stopping"]
            if (
                not smoke_steps
                and bool(early["enabled"])
                and step >= int(early["minimum_step"])
                and validate
                and stale >= int(early["patience_evaluations"])
            ):
                break
    except KeyboardInterrupt:
        checkpoint_path = manager.save(checkpoint_number=int(global_step.numpy()))
        status.update(
            {"status": "INTERRUPTED_RESUMABLE", "last_checkpoint": checkpoint_path, "test_loaded": False}
        )
        atomic_write_json(status_path, status)
        raise

    if not manager.latest_checkpoint:
        manager.save(checkpoint_number=int(global_step.numpy()))
    export_name = f"qingxiaojia_{model_name}_{'smoke' if smoke_steps else 'formal'}_full_int8.tflite"
    if not smoke_steps:
        best_weights = run_dir / "best_single.weights.h5"
        if not best_weights.is_file():
            raise RuntimeError("Formal run ended without a Validation-eligible best model")
        model.load_weights(best_weights)
    export = export_full_int8(tf, model, store, run_dir / "export" / export_name)
    if smoke_steps:
        roundtrip = checkpoint_roundtrip(
            tf,
            checkpoint_path=str(manager.latest_checkpoint),
            model_name=model_name,
            model_config=model_config,
            store=store,
            original_model=model,
        )
        report = {
            "schema": "wakeword-studio.fair-binary-kws-smoke/v1",
            "status": "PASS",
            "formal_result": False,
            "model_name": model_name,
            "optimization_steps": int(global_step.numpy()),
            "validation_smoke": latest_validation,
            "checkpoint": roundtrip,
            "export": export,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "input_shape": [1, *store.input_shape],
            "device": device,
            "test_loaded": False,
        }
        atomic_write_json(run_dir / "SMOKE_REPORT.json", report)
        status.update({"status": "SMOKE_COMPLETED", "smoke_report": str(run_dir / "SMOKE_REPORT.json")})
    else:
        int8_metrics = evaluate_int8(
            tf,
            Path(str(export["path"])),
            store,
            maximum_overall_fpr=float(formal["checkpoint_selection"]["maximum_overall_fpr"]),
        )
        report = {
            "schema": "wakeword-studio.fair-binary-kws-formal/v1",
            "status": "PASS",
            "formal_result": True,
            "model_name": model_name,
            "keyword": experiment["keyword"],
            "task": experiment["task"],
            "metrics": int8_metrics,
            "export": export,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "input_shape": [1, *store.input_shape],
            "device": device,
            "test_loaded": False,
        }
        atomic_write_json(run_dir / "FORMAL_RESULT.json", report)
        atomic_write_json(
            run_dir / "threshold_freeze.json",
            {
                "threshold": int8_metrics["threshold"],
                "threshold_source": "validation_only_final_int8_scores",
                "selection_policy": "frozen RepCNN B2 Validation ranking with overall FPR <= 0.10",
                "model_sha256": export["sha256"],
                "test_loaded": False,
            },
        )
        status.update({"status": "COMPLETED", "formal_result": str(run_dir / "FORMAL_RESULT.json")})
    status.update({"final_step": int(global_step.numpy()), "export": export, "test_loaded": False})
    atomic_write_json(status_path, status)
    return report
