"""Short GPU smoke trainer and deployable Full-INT8 exporter for Multi-KWS."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .binary_kws_models import count_trainable_parameters
from .multikws_evaluator import calibrate_validation, metrics_from_predictions, runtime_decision
from .multikws_models import build_multikws_model, estimate_macs
from .multikws_sampler import DeterministicEpochSampler
from .multikws_vocabulary import MultiKWSVocabulary


def _tensorflow() -> Any:
    import tensorflow as tf

    return tf


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_gpu(tf: Any) -> dict[str, Any]:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    names = [
        str(tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name))
        for gpu in gpus
    ]
    device = None
    if gpus:
        with tf.device("/GPU:0"):
            probe = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
        _ = probe.numpy()
        device = str(probe.device)
    return {
        "GPU_DETECTED": bool(gpus),
        "GPU_NAME": names,
        "TRAIN_DEVICE": "GPU" if gpus else "CPU",
        "GPU_OP_EXECUTED": bool(device and "GPU:0" in device.upper()),
        "GPU_OP_DEVICE": device,
        "TENSORFLOW_VERSION": tf.__version__,
    }


def predict_in_batches(
    model: Any, features: np.ndarray, batch_size: int = 32,
) -> np.ndarray:
    """Run ordered inference without placing an entire evaluation split on GPU."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim < 1:
        raise ValueError("features must have a sample dimension")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if len(values) == 0:
        raise ValueError("features must not be empty")
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        prediction = model(values[start:stop], training=False)
        if hasattr(prediction, "numpy"):
            prediction = prediction.numpy()
        batch_output = np.asarray(prediction)
        if len(batch_output) != stop - start:
            raise RuntimeError("Batched inference changed the sample count")
        outputs.append(batch_output)
    result = np.concatenate(outputs, axis=0)
    if len(result) != len(values):
        raise RuntimeError("Batched inference did not preserve the full input order")
    return result


def _predict_int8(tf: Any, model_path: Path, features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    scores = []
    for feature in features:
        value = np.clip(np.rint(feature[np.newaxis] / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        scores.append((output.astype(np.float32) - output_zero) * output_scale)
    # XNNPACK may add a runtime-only DELEGATE node; it is not a FlatBuffer op.
    ops = sorted(
        {
            str(item["op_name"])
            for item in interpreter._get_ops_details()
            if str(item["op_name"]) != "DELEGATE"
        }
    )
    detail = {
        "input_shape": [int(value) for value in input_detail["shape"]],
        "output_shape": [int(value) for value in output_detail["shape"]],
        "input_dtype": str(np.dtype(input_detail["dtype"])),
        "output_dtype": str(np.dtype(output_detail["dtype"])),
        "input_quantization": [float(input_scale), int(input_zero)],
        "output_quantization": [float(output_scale), int(output_zero)],
        "operator_list": ops,
        "tflite_micro_preliminary_audit": "BUILTINS_INT8_ONLY; compile/device runtime not verified",
        "HARDWARE_RUNTIME_VERIFIED": False,
    }
    return np.concatenate(scores, axis=0), detail


def export_full_int8(
    tf: Any,
    model: Any,
    representatives: np.ndarray,
    destination: Path,
    num_classes: int,
) -> dict[str, Any]:
    @tf.function(input_signature=[tf.TensorSpec((1, 99, 40), tf.float32)])
    def serving(value: Any) -> Any:
        return model(value, training=False)

    def representative_dataset() -> Iterator[list[np.ndarray]]:
        for feature in representatives:
            yield [feature[np.newaxis].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serving.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    content = converter.convert()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    _, detail = _predict_int8(tf, destination, representatives[:1])
    if detail["input_shape"] != [1, 99, 40] or detail["output_shape"] != [1, num_classes]:
        raise RuntimeError(f"Unexpected Full-INT8 shape: {detail}")
    if detail["input_dtype"] != "int8" or detail["output_dtype"] != "int8":
        raise RuntimeError("Export is not Full INT8")
    detail.update(
        {
            "path": str(destination.resolve()),
            "bytes": len(content),
            "KiB": len(content) / 1024.0,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    return detail


def train_multikws(
    *,
    model_name: str,
    vocabulary_path: Path,
    feature_store_path: Path,
    run_dir: Path,
    smoke_steps: int = 2,
    require_gpu: bool = True,
    seed: int = 20260901,
    run_mode: str = "smoke",
    validation_interval: int | None = None,
    early_stopping_patience: int | None = None,
    resume: bool = False,
    architecture_config: Mapping[str, Any] | None = None,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    sampler_kind: str = "deterministic_epoch_shuffle",
    drop_last: bool = False,
    max_epochs: int | None = None,
    validation_every_epochs: int | None = None,
    interrupt_after_step: int | None = None,
    evaluation_batch_size: int = 32,
) -> dict[str, Any]:
    """Train with one fair protocol; defaults remain a bounded two-step smoke."""

    started = time.perf_counter()
    tf = _tensorflow()
    gpu = configure_gpu(tf)
    if require_gpu and not gpu["GPU_DETECTED"]:
        raise RuntimeError("GPU is required for this smoke/formal path")
    tf.keras.utils.set_random_seed(seed)
    vocab = MultiKWSVocabulary.load(vocabulary_path)
    arrays = np.load(feature_store_path)
    x_train = np.asarray(arrays["x_train"], np.float32)
    y_train = np.asarray(arrays["y_train"], np.int32)
    x_val = np.asarray(arrays["x_validation"], np.float32)
    y_val = np.asarray(arrays["y_validation"], np.int32)
    metadata = json.loads(feature_store_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata.get("TEST_READ") is not False:
        raise RuntimeError("Feature store does not prove TEST_READ=false")
    sources = [str(row["source"]) for row in metadata["metadata"]["validation"]]

    shared = {"dropout": 0.0, "activation": "relu"}
    architecture = dict(architecture_config) if architecture_config is not None else (
        {**shared, "channels": 12, "depth": 2, "subbands": 4, "temporal_dilations": [1, 2]}
        if model_name == "bcresnet"
        else {**shared, "hidden_dim": 16, "depth": 2, "kernel_size": [5, 3], "patch_size": [3, 2], "stride": [2, 2]}
    )
    model = build_multikws_model(model_name, (99, 40), vocab.num_classes, architecture)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(learning_rate), weight_decay=float(weight_decay)
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    if smoke_steps < 1:
        raise ValueError("training steps must be positive")
    interval = int(validation_interval or smoke_steps)
    if interval < 1:
        raise ValueError("validation_interval must be positive")
    if int(evaluation_batch_size) < 1:
        raise ValueError("evaluation_batch_size must be positive")
    run_dir.mkdir(parents=True, exist_ok=True)
    step_variable = tf.Variable(0, dtype=tf.int64, trainable=False, name="training_step")
    latest_checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step=step_variable
    )
    latest_manager = tf.train.CheckpointManager(
        latest_checkpoint, str(run_dir / "checkpoints" / "latest"), max_to_keep=3
    )
    state_path = run_dir / "TRAINING_STATE.json"
    best_checkpoint_prefix = str((run_dir / "checkpoints" / "best" / "best").resolve())
    Path(best_checkpoint_prefix).parent.mkdir(parents=True, exist_ok=True)
    best_rank: tuple[float, ...] | None = None
    stale_validations = 0
    best_checkpoint_path: str | None = None
    if resume:
        if not latest_manager.latest_checkpoint:
            raise RuntimeError("--resume requested but no latest checkpoint exists")
        latest_checkpoint.restore(latest_manager.latest_checkpoint).expect_partial()
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if int(state.get("current_step", -1)) != int(step_variable.numpy()):
                raise RuntimeError("Checkpoint step and TRAINING_STATE.json disagree")
            rank = state.get("best_validation_rank")
            best_rank = tuple(float(value) for value in rank) if rank is not None else None
            stale_validations = int(state.get("stale_validations", 0))
            best_checkpoint_path = state.get("best_checkpoint_path")
    if sampler_kind != "deterministic_epoch_shuffle":
        raise ValueError(f"Unsupported sampler: {sampler_kind}")
    sampler = DeterministicEpochSampler(
        sample_count=len(x_train), batch_size=int(batch_size), seed=seed,
        drop_last=bool(drop_last),
    )
    sampler_audit = sampler.first_epoch_audit()
    loss = tf.constant(np.nan, tf.float32)
    stopped_early = False
    def write_training_state(interrupted: bool = False) -> None:
        current_step = int(step_variable.numpy())
        _json_write(
            state_path,
            {
                "run_mode": run_mode,
                "current_step": current_step,
                "maximum_steps": smoke_steps,
                "steps_per_epoch": sampler.steps_per_epoch,
                "current_epoch": current_step // sampler.steps_per_epoch,
                "step_in_epoch": current_step % sampler.steps_per_epoch,
                "next_absolute_step": current_step,
                "best_validation_rank": None if best_rank is None else list(best_rank),
                "best_checkpoint_path": best_checkpoint_path,
                "stale_validations": stale_validations,
                "optimizer_state_checkpointed": True,
                "sampler_resume_from_absolute_step": True,
                "interrupted": interrupted,
                "TEST_READ": False,
            },
        )

    try:
        for step in range(int(step_variable.numpy()), smoke_steps):
            batch_indices = sampler.batch_indices(step)
            with tf.GradientTape() as tape:
                scores = model(tf.convert_to_tensor(x_train[batch_indices]), training=True)
                loss = loss_fn(y_train[batch_indices], scores)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step_variable.assign(step + 1)
            if interrupt_after_step is not None and int(step_variable.numpy()) >= interrupt_after_step:
                raise KeyboardInterrupt("intentional checkpoint/resume regression interruption")
            should_validate = (step + 1) % interval == 0 or step + 1 == smoke_steps
            if should_validate:
                validation_scores = predict_in_batches(
                    model, x_val, batch_size=int(evaluation_batch_size)
                )
                validation_metrics = calibrate_validation(
                    validation_scores, y_val, vocab.class_names, sources
                )
                rank = (
                    float(validation_metrics["macro_f1"]),
                    float(validation_metrics["worst_keyword_recall"]),
                    -float(validation_metrics["background_false_accept_rate"]),
                    float(validation_metrics["micro_accuracy"]),
                )
                latest_manager.save(checkpoint_number=step + 1)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    stale_validations = 0
                    best_checkpoint_path = tf.train.Checkpoint(model=model).write(
                        best_checkpoint_prefix
                    )
                else:
                    stale_validations += 1
                write_training_state()
                if (
                    early_stopping_patience is not None
                    and stale_validations >= early_stopping_patience
                ):
                    stopped_early = True
                    break
    except KeyboardInterrupt:
        latest_manager.save(checkpoint_number=int(step_variable.numpy()))
        write_training_state(interrupted=True)
        raise
    if not np.isfinite(float(loss.numpy())):
        raise RuntimeError("Non-finite training loss")

    if best_checkpoint_path is None:
        raise RuntimeError("No Validation checkpoint was selected")
    tf.train.Checkpoint(model=model).restore(best_checkpoint_path).expect_partial()

    checkpoint_prefix = str((run_dir / "checkpoints" / "restore_audit").resolve())
    Path(checkpoint_prefix).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tf.train.Checkpoint(model=model).write(checkpoint_prefix)
    restored = build_multikws_model(model_name, (99, 40), vocab.num_classes, architecture)
    # ``Checkpoint.write`` intentionally omits Checkpoint's own save_counter;
    # restored inference below is the stronger model-weight equality check.
    tf.train.Checkpoint(model=restored).restore(checkpoint_path).expect_partial()
    reference = model(x_val[:1], training=False).numpy()
    checkpoint_max_abs_diff = float(np.max(np.abs(reference - restored(x_val[:1], training=False).numpy())))
    if checkpoint_max_abs_diff > 1e-6:
        raise RuntimeError("Checkpoint restore mismatch")

    float_scores = predict_in_batches(model, x_val, batch_size=int(evaluation_batch_size))
    float_metrics = calibrate_validation(float_scores, y_val, vocab.class_names, sources)
    threshold = float(float_metrics["threshold"])
    margin = float(float_metrics["margin_threshold"])
    threshold_freeze = {
        "source": "validation_only",
        "top1_threshold": threshold,
        "margin_threshold": margin,
        "TEST_READ": False,
    }
    _json_write(run_dir / "threshold_freeze.json", threshold_freeze)
    _json_write(run_dir / "multikws_calibration.json", float_metrics)

    export_path = run_dir / "export" / f"teacher_six_{model_name}_{run_mode}_full_int8.tflite"
    representatives = x_train
    int8_export = export_full_int8(tf, model, representatives, export_path, vocab.num_classes)
    int8_scores, _ = _predict_int8(tf, export_path, x_val)
    int8_predictions = np.asarray(
        [runtime_decision(row, threshold=threshold, margin_threshold=margin).class_index for row in int8_scores],
        np.int32,
    )
    int8_metrics = metrics_from_predictions(
        int8_scores, y_val, int8_predictions, vocab.class_names, sources, threshold, margin
    )
    degradation = {
        "macro_recall_pp": 100.0 * (float_metrics["macro_recall"] - int8_metrics["macro_recall"]),
        "macro_f1_pp": 100.0 * (float_metrics["macro_f1"] - int8_metrics["macro_f1"]),
        "worst_keyword_recall_pp": 100.0 * (
            float_metrics["worst_keyword_recall"] - int8_metrics["worst_keyword_recall"]
        ),
    }
    _json_write(run_dir / "confusion_float_validation.json", float_metrics)
    _json_write(run_dir / "confusion_int8_validation.json", int8_metrics)
    report = {
        "schema": f"wakeword-studio.multikws-{run_mode}/v1",
        "run_mode": run_mode,
        "model_name": model_name,
        "num_classes": vocab.num_classes,
        "class_names": list(vocab.class_names),
        "input_shape": [99, 40],
        "objective": "sparse_categorical_crossentropy",
        "optimizer_policy": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
        },
        "sampler": {
            "kind": sampler_kind,
            "seed": seed,
            "drop_last": bool(drop_last),
            "steps_per_epoch": sampler.steps_per_epoch,
            "first_epoch": sampler_audit,
            "resume_order_reproducible_from_absolute_step": True,
        },
        "architecture_config": architecture,
        "maximum_steps": smoke_steps,
        "completed_steps": int(step_variable.numpy()),
        "optimizer_iterations": int(optimizer.iterations.numpy()),
        "validation_interval": interval,
        "evaluation_batch_size": int(evaluation_batch_size),
        "early_stopping_patience": early_stopping_patience,
        "max_epochs": max_epochs,
        "validation_every_epochs": validation_every_epochs,
        "stopped_early": stopped_early,
        "best_validation_rank": list(best_rank),
        "best_checkpoint_path": best_checkpoint_path,
        "loss": float(loss.numpy()),
        "forward_backward": True,
        "checkpoint_path": checkpoint_path,
        "checkpoint_restore_max_abs_diff": checkpoint_max_abs_diff,
        "parameter_count": count_trainable_parameters(model),
        "estimated_macs": estimate_macs(model_name, (99, 40), vocab.num_classes, architecture),
        "float_validation": float_metrics,
        "int8_validation": int8_metrics,
        "float_to_int8_degradation": degradation,
        "int8_export": int8_export,
        "PTQ_REPRESENTATIVE_SPLIT": "train",
        "TEST_READ": False,
        "HARDWARE_RUNTIME_VERIFIED": False,
        "elapsed_seconds": time.perf_counter() - started,
        **gpu,
    }
    _json_write(run_dir / ("SMOKE_REPORT.json" if run_mode == "smoke" else "TRAINING_REPORT.json"), report)
    return report
