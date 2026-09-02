"""Monitored, resumable microWakeWord Tiny benchmark/formal training runner."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microwakeword import mixednet
from microwakeword.data import FeatureHandler, spec_augment
from wakeword_studio.json_utils import json_dumps


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json_dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def system_memory() -> dict[str, float | None]:
    if os.name != "nt":
        return {"total_ram_gib": None, "available_ram_gib": None}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"total_ram_gib": None, "available_ram_gib": None}
    gib = 1024**3
    return {
        "total_ram_gib": round(status.ullTotalPhys / gib, 3),
        "available_ram_gib": round(status.ullAvailPhys / gib, 3),
    }


def working_set_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.WorkingSetSize)


def device_info() -> dict[str, object]:
    physical = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    hardware_gpu: dict[str, object] | None = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, total, free, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            hardware_gpu = {
                "name": name,
                "vram_total_mib": int(total),
                "vram_free_mib": int(free),
                "driver": driver,
            }
    return {
        "tensorflow_version": tf.__version__,
        "tensorflow_devices": [{"name": item.name, "type": item.device_type} for item in physical],
        "tensorflow_gpu_available": bool(gpus),
        "selected_device": "GPU" if gpus else "CPU",
        "hardware_gpu": hardware_gpu,
        **system_memory(),
    }


def effective_stage(values: list[object], training_steps: list[int], step: int) -> object:
    boundary = 0
    for index, count in enumerate(training_steps):
        boundary += int(count)
        if step <= boundary:
            return values[min(index, len(values) - 1)]
    return values[-1]


def binary_metrics(labels: np.ndarray, scores: np.ndarray, groups: list[str]) -> dict[str, object]:
    labels = labels.astype(np.int32).reshape(-1)
    scores = np.clip(scores.astype(np.float64).reshape(-1), 1e-7, 1 - 1e-7)
    predicted = scores >= 0.5
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    group_false_accepts = {
        group: int(sum(score >= 0.5 for score, item_group in zip(scores, groups) if item_group == group))
        for group in sorted(set(groups) - {"positive"})
    }
    return {
        "threshold": 0.5,
        "loss": float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))),
        "accuracy": (tp + tn) / len(labels),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "false_accepts_by_group": group_false_accepts,
        "score_statistics": {
            "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "standard_deviation": float(np.std(scores)),
            "distinct_rounded_1e7": int(len(np.unique(np.round(scores, 7)))),
            "all_identical": bool(np.ptp(scores) <= 1e-12),
        },
    }


def predict_fixed_batch(model: tf.keras.Model, data: np.ndarray, batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for offset in range(0, len(data), batch_size):
        chunk = data[offset : offset + batch_size]
        count = len(chunk)
        if count < batch_size:
            chunk = np.pad(chunk, ((0, batch_size - count), (0, 0), (0, 0)))
        predictions.append(np.asarray(model(chunk, training=False)).reshape(-1)[:count])
    return np.concatenate(predictions)


def load_validation(
    handler: FeatureHandler, feature_specs: list[dict[str, object]], features_length: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[str] = []
    for spec, provider in zip(feature_specs, handler.feature_providers):
        group = str(spec["label"])
        mode = "validation_ambient" if group == "ambient" else "validation"
        for feature in provider.get_feature_generator(mode, features_length, "truncate_start"):
            data.append(feature)
            labels.append(float(provider.label))
            groups.append(group)
    return np.asarray(data), np.asarray(labels), groups


def sample_batch(
    handler: FeatureHandler,
    feature_specs: list[dict[str, object]],
    batch_size: int,
    features_length: int,
    augmentation: dict[str, int],
    sampling_counts: Counter[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    available = [
        (spec, provider)
        for spec, provider in zip(feature_specs, handler.feature_providers)
        if provider.get_mode_size("training")
    ]
    chosen = random.choices(
        available,
        weights=[float(provider.sampling_weight) for _, provider in available],
        k=batch_size,
    )
    data: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    for spec, provider in chosen:
        feature = provider.get_random_spectrogram("training", features_length, "default")
        feature = spec_augment(feature, **augmentation)
        data.append(feature)
        labels.append(float(provider.label))
        weights.append(float(provider.penalty_weight))
        sampling_counts[str(spec["label"])] += 1
    return np.asarray(data), np.asarray(labels).reshape(-1, 1), np.asarray(weights)


def build_runtime_config(raw: dict[str, object], run_dir: Path) -> tuple[dict[str, object], object]:
    architecture = raw["architecture"]
    flags = SimpleNamespace(
        pointwise_filters=architecture["pointwise_filters"],
        repeat_in_block=architecture["repeat_in_block"],
        mixconv_kernel_sizes=architecture["mixconv_kernel_sizes"],
        residual_connection=architecture["residual_connection"],
        first_conv_filters=int(architecture["first_conv_filters"]),
        first_conv_kernel_size=int(architecture["first_conv_kernel_size"]),
        stride=int(architecture["stride"]),
        pooled=int(architecture["pooled"]),
        max_pool=int(architecture["max_pool"]),
        spatial_attention=int(architecture["spatial_attention"]),
    )
    frontend = raw["frontend"]
    desired_samples = int(frontend["sample_rate_hz"] * frontend["clip_duration_ms"] / 1000)
    window_samples = int(frontend["sample_rate_hz"] * frontend["window_size_ms"] / 1000)
    step_samples = int(
        flags.stride * frontend["sample_rate_hz"] * frontend["window_step_ms"] / 1000
    )
    final_length = 1 + int((desired_samples - window_samples) / step_samples)
    spectrogram_length = final_length + mixednet.spectrogram_slices_dropped(flags)
    runtime = {
        **raw,
        "train_dir": str(run_dir),
        "summaries_dir": str(run_dir / "tensorboard"),
        "stride": flags.stride,
        "window_step_ms": int(frontend["window_step_ms"]),
        "clip_duration_ms": int(frontend["clip_duration_ms"]),
        "spectrogram_length_final_layer": final_length,
        "spectrogram_length": spectrogram_length,
        "training_input_shape": (spectrogram_length, int(frontend["feature_bins"])),
    }
    return runtime, flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "formal"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-formal-training", action="store_true")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Benchmark-only interruption probe; saves a checkpoint and exits PAUSED.",
    )
    args = parser.parse_args()
    if args.mode == "formal" and not args.allow_formal_training:
        raise SystemExit("Formal training is gated; pass --allow-formal-training only after user approval")
    if args.stop_after_step is not None and args.mode != "benchmark":
        raise SystemExit("--stop-after-step is only allowed for benchmark resume verification")

    config_path = args.config.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(raw["dataset_manifest"]).resolve()
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != str(raw["dataset_manifest_sha256"]).lower():
        raise RuntimeError("DatasetManifest hash changed; refusing to train")
    feature_summary = json.loads(
        (Path(raw["features_root"]) / "summary.json").read_text(encoding="utf-8")
    )
    if feature_summary["manifest_sha256"] != actual_manifest_hash:
        raise RuntimeError("Feature store was not built from the configured DatasetManifest")

    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        # The independent-process launcher must create the directory first so
        # Start-Process can open stdout/stderr redirects. Accept only that exact
        # inert scaffold; refuse any directory containing training state.
        launcher_scaffold = {
            "launcher.stdout.log",
            "launcher.stderr.log",
            "RESUME_COMMAND.txt",
        }
        unexpected = {item.name for item in run_dir.iterdir()} - launcher_scaffold
        if unexpected:
            raise FileExistsError(
                f"Refusing non-empty existing run directory; unexpected={sorted(unexpected)}"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=args.resume)
    status_path = run_dir / "TRAINING_STATUS.json"
    log_path = run_dir / "training.log"
    checkpoint_dir = run_dir / "checkpoints"
    best_path = run_dir / "best_weights.weights.h5"
    last_path = run_dir / "last_weights.weights.h5"
    traceback_path = run_dir / "traceback.txt"
    previous_status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if args.resume and status_path.exists()
        else {}
    )
    runtime, flags = build_runtime_config(raw, run_dir)
    effective_config_path = run_dir / "effective_config.yaml"
    effective_config_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")

    np.random.seed(int(raw["seed"]))
    random.seed(int(raw["seed"]))
    tf.random.set_seed(int(raw["seed"]))
    devices = device_info()
    batch_size = int(raw["batch_size"])
    planned_steps = int(raw["planned_steps"])
    if args.mode == "benchmark":
        total_steps = int(raw["benchmark"]["steps"])
        eval_interval = int(raw["benchmark"]["validation_interval"])
        checkpoint_interval = int(raw["benchmark"]["checkpoint_interval"])
    else:
        total_steps = planned_steps
        eval_interval = int(raw["eval_step_interval"])
        checkpoint_interval = int(raw["checkpoint_interval"])

    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    status: dict[str, object] = {
        **previous_status,
        "status": "RUNNING",
        "mode": args.mode,
        "pid": os.getpid(),
        "start_time": previous_status.get("start_time", utc_now()),
        "resume_time": utc_now() if args.resume else None,
        "last_update": utc_now(),
        "planned_steps": total_steps,
        "formal_planned_steps": planned_steps,
        "current_step": 0,
        "command": command,
        "config_path": str(config_path),
        "effective_config_path": str(effective_config_path),
        "dataset_manifest_sha256": actual_manifest_hash,
        "log_path": str(log_path),
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": previous_status.get("best_checkpoint"),
        "best_metric": previous_status.get("best_metric"),
        "device": devices,
    }
    atomic_json(status_path, status)

    def log(event: str, **values: object) -> None:
        payload = " ".join(f"{key}={value}" for key, value in values.items())
        line = f"{utc_now()} {event} {payload}".rstrip()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    started = time.perf_counter()
    step_times: list[float] = list(previous_status.get("benchmark_step_times_seconds", []))
    validation_times: list[float] = list(previous_status.get("benchmark_validation_times_seconds", []))
    checkpoint_times: list[float] = list(previous_status.get("benchmark_checkpoint_times_seconds", []))
    best_save_times: list[float] = list(previous_status.get("benchmark_best_save_times_seconds", []))
    loss_history: list[float] = list(previous_status.get("benchmark_loss_history", []))
    gradient_norms: list[float] = list(previous_status.get("benchmark_gradient_norms", []))
    elapsed_before_resume = float(previous_status.get("benchmark_elapsed_seconds", 0.0))
    sampling_counts: Counter[str] = Counter(previous_status.get("actual_sampler_counts", {}))
    max_working_set = working_set_bytes() or 0
    last_heartbeat = started
    previous_best = previous_status.get("best_metric") or {}
    best_f1 = float(previous_best.get("f1", -1.0))
    best_recall = float(previous_best.get("recall", -1.0))
    stale_evaluations = 0
    last_completed_step = 0

    try:
        handler = FeatureHandler(runtime)
        validation_data, validation_labels, validation_groups = load_validation(
            handler, raw["features"], int(runtime["spectrogram_length"])
        )
        log(
            "DATA_READY",
            training=sum(provider.get_mode_size("training") for provider in handler.feature_providers),
            validation=len(validation_labels),
            sampler=json.dumps(raw["class_sampling"], separators=(",", ":")),
        )
        model = mixednet.model(flags, runtime["training_input_shape"], batch_size)
        parameter_count = int(model.count_params())
        if parameter_count != int(raw["architecture"]["parameter_count"]):
            raise RuntimeError(
                f"Architecture parameter mismatch: expected={raw['architecture']['parameter_count']} "
                f"actual={parameter_count}"
            )
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(raw["learning_rates"][0]))
        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy())
        optimizer.build(model.trainable_variables)
        step_variable = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        checkpoint = tf.train.Checkpoint(step=step_variable, optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=5)
        if args.resume:
            if not manager.latest_checkpoint:
                raise RuntimeError("--resume requested but no checkpoint exists")
            restore_status = checkpoint.restore(manager.latest_checkpoint)
            restore_status.assert_consumed()
            last_completed_step = int(step_variable.numpy())
            optimizer_iterations = int(optimizer.iterations.numpy())
            if optimizer_iterations != last_completed_step:
                raise RuntimeError(
                    "Optimizer iteration did not strictly resume with global step: "
                    f"optimizer={optimizer_iterations} step={last_completed_step}"
                )
            status["current_step"] = last_completed_step
            status["resume_verified_at"] = utc_now()
            status["resume_checkpoint"] = manager.latest_checkpoint
            log("RESUMED", step=last_completed_step, checkpoint=manager.latest_checkpoint)

        if last_completed_step >= total_steps:
            status.update(
                {
                    "status": "COMPLETED",
                    "current_step": last_completed_step,
                    "last_update": utc_now(),
                    "resume_verified_at": utc_now(),
                    "last_checkpoint": manager.latest_checkpoint,
                }
            )
            atomic_json(status_path, status)
            log(
                "RESUME_VERIFIED_ALREADY_COMPLETE",
                step=last_completed_step,
                checkpoint=manager.latest_checkpoint,
            )
            return

        segment_end_step = min(total_steps, args.stop_after_step or total_steps)
        if segment_end_step <= last_completed_step:
            raise RuntimeError(
                f"--stop-after-step={segment_end_step} must exceed resumed step {last_completed_step}"
            )
        log("MODEL_READY", parameters=parameter_count, input_shape=runtime["training_input_shape"])
        for step in range(last_completed_step + 1, segment_end_step + 1):
            lr = float(effective_stage(raw["learning_rates"], raw["training_steps"], step))
            optimizer.learning_rate.assign(lr)
            augmentation = {
                "time_mask_max_size": int(effective_stage(raw["time_mask_max_size"], raw["training_steps"], step)),
                "time_mask_count": int(effective_stage(raw["time_mask_count"], raw["training_steps"], step)),
                "freq_mask_max_size": int(effective_stage(raw["freq_mask_max_size"], raw["training_steps"], step)),
                "freq_mask_count": int(effective_stage(raw["freq_mask_count"], raw["training_steps"], step)),
            }
            batch, labels, sample_weights = sample_batch(
                handler,
                raw["features"],
                batch_size,
                int(runtime["spectrogram_length"]),
                augmentation,
                sampling_counts,
            )
            step_started = time.perf_counter()
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
            weight_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
            with tf.GradientTape() as tape:
                predictions = model(batch_tensor, training=True)
                per_sample_loss = tf.keras.losses.binary_crossentropy(label_tensor, predictions)
                weighted_loss = tf.reduce_sum(per_sample_loss * weight_tensor) / tf.reduce_sum(
                    weight_tensor
                )
            gradients = tape.gradient(weighted_loss, model.trainable_variables)
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("Gradient sanity failed: at least one gradient is None")
            if not all(bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient in gradients):
                raise FloatingPointError("Gradient sanity failed: non-finite gradient")
            gradient_norm = float(tf.linalg.global_norm(gradients).numpy())
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise FloatingPointError(f"Gradient sanity failed: norm={gradient_norm}")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss = float(weighted_loss.numpy())
            if not math.isfinite(loss):
                raise FloatingPointError(f"Training loss is not finite at step {step}: {loss}")
            if int(optimizer.iterations.numpy()) != step:
                raise RuntimeError(
                    f"Optimizer iteration mismatch at step {step}: "
                    f"{int(optimizer.iterations.numpy())}"
                )
            step_elapsed = time.perf_counter() - step_started
            step_times.append(step_elapsed)
            loss_history.append(loss)
            gradient_norms.append(gradient_norm)
            last_completed_step = step
            step_variable.assign(step)
            now = time.perf_counter()
            max_working_set = max(max_working_set, working_set_bytes() or 0)

            is_last = step == total_steps
            is_segment_end = step == segment_end_step
            should_heartbeat = (
                step % int(raw["heartbeat_interval_steps"]) == 0
                or now - last_heartbeat >= float(raw["heartbeat_interval_seconds"])
                or is_last
                or is_segment_end
            )
            if should_heartbeat:
                elapsed = now - started
                mean_step = statistics.mean(step_times[-50:])
                eta = max(0.0, (total_steps - step) * mean_step)
                status.update(
                    {
                        "current_step": step,
                        "last_update": utc_now(),
                        "last_loss": loss,
                        "elapsed_seconds": round(elapsed, 3),
                        "eta_seconds_training_only": round(eta, 3),
                        "actual_sampler_counts": dict(sampling_counts),
                    }
                )
                atomic_json(status_path, status)
                log(
                    "HEARTBEAT",
                    step=f"{step}/{total_steps}",
                    loss=f"{loss:.6f}",
                    sec_per_step=f"{mean_step:.6f}",
                    elapsed=f"{elapsed:.1f}",
                    eta=f"{eta:.1f}",
                )
                last_heartbeat = now

            if step % eval_interval == 0 or is_last:
                validation_started = time.perf_counter()
                scores = predict_fixed_batch(model, validation_data, batch_size)
                metrics = binary_metrics(validation_labels, scores, validation_groups)
                validation_elapsed = time.perf_counter() - validation_started
                validation_times.append(validation_elapsed)
                log(
                    "VALIDATION",
                    step=step,
                    loss=f"{metrics['loss']:.6f}",
                    recall=f"{metrics['recall']:.6f}",
                    precision=f"{metrics['precision']:.6f}",
                    f1=f"{metrics['f1']:.6f}",
                    roc_auc=f"{metrics['roc_auc']:.6f}",
                    elapsed=f"{validation_elapsed:.3f}",
                )
                improved = (
                    float(metrics["f1"]) > best_f1 + float(raw["early_stopping"]["min_delta"])
                    or (
                        math.isclose(float(metrics["f1"]), best_f1, abs_tol=1e-12)
                        and float(metrics["recall"]) > best_recall
                    )
                )
                if improved:
                    best_started = time.perf_counter()
                    model.save_weights(best_path)
                    best_save_times.append(time.perf_counter() - best_started)
                    best_f1 = float(metrics["f1"])
                    best_recall = float(metrics["recall"])
                    stale_evaluations = 0
                    status["best_checkpoint"] = str(best_path)
                    status["best_metric"] = {"f1": best_f1, "recall": best_recall, "step": step}
                else:
                    stale_evaluations += 1
                status["last_validation"] = metrics
                atomic_json(status_path, status)
                early = raw["early_stopping"]
                if (
                    args.mode == "formal"
                    and bool(early["enabled"])
                    and step >= int(early["warmup_steps"])
                    and stale_evaluations >= int(early["patience_evaluations"])
                ):
                    log("EARLY_STOPPING", step=step, stale_evaluations=stale_evaluations)
                    total_steps = step
                    status["planned_steps_after_early_stop"] = step
                    is_last = True

            if step % checkpoint_interval == 0 or is_last or is_segment_end:
                checkpoint_started = time.perf_counter()
                last_checkpoint = manager.save(checkpoint_number=step)
                model.save_weights(last_path)
                checkpoint_times.append(time.perf_counter() - checkpoint_started)
                status["last_checkpoint"] = last_checkpoint
                status["last_successful_checkpoint"] = last_checkpoint
                status["last_successful_step"] = step
                atomic_json(status_path, status)
                log("CHECKPOINT", step=step, path=last_checkpoint, elapsed=f"{checkpoint_times[-1]:.3f}")
            if is_last or is_segment_end:
                break

        invocation_elapsed = time.perf_counter() - started
        accumulated_elapsed = elapsed_before_resume + invocation_elapsed
        if segment_end_step < total_steps:
            resume_command = subprocess.list2cmdline(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config_path),
                    "--mode",
                    args.mode,
                    "--run-dir",
                    str(run_dir),
                    "--resume",
                ]
            )
            status.update(
                {
                    "status": "PAUSED_FOR_RESUME_PROBE",
                    "current_step": last_completed_step,
                    "last_update": utc_now(),
                    "benchmark_elapsed_seconds": accumulated_elapsed,
                    "benchmark_step_times_seconds": step_times,
                    "benchmark_validation_times_seconds": validation_times,
                    "benchmark_checkpoint_times_seconds": checkpoint_times,
                    "benchmark_best_save_times_seconds": best_save_times,
                    "benchmark_loss_history": loss_history,
                    "benchmark_gradient_norms": gradient_norms,
                    "resume_command": resume_command,
                }
            )
            atomic_json(status_path, status)
            log(
                "PAUSED_FOR_RESUME_PROBE",
                step=last_completed_step,
                checkpoint=manager.latest_checkpoint,
            )
            print(resume_command, flush=True)
            return

        total_elapsed = accumulated_elapsed
        warmup = int(raw["benchmark"]["warmup_steps_excluded_from_timing"])
        measured = step_times[warmup:] if len(step_times) > warmup else step_times
        mean_step = statistics.mean(measured)
        validation_mean = statistics.mean(validation_times) if validation_times else 0.0
        checkpoint_mean = statistics.mean(checkpoint_times) if checkpoint_times else 0.0
        best_save_mean = statistics.mean(best_save_times) if best_save_times else 0.0
        formal_validation_count = math.ceil(planned_steps / int(raw["eval_step_interval"]))
        formal_checkpoint_count = math.ceil(planned_steps / int(raw["checkpoint_interval"]))
        estimated_seconds = (
            planned_steps * mean_step
            + formal_validation_count * validation_mean
            + formal_checkpoint_count * checkpoint_mean
            + formal_validation_count * best_save_mean
        )
        loss_head = statistics.mean(loss_history[: min(20, len(loss_history))])
        loss_tail = statistics.mean(loss_history[-min(20, len(loss_history)) :])
        output_statistics = (status.get("last_validation") or {}).get("score_statistics", {})
        benchmark_report = {
            "schema": "wakeword-studio.microwakeword-benchmark/v1",
            "mode": args.mode,
            "dataset_manifest_sha256": actual_manifest_hash,
            "steps_completed": last_completed_step,
            "warmup_steps_excluded": warmup,
            "measured_seconds_per_step_mean": mean_step,
            "measured_seconds_per_step_median": statistics.median(measured),
            "measured_seconds_per_step_p95": float(np.percentile(measured, 95)),
            "validation_seconds_mean": validation_mean,
            "checkpoint_seconds_mean": checkpoint_mean,
            "best_weight_save_seconds_mean": best_save_mean,
            "benchmark_total_elapsed_seconds": total_elapsed,
            "formal_planned_steps": planned_steps,
            "formal_validation_count": formal_validation_count,
            "formal_checkpoint_count": formal_checkpoint_count,
            "estimated_formal_runtime_seconds": estimated_seconds,
            "estimated_completion_if_started_now": (
                datetime.now(timezone.utc) + timedelta(seconds=estimated_seconds)
            ).isoformat(),
            "device": devices,
            "process_peak_observed_working_set_mib": round(max_working_set / (1024**2), 3),
            "actual_sampler_counts": dict(sampling_counts),
            "actual_sampler_distribution": {
                key: count / sum(sampling_counts.values()) for key, count in sampling_counts.items()
            },
            "last_validation": status.get("last_validation"),
            "sanity": {
                "loss_all_finite": all(math.isfinite(value) for value in loss_history),
                "loss_first_20_mean": loss_head,
                "loss_last_20_mean": loss_tail,
                "loss_decreased_first20_to_last20": loss_tail < loss_head,
                "gradient_norm_all_finite_positive": all(
                    math.isfinite(value) and value > 0.0 for value in gradient_norms
                ),
                "gradient_norm_minimum": min(gradient_norms),
                "gradient_norm_maximum": max(gradient_norms),
                "optimizer_iterations": int(optimizer.iterations.numpy()),
                "strict_resume_verified": bool(status.get("resume_verified_at")),
                "validation_outputs_all_identical": output_statistics.get("all_identical"),
                "validation_output_standard_deviation": output_statistics.get(
                    "standard_deviation"
                ),
            },
        }
        atomic_json(run_dir / "benchmark_report.json", benchmark_report)
        status.update(
            {
                "status": "COMPLETED",
                "current_step": last_completed_step,
                "last_update": utc_now(),
                "end_time": utc_now(),
                "total_elapsed_time_seconds": round(total_elapsed, 3),
                "best_checkpoint": str(best_path) if best_path.exists() else None,
                "last_checkpoint": manager.latest_checkpoint,
                "benchmark_report": str(run_dir / "benchmark_report.json"),
            }
        )
        atomic_json(status_path, status)
        log("COMPLETED", step=last_completed_step, elapsed=f"{total_elapsed:.3f}")
        print(json.dumps(benchmark_report, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        status.update(
            {
                "status": "INTERRUPTED" if interrupted else "FAILED",
                "current_step": last_completed_step,
                "last_update": utc_now(),
                "end_time": utc_now(),
                "error": repr(exc),
                "traceback_path": str(traceback_path),
                "last_successful_step": last_completed_step,
                "resume_command": subprocess.list2cmdline(
                    [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--mode", args.mode, "--run-dir", str(run_dir), "--resume"]
                    + (["--allow-formal-training"] if args.mode == "formal" else [])
                ),
            }
        )
        atomic_json(status_path, status)
        log(status["status"], step=last_completed_step, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
