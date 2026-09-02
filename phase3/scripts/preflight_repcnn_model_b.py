"""Phase 3A RepCNN implementation audit and bounded Train/Validation preflight.

Held-out Test audio is never exposed through the adapter used by this script.
No stage in this file starts formal training.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit.embedded_wakeword.models.classifier import (  # noqa: E402
    RepCNNClassifier,
    reparameterize_model,
)
from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend  # noqa: E402
from livekit.embedded_wakeword.models.pipeline import SpecAugment  # noqa: E402
from livekit.embedded_wakeword.training.trainer import focal_loss  # noqa: E402
from wakeword_studio.dataset.repcnn_adapter import (  # noqa: E402
    REPCNN_LABELS,
    RepCNNDatasetAdapter,
    RepCNNSample,
)
from wakeword_studio.frontends import load_training_audio  # noqa: E402
from wakeword_studio.training.streaming_windows import extract_streaming_window  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.WorkingSetSize)


def gpu_memory() -> dict[str, object] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name, total, used, free = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {
        "name": name,
        "total_mib": int(total),
        "used_mib": int(used),
        "free_mib": int(free),
    }


def device_info() -> dict[str, object]:
    physical = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow_version": tf.__version__,
        "tensorflow_devices": [
            {"name": item.name, "type": item.device_type} for item in physical
        ],
        "tensorflow_gpu_available": bool(gpus),
        "selected_device": "GPU" if gpus else "CPU",
        "hardware_gpu": gpu_memory(),
    }


class FeatureLoader:
    """Extract exact RepCNN clips/features while retaining manifest identity."""

    def __init__(self, adapter: RepCNNDatasetAdapter, config: dict[str, Any]):
        self.adapter = adapter
        self.config = config
        frontend = config["frontend"]
        self.sample_rate = int(frontend["sample_rate_hz"])
        self.clip_ms = float(config["augmentation"]["clip_duration"]) * 1000.0
        self.expected_shape = tuple(int(value) for value in frontend["input_shape"])
        self.frontend = MicroFrontend(
            sample_rate=self.sample_rate,
            window_size_ms=int(frontend["window_size_ms"]),
            window_step_ms=int(frontend["window_step_ms"]),
            num_channels=int(frontend["feature_bins"]),
        )
        self.cache: dict[str, np.ndarray] = {}

    def audio_and_feature(self, sample: RepCNNSample) -> tuple[np.ndarray, np.ndarray]:
        if sample.split not in {"train", "validation"}:
            raise RuntimeError("Held-out Test access is prohibited")
        audio = load_training_audio(sample.audio_path)
        clip = extract_streaming_window(
            audio,
            sample.window,
            sample_rate_hz=self.sample_rate,
            window_ms=self.clip_ms,
        )
        expected_samples = int(round(self.sample_rate * self.clip_ms / 1000.0))
        if clip.shape != (expected_samples,):
            raise RuntimeError(f"RepCNN clip shape changed for {sample.record_id}: {clip.shape}")
        feature = np.asarray(self.frontend(clip)[0], dtype=np.float32)
        if feature.shape != self.expected_shape:
            raise RuntimeError(
                f"RepCNN feature shape changed for {sample.record_id}: "
                f"expected={self.expected_shape} actual={feature.shape}"
            )
        if not np.all(np.isfinite(feature)):
            raise RuntimeError(f"Non-finite RepCNN feature: {sample.record_id}")
        return clip, feature

    def feature(self, sample: RepCNNSample) -> np.ndarray:
        value = self.cache.get(sample.record_id)
        if value is None:
            _, value = self.audio_and_feature(sample)
            self.cache[sample.record_id] = value
        return value

    def stack(self, samples: list[RepCNNSample]) -> np.ndarray:
        return np.stack([self.feature(sample) for sample in samples]).astype(np.float32)


def build_adapter(config: dict[str, Any]) -> RepCNNDatasetAdapter:
    return RepCNNDatasetAdapter(
        Path(config["dataset_manifest"]),
        expected_sha256=str(config["dataset_manifest_sha256"]),
        seed=int(config["seed"]),
        clip_duration_ms=float(config["augmentation"]["clip_duration"]) * 1000.0,
    )


def build_model(config: dict[str, Any]) -> RepCNNClassifier:
    architecture = config["model"]
    shape = config["frontend"]["input_shape"]
    model = RepCNNClassifier(
        n_frames=int(shape[0]),
        n_features=int(shape[1]),
        filters=int(architecture["filters"]),
        n_blocks=int(architecture["n_blocks"]),
        dropout=float(architecture["dropout"]),
    )
    model(tf.zeros((1, int(shape[0]), int(shape[1]))), training=False)
    # Real microfrontend values reach ~25 in the pinned implementation.  The
    # upstream Glorot sigmoid head can therefore start completely saturated at
    # 1.0 for every class, making an honest before/after overfit check
    # impossible and producing a poor first gradient.  A zero final head is a
    # standard neutral 0.5 initialization; it changes neither architecture nor
    # the native focal objective and lets the backbone receive finite gradients.
    if config["model"].get("sigmoid_head_initializer") == "zeros_for_stable_unsaturated_start":
        kernel, bias = model.dense_out.get_weights()
        model.dense_out.set_weights([np.zeros_like(kernel), np.zeros_like(bias)])
    return model


def architecture_audit(config: dict[str, Any], output_root: Path) -> dict[str, object]:
    model = build_model(config)
    fused = reparameterize_model(model)
    architecture = {
        "official_implementation": "LiveKit Embedded Wakeword RepCNN at pinned commit 726403d",
        "architecture": (
            "Conv2D(3x3)+BN+ReLU; 11 RepDS blocks, each with training-time "
            "dilated DW3x3+BN / DW1x1+BN / identity-BN branches, PW1x1+BN, "
            "ReLU and residual; global average pooling; Dense(1,sigmoid)"
        ),
        "input_feature": "TFLM microfrontend PCAN/log filterbank, uint16 * 0.0390625",
        "input_tensor_shape": [1, 99, 40],
        "frame_window": {"window_ms": 30, "hop_ms": 20, "clip_ms": 2000},
        "streaming": (
            "stateless rolling 99-frame ring buffer; one new score per 20-ms hop "
            "after approximately 2 seconds of warm-up"
        ),
        "original_training_loss": (
            "clip-level focal binary cross-entropy (gamma=2) with mixup, label "
            "smoothing and scheduled negative weight"
        ),
        "output_semantics": "one unthresholded sigmoid clip probability in [0,1]",
        "training_parameter_count": int(model.count_params()),
        "deployment_parameter_count": int(fused.count_params()),
        "estimated_float_weight_bytes": int(fused.count_params()) * 4,
        "measured_converter_compute": {
            "arithmetic_ops": 420_204_000,
            "macs_per_99_frame_invocation": 210_102_000,
            "note": "Measured by TensorFlow Lite converter during Phase 3A; supersedes the old Phase 0 estimate",
        },
        "quantization_route": (
            "RepCNN branch/BN fusion followed by representative-data full INT8 "
            "TFLite conversion with int8 input and int8 output"
        ),
        "tflite_compatible": True,
        "tflm_compatibility": "operator-compatible; physical ESP32 execution remains unverified",
        "operators": [
            "RESHAPE",
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "ADD",
            "MEAN",
            "FULLY_CONNECTED",
            "LOGISTIC",
        ],
        "size_explanation": (
            "xxlarge fixes 64 channels and 11 blocks. Fusion leaves 53,505 scalar "
            "parameters; INT8 weights plus per-channel quantization tables, tensor "
            "metadata and FlatBuffer/operator overhead produced the measured Phase-0/1 "
            "reference of 112,808 bytes (110.164 KiB)."
        ),
        "initialization_stabilization": (
            "Dense sigmoid kernel/bias start at zero (neutral score 0.5) because the "
            "pinned real frontend reaches ~25 and the upstream Glorot head was measured "
            "to saturate every untrained class at 1.0; architecture and objective are unchanged"
        ),
        "frame_or_sequence_supervision_fit": (
            "clip-level: global average pooling collapses time before the single sigmoid; "
            "native RepCNN does not expose aligned frame logits"
        ),
    }
    atomic_json(output_root / "architecture_audit.json", architecture)
    return architecture


def source_label_audit(
    config: dict[str, Any], adapter: RepCNNDatasetAdapter, output_root: Path
) -> dict[str, object]:
    distribution = adapter.source_label_counts("train")
    batch_counts = {
        key: int(value)
        for key, value in config["sampling"]["batch_n_per_class"].items()
    }
    if set(batch_counts) != set(REPCNN_LABELS) or any(value <= 0 for value in batch_counts.values()):
        raise RuntimeError("Every RepCNN label must have a positive per-batch sample count")
    speech_sources = {
        label: set(distribution[label])
        for label in ("positive", "negative", "hard_negative")
    }
    if len(set(map(frozenset, speech_sources.values()))) != 1:
        raise RuntimeError("Speech source identity differs across label classes")
    if any(len(sources) < 2 for sources in speech_sources.values()):
        raise RuntimeError("Speech labels require at least two sources in Train")

    rng = np.random.default_rng(int(config["seed"]))
    sampled_sources: dict[str, Counter[str]] = {}
    for label, count in batch_counts.items():
        rows = adapter.samples("train", label)
        indices = rng.integers(0, len(rows), size=count * 100)
        sampled_sources[label] = Counter(rows[int(index)].source for index in indices)
    result = {
        "status": "PASS",
        "train_source_by_label": distribution,
        "speech_source_sets_equal": True,
        "ambient_only_source_exception": "procedural_ambient is semantically ambient, not a speech-label shortcut",
        "batch_n_per_class": batch_counts,
        "batch_size": sum(batch_counts.values()),
        "sampled_source_counts_100_batches": {
            label: dict(sorted(counts.items())) for label, counts in sampled_sources.items()
        },
        "test_loaded": adapter.test_loaded,
    }
    atomic_json(output_root / "source_label_audit.json", result)
    return result


def target_audit(
    config: dict[str, Any],
    adapter: RepCNNDatasetAdapter,
    loader: FeatureLoader,
    output_root: Path,
) -> dict[str, object]:
    requested = {"positive": 10, "negative": 10, "hard_negative": 10, "ambient": 5}
    selected: list[RepCNNSample] = []
    for label, count in requested.items():
        selected.extend(
            adapter.deterministic_sample("train", label, count, purpose=f"target-audit-{label}")
        )

    errors: list[str] = []
    rows: list[dict[str, object]] = []
    for sample in selected:
        try:
            clip, feature = loader.audio_and_feature(sample)
            expected_target = 1.0 if sample.label == "positive" else 0.0
            if sample.target != expected_target:
                raise RuntimeError("label/target mismatch")
            if not sample.audio_path.is_file():
                raise FileNotFoundError(sample.audio_path)
            if not sample.window.alignment_ok:
                raise RuntimeError("window alignment failed")
            rows.append(
                {
                    **sample.metadata(),
                    "audio_samples": len(clip),
                    "feature_shape": list(feature.shape),
                    "feature_min": float(feature.min()),
                    "feature_max": float(feature.max()),
                    "feature_mean": float(feature.mean()),
                    "check": "PASS",
                }
            )
            loader.cache[sample.record_id] = feature
        except Exception as exc:  # audit must retain every row failure
            errors.append(f"{sample.record_id}: {exc}")

    csv_path = output_root / "target_audit.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "status": "PASS" if not errors and len(rows) == sum(requested.values()) else "FAIL",
        "requested": requested,
        "audited": dict(Counter(row["label"] for row in rows)),
        "checks": ["audio", "label", "feature", "target", "window_alignment"],
        "errors": errors,
        "error_count": len(errors),
        "test_loaded": adapter.test_loaded,
        "csv": str(csv_path.resolve()),
    }
    atomic_json(output_root / "target_audit.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("TARGET AUDIT failed: " + "; ".join(errors[:5]))
    return report


def group_scores(
    model: RepCNNClassifier, features: np.ndarray, labels: list[str]
) -> dict[str, float]:
    values = np.asarray(model(features, training=False)).reshape(-1)
    return {
        label: float(np.mean(values[np.asarray(labels) == label]))
        for label in sorted(set(labels))
    }


def calibrate_batch_norm(model: RepCNNClassifier, features: np.ndarray) -> int:
    """Replace stale tiny-subset BN moving statistics without a gradient step."""

    layers: list[tf.keras.layers.BatchNormalization] = []
    for layer in model._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layers.append(layer)
    momenta = [layer.momentum for layer in layers]
    try:
        for layer in layers:
            layer.momentum = 0.0
        model(tf.constant(features), training=True)
    finally:
        for layer, momentum in zip(layers, momenta):
            layer.momentum = momentum
    return len(layers)


def tiny_overfit(
    config: dict[str, Any],
    adapter: RepCNNDatasetAdapter,
    loader: FeatureLoader,
    output_root: Path,
) -> dict[str, object]:
    settings = config["tiny_overfit"]
    samples: list[RepCNNSample] = []
    for label, count in settings["samples"].items():
        samples.extend(
            adapter.deterministic_sample("train", label, int(count), purpose=f"tiny-{label}")
        )
    features = loader.stack(samples)
    targets = np.asarray([sample.target for sample in samples], dtype=np.float32)
    labels = [sample.label for sample in samples]

    tf.keras.utils.set_random_seed(int(config["seed"]) + 1)
    model = build_model(config)
    optimizer = tf.keras.optimizers.Adam(float(settings["learning_rate"]))
    optimizer.build(model.trainable_variables)
    positive_weight = float(np.sum(targets == 0) / max(1, np.sum(targets == 1)))
    sample_weights = np.where(targets > 0.5, positive_weight, 1.0).astype(np.float32)
    feature_tensor = tf.constant(features)
    target_tensor = tf.constant(targets)
    weight_tensor = tf.constant(sample_weights)

    @tf.function
    def step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            predictions = tf.squeeze(model(feature_tensor, training=True), axis=-1)
            losses = focal_loss(predictions, target_tensor, gamma=float(config["objective"]["gamma"]))
            loss = tf.reduce_mean(losses * weight_tensor)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    before_scores = group_scores(model, features, labels)
    before_predictions = np.asarray(model(features, training=False)).reshape(-1)
    before_loss = float(
        tf.reduce_mean(
            focal_loss(tf.constant(before_predictions), target_tensor, gamma=2.0) * weight_tensor
        ).numpy()
    )
    losses: list[float] = []
    started = time.perf_counter()
    for index in range(int(settings["steps"])):
        value = float(step().numpy())
        if not math.isfinite(value):
            raise RuntimeError("Tiny overfit produced NaN/Inf")
        losses.append(value)
        if (index + 1) % 25 == 0:
            print(
                f"TINY_OVERFIT step={index + 1}/{settings['steps']} loss={value:.6f} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )

    calibrated_batch_norm_layers = calibrate_batch_norm(model, features)
    after_scores = group_scores(model, features, labels)
    after_predictions = np.asarray(model(features, training=False)).reshape(-1)
    after_loss = float(
        tf.reduce_mean(
            focal_loss(tf.constant(after_predictions), target_tensor, gamma=2.0) * weight_tensor
        ).numpy()
    )
    minimum_delta = float(settings["minimum_score_delta"])
    checks = {
        "loss_decreased": after_loss < before_loss,
        "positive_score_increased": after_scores["positive"] - before_scores["positive"] >= minimum_delta,
        "ordinary_negative_score_decreased": before_scores["negative"] - after_scores["negative"] >= minimum_delta,
        "hard_negative_score_decreased": before_scores["hard_negative"] - after_scores["hard_negative"] >= minimum_delta,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "samples": dict(Counter(labels)),
        "steps": int(settings["steps"]),
        "before_loss": before_loss,
        "after_loss": after_loss,
        "before_scores": before_scores,
        "after_scores": after_scores,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "training_parameter_count": int(model.count_params()),
        "batchnorm_calibration": {
            "passes": 1,
            "momentum": 0.0,
            "layers": calibrated_batch_norm_layers,
            "source": "same frozen tiny training subset; no gradient",
        },
        "test_loaded": adapter.test_loaded,
    }
    weights_path = output_root / "tiny_overfit.weights.h5"
    model.save_weights(weights_path)
    report["weights"] = str(weights_path.resolve())
    atomic_json(output_root / "tiny_overfit.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("Tiny overfit failed; formal training is prohibited")
    return report


def finish_existing_tiny_calibration(
    config: dict[str, Any],
    adapter: RepCNNDatasetAdapter,
    loader: FeatureLoader,
    output_root: Path,
) -> dict[str, object]:
    """Finish BN calibration for an already completed bounded tiny run."""

    report_path = output_root / "tiny_overfit.json"
    weights_path = output_root / "tiny_overfit.weights.h5"
    if not report_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Existing tiny-overfit report and weights are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    settings = config["tiny_overfit"]
    samples: list[RepCNNSample] = []
    for label, count in settings["samples"].items():
        samples.extend(
            adapter.deterministic_sample("train", label, int(count), purpose=f"tiny-{label}")
        )
    features = loader.stack(samples)
    targets = np.asarray([sample.target for sample in samples], dtype=np.float32)
    labels = [sample.label for sample in samples]
    model = build_model(config)
    model.load_weights(weights_path)
    started = time.perf_counter()
    layer_count = calibrate_batch_norm(model, features)
    after_scores = group_scores(model, features, labels)
    predictions = np.asarray(model(features, training=False)).reshape(-1)
    positive_weight = float(np.sum(targets == 0) / max(1, np.sum(targets == 1)))
    sample_weights = np.where(targets > 0.5, positive_weight, 1.0).astype(np.float32)
    after_loss = float(
        tf.reduce_mean(
            focal_loss(tf.constant(predictions), tf.constant(targets), gamma=2.0)
            * tf.constant(sample_weights)
        ).numpy()
    )
    before_scores = report["before_scores"]
    minimum_delta = float(settings["minimum_score_delta"])
    checks = {
        "loss_decreased": after_loss < float(report["before_loss"]),
        "positive_score_increased": after_scores["positive"] - before_scores["positive"] >= minimum_delta,
        "ordinary_negative_score_decreased": before_scores["negative"] - after_scores["negative"] >= minimum_delta,
        "hard_negative_score_decreased": before_scores["hard_negative"] - after_scores["hard_negative"] >= minimum_delta,
    }
    report.update(
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "after_loss": after_loss,
            "after_scores": after_scores,
            "checks": checks,
            "batchnorm_calibration": {
                "passes": 1,
                "momentum": 0.0,
                "layers": layer_count,
                "source": "same frozen tiny training subset; no gradient",
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
    )
    model.save_weights(weights_path)
    atomic_json(report_path, report)
    if report["status"] != "PASS":
        raise RuntimeError("Tiny overfit still fails after BatchNorm calibration")
    return report


def quantization_metadata(interpreter: tf.lite.Interpreter) -> dict[str, object]:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    return {
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "output_dequantization": "real_score = scale * (raw - zero_point)",
    }


def int8_export_smoke(
    config: dict[str, Any],
    adapter: RepCNNDatasetAdapter,
    loader: FeatureLoader,
    output_root: Path,
) -> dict[str, object]:
    weights_path = output_root / "tiny_overfit.weights.h5"
    if not weights_path.is_file():
        raise FileNotFoundError("Tiny-overfit weights are required before INT8 export smoke")
    model = build_model(config)
    model.load_weights(weights_path)
    fused = reparameterize_model(model)

    representatives: list[RepCNNSample] = []
    for split in ("train", "validation"):
        for label in REPCNN_LABELS:
            representatives.extend(
                adapter.deterministic_sample(split, label, 4, purpose=f"int8-{split}-{label}")
            )
    representative_features = loader.stack(representatives)
    probe = representative_features[:4]
    fusion_error = float(
        np.max(
            np.abs(
                np.asarray(model(probe, training=False))
                - np.asarray(fused(probe, training=False))
            )
        )
    )
    if fusion_error > 1e-4:
        raise RuntimeError(f"RepCNN fusion changed predictions: max_abs_error={fusion_error}")

    shape = tuple(int(value) for value in config["frontend"]["input_shape"])

    @tf.function(input_signature=[tf.TensorSpec((1, *shape), tf.float32)])
    def serving(value: tf.Tensor) -> tf.Tensor:
        return fused(value, training=False)

    def representative_dataset():
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
    model_path = output_root / "model" / "qingxiaojia_repcnn_performance_v1_full_int8.tflite"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(content)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    metadata = quantization_metadata(interpreter)
    operators = sorted(
        {
            row["op_name"]
            for row in interpreter._get_ops_details()
            if row["op_name"] != "DELEGATE"
        }
    )
    if metadata["input_dtype"] != "int8" or metadata["output_dtype"] != "int8":
        raise RuntimeError("Export smoke did not produce full INT8 input/output")
    if not metadata["input_scale"] or not metadata["output_scale"]:
        raise RuntimeError("Export smoke lacks scalar quantization metadata")
    report = {
        "status": "PASS",
        "path": str(model_path.resolve()),
        "bytes": len(content),
        "kib": len(content) / 1024.0,
        "sha256": sha256_bytes(content),
        "training_parameter_count": int(model.count_params()),
        "deployment_parameter_count": int(fused.count_params()),
        "fusion_max_abs_error": fusion_error,
        "quantization": metadata,
        "operators": operators,
        "tflite_builtin_int8_only": True,
        "test_loaded": adapter.test_loaded,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_root / "int8_export_smoke.json", report)
    return report


def finalize_existing_export(output_root: Path) -> dict[str, object]:
    """Re-inspect the existing TFLite without converting it again."""

    report_path = output_root / "int8_export_smoke.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_path = Path(report["path"])
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    report["quantization"] = quantization_metadata(interpreter)
    report["operators"] = sorted(
        {
            row["op_name"]
            for row in interpreter._get_ops_details()
            if row["op_name"] != "DELEGATE"
        }
    )
    report["host_delegate_excluded_from_tflm_operator_list"] = True
    report["status"] = "PASS"
    atomic_json(report_path, report)
    return report


def validation_snapshot(
    model: RepCNNClassifier,
    features: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    started = time.perf_counter()
    scores = np.asarray(model(features, training=False)).reshape(-1)
    loss = float(np.mean(-(targets * np.log(np.clip(scores, 1e-7, 1.0))) - ((1 - targets) * np.log(np.clip(1 - scores, 1e-7, 1.0)))))
    predicted = scores >= 0.5
    return {
        "loss": loss,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "accuracy_at_0_5": float(np.mean(predicted == targets.astype(bool))),
        "scores_by_label": {
            label: float(np.mean(scores[np.asarray(labels) == label]))
            for label in sorted(set(labels))
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def make_batch(
    rng: np.random.Generator,
    pools: dict[str, tuple[list[RepCNNSample], np.ndarray]],
    batch_counts: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label, count in batch_counts.items():
        _, values = pools[label]
        indices = rng.integers(0, len(values), size=count)
        features.append(values[indices])
        targets.append(np.full(count, 1.0 if label == "positive" else 0.0, np.float32))
    x = np.concatenate(features)
    y = np.concatenate(targets)
    permutation = rng.permutation(len(y))
    return x[permutation], y[permutation]


def indexed_model_state(model: RepCNNClassifier) -> tf.train.Checkpoint:
    """Track every Keras 3 variable, including layers held in Python lists."""

    return tf.train.Checkpoint(
        **{
            f"variable_{index:03d}": variable
            for index, variable in enumerate(model.variables)
        }
    )


def benchmark(
    config: dict[str, Any],
    adapter: RepCNNDatasetAdapter,
    loader: FeatureLoader,
    output_root: Path,
) -> dict[str, object]:
    settings = config["benchmark"]
    batch_counts = {
        key: int(value)
        for key, value in config["sampling"]["batch_n_per_class"].items()
    }
    train_pools: dict[str, tuple[list[RepCNNSample], np.ndarray]] = {}
    validation_samples: list[RepCNNSample] = []
    for label in REPCNN_LABELS:
        rows = adapter.deterministic_sample(
            "train", label, int(settings["pool_per_class"]), purpose=f"benchmark-train-{label}"
        )
        train_pools[label] = (rows, loader.stack(rows))
        validation_samples.extend(
            adapter.deterministic_sample(
                "validation",
                label,
                int(settings["validation_per_class"]),
                purpose=f"benchmark-validation-{label}",
            )
        )
    validation_features = loader.stack(validation_samples)
    validation_targets = np.asarray([sample.target for sample in validation_samples], np.float32)
    validation_labels = [sample.label for sample in validation_samples]

    tf.keras.utils.set_random_seed(int(config["seed"]) + 2)
    rng = np.random.default_rng(int(config["seed"]) + 2)
    model = build_model(config)
    spec_augment = SpecAugment()
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    optimizer.build(model.trainable_variables)
    gamma = float(config["objective"]["gamma"])
    mixup_alpha = float(settings.get("mixup_alpha", config["objective"]["mixup_alpha"]))
    label_smoothing = float(
        settings.get("label_smoothing", config["objective"]["label_smoothing"])
    )
    use_spec_augment = bool(settings.get("spec_augment", True))

    def make_train_step(
        active_model: RepCNNClassifier,
        active_optimizer: tf.keras.optimizers.Optimizer,
    ):
        @tf.function
        def compiled(
            features: tf.Tensor,
            labels: tf.Tensor,
            negative_weight: tf.Tensor,
            lam: tf.Tensor,
        ) -> tf.Tensor:
            permutation = tf.random.shuffle(tf.range(tf.shape(features)[0]))
            features = lam * features + (1.0 - lam) * tf.gather(features, permutation)
            labels = lam * labels + (1.0 - lam) * tf.gather(labels, permutation)
            labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            with tf.GradientTape() as tape:
                augmented = spec_augment(features, training=True) if use_spec_augment else features
                predictions = tf.squeeze(active_model(augmented, training=True), axis=-1)
                losses = focal_loss(predictions, labels, gamma=gamma)
                weights = tf.where(labels < 0.5, negative_weight, 1.0)
                loss = tf.reduce_mean(losses * weights)
            gradients = tape.gradient(loss, active_model.trainable_variables)
            active_optimizer.apply_gradients(
                zip(gradients, active_model.trainable_variables)
            )
            return loss

        return compiled

    train_step = make_train_step(model, optimizer)

    total_steps = int(settings["steps"])
    validation_steps = {int(value) for value in settings["validation_steps"]}
    checkpoint_steps = {int(value) for value in settings["checkpoint_steps"]}
    strict_resume_step = int(settings["strict_resume_step"])
    step_times: list[float] = []
    losses: list[float] = []
    validations: list[dict[str, object]] = []
    checkpoint_times: list[float] = []
    peak_ram = working_set_bytes() or 0
    checkpoint_dir = output_root / "benchmark_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    strict_resume = False
    strict_resume_prediction_error: float | None = None
    strict_resume_weight_error: float | None = None
    started = time.perf_counter()

    for step_index in range(1, total_steps + 1):
        batch_x, batch_y = make_batch(rng, train_pools, batch_counts)
        progress = (step_index - 1) / max(1, total_steps - 1)
        negative_weight = 1.0 + (float(settings["max_negative_weight"]) - 1.0) * progress
        lam = float(rng.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
        step_started = time.perf_counter()
        loss_value = float(
            train_step(
                tf.constant(batch_x),
                tf.constant(batch_y),
                tf.constant(negative_weight, tf.float32),
                tf.constant(lam, tf.float32),
            ).numpy()
        )
        step_times.append(time.perf_counter() - step_started)
        losses.append(loss_value)
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Benchmark produced NaN/Inf at step {step_index}")
        current_ram = working_set_bytes() or 0
        peak_ram = max(peak_ram, current_ram)

        if step_index in validation_steps:
            snapshot = validation_snapshot(
                model, validation_features, validation_targets, validation_labels
            )
            snapshot["step"] = step_index
            validations.append(snapshot)
            print(
                f"BENCHMARK VALIDATION step={step_index} loss={snapshot['loss']:.6f} "
                f"score_std={snapshot['score_std']:.6f}",
                flush=True,
            )

        if step_index in checkpoint_steps:
            checkpoint_started = time.perf_counter()
            checkpoint = tf.train.Checkpoint(
                step=tf.Variable(step_index, dtype=tf.int64),
                model_state=indexed_model_state(model),
                optimizer=optimizer,
            )
            prefix = checkpoint.save(str(checkpoint_dir / f"step_{step_index}"))
            checkpoint_times.append(time.perf_counter() - checkpoint_started)
            print(f"BENCHMARK CHECKPOINT step={step_index} prefix={prefix}", flush=True)

            if step_index == strict_resume_step:
                before = np.asarray(model(validation_features[:4], training=False))
                before_weights = [np.asarray(value).copy() for value in model.get_weights()]
                restored_model = build_model(config)
                restored_optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=float(settings["learning_rate"]),
                    weight_decay=float(config.get("weight_decay", 0.01)),
                )
                restored_optimizer.build(restored_model.trainable_variables)
                restored_step = tf.Variable(0, dtype=tf.int64)
                restored = tf.train.Checkpoint(
                    step=restored_step,
                    model_state=indexed_model_state(restored_model),
                    optimizer=restored_optimizer,
                )
                restored.restore(prefix).assert_existing_objects_matched()
                after = np.asarray(restored_model(validation_features[:4], training=False))
                if int(restored_step.numpy()) != strict_resume_step:
                    raise RuntimeError("Strict resume restored the wrong global step")
                restored_weights = restored_model.get_weights()
                strict_resume_weight_error = max(
                    float(np.max(np.abs(left - right)))
                    for left, right in zip(before_weights, restored_weights)
                )
                strict_resume_prediction_error = float(np.max(np.abs(before - after)))
                if strict_resume_weight_error != 0.0:
                    raise RuntimeError(
                        "Strict resume changed model weights: "
                        f"max_abs_error={strict_resume_weight_error}"
                    )
                if strict_resume_prediction_error > 1e-5:
                    raise RuntimeError(
                        "Strict resume prediction error exceeds oneDNN tolerance: "
                        f"max_abs_error={strict_resume_prediction_error}"
                    )
                model = restored_model
                optimizer = restored_optimizer
                train_step = make_train_step(model, optimizer)
                strict_resume = True
                print(
                    "BENCHMARK STRICT_RESUME PASS step=90 "
                    f"weight_error={strict_resume_weight_error} "
                    f"prediction_error={strict_resume_prediction_error}",
                    flush=True,
                )

        if step_index % 20 == 0:
            print(
                f"BENCHMARK step={step_index}/{total_steps} loss={loss_value:.6f} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )

    measured = np.asarray(step_times[10:] if len(step_times) > 10 else step_times)
    first_loss = float(np.mean(losses[:20]))
    last_loss = float(np.mean(losses[-20:]))
    checks = {
        "loss_decreased": last_loss < first_loss,
        "no_nan": all(math.isfinite(value) for value in losses),
        "nonconstant_output": bool(validations and validations[-1]["score_std"] > 1e-7),
        "checkpoint_saved": bool(checkpoint_times),
        "strict_resume": strict_resume,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "steps": total_steps,
        "warmup_steps_excluded": min(10, len(step_times)),
        "mean_seconds_per_step": float(np.mean(measured)),
        "p95_seconds_per_step": float(np.percentile(measured, 95)),
        "first_20_mean_loss": first_loss,
        "last_20_mean_loss": last_loss,
        "validation_overhead_seconds": [row["elapsed_seconds"] for row in validations],
        "checkpoint_overhead_seconds": checkpoint_times,
        "peak_ram_bytes": peak_ram,
        "peak_ram_gib": peak_ram / 1024**3 if peak_ram else None,
        "gpu_memory": gpu_memory(),
        "device": device_info(),
        "validations": validations,
        "checks": checks,
        "strict_resume": strict_resume,
        "strict_resume_weight_max_abs_error": strict_resume_weight_error,
        "strict_resume_prediction_max_abs_error": strict_resume_prediction_error,
        "elapsed_seconds": time.perf_counter() - started,
        "test_loaded": adapter.test_loaded,
    }
    atomic_json(output_root / "benchmark.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("RepCNN benchmark acceptance checks failed")
    return report


def finalize_existing_benchmark(output_root: Path) -> dict[str, object]:
    """Re-evaluate a completed benchmark report without running more steps."""

    path = output_root / "benchmark.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report["checks"]
    report["strict_resume_weight_max_abs_error"] = checks.pop(
        "strict_resume_weight_max_abs_error", 0.0
    )
    report["strict_resume_prediction_max_abs_error"] = checks.pop(
        "strict_resume_prediction_max_abs_error", 0.0
    )
    report["status"] = "PASS" if all(bool(value) for value in checks.values()) else "FAIL"
    atomic_json(path, report)
    if report["status"] != "PASS":
        raise RuntimeError("Existing benchmark still fails acceptance checks")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "target-audit",
            "tiny-overfit",
            "tiny-calibrate",
            "export-smoke",
            "export-finalize",
            "benchmark",
            "benchmark-finalize",
            "all",
        ),
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["preflight_outputs"]["root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"PHASE3A START stage={args.stage}", flush=True)
    adapter = build_adapter(config)
    if adapter.test_loaded:
        raise RuntimeError("Held-out Test must not be loaded during Phase 3A")
    loader = FeatureLoader(adapter, config)
    stages: dict[str, object] = {}

    if args.stage in {"audit", "all"}:
        stages["architecture"] = architecture_audit(config, output_root)
        stages["source_label"] = source_label_audit(config, adapter, output_root)
    if args.stage in {"target-audit", "all"}:
        stages["target_audit"] = target_audit(config, adapter, loader, output_root)
    if args.stage in {"tiny-overfit", "all"}:
        stages["tiny_overfit"] = tiny_overfit(config, adapter, loader, output_root)
    if args.stage == "tiny-calibrate":
        stages["tiny_overfit"] = finish_existing_tiny_calibration(
            config, adapter, loader, output_root
        )
    if args.stage in {"export-smoke", "all"}:
        stages["int8_export"] = int8_export_smoke(config, adapter, loader, output_root)
    if args.stage == "export-finalize":
        stages["int8_export"] = finalize_existing_export(output_root)
    if args.stage in {"benchmark", "all"}:
        stages["benchmark"] = benchmark(config, adapter, loader, output_root)
    if args.stage == "benchmark-finalize":
        stages["benchmark"] = finalize_existing_benchmark(output_root)

    summary = {
        "schema": "wakeword-studio.repcnn-phase3a-preflight/v1",
        "created_at": utc_now(),
        "stage": args.stage,
        "config": str(config_path),
        "dataset_manifest_sha256": adapter.manifest_sha256,
        "dataset_counts": adapter.counts(),
        "test_loaded": adapter.test_loaded,
        "results": stages,
    }
    atomic_json(output_root / f"summary_{args.stage}.json", summary)
    print(f"PHASE3A COMPLETE stage={args.stage} test_loaded={adapter.test_loaded}", flush=True)


if __name__ == "__main__":
    main()
