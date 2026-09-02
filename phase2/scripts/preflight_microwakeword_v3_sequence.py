"""Phase 2H sequence-objective audit, tiny overfit, and short benchmark.

This script deliberately indexes only the existing Train and Validation feature
stores. It never opens v2 Test or v1 external Test audio/feature metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from mmap_ninja.ragged import RaggedMmap
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microwakeword import mixednet
from microwakeword.data import spec_augment
from wakeword_studio.training.sequence_objective import (
    SequenceTarget,
    build_sequence_target,
    consecutive_trigger_score,
)

from phase2.scripts.run_microwakeword_training import (
    atomic_json,
    build_runtime_config,
    device_info,
    sha256_file,
    working_set_bytes,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deterministic_key(seed: int, record_id: str, purpose: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{record_id}".encode()).hexdigest()


@dataclass(frozen=True)
class IndexedFeature:
    metadata: dict[str, Any]
    mmap_path: Path

    @property
    def record_id(self) -> str:
        return str(self.metadata["record_id"])

    @property
    def label(self) -> str:
        return str(self.metadata["label"])


class FrozenFeatureStore:
    """Read-only Train/Validation feature index with no Test path traversal."""

    MODES = {"train": "training", "validation": "validation"}

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.records: dict[tuple[str, str], list[IndexedFeature]] = {}
        self._mmaps: dict[Path, RaggedMmap] = {}
        for split, default_mode in self.MODES.items():
            for label in ("positive", "negative", "hard_negative", "ambient"):
                mode = "validation_ambient" if split == "validation" and label == "ambient" else default_mode
                directory = self.root / label / mode
                indexed: list[IndexedFeature] = []
                for metadata_path in sorted(directory.glob("shard_*_records.jsonl")):
                    mmap_path = metadata_path.with_name(
                        metadata_path.name.replace("_records.jsonl", "_mmap")
                    )
                    if not mmap_path.is_dir():
                        raise FileNotFoundError(f"Feature mmap missing: {mmap_path}")
                    with metadata_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            item = json.loads(line)
                            if item["split"] != split or item["label"] != label:
                                raise RuntimeError(f"Feature metadata split/label mismatch: {metadata_path}")
                            indexed.append(IndexedFeature(item, mmap_path))
                if not indexed:
                    raise RuntimeError(f"No frozen features found for {split}/{label}")
                self.records[(split, label)] = indexed

    def feature(self, record: IndexedFeature) -> np.ndarray:
        mmap = self._mmaps.get(record.mmap_path)
        if mmap is None:
            mmap = RaggedMmap(record.mmap_path)
            self._mmaps[record.mmap_path] = mmap
        value = np.asarray(mmap[int(record.metadata["feature_index"])], dtype=np.float32)
        return value


def sequence_target(record: IndexedFeature, config: dict[str, Any]) -> SequenceTarget:
    frontend = config["frontend"]
    objective = config["sequence_objective"]
    return build_sequence_target(
        label=record.label,
        phrase_start_ms=record.metadata.get("effective_phrase_start_ms"),
        phrase_end_ms=record.metadata.get("effective_phrase_end_ms"),
        window_start_ms=float(record.metadata["window_start_ms"]),
        original_feature_frames=int(frontend["stored_feature_frames"]),
        tail_padding_feature_frames=int(objective["tail_padding_feature_frames"]),
        frontend_window_ms=float(frontend["window_size_ms"]),
        frontend_step_ms=float(frontend["window_step_ms"]),
        model_stride=int(config["architecture"]["stride"]),
        positive_frames=int(objective["positive_target_frames"]),
    )


def make_long_input(feature: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    frontend = config["frontend"]
    objective = config["sequence_objective"]
    expected = (int(frontend["stored_feature_frames"]), int(frontend["feature_bins"]))
    if feature.shape != expected:
        raise RuntimeError(f"Frozen feature shape changed: expected={expected} actual={feature.shape}")
    return np.pad(
        feature,
        (
            (int(objective["left_context_feature_frames"]), int(objective["tail_padding_feature_frames"])),
            (0, 0),
        ),
    ).astype(np.float32, copy=False)


def build_sequence_model(
    config: dict[str, Any], *, batch_size: int
) -> tuple[tf.keras.Model, tf.keras.Model, dict[str, Any]]:
    runtime, flags = build_runtime_config(config, Path(config["run_root"]) / "_shape_only")
    base = mixednet.model(flags, runtime["training_input_shape"], batch_size)
    if base.count_params() != int(config["architecture"]["parameter_count"]):
        raise RuntimeError("Frozen Tiny parameter count changed")
    if base.layers[-2].__class__.__name__ != "Flatten" or base.layers[-1].__class__.__name__ != "Dense":
        raise RuntimeError("Unexpected upstream MixedNet output structure")

    body = tf.keras.Model(base.input, base.layers[-4].output, name="mixednet_tiny_body")
    long_length = (
        int(config["sequence_objective"]["left_context_feature_frames"])
        + int(config["frontend"]["stored_feature_frames"])
        + int(config["sequence_objective"]["tail_padding_feature_frames"])
    )
    sequence_input = tf.keras.Input(
        batch_shape=(batch_size, long_length, int(config["frontend"]["feature_bins"])),
        name="sequence_features",
    )
    long_body = tf.keras.models.clone_model(body, input_tensors=sequence_input)
    long_body.set_weights(body.get_weights())
    latent = long_body.output
    latent = tf.keras.layers.Lambda(lambda value: tf.squeeze(value, axis=2), name="remove_frequency_axis")(latent)
    latent_context_frames = int(base.layers[-4].output.shape[1])
    latent_channels = int(base.layers[-4].output.shape[-1])
    framed = tf.keras.layers.Lambda(
        lambda value: tf.signal.frame(value, latent_context_frames, 1, axis=1),
        name="rolling_deployment_context",
    )(latent)
    flattened = tf.keras.layers.Reshape(
        (-1, latent_context_frames * latent_channels), name="flatten_each_context"
    )(framed)
    decision_dense = tf.keras.layers.Dense(1, activation="sigmoid", name="sequence_decision")
    scores = decision_dense(flattened)
    scores = tf.keras.layers.Lambda(lambda value: tf.squeeze(value, axis=-1), name="decision_scores")(scores)
    sequence = tf.keras.Model(sequence_input, scores, name="mixednet_tiny_sequence_training")
    decision_dense.set_weights(base.layers[-1].get_weights())
    if sequence.count_params() != base.count_params():
        raise RuntimeError("Sequence wrapper changed trainable parameter count")
    if len(sequence.get_weights()) != len(base.get_weights()) or any(
        left.shape != right.shape for left, right in zip(sequence.get_weights(), base.get_weights())
    ):
        raise RuntimeError("Sequence weights are not export-compatible with frozen Tiny")

    details = {
        "training_input_shape": list(runtime["training_input_shape"]),
        "long_sequence_input_shape": [long_length, int(config["frontend"]["feature_bins"])],
        "latent_context_frames": latent_context_frames,
        "sequence_decision_frames": int(sequence.output.shape[1]),
        "parameters": int(sequence.count_params()),
    }
    return sequence, base, details


def transfer_to_base(sequence: tf.keras.Model, base: tf.keras.Model) -> None:
    base.set_weights(sequence.get_weights())


def choose_records(
    records: list[IndexedFeature], count: int, *, seed: int, purpose: str
) -> list[IndexedFeature]:
    ordered = sorted(records, key=lambda row: deterministic_key(seed, row.record_id, purpose))
    if len(ordered) < count:
        raise RuntimeError(f"Not enough records for {purpose}: requested={count} available={len(ordered)}")
    return ordered[:count]


def audit_targets(
    config: dict[str, Any], store: FrozenFeatureStore, run_dir: Path
) -> dict[str, Any]:
    seed = int(config["seed"])
    selected: list[IndexedFeature] = []
    for label, count in (("positive", 10), ("hard_negative", 10), ("negative", 5)):
        selected.extend(
            choose_records(store.records[("train", label)], count, seed=seed, purpose=f"target-audit-{label}")
        )
    frame_csv = run_dir / str(config["outputs"]["target_audit_csv"])
    sample_csv = run_dir / str(config["outputs"]["target_audit_summary_csv"])
    frame_fields = [
        "record_id", "audio_path", "label", "text", "phrase_start_ms", "phrase_end_ms",
        "window_start_ms", "window_end_ms", "decision_frame", "frame_timestamp_ms",
        "source_timestamp_ms", "target",
    ]
    sample_fields = [
        "record_id", "audio_path", "label", "text", "phrase_start_ms", "phrase_end_ms",
        "window_start_ms", "window_end_ms", "decision_frames", "positive_frames",
        "positive_frame_timestamps_ms", "audit_error",
    ]
    errors: list[str] = []
    if not run_dir.is_dir():
        raise FileNotFoundError(
            "Preflight run scaffold is missing; create the explicit project directory first"
        )
    unexpected = {
        item.name
        for item in run_dir.iterdir()
        if item.name not in {".preflight_scaffold", "benchmark"}
    }
    if unexpected:
        raise FileExistsError(f"Preflight run directory is not clean: {sorted(unexpected)}")
    with frame_csv.open("w", newline="", encoding="utf-8-sig") as frame_handle, sample_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as sample_handle:
        frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields)
        sample_writer = csv.DictWriter(sample_handle, fieldnames=sample_fields)
        frame_writer.writeheader()
        sample_writer.writeheader()
        for record in selected:
            target = sequence_target(record, config)
            local_errors: list[str] = []
            positives = np.flatnonzero(target.targets)
            if record.label == "positive":
                expected = int(config["sequence_objective"]["positive_target_frames"])
                if len(positives) != expected or not np.all(np.diff(positives) == 1):
                    local_errors.append("positive target is not the configured consecutive region")
                if np.any(target.targets[target.decision_timestamps_ms < target.phrase_end_relative_ms]):
                    local_errors.append("positive target leaked before phrase end")
            elif np.any(target.targets):
                local_errors.append("negative-class sequence contains a positive frame")
            errors.extend(f"{record.record_id}: {item}" for item in local_errors)
            audio_path = Path(config["dataset_path"]) / str(record.metadata["audio_path"])
            for frame_index, (timestamp, value) in enumerate(
                zip(target.decision_timestamps_ms, target.targets)
            ):
                frame_writer.writerow(
                    {
                        "record_id": record.record_id,
                        "audio_path": str(audio_path),
                        "label": record.label,
                        "text": record.metadata.get("text"),
                        "phrase_start_ms": record.metadata.get("phrase_start_ms"),
                        "phrase_end_ms": record.metadata.get("phrase_end_ms"),
                        "window_start_ms": round(float(record.metadata["window_start_ms"]), 3),
                        "window_end_ms": round(float(record.metadata["window_end_ms"]), 3),
                        "decision_frame": frame_index,
                        "frame_timestamp_ms": round(float(timestamp), 3),
                        "source_timestamp_ms": round(float(record.metadata["window_start_ms"]) + float(timestamp), 3),
                        "target": int(value),
                    }
                )
            sample_writer.writerow(
                {
                    "record_id": record.record_id,
                    "audio_path": str(audio_path),
                    "label": record.label,
                    "text": record.metadata.get("text"),
                    "phrase_start_ms": record.metadata.get("phrase_start_ms"),
                    "phrase_end_ms": record.metadata.get("phrase_end_ms"),
                    "window_start_ms": round(float(record.metadata["window_start_ms"]), 3),
                    "window_end_ms": round(float(record.metadata["window_end_ms"]), 3),
                    "decision_frames": len(target.targets),
                    "positive_frames": len(positives),
                    "positive_frame_timestamps_ms": "|".join(
                        f"{target.decision_timestamps_ms[index]:.1f}" for index in positives
                    ),
                    "audit_error": "|".join(local_errors),
                }
            )
    if errors:
        raise RuntimeError("Target audit failed: " + "; ".join(errors[:10]))
    return {
        "status": "PASSED",
        "samples": len(selected),
        "counts": dict(Counter(record.label for record in selected)),
        "decision_frame_step_ms": int(config["sequence_objective"]["decision_frame_step_ms"]),
        "positive_target_frames": int(config["sequence_objective"]["positive_target_frames"]),
        "positive_target_duration_ms": int(config["sequence_objective"]["positive_target_duration_ms"]),
        "errors": 0,
        "frame_csv": str(frame_csv),
        "sample_csv": str(sample_csv),
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }


def objective_loss(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    frame_weights: tf.Tensor,
    hard_mask: tf.Tensor,
    config: dict[str, Any],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    clipped = tf.clip_by_value(predictions, 1e-7, 1.0 - 1e-7)
    frame_bce = -(targets * tf.math.log(clipped) + (1.0 - targets) * tf.math.log(1.0 - clipped))
    frame_loss = tf.reduce_sum(frame_bce * frame_weights) / tf.reduce_sum(frame_weights)
    hard_max = tf.reduce_max(clipped, axis=1)
    hard_losses = -tf.math.log(1.0 - hard_max)
    hard_count = tf.reduce_sum(hard_mask)
    hard_loss = tf.where(
        hard_count > 0,
        tf.reduce_sum(hard_losses * hard_mask) / tf.maximum(hard_count, 1.0),
        0.0,
    )
    total = frame_loss + float(config["sequence_objective"]["hard_negative_max_penalty_weight"]) * hard_loss
    return total, frame_loss, hard_loss


def prepare_batch(
    records: list[IndexedFeature],
    store: FrozenFeatureStore,
    config: dict[str, Any],
    *,
    augment: bool,
    required_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(records) > required_batch_size:
        raise ValueError("records exceed fixed model batch size")
    padded_records = list(records)
    while len(padded_records) < required_batch_size:
        padded_records.append(records[len(padded_records) % len(records)])
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    hard_masks: list[float] = []
    objective = config["sequence_objective"]
    augmentation = config["augmentation"]
    for record in padded_records:
        feature = store.feature(record).copy()
        if augment:
            feature = spec_augment(
                feature,
                time_mask_max_size=int(augmentation["time_mask_max_size"]),
                time_mask_count=int(augmentation["time_mask_count"]),
                freq_mask_max_size=int(augmentation["freq_mask_max_size"]),
                freq_mask_count=int(augmentation["freq_mask_count"]),
            )
        target = sequence_target(record, config)
        frame_weight = np.full_like(target.targets, float(objective["other_frame_weight"]))
        frame_weight[target.targets == 1] = float(objective["positive_frame_weight"])
        if record.label == "hard_negative":
            frame_weight *= float(objective["hard_negative_frame_weight"])
        inputs.append(make_long_input(feature, config))
        targets.append(target.targets)
        weights.append(frame_weight)
        hard_masks.append(float(record.label == "hard_negative"))
    return (
        np.asarray(inputs),
        np.asarray(targets),
        np.asarray(weights),
        np.asarray(hard_masks, dtype=np.float32),
    )


def sample_training_records(
    pools: dict[str, list[IndexedFeature]], config: dict[str, Any], rng: random.Random
) -> list[IndexedFeature]:
    labels = ["positive", "negative", "hard_negative", "ambient"]
    sampling = config["class_sampling"]
    weights = [
        float(sampling["positive"]),
        float(sampling["ordinary_negative"]),
        float(sampling["hard_negative"]),
        float(sampling["ambient"]),
    ]
    chosen_labels = rng.choices(labels, weights=weights, k=int(config["batch_size"]))
    return [rng.choice(pools[label]) for label in chosen_labels]


def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: dict[str, Any],
) -> tuple[float, float, float, float]:
    inputs, targets, weights, hard_masks = batch
    with tf.GradientTape() as tape:
        predictions = model(tf.convert_to_tensor(inputs), training=True)
        total, frame_loss, hard_loss = objective_loss(
            predictions,
            tf.convert_to_tensor(targets),
            tf.convert_to_tensor(weights),
            tf.convert_to_tensor(hard_masks),
            config,
        )
    gradients = tape.gradient(total, model.trainable_variables)
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("Sequence objective produced a missing gradient")
    if not all(bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient in gradients):
        raise FloatingPointError("Sequence objective produced a non-finite gradient")
    gradient_norm = float(tf.linalg.global_norm(gradients).numpy())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise FloatingPointError(f"Invalid gradient norm: {gradient_norm}")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    values = tuple(float(value.numpy()) for value in (total, frame_loss, hard_loss))
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(f"Non-finite sequence loss: {values}")
    return (*values, gradient_norm)


def predict_records(
    model: tf.keras.Model,
    records: list[IndexedFeature],
    store: FrozenFeatureStore,
    config: dict[str, Any],
) -> np.ndarray:
    batch_size = int(config["batch_size"])
    outputs: list[np.ndarray] = []
    for offset in range(0, len(records), batch_size):
        chunk = records[offset : offset + batch_size]
        inputs, _, _, _ = prepare_batch(
            chunk, store, config, augment=False, required_batch_size=batch_size
        )
        scores = np.asarray(model(inputs, training=False))
        outputs.append(scores[: len(chunk)])
    return np.concatenate(outputs, axis=0)


def score_diagnostics(
    records: list[IndexedFeature], scores: np.ndarray, config: dict[str, Any]
) -> dict[str, float]:
    positive_end: list[float] = []
    positive_pre: list[float] = []
    hard_max: list[float] = []
    negative_max: list[float] = []
    ambient_max: list[float] = []
    for record, sequence_scores in zip(records, scores):
        target = sequence_target(record, config)
        if record.label == "positive":
            positive_end.extend(sequence_scores[target.targets == 1].tolist())
            pre = sequence_scores[target.decision_timestamps_ms < target.phrase_end_relative_ms]
            if len(pre):
                positive_pre.append(float(np.max(pre)))
        elif record.label == "hard_negative":
            hard_max.append(float(np.max(sequence_scores)))
        elif record.label == "negative":
            negative_max.append(float(np.max(sequence_scores)))
        else:
            ambient_max.append(float(np.max(sequence_scores)))
    mean = lambda values: float(np.mean(values)) if values else 0.0
    return {
        "positive_end_region_mean": mean(positive_end),
        "positive_pre_phrase_max_mean": mean(positive_pre),
        "hard_negative_sequence_max_mean": mean(hard_max),
        "ordinary_negative_sequence_max_mean": mean(negative_max),
        "ambient_sequence_max_mean": mean(ambient_max),
    }


def validation_metrics(
    model: tf.keras.Model,
    store: FrozenFeatureStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    records = sum(
        (store.records[("validation", label)] for label in ("positive", "negative", "hard_negative", "ambient")),
        [],
    )
    scores = predict_records(model, records, store, config)
    consecutive = int(config["sequence_objective"]["deployment_consecutive_frames"])
    sequence_scores = np.asarray(
        [consecutive_trigger_score(row, consecutive) for row in scores], dtype=np.float64
    )
    labels = np.asarray([record.label == "positive" for record in records], dtype=np.int32)
    predicted = sequence_scores >= float(config["validation_policy"]["threshold_during_training"])
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "count": len(records),
        "threshold": float(config["validation_policy"]["threshold_during_training"]),
        "trigger_logic": f"any {consecutive} consecutive frames >= threshold",
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": float(roc_auc_score(labels, sequence_scores)),
        "pr_auc": float(average_precision_score(labels, sequence_scores)),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "sequence_score_statistics": {
            "mean": float(np.mean(sequence_scores)),
            "std": float(np.std(sequence_scores)),
            "min": float(np.min(sequence_scores)),
            "max": float(np.max(sequence_scores)),
            "all_identical": bool(np.ptp(sequence_scores) <= 1e-12),
        },
        "frame_sequence_diagnostics": score_diagnostics(records, scores, config),
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }


def overfit_subset(store: FrozenFeatureStore, config: dict[str, Any]) -> list[IndexedFeature]:
    seed = int(config["seed"])
    tiny = config["tiny_overfit"]
    positives = choose_records(
        store.records[("train", "positive")], int(tiny["positive"]), seed=seed, purpose="overfit-positive"
    )
    hard_pool = store.records[("train", "hard_negative")]
    required_texts = ("你好，小甲", "你好，青甲")
    hard: list[IndexedFeature] = []
    for text in required_texts:
        candidates = [record for record in hard_pool if record.metadata.get("text") == text]
        hard.extend(choose_records(candidates, 1, seed=seed, purpose=f"overfit-{text}"))
    remaining = [record for record in hard_pool if record.record_id not in {item.record_id for item in hard}]
    hard.extend(
        choose_records(
            remaining,
            int(tiny["hard_negative"]) - len(hard),
            seed=seed,
            purpose="overfit-hard-fill",
        )
    )
    negative = choose_records(
        store.records[("train", "negative")], int(tiny["ordinary_negative"]), seed=seed, purpose="overfit-negative"
    )
    ambient = choose_records(
        store.records[("train", "ambient")], int(tiny["ambient"]), seed=seed, purpose="overfit-ambient"
    )
    result = positives + hard + negative + ambient
    if len(result) != int(tiny["samples"]):
        raise RuntimeError("Tiny overfit subset count mismatch")
    return result


def write_traces(
    path: Path,
    records: list[IndexedFeature],
    stages: dict[str, np.ndarray],
    config: dict[str, Any],
) -> None:
    fields = ["stage", "record_id", "audio_path", "label", "text", "timestamp_ms", "target", "score"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage, matrix in stages.items():
            for record, scores in zip(records, matrix):
                target = sequence_target(record, config)
                for timestamp, target_value, score in zip(
                    target.decision_timestamps_ms, target.targets, scores
                ):
                    writer.writerow(
                        {
                            "stage": stage,
                            "record_id": record.record_id,
                            "audio_path": str(Path(config["dataset_path"]) / str(record.metadata["audio_path"])),
                            "label": record.label,
                            "text": record.metadata.get("text"),
                            "timestamp_ms": round(float(timestamp), 3),
                            "target": int(target_value),
                            "score": float(score),
                        }
                    )


def run_tiny_overfit(
    config: dict[str, Any], store: FrozenFeatureStore, run_dir: Path
) -> dict[str, Any]:
    tf.keras.backend.clear_session()
    np.random.seed(int(config["seed"]))
    tf.random.set_seed(int(config["seed"]))
    rng = random.Random(int(config["seed"]))
    model, _, model_details = build_sequence_model(config, batch_size=int(config["batch_size"]))
    optimizer = tf.keras.optimizers.Adam(float(config["tiny_overfit"]["learning_rate"]))
    optimizer.build(model.trainable_variables)
    subset = overfit_subset(store, config)
    pools = {
        label: [record for record in subset if record.label == label]
        for label in ("positive", "negative", "hard_negative", "ambient")
    }
    before_scores = predict_records(model, subset, store, config)
    before = score_diagnostics(subset, before_scores, config)
    losses: list[float] = []
    steps = int(config["tiny_overfit"]["steps"])
    for step in range(1, steps + 1):
        records = sample_training_records(pools, config, rng)
        batch = prepare_batch(
            records, store, config, augment=False, required_batch_size=int(config["batch_size"])
        )
        total, frame, hard, gradient = train_step(model, optimizer, batch, config)
        losses.append(total)
        if step % 25 == 0 or step == steps:
            print(
                f"OVERFIT_HEARTBEAT step={step}/{steps} loss={total:.6f} "
                f"frame={frame:.6f} hard_max={hard:.6f} gradient={gradient:.6f}",
                flush=True,
            )
    after_before_calibration_scores = predict_records(model, subset, store, config)
    after_before_calibration = score_diagnostics(
        subset, after_before_calibration_scores, config
    )
    calibration_records = [subset[index % len(subset)] for index in range(int(config["batch_size"]))]
    calibration_inputs, _, _, _ = prepare_batch(
        calibration_records,
        store,
        config,
        augment=False,
        required_batch_size=int(config["batch_size"]),
    )
    calibration_passes = int(config["tiny_overfit"]["batchnorm_calibration_passes"])
    for _ in range(calibration_passes):
        model(calibration_inputs, training=True)
    after_scores = predict_records(model, subset, store, config)
    after = score_diagnostics(subset, after_scores, config)
    deltas = {key: after[key] - before[key] for key in before}
    tiny = config["tiny_overfit"]
    checks = {
        "positive_end_region_increased": deltas["positive_end_region_mean"]
        >= float(tiny["minimum_positive_end_increase"]),
        "positive_pre_phrase_decreased": deltas["positive_pre_phrase_max_mean"]
        <= -float(tiny["minimum_positive_pre_phrase_decrease"]),
        "hard_negative_max_decreased": deltas["hard_negative_sequence_max_mean"]
        <= -float(tiny["minimum_hard_negative_max_decrease"]),
        "loss_finite": all(math.isfinite(value) for value in losses),
        "loss_decreased": statistics.mean(losses[-20:]) < statistics.mean(losses[:20]),
    }
    passed = all(checks.values())
    trace_records = [next(record for record in subset if record.label == "positive")]
    for text in ("你好，小甲", "你好，青甲"):
        trace_records.append(
            next(record for record in subset if record.label == "hard_negative" and record.metadata.get("text") == text)
        )
    indices = [subset.index(record) for record in trace_records]
    trace_path = run_dir / str(config["outputs"]["score_trace_csv"])
    write_traces(
        trace_path,
        trace_records,
        {
            "before": before_scores[indices],
            "after": after_scores[indices],
        },
        config,
    )
    report = {
        "status": "PASSED" if passed else "FAILED",
        "samples": len(subset),
        "counts": dict(Counter(record.label for record in subset)),
        "steps": steps,
        "augmentation_enabled": False,
        "batchnorm_calibration_passes": calibration_passes,
        "after_before_batchnorm_calibration": after_before_calibration,
        "before": before,
        "after": after,
        "deltas": deltas,
        "checks": checks,
        "loss_first_20_mean": statistics.mean(losses[:20]),
        "loss_last_20_mean": statistics.mean(losses[-20:]),
        "model": model_details,
        "trace_records": [
            {"record_id": record.record_id, "label": record.label, "text": record.metadata.get("text")}
            for record in trace_records
        ],
        "trace_csv": str(trace_path),
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }
    atomic_json(run_dir / str(config["outputs"]["tiny_overfit_report"]), report)
    if not passed:
        raise RuntimeError(f"Tiny overfit gate failed: {checks}")
    return report


def verify_wrapper_equivalence(
    sequence: tf.keras.Model,
    base: tf.keras.Model,
    records: list[IndexedFeature],
    store: FrozenFeatureStore,
    config: dict[str, Any],
) -> float:
    batch_size = int(config["batch_size"])
    inputs, _, _, _ = prepare_batch(
        records[:2], store, config, augment=False, required_batch_size=batch_size
    )
    sequence_scores = np.asarray(sequence(inputs, training=False))
    decision_index = min(50, sequence_scores.shape[1] - 1)
    stride = int(config["architecture"]["stride"])
    context_length = int(base.input.shape[1])
    contexts = np.zeros((batch_size, context_length, int(config["frontend"]["feature_bins"])), dtype=np.float32)
    start = decision_index * stride
    contexts[:2] = inputs[:2, start : start + context_length]
    base_scores = np.asarray(base(contexts, training=False)).reshape(-1)
    error = float(np.max(np.abs(sequence_scores[:2, decision_index] - base_scores[:2])))
    if error > 1e-6:
        raise RuntimeError(f"Sequence wrapper differs from deployment context: {error}")
    return error


def run_benchmark(
    config: dict[str, Any], store: FrozenFeatureStore, run_dir: Path
) -> dict[str, Any]:
    tf.keras.backend.clear_session()
    seed = int(config["seed"])
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = random.Random(seed)
    batch_size = int(config["batch_size"])
    benchmark_dir = run_dir / "benchmark"
    if not benchmark_dir.is_dir() or not (benchmark_dir / "checkpoints").is_dir():
        raise FileNotFoundError("Benchmark/checkpoint scaffold is missing")
    status_path = benchmark_dir / "TRAINING_STATUS.json"
    log_path = benchmark_dir / "training.log"
    checkpoint_dir = benchmark_dir / "checkpoints"

    def log(event: str, **values: Any) -> None:
        line = f"{utc_now()} {event} " + " ".join(f"{key}={value}" for key, value in values.items())
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")
        print(line.rstrip(), flush=True)

    model, base, model_details = build_sequence_model(config, batch_size=batch_size)
    equivalence_error = verify_wrapper_equivalence(
        model, base, store.records[("train", "positive")][:2], store, config
    )
    optimizer = tf.keras.optimizers.Adam(float(config["learning_rates"][0]))
    optimizer.build(model.trainable_variables)
    step_variable = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    checkpoint = tf.train.Checkpoint(step=step_variable, optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=3)
    pools = {
        label: store.records[("train", label)]
        for label in ("positive", "negative", "hard_negative", "ambient")
    }
    total_steps = int(config["benchmark"]["steps"])
    resume_step = int(config["benchmark"]["strict_resume_step"])
    validation_steps = set(int(value) for value in config["benchmark"]["validation_steps"])
    checkpoint_steps = set(int(value) for value in config["benchmark"]["checkpoint_steps"])
    step_times: list[float] = []
    losses: list[float] = []
    validation_times: list[float] = []
    checkpoint_times: list[float] = []
    validations: dict[str, Any] = {}
    max_working_set = working_set_bytes() or 0
    status = {
        "status": "RUNNING",
        "current_step": 0,
        "planned_steps": total_steps,
        "formal_planned_steps": int(config["planned_steps"]),
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }
    atomic_json(status_path, status)
    log("SEQUENCE_BENCHMARK_START", device=device_info()["selected_device"], steps=total_steps)

    for step in range(1, total_steps + 1):
        records = sample_training_records(pools, config, rng)
        batch = prepare_batch(records, store, config, augment=True, required_batch_size=batch_size)
        started = time.perf_counter()
        total, frame, hard, gradient = train_step(model, optimizer, batch, config)
        step_times.append(time.perf_counter() - started)
        losses.append(total)
        step_variable.assign(step)
        max_working_set = max(max_working_set, working_set_bytes() or 0)

        if step in validation_steps:
            validation_started = time.perf_counter()
            metrics = validation_metrics(model, store, config)
            validation_times.append(time.perf_counter() - validation_started)
            validations[str(step)] = metrics
            log(
                "VALIDATION",
                step=step,
                recall=f"{metrics['recall']:.6f}",
                fpr=f"{metrics['fpr']:.6f}",
                roc_auc=f"{metrics['roc_auc']:.6f}",
                elapsed=f"{validation_times[-1]:.3f}",
            )
        if step in checkpoint_steps:
            checkpoint_started = time.perf_counter()
            saved = manager.save(checkpoint_number=step)
            checkpoint_times.append(time.perf_counter() - checkpoint_started)
            log("CHECKPOINT", step=step, path=saved, elapsed=f"{checkpoint_times[-1]:.3f}")
        if step % int(config["heartbeat_interval_steps"]) == 0 or step == total_steps:
            status.update(
                {
                    "current_step": step,
                    "last_loss": total,
                    "last_update": utc_now(),
                    "last_checkpoint": manager.latest_checkpoint,
                }
            )
            atomic_json(status_path, status)
            log(
                "HEARTBEAT",
                step=f"{step}/{total_steps}",
                loss=f"{total:.6f}",
                frame=f"{frame:.6f}",
                hard_max=f"{hard:.6f}",
                gradient=f"{gradient:.6f}",
            )

        if step == resume_step:
            saved_iterations = int(optimizer.iterations.numpy())
            del checkpoint, manager, optimizer, model, base
            tf.keras.backend.clear_session()
            model, base, resumed_details = build_sequence_model(config, batch_size=batch_size)
            if resumed_details != model_details:
                raise RuntimeError("Model details changed during strict resume")
            optimizer = tf.keras.optimizers.Adam(float(config["learning_rates"][0]))
            optimizer.build(model.trainable_variables)
            step_variable = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            checkpoint = tf.train.Checkpoint(step=step_variable, optimizer=optimizer, model=model)
            manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=3)
            if not manager.latest_checkpoint:
                raise RuntimeError("Strict resume checkpoint missing")
            restore = checkpoint.restore(manager.latest_checkpoint)
            restore.assert_consumed()
            if int(step_variable.numpy()) != resume_step:
                raise RuntimeError("Strict resume global step mismatch")
            if int(optimizer.iterations.numpy()) != saved_iterations:
                raise RuntimeError("Strict resume optimizer iteration mismatch")
            log("STRICT_RESUME_VERIFIED", step=resume_step, checkpoint=manager.latest_checkpoint)

    warmup = int(config["benchmark"]["warmup_steps_excluded_from_timing"])
    measured = step_times[warmup:]
    mean_step = statistics.mean(measured)
    p95_step = float(np.percentile(measured, 95))
    validation_mean = statistics.mean(validation_times)
    checkpoint_mean = statistics.mean(checkpoint_times)
    planned_steps = int(config["planned_steps"])
    formal_validation_count = math.ceil(planned_steps / int(config["eval_step_interval"]))
    formal_checkpoint_count = math.ceil(planned_steps / int(config["checkpoint_interval"]))
    eta = (
        planned_steps * mean_step
        + formal_validation_count * validation_mean
        + formal_checkpoint_count * checkpoint_mean
    )
    report = {
        "status": "PASSED",
        "device": device_info(),
        "steps": total_steps,
        "mean_seconds_per_step": mean_step,
        "p95_seconds_per_step": p95_step,
        "validation_seconds_mean": validation_mean,
        "checkpoint_seconds_mean": checkpoint_mean,
        "process_peak_working_set_mib": max_working_set / (1024**2),
        "loss_first_20_mean": statistics.mean(losses[:20]),
        "loss_last_20_mean": statistics.mean(losses[-20:]),
        "loss_decreased": statistics.mean(losses[-20:]) < statistics.mean(losses[:20]),
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "strict_resume_verified": True,
        "strict_resume_step": resume_step,
        "last_checkpoint": manager.latest_checkpoint,
        "validations": validations,
        "model": model_details,
        "sequence_wrapper_deployment_context_max_error": equivalence_error,
        "planned_steps": planned_steps,
        "validation_interval": int(config["eval_step_interval"]),
        "checkpoint_interval": int(config["checkpoint_interval"]),
        "estimated_formal_seconds": eta,
        "estimated_formal_hhmmss": time.strftime("%H:%M:%S", time.gmtime(eta)),
        "expected_int8_bytes": int(config["quantization"]["expected_nominal_bytes"]),
        "expected_int8_kib": float(config["quantization"]["expected_nominal_size_kib"]),
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }
    atomic_json(run_dir / str(config["outputs"]["benchmark_report"]), report)
    status.update(
        {
            "status": "COMPLETED",
            "current_step": total_steps,
            "last_update": utc_now(),
            "last_checkpoint": manager.latest_checkpoint,
            "strict_resume_verified": True,
            "report": str(run_dir / str(config["outputs"]["benchmark_report"])),
        }
    )
    atomic_json(status_path, status)
    log("SEQUENCE_BENCHMARK_COMPLETE", mean_sec_step=f"{mean_step:.6f}", eta=report["estimated_formal_hhmmss"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["dataset_manifest"]).resolve()
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != str(config["dataset_manifest_sha256"]).lower():
        raise RuntimeError("Frozen qingxiaojia_v2 DatasetManifest hash changed")
    if "test" in set(config.get("included_splits", [])):
        raise RuntimeError("Phase 2H must not include Test")
    feature_summary = json.loads(
        (Path(config["features_root"]) / "summary.json").read_text(encoding="utf-8")
    )
    if feature_summary["manifest_sha256"] != actual_manifest_hash:
        raise RuntimeError("Frozen feature store manifest hash mismatch")

    run_dir = args.run_dir.resolve()
    print("PHASE2H START", flush=True)
    print(f"CONFIG={config_path}", flush=True)
    print(f"MANIFEST_SHA256={actual_manifest_hash}", flush=True)
    print("TEST ACCESS GUARD=v2 Test and v1 external Test excluded", flush=True)
    store = FrozenFeatureStore(Path(config["features_root"]))
    print(
        "FEATURE INDEX READY "
        + " ".join(
            f"{split}_{label}={len(store.records[(split, label)])}"
            for split in ("train", "validation")
            for label in ("positive", "negative", "hard_negative", "ambient")
        ),
        flush=True,
    )
    audit = audit_targets(config, store, run_dir)
    print("TARGET AUDIT PASSED errors=0", flush=True)
    overfit = run_tiny_overfit(config, store, run_dir)
    print("TINY OVERFIT PASSED", flush=True)
    benchmark = run_benchmark(config, store, run_dir)
    report = {
        "schema": "wakeword-studio.phase2h-sequence-objective-preflight/v1",
        "status": "READY_AWAITING_START_V3_SEQUENCE_FORMAL_TRAINING",
        "completed_at": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": actual_manifest_hash,
        "only_changed_variable": "training_objective",
        "objective": config["sequence_objective"],
        "target_audit": audit,
        "tiny_overfit": overfit,
        "benchmark": benchmark,
        "frozen_variables": config["frozen_variables"],
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
        "formal_training_started": False,
    }
    atomic_json(run_dir / "PHASE2H_PREFLIGHT_REPORT.json", report)
    print("PHASE2H PREFLIGHT COMPLETE", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
