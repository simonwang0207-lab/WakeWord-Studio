"""Phase 2I frozen export and deployment-sequence evaluation for Model A v3.

The smoke stage reads only the frozen Validation feature index. Held-out Test
audio is inaccessible until a Validation threshold freeze artifact exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.data import FeatureHandler
from microwakeword.layers import modes
from microwakeword.utils import convert_model_saved, convert_saved_model_to_tflite

from phase2.scripts.evaluate_microwakeword_v2_frozen import (
    binary_metrics,
    category_metrics,
    describe,
    error_analysis,
    read_csv,
    score_distributions,
    sha256_file,
    source_breakdown,
    threshold_sweep,
    write_csv,
)
from phase2.scripts.preflight_microwakeword_v3_sequence import (
    FrozenFeatureStore,
    build_sequence_model,
    transfer_to_base,
)
from phase2.scripts.run_microwakeword_training import build_runtime_config
from wakeword_studio.dataset.manifest import DatasetManifest
from wakeword_studio.frontends import load_inference_audio
from wakeword_studio.json_utils import json_dumps, normalize_json_value
from wakeword_studio.training.frozen_deployment import assert_frozen_deployment_contract
from wakeword_studio.training.sequence_objective import consecutive_trigger_score


EXPECTED_BEST_STEP = 2500
EXPECTED_BEST_F1 = 0.6537966537966537
EXPECTED_FINAL_STEP = 7500
EXPECTED_PARAMETERS = 19697
EXPECTED_MODEL_BYTES = 52840
V1_MANIFEST_SHA256 = "70b089652a7f8eb407c9d23ccc0efe7e33ce241fad2309f87f35702dc4752391"
SEQUENCE_FRAMES = 3
REASONABLE_FPR_LIMIT = 0.01
SPECIAL_FALSE_ACCEPT_TEXTS = ("你好，小甲", "你好，青甲")
V2_BASELINE = {
    "validation_f1": 0.6053550640279395,
    "v2_test_recall": 0.7575,
    "v2_test_fpr": 0.2025,
    "hard_negative_fpr": 0.4305555555555556,
    "ordinary_negative_fpr": 0.1375,
    "ambient_fpr": 0.0,
    "v1_external_recall": 0.615,
    "v1_external_fpr": 0.21142857142857144,
    "roc_auc": 0.838821875,
    "int8_size_bytes": 52840,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json_dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def checkpoint_metadata(prefix: Path) -> dict[str, Any]:
    variables = tf.train.list_variables(str(prefix))
    names = {name for name, _ in variables}
    step_key = "step/.ATTRIBUTES/VARIABLE_VALUE"
    iteration_key = "optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE"
    learning_rate_key = "optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE"
    if step_key not in names or iteration_key not in names or learning_rate_key not in names:
        raise RuntimeError(f"Incomplete trainer state in checkpoint: {prefix}")
    return {
        "prefix": str(prefix),
        "global_step": int(tf.train.load_variable(str(prefix), step_key)),
        "optimizer_iterations": int(tf.train.load_variable(str(prefix), iteration_key)),
        "learning_rate": float(tf.train.load_variable(str(prefix), learning_rate_key)),
        "optimizer_slot_variable_count": sum(
            name.startswith("optimizer/_variables/") for name in names
        ),
        "model_variable_count": sum(name.startswith("model/") for name in names),
    }


def dependency_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "phase2i_script": script_path,
        "phase2g_metric_helpers": SCRIPT_DIR / "evaluate_microwakeword_v2_frozen.py",
        "sequence_model_builder": SCRIPT_DIR / "preflight_microwakeword_v3_sequence.py",
        "runtime_config": SCRIPT_DIR / "run_microwakeword_training.py",
        "sequence_score": PROJECT_ROOT / "src/wakeword_studio/training/sequence_objective.py",
        "json_normalization": PROJECT_ROOT / "src/wakeword_studio/json_utils.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def load_context(config_path: Path, run_dir: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status = json.loads((run_dir / "TRAINING_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETED":
        raise RuntimeError(f"Training status must be COMPLETED, got {status.get('status')}")
    if int(status.get("final_step", -1)) != EXPECTED_FINAL_STEP or not status.get("early_stopped"):
        raise RuntimeError("Expected the confirmed early-stopped step-7500 training run")
    if int(status.get("best_step", -1)) != EXPECTED_BEST_STEP:
        raise RuntimeError("Frozen best step is not 2500")
    if abs(float(status.get("best_validation_sequence_f1", -1.0)) - EXPECTED_BEST_F1) > 1e-12:
        raise RuntimeError("Frozen best Validation sequence F1 changed")
    if int(raw["architecture"]["parameter_count"]) != EXPECTED_PARAMETERS:
        raise RuntimeError("Frozen Tiny parameter count changed")
    if raw.get("frozen_variables", {}).get("dataset") != "qingxiaojia_v2":
        raise RuntimeError("V3 is no longer pinned to frozen qingxiaojia_v2")
    if int(raw["sequence_objective"]["deployment_consecutive_frames"]) != SEQUENCE_FRAMES:
        raise RuntimeError("Deployment sequence logic is no longer three consecutive frames")

    config_hash = sha256_file(config_path)
    if config_hash != str(status.get("config_sha256", "")).lower():
        raise RuntimeError("V3 config differs from the completed training run")
    manifest_path = Path(raw["dataset_manifest"]).resolve()
    manifest_hash = sha256_file(manifest_path)
    expected_manifest = str(raw["dataset_manifest_sha256"]).lower()
    if manifest_hash != expected_manifest or manifest_hash != status.get("dataset_manifest_sha256"):
        raise RuntimeError("Frozen qingxiaojia_v2 DatasetManifest hash changed")
    feature_summary = json.loads(
        (Path(raw["features_root"]) / "summary.json").read_text(encoding="utf-8")
    )
    if feature_summary.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Frozen feature store DatasetManifest hash mismatch")
    if feature_summary.get("held_out_test_loaded") is not False:
        raise RuntimeError("Frozen Train/Validation feature store unexpectedly includes Test")

    best_weights = Path(status["best_checkpoint"]).resolve()
    last_weights = (run_dir / "last_weights.weights.h5").resolve()
    if not best_weights.is_file() or best_weights == last_weights:
        raise RuntimeError("Best step-2500 weights are missing or alias the final weights")
    log_lines = (run_dir / "training.log").read_text(encoding="utf-8").splitlines()
    best_validation = next((line for line in log_lines if "VALIDATION step=2500 " in line), None)
    best_checkpoint = next((line for line in log_lines if "CHECKPOINT step=2500 " in line), None)
    completed = next((line for line in log_lines if "TRAINING_COMPLETED step=7500 " in line), None)
    if not best_validation or "f1=0.653797" not in best_validation or "best=True" not in best_validation:
        raise RuntimeError("Training log does not prove best Validation step 2500")
    if not best_checkpoint or not completed:
        raise RuntimeError("Training log lacks checkpoint/completion evidence")

    best_prefix = run_dir / "checkpoints" / "ckpt-2500"
    final_prefix = run_dir / "checkpoints" / "ckpt-7500"
    best_trainer_state_present = best_prefix.with_suffix(".index").is_file()
    final_metadata = checkpoint_metadata(final_prefix)
    if final_metadata["global_step"] != EXPECTED_FINAL_STEP:
        raise RuntimeError("Final checkpoint global step is not 7500")
    if final_metadata["optimizer_iterations"] != EXPECTED_FINAL_STEP:
        raise RuntimeError("Final checkpoint optimizer iterations are not 7500")

    return {
        "raw": raw,
        "status": status,
        "config_path": config_path,
        "config_sha256": config_hash,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "feature_summary": feature_summary,
        "best_weights": best_weights,
        "best_weights_sha256": sha256_file(best_weights),
        "last_weights_sha256": sha256_file(last_weights),
        "best_log_evidence": [best_validation, best_checkpoint],
        "checkpoint_metadata": {
            "best_step": EXPECTED_BEST_STEP,
            "best_trainer_checkpoint_present": best_trainer_state_present,
            "best_trainer_state_note": (
                "ckpt-2500 retained" if best_trainer_state_present else
                "ckpt-2500 pruned by max_to_keep=5; standalone best weights retained"
            ),
            "final_checkpoint_not_for_evaluation": final_metadata,
        },
    }


def representative_runtime(v3_raw: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], Any]:
    v2_config_path = PROJECT_ROOT / "configs/models/microwakeword_tiny_v2.yaml"
    v2_raw = yaml.safe_load(v2_config_path.read_text(encoding="utf-8"))
    assert_frozen_deployment_contract(v2_raw, v3_raw)
    if v2_raw["dataset_manifest_sha256"] != v3_raw["dataset_manifest_sha256"]:
        raise RuntimeError("INT8 representative config uses a different DatasetManifest")
    if Path(v2_raw["features_root"]).resolve() != Path(v3_raw["features_root"]).resolve():
        raise RuntimeError("INT8 representative config uses a different feature store")
    runtime, flags = build_runtime_config(v2_raw, output_root)
    return runtime, flags


class FreshInterpreterSequenceScorer:
    """Full-INT8 streaming scorer with a fresh interpreter for every WAV."""

    def __init__(self, model_path: Path, stride: int, step_ms: int, consecutive_frames: int = 3):
        self.model_path = model_path
        self.stride = int(stride)
        self.step_ms = int(step_ms)
        self.consecutive_frames = int(consecutive_frames)
        probe = self._new_interpreter()
        self.input_detail = probe.get_input_details()[0]
        self.output_detail = probe.get_output_details()[0]
        self.input_scale, self.input_zero_point = self.input_detail["quantization"]
        self.output_scale, self.output_zero_point = self.output_detail["quantization"]
        if not self.input_scale or not self.output_scale:
            raise RuntimeError("Full-INT8 model lacks scalar quantization metadata")
        if np.dtype(self.input_detail["dtype"]) != np.dtype(np.int8):
            raise RuntimeError("Frozen full-INT8 input must be int8")
        if np.dtype(self.output_detail["dtype"]) != np.dtype(np.uint8):
            raise RuntimeError("Frozen full-INT8 output must be uint8")

    def _new_interpreter(self) -> tf.lite.Interpreter:
        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        return interpreter

    def metadata(self) -> dict[str, Any]:
        input_q = self.input_detail["quantization_parameters"]
        output_q = self.output_detail["quantization_parameters"]
        return {
            "input_dtype": np.dtype(self.input_detail["dtype"]).name,
            "input_shape": self.input_detail["shape"].tolist(),
            "input_scale": float(self.input_scale),
            "input_zero_point": int(self.input_zero_point),
            "input_scales": np.asarray(input_q["scales"]).astype(float).tolist(),
            "input_zero_points": np.asarray(input_q["zero_points"]).astype(int).tolist(),
            "input_quantization_formula": "q = clip(round(real / scale + zero_point))",
            "output_dtype": np.dtype(self.output_detail["dtype"]).name,
            "output_shape": self.output_detail["shape"].tolist(),
            "output_scale": float(self.output_scale),
            "output_zero_point": int(self.output_zero_point),
            "output_scales": np.asarray(output_q["scales"]).astype(float).tolist(),
            "output_zero_points": np.asarray(output_q["zero_points"]).astype(int).tolist(),
            "output_dequantization_formula": "real_score = scale * (raw - zero_point)",
            "raw_div_255_used": False,
            "frame_score_semantics": "sigmoid probability after TFLite dequantization",
            "sequence_score_formula": "max_t(min(p[t], p[t+1], p[t+2]))",
            "stream_state_reset": "fresh TFLite interpreter per WAV",
        }

    def score_audio(self, audio: np.ndarray, *, include_trace: bool = False) -> dict[str, Any]:
        interpreter = self._new_interpreter()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        features = generate_features_for_clip(audio.astype(np.float32), step_ms=self.step_ms)
        usable = len(features) - len(features) % self.stride
        raw_values: list[int] = []
        limits = np.iinfo(input_detail["dtype"])
        for offset in range(0, usable, self.stride):
            chunk = features[offset : offset + self.stride]
            quantized = np.rint(chunk / self.input_scale + self.input_zero_point)
            quantized = np.clip(quantized, limits.min, limits.max).astype(input_detail["dtype"])
            interpreter.set_tensor(input_detail["index"], quantized.reshape(input_detail["shape"]))
            interpreter.invoke()
            raw_values.append(int(interpreter.get_tensor(output_detail["index"])[0][0]))
        frame_scores = self.output_scale * (
            np.asarray(raw_values, dtype=np.float64) - self.output_zero_point
        )
        if len(frame_scores) and (np.min(frame_scores) < -1e-9 or np.max(frame_scores) > 1.0 + 1e-9):
            raise RuntimeError("Dequantized sigmoid frame score is outside probability range")
        sequence_score = consecutive_trigger_score(frame_scores, self.consecutive_frames)
        result: dict[str, Any] = {
            "score": float(sequence_score),
            "sequence_score": float(sequence_score),
            "frame_score_max": float(np.max(frame_scores)) if len(frame_scores) else 0.0,
            "frame_score_min": float(np.min(frame_scores)) if len(frame_scores) else 0.0,
            "frame_score_std": float(np.std(frame_scores)) if len(frame_scores) else 0.0,
            "final_frame_score": float(frame_scores[-1]) if len(frame_scores) else 0.0,
            "int8_raw_max": max(raw_values, default=int(self.output_zero_point)),
            "feature_frames": int(len(features)),
            "stream_packets": int(usable // self.stride),
            "discarded_feature_frames": int(len(features) - usable),
        }
        if include_trace:
            result["raw_trace"] = raw_values
            result["frame_score_trace"] = frame_scores.tolist()
        return result


def export_frozen(context: dict[str, Any], output_root: Path) -> dict[str, Any]:
    frozen_root = output_root / "frozen_checkpoint"
    frozen_root.mkdir(parents=True, exist_ok=True)
    frozen_weights = frozen_root / "best_step_2500.weights.h5"
    if not frozen_weights.exists():
        shutil.copy2(context["best_weights"], frozen_weights)
    if sha256_file(frozen_weights) != context["best_weights_sha256"]:
        raise RuntimeError("Frozen checkpoint copy differs from best step-2500 weights")

    freeze = {
        "frozen_at": utc_now(),
        "source": str(context["best_weights"]),
        "frozen_copy": str(frozen_weights),
        "sha256": context["best_weights_sha256"],
        "best_step": EXPECTED_BEST_STEP,
        "best_validation_sequence_f1": EXPECTED_BEST_F1,
        "last_step_7500_used": False,
        "checkpoint_metadata": context["checkpoint_metadata"],
        "training_log_evidence": context["best_log_evidence"],
    }
    atomic_json(frozen_root / "checkpoint_freeze.json", freeze)

    runtime, _ = representative_runtime(context["raw"], output_root)
    final_root = output_root / "final_model"
    saved_name = "stream_state_internal"
    tflite_root = final_root / "tflite_stream_state_internal_quant"
    tflite_path = tflite_root / "stream_state_internal_quant.tflite"
    if not tflite_path.exists():
        sequence, base, details = build_sequence_model(context["raw"], batch_size=1)
        sequence.load_weights(frozen_weights)
        transfer_to_base(sequence, base)
        if int(base.count_params()) != EXPECTED_PARAMETERS or int(details["parameters"]) != EXPECTED_PARAMETERS:
            raise RuntimeError("Parameter count changed before export")
        # TensorFlow's Windows checkpoint writer still uses legacy path limits.
        # Build in a short temporary directory, then retain only the final TFLite.
        with tempfile.TemporaryDirectory(prefix="p2i_", dir=PROJECT_ROOT / "phase2") as temp:
            staging_root = Path(temp)
            export_config = {**runtime, "train_dir": str(staging_root)}
            staging_tflite_root = staging_root / "tflite"
            staging_tflite = staging_tflite_root / tflite_path.name
            print("PHASE2I EXPORT saved_model from best step 2500", flush=True)
            convert_model_saved(
                base, export_config, saved_name, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
            )
            print("PHASE2I EXPORT full_int8_streaming_tflite", flush=True)
            convert_saved_model_to_tflite(
                config=export_config,
                audio_processor=FeatureHandler(runtime),
                path_to_model=str(staging_root / saved_name),
                folder=str(staging_tflite_root),
                fname=staging_tflite.name,
                quantize=True,
            )
            tflite_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging_tflite, tflite_path)

    scorer = FreshInterpreterSequenceScorer(
        tflite_path, int(runtime["stride"]), int(runtime["window_step_ms"]), SEQUENCE_FRAMES
    )
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    operators = sorted({row["op_name"] for row in interpreter._get_ops_details()})
    model_info = {
        "path": str(tflite_path),
        "bytes": tflite_path.stat().st_size,
        "kib": tflite_path.stat().st_size / 1024.0,
        "expected_nominal_bytes": EXPECTED_MODEL_BYTES,
        "matches_expected_nominal_bytes": tflite_path.stat().st_size == EXPECTED_MODEL_BYTES,
        "sha256": sha256_file(tflite_path),
        "parameter_count": EXPECTED_PARAMETERS,
        "full_int8": True,
        "operators": operators,
        "quantization": scorer.metadata(),
    }
    atomic_json(output_root / "model_info.json", model_info)
    atomic_json(
        output_root / "export_provenance.json",
        {
            "created_at": utc_now(),
            "checkpoint": freeze,
            "config_sha256": context["config_sha256"],
            "dataset_manifest_sha256": context["manifest_sha256"],
            "representative_data": "frozen qingxiaojia_v2 Train/Validation feature store; Test excluded",
            "model": model_info,
            "training_api_used": False,
        },
    )
    return {"runtime": runtime, "tflite_path": tflite_path, "model_info": model_info}


def validation_row(record: Any, dataset_root: Path, score: dict[str, Any]) -> dict[str, Any]:
    item = record.metadata
    if item.get("split") != "validation":
        raise RuntimeError("Validation scorer received a non-Validation record")
    return {
        "path": str((dataset_root / item["audio_path"]).resolve()),
        "record_id": item["record_id"],
        "split": "validation",
        "label": item["label"],
        "speaker_id": item.get("speaker_id"),
        "source": item.get("source"),
        "text": item.get("text"),
        "noise": item.get("noise_id"),
        "snr_db": item.get("snr_db"),
        "hard_negative_tier": item.get("hard_negative_tier"),
        **score,
    }


def heldout_row(record: Any, dataset_root: Path, score: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str((dataset_root / record.audio_path).resolve()),
        "record_id": record.record_id,
        "split": record.split,
        "label": record.label,
        "speaker_id": record.speaker.speaker_id,
        "source": record.speaker.source,
        "text": record.text,
        "noise": record.acoustic.noise_id,
        "snr_db": record.acoustic.snr_db,
        "duration_seconds": record.duration_seconds,
        "hard_negative_tier": record.hard_negative_tier,
        **score,
    }


def score_validation_records(
    records: list[Any], dataset_root: Path, scorer: FreshInterpreterSequenceScorer, name: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        audio = load_inference_audio(dataset_root / record.metadata["audio_path"])
        rows.append(validation_row(record, dataset_root, scorer.score_audio(audio)))
        if index % 50 == 0 or index == len(records):
            print(f"PHASE2I HEARTBEAT dataset={name} records={index}/{len(records)}", flush=True)
    return rows


def score_heldout_records(
    records: list[Any], dataset_root: Path, scorer: FreshInterpreterSequenceScorer, name: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if record.split != "test":
            raise RuntimeError("Held-out scorer received a non-Test record")
        audio = load_inference_audio(dataset_root / record.audio_path)
        rows.append(heldout_row(record, dataset_root, scorer.score_audio(audio)))
        if index % 50 == 0 or index == len(records):
            print(f"PHASE2I HEARTBEAT dataset={name} records={index}/{len(records)}", flush=True)
    return rows


def validation_records(context: dict[str, Any]) -> tuple[list[Any], Path]:
    store = FrozenFeatureStore(Path(context["raw"]["features_root"]))
    records = sum(
        (store.records[("validation", label)] for label in ("positive", "negative", "hard_negative", "ambient")),
        [],
    )
    if any(record.metadata.get("split") != "validation" for record in records):
        raise RuntimeError("Frozen Validation index contains a non-Validation record")
    return records, Path(context["raw"]["dataset_path"]).resolve()


def special_false_accepts(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for text in SPECIAL_FALSE_ACCEPT_TEXTS:
        selected = [row for row in rows if row["label"] == "hard_negative" and row["text"] == text]
        accepted = sum(float(row["score"]) >= threshold for row in selected)
        result[text] = {
            "count": len(selected),
            "false_accepts": accepted,
            "false_positive_rate": accepted / len(selected) if selected else None,
        }
    return result


def external_sources(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    groups = {
        "Kokoro zm_053": ("kokoro", "zm_053"),
        "Kokoro zm_056": ("kokoro", "zm_056"),
        "MeloTTS ZH": ("melotts", "ZH"),
    }
    return {
        name: binary_metrics(
            [row for row in rows if row["source"] == source and row["speaker_id"] == speaker],
            threshold,
        )
        for name, (source, speaker) in groups.items()
    }


def comparison(v3_validation: dict[str, Any], v2_metrics: dict[str, Any], v2_categories: dict[str, Any],
               v1_metrics: dict[str, Any], model_bytes: int) -> dict[str, Any]:
    v3 = {
        "validation_f1": v3_validation["f1"],
        "v2_test_recall": v2_metrics["recall_tpr"],
        "v2_test_fpr": v2_metrics["false_positive_rate"],
        "hard_negative_fpr": v2_categories["hard_negative"]["false_positive_rate"],
        "ordinary_negative_fpr": v2_categories["negative"]["false_positive_rate"],
        "ambient_fpr": v2_categories["ambient"]["false_positive_rate"],
        "v1_external_recall": v1_metrics["recall_tpr"],
        "v1_external_fpr": v1_metrics["false_positive_rate"],
        "roc_auc": v2_metrics["roc_auc"],
        "int8_size_bytes": model_bytes,
    }
    return {"columns": ["metric", "v2", "v3"], "v2": V2_BASELINE, "v3": v3}


def stage_export(context: dict[str, Any], output_root: Path) -> None:
    exported = export_frozen(context, output_root)
    print(json_dumps(exported["model_info"], ensure_ascii=False, indent=2), flush=True)
    print("PHASE2I EXPORT COMPLETE", flush=True)


def stage_smoke(context: dict[str, Any], output_root: Path) -> None:
    started = time.perf_counter()
    exported = export_frozen(context, output_root)
    records, dataset_root = validation_records(context)
    selected = []
    for label in ("positive", "negative", "hard_negative", "ambient"):
        selected.extend([record for record in records if record.label == label][:4])
    scorer = FreshInterpreterSequenceScorer(
        exported["tflite_path"], exported["runtime"]["stride"],
        exported["runtime"]["window_step_ms"], SEQUENCE_FRAMES
    )
    rows = score_validation_records(selected, dataset_root, scorer, "v2_validation_smoke")
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(scores)) or np.ptp(scores) <= 0:
        raise RuntimeError("Validation smoke sequence scores are non-finite or constant")
    repeat_audio = load_inference_audio(dataset_root / selected[0].metadata["audio_path"])
    first = scorer.score_audio(repeat_audio, include_trace=True)
    second = scorer.score_audio(repeat_audio, include_trace=True)
    if first["raw_trace"] != second["raw_trace"]:
        raise RuntimeError("Fresh-interpreter state isolation regression")
    report = {
        "schema": "wakeword-studio.phase2i-validation-smoke/v1",
        "completed_at": utc_now(),
        "status": "PASSED",
        "checkpoint_step": EXPECTED_BEST_STEP,
        "checkpoint_sha256": context["best_weights_sha256"],
        "model": exported["model_info"],
        "samples": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "sequence_score_statistics": describe(scores.tolist()),
        "score_semantics": scorer.metadata(),
        "fresh_interpreter_repeat_identical": True,
        "json_numpy_normalization_probe": normalize_json_value({
            "integer": np.int64(1), "float": np.float32(0.25),
            "boolean": np.bool_(True), "array": np.asarray([1, 2]),
        }),
        "test_loaded": False,
        "v2_test_audio_accessed": False,
        "v1_external_test_audio_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_csv(output_root / "validation_smoke_scores.csv", rows)
    atomic_json(output_root / "validation_smoke_report.json", report)
    print(json_dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("PHASE2I VALIDATION-ONLY SMOKE COMPLETE", flush=True)


def stage_validation_freeze(
    context: dict[str, Any], output_root: Path, script_path: Path
) -> None:
    freeze_path = output_root / "threshold_freeze.json"
    if freeze_path.exists():
        raise RuntimeError("Threshold is already frozen; refusing to select it again")
    exported = export_frozen(context, output_root)
    records, dataset_root = validation_records(context)
    scorer = FreshInterpreterSequenceScorer(
        exported["tflite_path"], exported["runtime"]["stride"],
        exported["runtime"]["window_step_ms"], SEQUENCE_FRAMES
    )
    rows = score_validation_records(records, dataset_root, scorer, "v2_validation")
    write_csv(output_root / "v2_validation_scores.csv", rows)
    sweep, operating = threshold_sweep(rows, scorer.output_detail)
    for point in operating["recall_targets"].values():
        if point.get("feasible") is False:
            continue
        reasonable = float(point["false_positive_rate"] or 0.0) <= REASONABLE_FPR_LIMIT
        target = int(round(float(point["recall_tpr"]) * 100))
        point.update({
            "reasonable_policy": f"validation_fpr <= {REASONABLE_FPR_LIMIT:.2f}",
            "reasonable": reasonable,
        })
        if not reasonable and float(point["recall_tpr"]) >= 0.98:
            point["verdict"] = "NO REASONABLE 98% OPERATING POINT"
    write_csv(output_root / "v2_validation_threshold_sweep.csv", sweep)
    freeze = {
        "schema": "wakeword-studio.phase2i-threshold-freeze/v1",
        "frozen_at": utc_now(),
        "selection_split": "v2_validation_only",
        "selection_rule": "best sequence F1; tie recall, precision, FPR, threshold",
        "sequence_score_formula": "max_t(min(p[t], p[t+1], p[t+2]))",
        "frame_score_semantics": "dequantized sigmoid probability",
        "selected_threshold": float(operating["best_f1"]["threshold"]),
        "selected_validation_metrics": operating["best_f1"],
        "operating_points": operating,
        "reasonable_fpr_limit": REASONABLE_FPR_LIMIT,
        "checkpoint_step": EXPECTED_BEST_STEP,
        "checkpoint_sha256": context["best_weights_sha256"],
        "tflite_sha256": exported["model_info"]["sha256"],
        "config_sha256": context["config_sha256"],
        "v2_manifest_sha256": context["manifest_sha256"],
        "evaluation_dependency_hashes": dependency_hashes(script_path),
        "quantization": scorer.metadata(),
        "v2_test_audio_accessed": False,
        "v1_external_test_audio_accessed": False,
    }
    atomic_json(output_root / "v2_validation_score_distributions.json", score_distributions(rows))
    atomic_json(freeze_path, freeze)
    print(json_dumps(freeze, ensure_ascii=False, indent=2), flush=True)
    print("PHASE2I VALIDATION THRESHOLD FROZEN", flush=True)


def stage_heldout_tests(context: dict[str, Any], output_root: Path, script_path: Path) -> None:
    result_path = output_root / "final_evaluation.json"
    if result_path.exists():
        raise RuntimeError("Held-out final evaluation already exists; refusing to run Test twice")
    freeze_path = output_root / "threshold_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("Validation threshold must be frozen before any Test access")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["evaluation_dependency_hashes"] != dependency_hashes(script_path):
        raise RuntimeError("Evaluation logic changed after threshold freeze")
    if freeze["checkpoint_sha256"] != context["best_weights_sha256"]:
        raise RuntimeError("Best checkpoint changed after threshold freeze")
    exported = export_frozen(context, output_root)
    if freeze["tflite_sha256"] != exported["model_info"]["sha256"]:
        raise RuntimeError("TFLite changed after threshold freeze")
    threshold = float(freeze["selected_threshold"])
    scorer = FreshInterpreterSequenceScorer(
        exported["tflite_path"], exported["runtime"]["stride"],
        exported["runtime"]["window_step_ms"], SEQUENCE_FRAMES
    )
    atomic_json(output_root / "heldout_access_log.json", {
        "started_at": utc_now(), "threshold": threshold,
        "threshold_frozen_at": freeze["frozen_at"],
        "threshold_reselection_allowed": False,
        "order": ["v2_heldout_test", "v1_external_test"],
    })

    v2_manifest = DatasetManifest.load(context["manifest_path"])
    v2_root = Path(v2_manifest.root).resolve()
    v2_records = [record for record in v2_manifest.records if record.split == "test"]
    v2_rows = score_heldout_records(v2_records, v2_root, scorer, "v2_heldout_test")
    write_csv(output_root / "v2_test_scores.csv", v2_rows)

    v1_manifest_path = PROJECT_ROOT / "datasets/projects/qingxiaojia_v1/DatasetManifest.json"
    if sha256_file(v1_manifest_path) != V1_MANIFEST_SHA256:
        raise RuntimeError("Immutable v1 external DatasetManifest changed")
    v1_manifest = DatasetManifest.load(v1_manifest_path)
    v1_root = Path(v1_manifest.root).resolve()
    v1_records = [record for record in v1_manifest.records if record.split == "test"]
    v1_rows = score_heldout_records(v1_records, v1_root, scorer, "v1_external_test")
    write_csv(output_root / "v1_external_test_scores.csv", v1_rows)

    v2_metrics = binary_metrics(v2_rows, threshold)
    v2_categories = category_metrics(v2_rows, threshold)
    v1_metrics = binary_metrics(v1_rows, threshold)
    v1_categories = category_metrics(v1_rows, threshold)
    result = {
        "schema": "wakeword-studio.phase2i-v3-frozen-final-evaluation/v1",
        "completed_at": utc_now(),
        "frozen_checkpoint": {
            "step": EXPECTED_BEST_STEP, "path": str(context["best_weights"]),
            "sha256": context["best_weights_sha256"], "ckpt_7500_used": False,
        },
        "model": exported["model_info"],
        "threshold": {
            "selected_threshold": threshold, "selected_on": "v2_validation_only",
            "selected_validation_metrics": freeze["selected_validation_metrics"],
            "operating_points": freeze["operating_points"], "changed_after_test": False,
        },
        "v2_test": {
            "metrics": v2_metrics, "categories": v2_categories,
            "sources": source_breakdown(v2_rows, threshold),
            "special_false_accepts": special_false_accepts(v2_rows, threshold),
            "score_distributions": score_distributions(v2_rows),
            "error_analysis": error_analysis(v2_rows, threshold, output_root / "errors/v2_test"),
        },
        "v1_external_test": {
            "manifest_sha256": V1_MANIFEST_SHA256,
            "metrics": v1_metrics, "categories": v1_categories,
            "required_source_breakdown": external_sources(v1_rows, threshold),
            "score_distributions": score_distributions(v1_rows),
            "error_analysis": error_analysis(v1_rows, threshold, output_root / "errors/v1_external_test"),
        },
        "comparison": comparison(
            freeze["selected_validation_metrics"], v2_metrics, v2_categories,
            v1_metrics, exported["model_info"]["bytes"]
        ),
        "evaluation_integrity": {
            "sequence_score_formula": "max_t(min(p[t], p[t+1], p[t+2]))",
            "clip_level_max_score_used": False,
            "output_formula": "scale * (raw - zero_point)",
            "raw_div_255_used": False,
            "fresh_interpreter_per_wav": True,
            "test_order": ["v2_heldout_test", "v1_external_test"],
            "threshold_changed_after_test": False,
            "evaluation_dependency_hashes": dependency_hashes(script_path),
        },
    }
    atomic_json(result_path, result)
    print(json_dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("PHASE2I V3 FROZEN EVALUATION COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("export", "smoke", "validation-freeze", "heldout-tests"),
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    script_path = Path(__file__).resolve()
    output_root = run_dir / "phase2i_v3_frozen_final"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"PHASE2I START stage={args.stage}", flush=True)
    context = load_context(config_path, run_dir)
    print(
        f"FROZEN INPUTS VERIFIED best_step={EXPECTED_BEST_STEP} "
        f"checkpoint_sha256={context['best_weights_sha256']} "
        f"manifest_sha256={context['manifest_sha256']}",
        flush=True,
    )
    if args.stage == "export":
        stage_export(context, output_root)
    elif args.stage == "smoke":
        stage_smoke(context, output_root)
    elif args.stage == "validation-freeze":
        stage_validation_freeze(context, output_root, script_path)
    else:
        stage_heldout_tests(context, output_root, script_path)


if __name__ == "__main__":
    main()
