"""One-shot held-out Test evaluation with a Validation-frozen INT8 model/op point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.training.multikws_evaluator import (  # noqa: E402
    metrics_from_predictions,
    runtime_decision,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def preflight(
    selection_path: Path, model_name: str, manifest_path: Path, output_dir: Path,
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    selection = _read(selection_path)
    if selection.get("selection_source") != "validation_only":
        raise ValueError("Model selection must be Validation-only")
    if selection.get("TEST_READ") is not False:
        raise ValueError("Selection artifact does not prove TEST_READ=false")
    if selection.get("98PCT") != "NOT_ACHIEVED":
        raise ValueError("Selection artifact has an unexpected 98PCT claim")
    if model_name not in selection.get("models", {}):
        raise ValueError(f"Model is not frozen in selection artifact: {model_name}")
    frozen = selection["models"][model_name]
    tflite = _resolve_project_path(str(frozen["tflite"]["path"]))
    if not tflite.is_file() or _sha256(tflite) != str(frozen["tflite"]["sha256"]):
        raise ValueError("Frozen TFLite is missing or its SHA256 changed")
    manifest = _read(manifest_path)
    dataset = selection["dataset"]
    if manifest.get("dataset_id") != dataset.get("id"):
        raise ValueError("Manifest dataset_id differs from frozen selection")
    if _sha256(manifest_path) != dataset.get("manifest_file_sha256"):
        raise ValueError("Manifest file SHA256 differs from frozen selection")
    threshold = float(frozen["frozen_int8_threshold"])
    margin = float(frozen["frozen_int8_margin_threshold"])
    if not np.isfinite(threshold) or not np.isfinite(margin):
        raise ValueError("Frozen operating point must be finite")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Test output: {output_dir}")
    protected = [
        _resolve_project_path(str(item["run"])) for item in selection["models"].values()
    ] + [manifest_path.parent]
    if any(_is_within(output_dir, path) for path in protected):
        raise ValueError("Test output must be outside dataset and Validation run directories")
    return {
        "selection": selection, "manifest": manifest, "model": frozen,
        "model_name": model_name, "tflite": tflite, "threshold": threshold,
        "margin_threshold": margin, "manifest_path": manifest_path,
        "output_dir": output_dir, "READY": True, "TEST_READ": False,
    }


def build_frozen_test_report(
    *, model_name: str, role: str, scores: np.ndarray, targets: Sequence[int],
    sources: Sequence[str], class_names: Sequence[str], threshold: float,
    margin_threshold: float, model_path: str, model_sha256: str,
    dataset_id: str, dataset_sha256: str,
) -> dict[str, Any]:
    """Apply one immutable operating point; this function has no calibration path."""

    values = np.asarray(scores, np.float32)
    predictions = np.asarray(
        [
            runtime_decision(row, threshold=threshold, margin_threshold=margin_threshold).class_index
            for row in values
        ],
        np.int32,
    )
    metrics = metrics_from_predictions(
        values, targets, predictions, class_names, sources, threshold, margin_threshold
    )
    metrics["test_loaded"] = True
    return {
        "schema": "wakeword-studio.multikws-heldout-test/v1",
        "model_name": model_name, "role": role,
        "selection_source": "validation_only",
        "dataset_id": dataset_id, "dataset_sha256": dataset_sha256,
        "model": {"path": model_path, "sha256": model_sha256, "full_int8": True},
        "frozen_threshold": float(threshold),
        "frozen_margin_threshold": float(margin_threshold),
        "threshold_used": float(metrics["threshold"]),
        "margin_threshold_used": float(metrics["margin_threshold"]),
        "THRESHOLD_CHANGED_AFTER_TEST": False,
        "sample_count": int(metrics["sample_count"]),
        "overall_metrics": {
            key: metrics[key] for key in (
                "macro_recall", "macro_precision", "macro_f1", "micro_accuracy",
                "worst_keyword_recall", "background_false_accept_rate",
                "background_rejection_rate",
            )
        },
        "per_keyword": {name: metrics["per_class"][name] for name in class_names[1:]},
        "confusion_matrix": metrics["confusion_matrix"],
        "normalized_confusion_matrix": metrics["normalized_confusion_matrix"],
        "top_confusion_pairs": metrics["top_confusion_pairs"],
        "per_source_per_keyword_recall": metrics["per_source_per_keyword_recall"],
        "background_false_accept_rate": metrics["background_false_accept_rate"],
        "metrics": metrics,
        "TEST_READ": True,
    }


def _extract_test_features(manifest: dict[str, Any], root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend

    frontend = MicroFrontend(
        sample_rate=16_000, window_size_ms=30, window_step_ms=20, num_channels=40
    )
    features: list[np.ndarray] = []
    targets: list[int] = []
    sources: list[str] = []
    for record in manifest["records"]:
        if str(record["split"]) != "test":
            continue
        audio, sample_rate = sf.read(
            root / str(record["path"]), dtype="float32", always_2d=False
        )
        if int(sample_rate) != 16_000:
            raise ValueError(f"Expected 16 kHz Test audio, got {sample_rate}")
        signal = np.asarray(audio, np.float32)
        if signal.ndim != 1:
            raise ValueError("Expected mono Test audio")
        if len(signal) > 32_000:
            start = (len(signal) - 32_000) // 2
            signal = signal[start:start + 32_000]
        elif len(signal) < 32_000:
            missing = 32_000 - len(signal)
            signal = np.pad(signal, (missing // 2, missing - missing // 2))
        feature = np.asarray(frontend(signal)[0], np.float32)
        if feature.shape != (99, 40):
            raise RuntimeError(f"Unexpected Test feature shape: {feature.shape}")
        features.append(feature)
        targets.append(int(record["class_index"]))
        speaker = record.get("speaker", {})
        sources.append(str(speaker.get("source", record.get("speech_source", "unknown"))))
    if not features:
        raise RuntimeError("Manifest contains no held-out Test records")
    return np.asarray(features, np.float32), np.asarray(targets, np.int32), sources


def _predict_int8(model_path: Path, features: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("Frozen model is not Full INT8")
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    outputs: list[np.ndarray] = []
    for feature in features:
        value = np.clip(
            np.rint(feature[np.newaxis] / input_scale + input_zero), -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()
        quantized = interpreter.get_tensor(output_detail["index"])
        outputs.append((quantized.astype(np.float32) - output_zero) * output_scale)
    return np.concatenate(outputs, axis=0)


def _atomic_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Test report: {path}")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate(preflight_result: dict[str, Any]) -> Path:
    manifest = preflight_result["manifest"]
    manifest_path = preflight_result["manifest_path"]
    features, targets, sources = _extract_test_features(manifest, manifest_path.parent)
    scores = _predict_int8(preflight_result["tflite"], features)
    selection = preflight_result["selection"]
    frozen = preflight_result["model"]
    report = build_frozen_test_report(
        model_name=preflight_result["model_name"], role=str(frozen["role"]),
        scores=scores, targets=targets, sources=sources,
        class_names=selection["class_names"], threshold=preflight_result["threshold"],
        margin_threshold=preflight_result["margin_threshold"],
        model_path=str(frozen["tflite"]["path"]),
        model_sha256=str(frozen["tflite"]["sha256"]),
        dataset_id=str(selection["dataset"]["id"]),
        dataset_sha256=str(selection["dataset"]["dataset_sha256"]),
    )
    if report["threshold_used"] != report["frozen_threshold"] or (
        report["margin_threshold_used"] != report["frozen_margin_threshold"]
    ):
        raise RuntimeError("Operating point changed during Test evaluation")
    output_dir = preflight_result["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "TEST_REPORT.json"
    _atomic_new_json(output_path, report)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", choices=("bcresnet", "convmixer"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    checked = preflight(args.selection, args.model, args.manifest, args.output_dir)
    if args.preflight:
        print(json.dumps({
            "READY": True, "model": args.model,
            "model_sha256": checked["model"]["tflite"]["sha256"],
            "threshold": checked["threshold"],
            "margin_threshold": checked["margin_threshold"],
            "selection_source": "validation_only", "TEST_READ": False,
        }, ensure_ascii=False))
        return
    output = evaluate(checked)
    print(json.dumps({"TEST_REPORT": str(output), "TEST_READ": True,
                      "THRESHOLD_CHANGED_AFTER_TEST": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
