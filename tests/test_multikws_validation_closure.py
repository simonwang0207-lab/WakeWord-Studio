from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLASS_NAMES = [
    "background", "qingxiaojia", "doudou", "diandian", "xiaorui", "duoduo", "jizhiwa"
]


def _metrics(macro_recall: float, threshold: float, margin: float) -> dict[str, object]:
    matrix = np.eye(7, dtype=np.int64) * 10
    matrix[0, 1] = 2
    matrix[2, 0] = 3
    per_class = {
        name: {"tp": 10, "fp": 1, "fn": 2, "recall": macro_recall,
               "precision": 0.8, "f1": 0.75}
        for name in CLASS_NAMES
    }
    sources = {
        "kokoro": {name: 0.9 for name in CLASS_NAMES[1:]},
        "voxcpm15": {name: 0.7 for name in CLASS_NAMES[1:]},
        "procedural_ambient": {name: None for name in CLASS_NAMES[1:]},
    }
    return {
        "threshold": threshold, "margin_threshold": margin,
        "confusion_matrix": matrix.tolist(), "normalized_confusion_matrix": matrix.tolist(),
        "per_class": per_class, "macro_recall": macro_recall, "macro_precision": 0.8,
        "macro_f1": 0.75, "micro_accuracy": 0.77, "worst_keyword_recall": 0.6,
        "per_source_per_keyword_recall": sources,
        "background_false_accept_rate": 0.1, "background_rejection_rate": 0.9,
        "top_confusion_pairs": [{"true": "doudou", "predicted": "background", "count": 3}],
        "top_k_scores": [], "sample_count": 1500, "test_loaded": False,
    }


def _write_run(root: Path, model: str, float_recall: float, int8_recall: float) -> Path:
    run = root / model
    export = run / "export" / f"{model}.tflite"
    export.parent.mkdir(parents=True)
    content = f"frozen-{model}".encode()
    export.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    threshold, margin = (0.2, 0.1)
    float_metrics = _metrics(float_recall, threshold, margin)
    int8_metrics = _metrics(int8_recall, threshold, margin)
    report = {
        "model_name": model, "class_names": CLASS_NAMES, "TEST_READ": False,
        "PTQ_REPRESENTATIVE_SPLIT": "train", "architecture_config": {"architecture": model},
        "completed_steps": 20, "sampler": {"steps_per_epoch": 10}, "stopped_early": True,
        "parameter_count": 123, "estimated_macs": 456,
        "HARDWARE_RUNTIME_VERIFIED": False,
        "int8_export": {"sha256": sha, "bytes": len(content), "KiB": len(content) / 1024},
    }
    for filename, value in (
        ("TRAINING_REPORT.json", report),
        ("confusion_float_validation.json", float_metrics),
        ("confusion_int8_validation.json", int8_metrics),
        ("threshold_freeze.json", {"source": "validation_only", "top1_threshold": threshold,
                                   "margin_threshold": margin, "TEST_READ": False}),
    ):
        (run / filename).write_text(json.dumps(value), encoding="utf-8")
    return run


def test_validation_report_reads_both_artifacts_and_generates_tables(tmp_path: Path) -> None:
    module = _module(
        "multikws_validation_report_test", "phase9/scripts/build_multikws_validation_report.py"
    )
    bc = _write_run(tmp_path, "bcresnet", 0.731, 0.612)
    conv = _write_run(tmp_path, "convmixer", 0.842, 0.823)
    dataset_info = tmp_path / "DATASET_INFO.json"
    dataset_info.write_text(json.dumps({
        "dataset_id": "fixture", "dataset_sha256": "dataset-sha",
        "manifest_sha256": "manifest-logical", "manifest_file_sha256": "manifest-file",
        "split_counts": {"train": 90, "validation": 15, "test": 15},
        "source_counts": {"kokoro": 54, "voxcpm15": 54, "procedural_ambient": 12},
        "TEST_READ": False,
    }), encoding="utf-8")

    markdown, selection = module.build_outputs(bc, conv, dataset_info)

    assert "73.10%" in markdown
    assert "61.20%" in markdown
    assert "| 模型 | 阶段 | Macro Recall" in markdown
    assert "| 关键词 | BC Recall" in markdown
    assert set(selection["models"]) == {"bcresnet", "convmixer"}
    assert selection["TEST_READ"] is False
    assert selection["selection_source"] == "validation_only"


def test_validation_report_rejects_any_test_read_artifact(tmp_path: Path) -> None:
    module = _module(
        "multikws_validation_report_guard_test", "phase9/scripts/build_multikws_validation_report.py"
    )
    bc = _write_run(tmp_path, "bcresnet", 0.7, 0.6)
    conv = _write_run(tmp_path, "convmixer", 0.8, 0.7)
    report_path = conv / "TRAINING_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["TEST_READ"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    dataset_info = tmp_path / "DATASET_INFO.json"
    dataset_info.write_text(json.dumps({
        "dataset_id": "fixture", "dataset_sha256": "x", "manifest_sha256": "y",
        "manifest_file_sha256": "z", "split_counts": {"train": 1, "validation": 1, "test": 1},
        "source_counts": {"kokoro": 1, "voxcpm15": 1, "procedural_ambient": 1},
        "TEST_READ": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="TEST_READ=false"):
        module.build_outputs(bc, conv, dataset_info)


def test_heldout_report_uses_frozen_operating_point_without_calibration() -> None:
    module = _module(
        "multikws_heldout_report_test", "phase9/scripts/evaluate_multikws_heldout_test.py"
    )
    scores = np.asarray([
        [0.9, 0.1, 0, 0, 0, 0, 0],
        [0.1, 0.8, 0.1, 0, 0, 0, 0],
        [0.1, 0.2, 0.7, 0, 0, 0, 0],
    ], np.float32)
    report = module.build_frozen_test_report(
        model_name="bcresnet", role="COMPUTE_LIGHT_BASELINE", scores=scores,
        targets=[0, 1, 2], sources=["procedural_ambient", "kokoro", "voxcpm15"],
        class_names=CLASS_NAMES, threshold=0.4, margin_threshold=0.2,
        model_path="model.tflite", model_sha256="abc", dataset_id="fixture",
        dataset_sha256="dataset-sha",
    )
    assert report["frozen_threshold"] == report["threshold_used"] == 0.4
    assert report["frozen_margin_threshold"] == report["margin_threshold_used"] == 0.2
    assert report["THRESHOLD_CHANGED_AFTER_TEST"] is False
    assert report["sample_count"] == 3
    assert report["TEST_READ"] is True


def test_heldout_preflight_reads_metadata_only_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _module(
        "multikws_heldout_preflight_test", "phase9/scripts/evaluate_multikws_heldout_test.py"
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    model = artifact_dir / "model.tflite"
    model.write_bytes(b"frozen")
    manifest = dataset_dir / "DatasetManifest.json"
    manifest.write_text(json.dumps({"dataset_id": "fixture", "records": []}), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "selection_source": "validation_only", "TEST_READ": False,
        "98PCT": "NOT_ACHIEVED", "class_names": CLASS_NAMES,
        "dataset": {"id": "fixture", "dataset_sha256": "d",
                    "manifest_file_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "models": {"bcresnet": {
            "role": "COMPUTE_LIGHT_BASELINE", "run": str(tmp_path / "run"),
            "tflite": {"path": str(model), "sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
            "frozen_int8_threshold": 0.4, "frozen_int8_margin_threshold": 0.2,
        }},
    }), encoding="utf-8")
    output = tmp_path / "outside" / "result"
    checked = module.preflight(selection, "bcresnet", manifest, output)
    assert checked["READY"] is True
    assert checked["TEST_READ"] is False
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.preflight(selection, "bcresnet", manifest, output)
