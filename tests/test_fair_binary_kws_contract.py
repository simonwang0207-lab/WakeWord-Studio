from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from wakeword_studio.training.binary_kws_models import MODEL_NAMES
from wakeword_studio.training.fair_evaluator import evaluate_validation_scores
from wakeword_studio.training.fair_feature_store import load_frozen_feature_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "bcresnet": PROJECT_ROOT / "configs/models/bcresnet_binary_fair.yaml",
    "convmixer": PROJECT_ROOT / "configs/models/convmixer_binary_fair.yaml",
}
B2_MODEL = (
    PROJECT_ROOT
    / "runs/qingxiaojia/repcnn_performance_v2_fasttrack/formal/user_run_01"
    / "phase6_finalization_v2/qingxiaojia_repcnn_performance_v2_full_int8.tflite"
)


def load_config(name: str) -> dict[str, object]:
    return yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))


def test_training_registry_and_configs_share_the_frozen_contract() -> None:
    assert MODEL_NAMES == ("repcnn", "bcresnet", "convmixer")
    bc = load_config("bcresnet")
    conv = load_config("convmixer")
    for config, name in ((bc, "bcresnet"), (conv, "convmixer")):
        assert config["experiment"] == {
            "model_name": name,
            "task": "binary_kws",
            "keyword": "qingxiaojia",
            "phrase": "你好，青小甲",
            "require_gpu_for_formal": True,
        }
        assert config["data"]["allowed_splits"] == ["train", "validation"]
        assert config["data"]["test_loaded"] is False
        assert config["frontend"]["input_shape"] == [99, 40]
        assert config["sampling"] == bc["sampling"]
        assert config["objective"] == bc["objective"]
        assert config["formal_training"] == bc["formal_training"]
        assert config["formal_training"]["checkpoint_selection"]["maximum_overall_fpr"] == 0.10
        assert config["formal_training"]["checkpoint_selection"]["formula"] == (
            bc["formal_training"]["checkpoint_selection"]["formula"]
        )


def test_feature_store_is_exact_b2_view_and_never_opens_test(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        assert not Path(path).name.startswith("test_")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    store = load_frozen_feature_store(load_config("bcresnet"), project_root=PROJECT_ROOT)
    assert store.test_loaded is False
    assert store.input_shape == (99, 40)
    counts = store.counts()
    assert counts["train:positive"] == 2350
    assert counts["validation:positive"] == 316
    assert counts["train:negative"] == 4800
    assert counts["validation:negative"] == 640
    assert len(opened) == 8
    assert all(name.startswith(("train_", "validation_")) for name in opened)


def test_unified_evaluator_reports_counts_auc_sources_and_operating_points() -> None:
    scores = [0.95, 0.80, 0.70, 0.20, 0.10, 0.60]
    targets = [1, 1, 1, 0, 0, 0]
    labels = ["positive", "positive", "positive", "negative", "hard_negative", "ambient"]
    sources = ["kokoro", "voxcpm15", "voxcpm15", "kokoro", "voxcpm15", "room"]
    result = evaluate_validation_scores(
        scores, targets, labels, sources, maximum_overall_fpr=0.0
    )
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (3, 0, 3, 0)
    assert result["recall"] == 1.0
    assert result["fpr"] == 0.0
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0
    assert result["per_source"]["kokoro"]["recall"] == 1.0
    assert result["per_source"]["room"]["recall"] is None
    assert result["operating_points"]["recall_targets"]["recall_at_least_98pct"]["feasible"] is True
    assert result["test_loaded"] is False


def test_final_int8_mode_can_scan_exact_quantized_score_boundaries() -> None:
    scores = [0.21875, 0.21875, 0.21875, 0.0]
    targets = [1, 0, 1, 0]
    labels = ["positive", "negative", "positive", "ambient"]
    sources = ["kokoro", "kokoro", "voxcpm15", "procedural_ambient"]
    result = evaluate_validation_scores(
        scores,
        targets,
        labels,
        sources,
        maximum_overall_fpr=0.5,
        thresholds=tuple(sorted(set(scores))) + (float(np.nextafter(max(scores), np.inf)),),
    )
    assert result["threshold"] == pytest.approx(0.21875)
    assert result["fpr"] == pytest.approx(0.5)


def test_frozen_b2_artifact_is_unchanged() -> None:
    assert B2_MODEL.stat().st_size == 112816
    assert hashlib.sha256(B2_MODEL.read_bytes()).hexdigest() == (
        "6acfecf7cc8651c1fba52809eee1d89abbcffa0a48bd46662b2e58ac23ce064f"
    )


def test_comparison_builder_rejects_smoke_as_formal(tmp_path: Path) -> None:
    path = PROJECT_ROOT / "phase8/scripts/build_fair_comparison.py"
    spec = importlib.util.spec_from_file_location("fair_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    smoke = tmp_path / "SMOKE_REPORT.json"
    smoke.write_text(
        json.dumps({"formal_result": False, "test_loaded": False}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="formal result"):
        module.formal_row(smoke)
