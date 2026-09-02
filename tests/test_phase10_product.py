from __future__ import annotations

import json
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wakeword_studio.backends import MicroWakeWordBackend, RepCNNBackend
from wakeword_studio.backends.multikws import MultiKWSBackend
from wakeword_studio.phase10 import (
    FalseWakeSession,
    MicAcceptanceSession,
    MultiKWSJob,
    RuntimeFeedbackStore,
    VocabularyManifest,
    build_keyword_expansion_plan,
    materialize_job_preflight,
    stable_keyword_id,
)
from wakeword_studio.registry import ActiveModelStore, ModelRegistry, teacher_six_model_configs
from wakeword_studio.runtime.detection_logic import DetectionConfig, DetectionLogic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = PROJECT_ROOT / "configs/multikws/teacher_six_keywords.json"


def classes(count: int = 7) -> list[dict[str, object]]:
    return [
        {"class_id": index, "keyword_id": "background" if index == 0 else f"k{index}", "display_name": "背景" if index == 0 else f"词{index}"}
        for index in range(count)
    ]


def test_multikws_backend_is_dynamic_and_reports_top2_margin() -> None:
    backend = MultiKWSBackend(classes(4), threshold=0.4, margin_threshold=0.1)
    prediction = backend.prediction_from_scores([0.05, 0.15, 0.70, 0.10])
    assert backend.num_classes == 4
    assert prediction.predicted_class_id == 2
    assert prediction.top2_class_id == 1
    assert prediction.margin == pytest.approx(0.55)
    assert prediction.accepted is True
    assert prediction.rejection_reason == "ACCEPTED"


@pytest.mark.parametrize(
    ("scores", "reason"),
    [([0.8, 0.1, 0.1], "BACKGROUND_TOP1"), ([0.1, 0.35, 0.2], "LOW_TOP1_SCORE"), ([0.1, 0.48, 0.43], "LOW_MARGIN")],
)
def test_multikws_rejection_reasons(scores: list[float], reason: str) -> None:
    backend = MultiKWSBackend(classes(3), threshold=0.4, margin_threshold=0.1)
    assert backend.prediction_from_scores(scores).rejection_reason == reason


def test_vocabulary_add_n_to_n_plus_one_preserves_old_ids() -> None:
    old = VocabularyManifest.from_legacy(VOCABULARY)
    new = old.expand("你好，小智", created_at="2026-09-01T00:00:00+00:00")
    assert stable_keyword_id("你好，小智") == "xiaozhi"
    assert [item.class_id for item in old.classes] == [item.class_id for item in new.classes[:-1]]
    assert [item.keyword_id for item in old.classes] == [item.keyword_id for item in new.classes[:-1]]
    assert new.classes[-1].class_id == 7
    assert new.classes[-1].keyword_id == "xiaozhi"


def test_add_keyword_plan_replays_old_data_and_never_overwrites_source() -> None:
    result = build_keyword_expansion_plan(VOCABULARY, "你好，小智")
    plan = result["plan"]
    assert result["ADD_KEYWORD_REQUIRES_RETRAIN"] is True
    assert plan["replay_existing_classes"] is True
    assert plan["source_dataset_immutable"] is True
    assert plan["source_dataset_id"] != plan["new_dataset_id"]
    protected = {"你好，青小甲", "你好，豆豆", "你好，点点", "你好，小瑞", "你好，多多", "你好，吉智娃"}
    assert not protected & {item["text"] for item in plan["hard_negatives"]}
    assert plan["starts_long_job"] is False


def test_job_creation_materializes_only_preflight_artifacts(tmp_path: Path) -> None:
    (tmp_path / "configs/multikws").mkdir(parents=True)
    for name in ("teacher_six_keywords.json", "teacher_six_formal_12k.json"):
        shutil.copy2(PROJECT_ROOT / "configs/multikws" / name, tmp_path / "configs/multikws" / name)
    preflight = build_keyword_expansion_plan(tmp_path / "configs/multikws/teacher_six_keywords.json", "你好，小智")
    job = MultiKWSJob.pending(tmp_path / "runs/multikws/user_expansions", "xiaozhi")
    artifacts = materialize_job_preflight(tmp_path, job, preflight)
    run_dir = tmp_path / artifacts["run_dir"]
    assert (run_dir / "EXPANSION_PLAN.json").is_file()
    assert (run_dir / "NEW_ONLY_GENERATION_CONFIG.json").is_file()
    assert (run_dir / "TRAINING_CONFIG.json").is_file()
    assert artifacts["long_job_started"] is False
    assert not (tmp_path / "datasets").exists()
    sys.path.insert(0, str(PROJECT_ROOT / "phase9/scripts"))
    import build_multikws_12k_dataset as builder

    builder.PROJECT_ROOT = tmp_path
    report = builder.planner_report(tmp_path / artifacts["new_only_config"])
    assert report["planned_effective_samples"] == 905
    assert report["planned_base_speech"] == 423


def test_replay_merge_keeps_parent_immutable_and_remaps_new_class(tmp_path: Path) -> None:
    old_root, new_root, output = tmp_path / "old", tmp_path / "new", tmp_path / "merged"
    old_root.mkdir(); new_root.mkdir()
    (old_root / "old.wav").write_bytes(b"R" * 64)
    (new_root / "new.wav").write_bytes(b"N" * 64)
    old = {"dataset_id": "teacher_six", "records": [{
        "record_id": "old-1", "path": "old.wav", "split": "test", "class_index": 1,
        "keyword_id": "qingxiaojia", "sha256": "1" * 64,
    }]}
    new = {"dataset_id": "new-only", "records": [{
        "record_id": "new-1", "path": "new.wav", "split": "train", "class_index": 1,
        "keyword_id": "xiaozhi", "sha256": "2" * 64,
    }]}
    (old_root / "DatasetManifest.json").write_text(json.dumps(old), encoding="utf-8")
    (new_root / "DatasetManifest.json").write_text(json.dumps(new), encoding="utf-8")
    vocabulary = {
        "vocabulary_id": "teacher_six_plus_xiaozhi_v1",
        "keywords": [
            {"class_index": i, "keyword_id": name}
            for i, name in enumerate(("qingxiaojia", "doudou", "diandian", "xiaorui", "duoduo", "jizhiwa"), 1)
        ] + [{"class_index": 7, "keyword_id": "xiaozhi"}],
    }
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    module_path = PROJECT_ROOT / "phase10/scripts/merge_keyword_expansion_dataset.py"
    spec = importlib.util.spec_from_file_location("phase10_merge", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module.merge(old_root / "DatasetManifest.json", new_root / "DatasetManifest.json", vocabulary_path, output)
    assert result["TEST_READ"] is False
    assert result["records"][0]["class_index"] == 1
    assert result["records"][1]["class_index"] == 7
    assert (old_root / "old.wav").read_bytes() == b"R" * 64


def test_teacher_registry_reads_frozen_metadata() -> None:
    configs = teacher_six_model_configs(PROJECT_ROOT)
    bc = configs["Teacher-Six BC-ResNet"]
    conv = configs["Teacher-Six ConvMixer"]
    assert bc["threshold"] == 0.0 and bc["margin_threshold"] == 0.2
    assert conv["threshold"] == 0.4 and conv["margin_threshold"] == 0.0
    assert bc["num_classes"] == conv["num_classes"] == 7
    assert conv["test_summary"]["macro_recall"] == pytest.approx(0.9422222222222222)


def _registry(tmp_path: Path) -> ModelRegistry:
    config = {"models": {
        "one": {"id": "one", "display_name": "One", "backend": "repcnn", "path": "one.tflite", "threshold": .5},
        "two": {"id": "two", "display_name": "Two", "backend": "repcnn", "path": "two.tflite", "threshold": .5},
    }}
    return ModelRegistry.from_config(tmp_path, config)


def test_active_model_activate_rollback_and_failed_job_isolated(tmp_path: Path) -> None:
    store = ActiveModelStore(tmp_path / "active.json", _registry(tmp_path), "one")
    store.activate("two")
    job = MultiKWSJob.pending(tmp_path / "runs", "xiaozhi")
    job.fail("synthetic failure")
    assert store.active_model_id == "two"
    assert json.loads((tmp_path / "active.json").read_text(encoding="utf-8"))["active_model_id"] == "two"
    store.rollback()
    assert store.active_model_id == "one"


def test_detection_logic_uses_multikws_reason() -> None:
    logic = DetectionLogic(DetectionConfig(wake_threshold=.4, consecutive_wake_frames=1, pre_silence_frames=0, post_silence_frames=0))
    inference = SimpleNamespace(
        predicted_keyword_id="background", top1_score=.8, top2_score=.1,
        top2_keyword_id="k1", margin=.7, background_score=.8,
        accepted=False, rejection_reason="BACKGROUND_TOP1",
    )
    decision = logic.update({"k1": .1}, True, 1.0, inference=inference)
    assert decision.wake_event is False
    assert decision.rejection_reason == "BACKGROUND_TOP1"


def test_runtime_feedback_privacy_default_and_schema(tmp_path: Path) -> None:
    store = RuntimeFeedbackStore(tmp_path / "runtime_feedback.jsonl")
    row = store.append({"audio_segment_path": "secret.wav", "top1": "k1"}, "wrong", "k2")
    assert row["audio_saved"] is False and row["audio_segment_path"] is None
    loaded = json.loads((tmp_path / "runtime_feedback.jsonl").read_text(encoding="utf-8"))
    assert loaded["schema"] == "wakeword-studio.runtime-feedback/v1"


def test_real_mic_acceptance_schema_is_not_test(tmp_path: Path) -> None:
    session = MicAcceptanceSession("m", "v", 10)
    session.record("k1", "correct")
    session.record("k1", "rejected")
    report = session.report()
    assert report["report_type"] == "REAL_MIC_ACCEPTANCE"
    assert report["is_held_out_test"] is False
    assert report["per_keyword"]["k1"]["hit_rate"] == .5


def test_false_wake_rate_is_distinct_from_background_far() -> None:
    session = FalseWakeSession(started_monotonic=100.0, false_wake_count=2)
    report = session.report(now=3700.0)
    assert report["false_wakes_per_hour"] == pytest.approx(2.0)
    assert report["distinct_from_background_far"] is True


def test_old_binary_backends_and_ui_routes_remain() -> None:
    assert MicroWakeWordBackend and RepCNNBackend
    source = (PROJECT_ROOT / "src/wakeword_studio/webapp.py").read_text(encoding="utf-8")
    for route in ("/api/generation/preflight", "/api/training/preflight", "/api/model/import", "/api/model/deploy"):
        assert route in source
    html = (PROJECT_ROOT / "phase7/webui/index.html").read_text(encoding="utf-8")
    for page in ("page-generation", "page-training", "page-live", "page-deployment"):
        assert f'id="{page}"' in html
