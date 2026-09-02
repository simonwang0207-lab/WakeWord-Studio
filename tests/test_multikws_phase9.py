from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wakeword_studio.dataset.multikws_dataset import resolve_dataset_plan
from wakeword_studio.training.multikws_evaluator import (
    calibrate_validation,
    confusion_matrix,
    runtime_decision,
)
from wakeword_studio.training.multikws_vocabulary import MultiKWSVocabulary, add_keyword


ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = ROOT / "configs" / "multikws" / "teacher_six_keywords.json"


def test_teacher_six_mapping_is_vocabulary_driven() -> None:
    vocabulary = MultiKWSVocabulary.load(VOCABULARY)
    assert vocabulary.num_classes == 7
    assert vocabulary.class_names == (
        "background",
        "qingxiaojia",
        "doudou",
        "diandian",
        "xiaorui",
        "duoduo",
        "jizhiwa",
    )
    assert [item.class_index for item in vocabulary.keywords] == list(range(1, 7))


def test_add_keyword_expands_seven_to_eight_and_requires_retrain(tmp_path: Path) -> None:
    plan_path = tmp_path / "add_keyword_plan.json"
    plan = add_keyword(
        VOCABULARY,
        keyword_id="xiaoxin",
        display_name="你好，小新",
        destination=plan_path,
    )
    assert plan["old_num_classes"] == 7
    assert plan["new_num_classes"] == 8
    assert plan["ADD_KEYWORD_REQUIRES_RETRAIN"] is True
    assert plan["validation_recalibration_required"] is True
    assert json.loads(plan_path.read_text(encoding="utf-8"))["new_num_classes"] == 8


def test_runtime_margin_confusion_and_validation_calibration() -> None:
    scores = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.02, 0.80, 0.18],
            [0.02, 0.49, 0.49],
            [0.05, 0.15, 0.80],
        ],
        dtype=np.float32,
    )
    assert runtime_decision(scores[1], threshold=0.5, margin_threshold=0.2).class_index == 1
    ambiguous = runtime_decision(scores[2], threshold=0.4, margin_threshold=0.1)
    assert ambiguous.state == "AMBIGUOUS"
    assert ambiguous.class_index == 0
    matrix = confusion_matrix([0, 1, 1, 2], [0, 1, 0, 2], 3)
    assert matrix.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    report = calibrate_validation(scores, [0, 1, 1, 2], ["background", "one", "two"], ["s"] * 4)
    assert report["calibration_source"] == "validation_only"
    assert report["test_loaded"] is False
    assert len(report["top_k_scores"]) == 4


@pytest.mark.parametrize("profile", ["quick", "small"])
def test_dataset_profiles_are_configurable(profile: str) -> None:
    plan = resolve_dataset_plan({"profile": profile, "seed": 3})
    assert plan.positive_per_keyword > 0
    assert plan.background_count > 0


def test_formal_profile_has_no_fake_fixed_size() -> None:
    with pytest.raises(ValueError, match="provider speed"):
        resolve_dataset_plan({"profile": "formal"})


def test_dynamic_model_head_seven_to_eight_without_source_change() -> None:
    tf = pytest.importorskip("tensorflow")
    from wakeword_studio.training.multikws_models import build_multikws_model

    configs = {
        "bcresnet": {"channels": 4, "depth": 1, "subbands": 4, "dropout": 0.0},
        "convmixer": {"hidden_dim": 4, "depth": 1, "dropout": 0.0},
    }
    for name, config in configs.items():
        seven = build_multikws_model(name, (99, 40), 7, config)
        eight = build_multikws_model(name, (99, 40), 8, config)
        assert seven(tf.zeros((1, 99, 40))).shape.as_list() == [1, 7]
        assert eight(tf.zeros((1, 99, 40))).shape.as_list() == [1, 8]
        with tf.GradientTape() as tape:
            output = seven(tf.zeros((2, 99, 40)), training=True)
            loss = tf.reduce_sum(output)
        assert any(gradient is not None for gradient in tape.gradient(loss, seven.trainable_variables))
