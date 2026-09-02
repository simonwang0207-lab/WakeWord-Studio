from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_microwakeword_tiny_formal_config_is_gated_and_reproducible() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (root / "configs" / "models" / "microwakeword_tiny_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["wake_word"] == "你好，青小甲"
    assert config["architecture"]["parameter_count"] == 19_697
    assert config["architecture"]["size"] == "Tiny"
    assert sum(config["training_steps"]) == config["planned_steps"] == 15_000
    assert len(config["dataset_manifest_sha256"]) == 64
    assert sum(config["class_sampling"].values()) == pytest.approx(1.0)
    assert (
        config["class_sampling"]["hard_negative"]
        >= config["class_sampling"]["ordinary_negative"]
    )
    assert config["validation_policy"]["test_during_training"] is False
    assert config["validation_policy"]["freeze_threshold_before_test"] is True
    assert config["quantization"]["expected_size_range_kib"] == [50, 100]
    assert config["benchmark"]["steps"] == 150
