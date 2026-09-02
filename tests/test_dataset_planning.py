from __future__ import annotations

import json
from pathlib import Path

from wakeword_studio.dataset.hard_negatives import HardNegativeGenerator, pinyin_signature
from wakeword_studio.dataset.planning import load_and_estimate


def test_hard_negative_curriculum_contains_required_phrases_and_tiers() -> None:
    phrases = HardNegativeGenerator().generate()
    by_text = {item.text: item for item in phrases}
    required = {
        "你好，青小佳",
        "你好，请小甲",
        "你好，青小架",
        "你好，青小杰",
        "你好，金小甲",
        "你好，星小甲",
        "你好，小甲",
        "青小甲",
        "你好，青甲",
        "你好吗，青小甲",
        "你好，小安",
        "你好，小瑞",
    }
    assert required <= set(by_text)
    assert len(phrases) == 12
    assert {item.tier for item in phrases} == {1, 2, 3}
    assert all(item.text != HardNegativeGenerator.TARGET for item in phrases)
    assert all(
        pinyin_signature(item.text) != pinyin_signature(HardNegativeGenerator.TARGET)
        for item in phrases
    )
    assert not (set(HardNegativeGenerator.EXCLUDED_EXACT_HOMOPHONES) & set(by_text))


def test_formal_plan_has_exact_counts_exclusive_groups_and_unseen_tts() -> None:
    root = Path(__file__).parents[1]
    estimate = load_and_estimate(root / "phase2" / "configs" / "qingxiaojia_v1.json").report
    assert estimate["total_samples"] == 9000
    assert sum(estimate["targets"].values()) == 9000
    for counts in estimate["split_counts_by_label"].values():
        assert counts == {"train": int(sum(counts.values()) * 0.8), "validation": int(sum(counts.values()) * 0.1), "test": int(sum(counts.values()) * 0.1)}
    groups = estimate["source_groups_by_split"]
    assert not (set(groups["train"]) & set(groups["validation"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert "melotts" in estimate["test_source_families"]
    assert estimate["speaker_source_group_exclusive"] is True
    assert estimate["estimated_total_artifacts_gib"] < 5.0
    assert estimate["estimated_generation_minutes"] < 120.0
    json.dumps(estimate, ensure_ascii=False)
