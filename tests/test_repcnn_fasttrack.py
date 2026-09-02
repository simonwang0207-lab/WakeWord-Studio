from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from wakeword_studio.training.repcnn_fasttrack import (
    HierarchicalBatchSampler,
    SamplingRecord,
    negative_weight,
    positive_is_eligible,
    preserve_checkpoint_prefix,
    production_learning_rate,
    select_validation_threshold,
    validation_improves_best,
)
from wakeword_studio.json_utils import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/models/repcnn_performance_v2_fasttrack.yaml"
VIEW_PATH = PROJECT_ROOT / "datasets/projects/qingxiaojia_v3_fasttrack/FASTTRACK_DATA_VIEW.json"


def schedule() -> dict[str, object]:
    return {
        "phase_steps": [8000, 1000, 1000],
        "base_learning_rate": 1e-4,
        "phase_lr_scales": [1.0, 0.1, 0.01],
        "phase1_warmup_fraction": 0.2,
        "phase1_hold_fraction": 1 / 3,
        "max_negative_weight": 5.0,
    }


def test_positive_eligibility_requires_duration_and_complete_containment() -> None:
    assert positive_is_eligible(
        phrase_start_ms=100.0, phrase_end_ms=2050.0, full_phrase_contained=True
    )
    assert not positive_is_eligible(
        phrase_start_ms=100.0, phrase_end_ms=2050.001, full_phrase_contained=True
    )
    assert not positive_is_eligible(
        phrase_start_ms=100.0, phrase_end_ms=1000.0, full_phrase_contained=False
    )
    assert not positive_is_eligible(
        phrase_start_ms=None, phrase_end_ms=1000.0, full_phrase_contained=True
    )


def test_production_lr_schedule_and_negative_weight_boundaries() -> None:
    value = schedule()
    assert production_learning_rate(value, 1) == pytest.approx(1e-4 / 1600)
    assert production_learning_rate(value, 1600) == pytest.approx(1e-4)
    assert production_learning_rate(value, 4267) == pytest.approx(1e-4)
    assert production_learning_rate(value, 8000) == pytest.approx(1e-5)
    assert production_learning_rate(value, 8001) == pytest.approx(1e-5)
    assert production_learning_rate(value, 9001) == pytest.approx(1e-6)
    assert production_learning_rate(value, 10000) == pytest.approx(1e-6)
    assert negative_weight(value, 1) == pytest.approx(1.0)
    assert negative_weight(value, 8000) == pytest.approx(5.0)
    assert negative_weight(value, 8001) == pytest.approx(5.0)


def synthetic_records() -> dict[str, list[SamplingRecord]]:
    records: dict[str, list[SamplingRecord]] = {
        "positive": [],
        "negative": [],
        "hard_negative": [],
        "ambient": [],
    }
    index = 0
    phrases = [f"near-miss-{number:02d}" for number in range(12)]
    for label in ("positive", "negative"):
        for source in ("kokoro", "voxcpm15"):
            for speaker in ("speaker-a", "speaker-b"):
                for record in range(8):
                    records[label].append(
                        SamplingRecord(
                            index=index,
                            record_id=f"{label}-{source}-{speaker}-{record}",
                            label=label,
                            source=source,
                            speaker_id=speaker,
                            text=None,
                        )
                    )
                    index += 1
    for source in ("kokoro", "voxcpm15"):
        for phrase in phrases:
            for speaker in ("speaker-a", "speaker-b"):
                records["hard_negative"].append(
                    SamplingRecord(
                        index=index,
                        record_id=f"hard-{source}-{phrase}-{speaker}",
                        label="hard_negative",
                        source=source,
                        speaker_id=speaker,
                        text=phrase,
                    )
                )
                index += 1
    for record in range(16):
        records["ambient"].append(
            SamplingRecord(
                index=index,
                record_id=f"ambient-{record}",
                label="ambient",
                source="procedural_ambient",
                speaker_id="none",
                text=None,
            )
        )
        index += 1
    return records


def test_hierarchical_sampler_has_exact_sources_and_phrase_cycle_balance() -> None:
    records = synthetic_records()
    by_index = {row.index: row for rows in records.values() for row in rows}
    sampler = HierarchicalBatchSampler(
        records,
        {"positive": 16, "negative": 4, "hard_negative": 8, "ambient": 4},
        seed=123,
        required_hard_phrases=["near-miss-00", "near-miss-11"],
    )
    hard_exposure: Counter[tuple[str, str | None]] = Counter()
    for step in range(1, 4):
        selected = sampler.sample(step)
        assert {label: len(indices) for label, indices in selected.items()} == {
            "positive": 16,
            "negative": 4,
            "hard_negative": 8,
            "ambient": 4,
        }
        for label, expected_per_source in (
            ("positive", 8),
            ("negative", 2),
            ("hard_negative", 4),
        ):
            sources = Counter(by_index[index].source for index in selected[label])
            assert sources == {"kokoro": expected_per_source, "voxcpm15": expected_per_source}
        for index in selected["hard_negative"]:
            row = by_index[index]
            hard_exposure[(row.source, row.text)] += 1
    assert len(hard_exposure) == 24
    assert set(hard_exposure.values()) == {1}
    speaker_counts = sampler.exposure_report()["speaker"]
    positive_counts = [
        count for key, count in speaker_counts.items() if key.startswith("positive:")
    ]
    assert max(positive_counts) - min(positive_counts) <= 1


def test_validation_threshold_obeys_fpr_cap_and_tracks_source_gap() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.65, 0.55, 0.1, 0.2, 0.3, 0.4])
    targets = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    labels = ["positive"] * 4 + ["negative"] * 2 + ["hard_negative"] * 2 + ["ambient"] * 2
    sources = ["kokoro", "kokoro", "voxcpm15", "voxcpm15"] + ["kokoro"] * 6
    result = select_validation_threshold(
        scores,
        targets,
        labels,
        sources,
        maximum_overall_fpr=0.0,
        thresholds=[0.5, 0.7],
    )
    assert result["threshold"] == pytest.approx(0.7)
    assert result["fpr"] == 0.0
    assert result["source_recall"] == {"kokoro": 1.0, "voxcpm15": 0.5}
    assert result["worst_source_recall"] == pytest.approx(0.5)
    assert result["source_gap"] == pytest.approx(0.5)
    assert result["fpr_cap_satisfied"] is True
    assert result["selection_kind"] == "configured_fpr_cap"
    assert result["operating_point_degenerate"] is False
    assert result["eligible_for_best"] is True
    assert len(result["threshold_sweep"]) == 2


def test_infeasible_custom_threshold_grid_is_diagnostic_and_cannot_be_best() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    targets = np.asarray([1, 1, 0, 0])
    labels = ["positive", "positive", "negative", "hard_negative"]
    sources = ["kokoro", "voxcpm15", "kokoro", "voxcpm15"]

    result = select_validation_threshold(
        scores,
        targets,
        labels,
        sources,
        maximum_overall_fpr=0.0,
        thresholds=[0.1, 0.5],
    )

    assert result["fpr_cap_satisfied"] is False
    assert result["selection_kind"] == "diagnostic_minimum_fpr_fallback"
    assert result["best_available_fpr"] == pytest.approx(1.0)
    assert result["threshold"] == pytest.approx(0.5)
    assert len(result["threshold_sweep"]) == 2
    assert validation_improves_best(result, ()) is False


def test_default_threshold_grid_has_sigmoid_and_reject_all_boundaries() -> None:
    scores = np.asarray([1.0, 1.0, 0.9, 0.8])
    targets = np.asarray([1, 1, 0, 0])
    labels = ["positive", "positive", "negative", "hard_negative"]
    sources = ["kokoro", "voxcpm15", "kokoro", "voxcpm15"]

    result = select_validation_threshold(
        scores, targets, labels, sources, maximum_overall_fpr=0.0
    )

    searched = [row["threshold"] for row in result["threshold_sweep"]]
    assert 1.0 in searched
    assert max(searched) > max(scores)
    assert result["fpr_cap_satisfied"] is True
    assert result["fpr"] == 0.0
    assert validation_improves_best(result, ()) is True


def test_reject_all_only_operating_point_is_degenerate_and_never_best() -> None:
    scores = np.asarray([1.0, 1.0, 1.0, 1.0])
    targets = np.asarray([1, 1, 0, 0])
    labels = ["positive", "positive", "negative", "hard_negative"]
    sources = ["kokoro", "voxcpm15", "kokoro", "voxcpm15"]

    result = select_validation_threshold(
        scores, targets, labels, sources, maximum_overall_fpr=0.0
    )

    assert result["fpr_cap_satisfied"] is True
    assert result["threshold"] > 1.0
    assert result["recall"] == 0.0
    assert result["fpr"] == 0.0
    assert result["operating_point_degenerate"] is True
    assert result["eligible_for_best"] is False
    assert validation_improves_best(result, ()) is False
    assert validation_improves_best(result, (0.1, 0.1, 0.0, 0.1, 0.1, -0.1)) is False
    assert all(row is not result for row in result["threshold_sweep"])


def test_step_3500_degenerate_status_is_atomic_json_safe_and_loop_can_advance(
    tmp_path: Path,
) -> None:
    scores = np.asarray([1.0, 1.0, 1.0, 1.0])
    targets = np.asarray([1, 1, 0, 0])
    labels = ["positive", "positive", "negative", "hard_negative"]
    sources = ["kokoro", "voxcpm15", "kokoro", "voxcpm15"]
    validation = select_validation_threshold(
        scores, targets, labels, sources, maximum_overall_fpr=0.0
    )
    status = {
        "status": "RUNNING",
        "current_step": 3500,
        "last_validation": validation,
        "best_validation_rank": [0.569444, 0.724684, -0.285207],
        "test_loaded": False,
    }
    json.dumps(status, ensure_ascii=False)
    status_path = tmp_path / "TRAINING_STATUS.json"
    atomic_write_json(status_path, status)
    restored = json.loads(status_path.read_text(encoding="utf-8"))
    assert restored["last_validation"]["operating_point_degenerate"] is True
    assert restored["last_validation"]["eligible_for_best"] is False
    assert len(restored["last_validation"]["threshold_sweep"]) >= 100
    assert validation_improves_best(validation, status["best_validation_rank"]) is False

    status["current_step"] = 3501
    atomic_write_json(status_path, status)
    assert json.loads(status_path.read_text(encoding="utf-8"))["current_step"] == 3501


def test_validation_sequence_continues_after_diagnostic_infeasible_result() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    targets = np.asarray([1, 1, 0, 0])
    labels = ["positive", "positive", "negative", "hard_negative"]
    sources = ["kokoro", "voxcpm15", "kokoro", "voxcpm15"]
    results = []
    for thresholds in ([0.1], [0.75]):
        results.append(
            select_validation_threshold(
                scores,
                targets,
                labels,
                sources,
                maximum_overall_fpr=0.0,
                thresholds=thresholds,
            )
        )

    assert len(results) == 2
    assert results[0]["fpr_cap_satisfied"] is False
    assert results[1]["fpr_cap_satisfied"] is True


def test_best_checkpoint_prefix_is_preserved_without_overwriting(tmp_path: Path) -> None:
    source_dir = tmp_path / "checkpoints"
    source_dir.mkdir()
    prefix = source_dir / "ckpt-2500"
    prefix.with_suffix(".index").write_bytes(b"index")
    (source_dir / "ckpt-2500.data-00000-of-00001").write_bytes(b"weights-and-optimizer")

    first = preserve_checkpoint_prefix(prefix, tmp_path / "preserved")
    second = preserve_checkpoint_prefix(prefix, tmp_path / "preserved")

    assert first == second
    assert {path.name for path in first} == {
        "ckpt-2500.index",
        "ckpt-2500.data-00000-of-00001",
    }
    assert (tmp_path / "preserved/ckpt-2500.data-00000-of-00001").read_bytes() == b"weights-and-optimizer"


def test_fasttrack_config_and_view_keep_test_closed() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    view = json.loads(VIEW_PATH.read_text(encoding="utf-8"))
    assert config["model_name"] == "qingxiaojia_repcnn_performance_v2_fasttrack"
    assert config["formal_training"]["phase_steps"] == [8000, 1000, 1000]
    assert config["formal_training"]["early_stopping"]["minimum_step"] == 8000
    assert config["objective"]["label_smoothing"] == 0.0
    assert config["formal_training"]["max_negative_weight"] == 5.0
    assert config["quantization"]["expected_input_shape"] == [1, 99, 40]
    assert config["quantization"]["expected_output_shape"] == [1, 1]
    assert config["frozen_data_contract"]["held_out_test_loaded"] is False
    assert view["test_loaded"] is False
    assert view["splits"]["train"]["eligible_positive"] == 2350
    assert view["splits"]["train"]["excluded_positive"] == 650
    assert view["splits"]["validation"]["eligible_positive"] == 316
    assert view["splits"]["validation"]["excluded_positive"] == 84
