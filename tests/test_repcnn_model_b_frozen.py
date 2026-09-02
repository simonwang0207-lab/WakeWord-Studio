from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from phase3.scripts.evaluate_repcnn_model_b_frozen import (
    binary_metrics,
    dequantize_output,
    detailed_report,
    external_source_breakdown,
    fullwav_window_starts,
    fullwav_windows,
    quantize_input,
    score_fullwav_clips,
    threshold_sweep,
    verify_threshold_gate,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"label": "positive", "score": 0.90},
        {"label": "positive", "score": 0.80},
        {"label": "negative", "score": 0.20},
        {"label": "hard_negative", "score": 0.70},
        {"label": "ambient", "score": 0.05},
    ]


def test_output_dequantization_uses_tflite_affine_formula() -> None:
    actual = dequantize_output(np.asarray([-128, -127, 127], np.int8), 1 / 256, -128)
    np.testing.assert_allclose(actual, [0.0, 1 / 256, 255 / 256])


def test_input_quantization_clips_to_int8() -> None:
    actual = quantize_input(np.asarray([-100.0, 0.0, 100.0]), 0.5, -3)
    np.testing.assert_array_equal(actual, [-128, -3, 127])


def test_binary_metrics_reports_required_fields() -> None:
    metrics = binary_metrics(_rows(), 0.75)
    assert metrics["recall_tpr"] == pytest.approx(1.0)
    assert metrics["false_rejection_rate"] == pytest.approx(0.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None


def test_threshold_sweep_reports_best_and_recall_targets() -> None:
    _, operating = threshold_sweep(_rows(), 1 / 256, -128)
    assert "best_f1" in operating
    assert set(operating["recall_targets"]) == {
        "recall_at_least_90", "recall_at_least_95", "recall_at_least_98"
    }
    assert operating["recall_targets"]["recall_at_least_98"]["verdict"]


def _context() -> dict[str, str]:
    return {
        "best_weights_sha256": "weights",
        "config_sha256": "config",
        "manifest_sha256": "manifest",
    }


def _freeze() -> dict[str, object]:
    return {
        "selection_split": "v2_validation_only",
        "selected_threshold": 0.84375,
        "checkpoint_sha256": "weights",
        "config_sha256": "config",
        "v2_manifest_sha256": "manifest",
        "tflite_sha256": "model",
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
    }


def test_test_gate_rejects_missing_validation_freeze(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Validation threshold"):
        verify_threshold_gate(_context(), tmp_path, {"sha256": "model"})


def test_test_gate_rejects_changed_artifact(tmp_path: Path) -> None:
    value = _freeze()
    value["checkpoint_sha256"] = "changed"
    (tmp_path / "threshold_freeze.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance"):
        verify_threshold_gate(_context(), tmp_path, {"sha256": "model"})


def test_test_gate_rejects_changed_threshold(tmp_path: Path) -> None:
    value = _freeze()
    value["selected_threshold"] = 0.5
    (tmp_path / "threshold_freeze.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly 0.84375"):
        verify_threshold_gate(_context(), tmp_path, {"sha256": "model"})


def test_test_gate_accepts_exact_frozen_provenance(tmp_path: Path) -> None:
    value = _freeze()
    (tmp_path / "threshold_freeze.json").write_text(json.dumps(value), encoding="utf-8")
    assert verify_threshold_gate(_context(), tmp_path, {"sha256": "model"}) == value


def test_detailed_report_has_required_v2_schema_and_utf8_texts() -> None:
    rows = _rows() + [
        {"label": "hard_negative", "score": 0.9, "source": "kokoro", "speaker_id": "s1", "text": "你好，小甲"},
        {"label": "hard_negative", "score": 0.1, "source": "voxcpm15", "speaker_id": "s2", "text": "你好，青甲"},
    ]
    for index, row in enumerate(rows[:5]):
        row.update({"source": "kokoro" if index % 2 else "voxcpm15", "speaker_id": f"s{index}", "text": "other"})
    report = detailed_report(rows, 0.75)
    assert set(report["categories"]) == {
        "positive", "ordinary_negative", "hard_negative", "ambient"
    }
    assert "Kokoro" in report["sources"]
    assert "VoxCPM1.5" in report["sources"]
    assert set(report["special_hard_negative_texts"]) == {"你好，小甲", "你好，青甲"}
    assert {"tp", "fp", "tn", "fn"} <= set(report["metrics"])


def test_external_source_breakdown_has_three_required_groups() -> None:
    rows = [
        {"label": "positive", "score": 0.9, "source": "kokoro", "speaker_id": "zm_053"},
        {"label": "negative", "score": 0.1, "source": "kokoro", "speaker_id": "zm_056"},
        {"label": "hard_negative", "score": 0.2, "source": "melotts", "speaker_id": "ZH"},
    ]
    assert set(external_source_breakdown(rows, 0.84375)) == {
        "Kokoro zm_053", "Kokoro zm_056", "MeloTTS ZH"
    }


def test_fullwav_short_audio_uses_symmetric_padding_with_odd_extra_at_end() -> None:
    audio = np.ones(9, dtype=np.float32)
    windows = fullwav_windows(audio, sample_rate_hz=10)
    assert len(windows) == 1
    start, clip = windows[0]
    assert start == -5
    np.testing.assert_array_equal(clip[:5], 0.0)
    np.testing.assert_array_equal(clip[5:14], 1.0)
    np.testing.assert_array_equal(clip[14:], 0.0)


def test_fullwav_exactly_two_seconds_has_one_window() -> None:
    assert fullwav_window_starts(20, 10) == [0]


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [
        (25, [0, 5]),
        (30, [0, 10]),
        (36, [0, 10, 16]),
    ],
)
def test_fullwav_regular_and_tail_anchored_windows(
    num_samples: int, expected: list[int]
) -> None:
    assert fullwav_window_starts(num_samples, 10) == expected


def test_fullwav_window_generation_is_deterministic() -> None:
    audio = np.arange(36, dtype=np.float32)
    first = fullwav_windows(audio, 10)
    second = fullwav_windows(audio, 10)
    assert [start for start, _ in first] == [start for start, _ in second]
    for (_, left), (_, right) in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_fullwav_generator_has_no_label_or_score_inputs() -> None:
    assert tuple(inspect.signature(fullwav_window_starts).parameters) == (
        "num_samples", "sample_rate_hz"
    )
    assert tuple(inspect.signature(fullwav_windows).parameters) == (
        "audio", "sample_rate_hz"
    )


def test_fullwav_scores_every_predetermined_window_before_max() -> None:
    windows = fullwav_windows(np.ones(36, dtype=np.float32), 10)
    predetermined_starts = [start for start, _ in windows]
    supplied_scores = iter([0.1, 0.9, 0.2])
    calls = 0

    def fake_score(_: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return next(supplied_scores)

    result = score_fullwav_clips(windows, fake_score)
    assert predetermined_starts == [0, 10, 16]
    assert calls == len(predetermined_starts)
    assert result["record_score"] == pytest.approx(0.9)
    assert result["winning_window_index"] == 1
    assert result["winning_window_start_sample"] == 10
