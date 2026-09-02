import json

import numpy as np
import pytest

from wakeword_studio.training.repcnn_finalization import (
    average_weight_sets,
    discover_finalization_candidates,
    operating_points,
    resolve_finalization_v2_output_dir,
    select_average_candidates,
)


def _data():
    scores = [0.95, 0.80, 0.60, 0.55, 0.20, 0.10]
    targets = [1, 1, 1, 0, 0, 0]
    labels = ["positive"] * 3 + ["negative", "hard_negative", "ambient"]
    sources = ["kokoro", "voxcpm15", "kokoro", "kokoro", "voxcpm15", "ambient"]
    return scores, targets, labels, sources


def test_operating_points_report_caps_and_recall_targets():
    result = operating_points(*_data())

    assert result["best_f1"]["f1"] == 1.0
    assert result["fpr_caps"]["fpr_at_most_10pct"]["fpr"] == 0.0
    assert result["fpr_caps"]["fpr_at_most_5pct"]["fpr"] == 0.0
    assert result["recall_targets"]["recall_at_least_90pct"]["feasible"] is True
    assert result["recall_targets"]["recall_at_least_98pct"]["fpr"] == 0.0
    assert len(result["threshold_sweep"]) == result["threshold_count"]
    assert all(row is not result["best_f1"] for row in result["threshold_sweep"])


def test_average_candidate_fallback_is_explicit():
    scores, targets, labels, sources = _data()
    rows = []
    for step, offset in ((100, 0.0), (200, -0.02), (300, -0.04)):
        points = operating_points(
            np.asarray(scores) + offset, targets, labels, sources
        )
        rows.append(
            {
                "checkpoint": f"ckpt-{step}",
                "selection_metrics": points["fpr_caps"]["fpr_at_most_10pct"],
            }
        )

    selection = select_average_candidates(rows)

    assert len(selection["selected_checkpoints"]) >= 2
    assert "fallback_used" in selection


def test_weight_averaging_is_tensorwise_arithmetic_mean():
    first = [np.asarray([1.0, 3.0], np.float32), np.asarray([[2]], np.int32)]
    second = [np.asarray([3.0, 5.0], np.float32), np.asarray([[4]], np.int32)]

    averaged = average_weight_sets([first, second])

    np.testing.assert_array_equal(averaged[0], np.asarray([2.0, 4.0], np.float32))
    np.testing.assert_array_equal(averaged[1], np.asarray([[3]], np.int32))


def test_reject_all_cap_point_is_marked_ineligible_for_checkpoint_selection():
    scores = [1.0, 1.0, 1.0, 1.0]
    targets = [1, 1, 0, 0]
    labels = ["positive", "positive", "negative", "hard_negative"]
    sources = ["kokoro", "voxcpm15", "kokoro", "voxcpm15"]

    result = operating_points(scores, targets, labels, sources)
    point = result["fpr_caps"]["fpr_at_most_10pct"]

    assert point["recall"] == 0.0
    assert point["fpr"] == 0.0
    assert point["operating_point_degenerate"] is True
    assert point["eligible_for_checkpoint_selection"] is False
    assert len(result["threshold_sweep"]) == result["threshold_count"]


def test_candidate_discovery_includes_preserved_live_and_best_weights(tmp_path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    preserved = run_dir / "preserved_best_checkpoint"
    checkpoints.mkdir(parents=True)
    preserved.mkdir()
    for step in (9000, 9500, 10000):
        (checkpoints / f"ckpt-{step}.index").write_bytes(b"index")
    (preserved / "ckpt-2500.index").write_bytes(b"index")
    (run_dir / "best_single.weights.h5").write_bytes(b"weights")
    (run_dir / "BEST_SINGLE_VALIDATION.json").write_text(
        json.dumps({"step": 8500, "test_loaded": False}), encoding="utf-8"
    )

    candidates = discover_finalization_candidates(run_dir)

    assert [(row.step, row.candidate_kind) for row in candidates] == [
        (2500, "preserved_checkpoint"),
        (8500, "best_single_weights"),
        (9000, "checkpoint"),
        (9500, "checkpoint"),
        (10000, "checkpoint"),
    ]


def test_best_weights_metadata_must_prove_test_not_loaded(tmp_path):
    (tmp_path / "best_single.weights.h5").write_bytes(b"weights")
    (tmp_path / "BEST_SINGLE_VALIDATION.json").write_text(
        json.dumps({"step": 8500, "test_loaded": True}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="test_loaded=false"):
        discover_finalization_candidates(tmp_path)


def test_v2_output_defaults_to_new_directory_and_refuses_original(tmp_path):
    assert resolve_finalization_v2_output_dir(tmp_path).name == "phase6_finalization_v2"

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        resolve_finalization_v2_output_dir(tmp_path, tmp_path / "phase6_finalization")
