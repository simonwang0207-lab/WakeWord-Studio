"""Unified Validation-only evaluator for binary wake-word models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from wakeword_studio.training.repcnn_fasttrack import select_validation_threshold
from wakeword_studio.training.repcnn_finalization import operating_points


def roc_auc(scores: Sequence[float], targets: Sequence[int]) -> float:
    """Compute ROC-AUC with tie-aware average ranks."""

    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int32)
    positive = int(np.sum(target == 1))
    negative = int(np.sum(target == 0))
    if not positive or not negative:
        return 0.0
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[target == 1]))
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def pr_auc(scores: Sequence[float], targets: Sequence[int]) -> float:
    """Compute step-wise average precision (the project PR-AUC convention)."""

    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int32)
    positive = int(np.sum(target == 1))
    if not positive:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    true_positive = np.cumsum(sorted_target == 1)
    false_positive = np.cumsum(sorted_target == 0)
    precision = true_positive / np.maximum(1, true_positive + false_positive)
    return float(np.sum(precision[sorted_target == 1]) / positive)


def per_source_metrics(
    scores: Sequence[float], targets: Sequence[int], sources: Sequence[str], threshold: float
) -> dict[str, dict[str, float | int | None]]:
    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int32)
    source = np.asarray(sources, dtype=object)
    result: dict[str, dict[str, float | int | None]] = {}
    for name in sorted({str(value) for value in source}):
        mask = source == name
        positives = mask & (target == 1)
        negatives = mask & (target == 0)
        result[name] = {
            "positive_count": int(np.sum(positives)),
            "negative_count": int(np.sum(negatives)),
            "recall": (
                float(np.mean(score[positives] >= threshold)) if np.any(positives) else None
            ),
            "fpr": (
                float(np.mean(score[negatives] >= threshold)) if np.any(negatives) else None
            ),
        }
    return result


def evaluate_validation_scores(
    scores: Sequence[float],
    targets: Sequence[int],
    labels: Sequence[str],
    sources: Sequence[str],
    *,
    maximum_overall_fpr: float,
    thresholds: Sequence[float] | None = None,
) -> dict[str, object]:
    """Apply the frozen B2 ranking and add model-independent diagnostics."""

    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int32)
    selected = select_validation_threshold(
        score,
        target,
        labels,
        sources,
        maximum_overall_fpr=maximum_overall_fpr,
        thresholds=thresholds,
    )
    threshold = float(selected["threshold"])
    selected["roc_auc"] = roc_auc(score, target)
    selected["pr_auc"] = pr_auc(score, target)
    selected["per_source"] = per_source_metrics(score, target, sources, threshold)
    selected["operating_points"] = operating_points(score, target, labels, sources)
    selected["split"] = "validation"
    selected["test_loaded"] = False
    return selected


def comparison_row(metrics: Mapping[str, object]) -> dict[str, object]:
    """Extract the stable formal-comparison fields from one result."""

    return {
        key: metrics[key]
        for key in (
            "tp",
            "fp",
            "tn",
            "fn",
            "recall",
            "precision",
            "f1",
            "fpr",
            "frr",
            "worst_source_recall",
            "source_gap",
            "roc_auc",
            "pr_auc",
        )
    }
