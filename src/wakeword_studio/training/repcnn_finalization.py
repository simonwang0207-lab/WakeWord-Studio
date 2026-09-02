"""Validation-only checkpoint finalization helpers for RepCNN.

This module deliberately has no dataset loader.  Callers must supply scores and
Validation metadata, which keeps held-out Test data outside the selection path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wakeword_studio.training.repcnn_fasttrack import (
    validation_at_threshold,
    validation_rank,
)


@dataclass(frozen=True)
class FinalizationCandidate:
    """A model-state candidate that can be ranked using Validation only."""

    path: Path
    step: int
    candidate_kind: str

    @property
    def key(self) -> str:
        return str(self.path.resolve())

    def report_metadata(self) -> dict[str, object]:
        return {
            "candidate_path": self.key,
            "step": self.step,
            "candidate_kind": self.candidate_kind,
        }


def discover_finalization_candidates(run_dir: Path) -> list[FinalizationCandidate]:
    """Discover retained checkpoints plus the independently saved best weights.

    TensorFlow CheckpointManager retention is deliberately not treated as the
    complete candidate inventory: formal training also saves the latest
    Validation-best model as ``best_single.weights.h5`` with its step recorded
    in ``BEST_SINGLE_VALIDATION.json``.
    """

    run_dir = Path(run_dir)
    candidates: list[FinalizationCandidate] = []
    seen: set[tuple[str, int]] = set()
    for directory_name, kind in (
        ("checkpoints", "checkpoint"),
        ("preserved_best_checkpoint", "preserved_checkpoint"),
    ):
        directory = run_dir / directory_name
        for index_path in sorted(directory.glob("ckpt-*.index")):
            prefix = index_path.with_suffix("")
            try:
                step = int(prefix.name.removeprefix("ckpt-"))
            except ValueError as error:
                raise ValueError(f"Unexpected checkpoint prefix: {prefix}") from error
            identity = (str(prefix.resolve()), step)
            if identity not in seen:
                candidates.append(FinalizationCandidate(prefix, step, kind))
                seen.add(identity)

    weights = run_dir / "best_single.weights.h5"
    metadata_path = run_dir / "BEST_SINGLE_VALIDATION.json"
    if weights.is_file() != metadata_path.is_file():
        raise FileNotFoundError(
            "best_single.weights.h5 and BEST_SINGLE_VALIDATION.json must exist together"
        )
    if weights.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("test_loaded") is not False:
            raise RuntimeError("Best-single metadata does not prove test_loaded=false")
        step = int(metadata["step"])
        candidates.append(
            FinalizationCandidate(weights, step, "best_single_weights")
        )

    kind_order = {
        "preserved_checkpoint": 0,
        "best_single_weights": 1,
        "checkpoint": 2,
    }
    return sorted(candidates, key=lambda item: (item.step, kind_order[item.candidate_kind]))


def resolve_finalization_v2_output_dir(
    run_dir: Path, requested: Path | None = None
) -> Path:
    """Resolve the v2 output and prohibit reuse of the original freeze directory."""

    run_dir = Path(run_dir).resolve()
    original = (run_dir / "phase6_finalization").resolve()
    destination = (
        Path(requested).resolve()
        if requested is not None
        else (run_dir / "phase6_finalization_v2").resolve()
    )
    if destination == original:
        raise RuntimeError("Refusing to overwrite the original phase6_finalization")
    return destination


def threshold_candidates(scores: Sequence[float]) -> tuple[float, ...]:
    """Return every score boundary plus a reject-all boundary."""

    values = np.unique(np.asarray(scores, dtype=np.float64))
    if values.size == 0:
        raise ValueError("scores must not be empty")
    return tuple(float(value) for value in values) + (
        float(np.nextafter(values[-1], np.inf)),
    )


def _best(rows: Sequence[dict[str, object]], key) -> dict[str, object]:
    if not rows:
        raise ValueError("operating-point candidate list is empty")
    return dict(max(rows, key=key))


def _annotate_operating_point(row: Mapping[str, object]) -> dict[str, object]:
    result = dict(row)
    degenerate = int(result["tp"]) == 0 and int(result["fp"]) == 0
    result["operating_point_degenerate"] = degenerate
    result["eligible_for_checkpoint_selection"] = not degenerate
    return result


def operating_points(
    scores: Sequence[float],
    targets: Sequence[int],
    labels: Sequence[str],
    sources: Sequence[str],
    *,
    thresholds: Sequence[float] | None = None,
) -> dict[str, object]:
    """Report useful Validation operating points without hiding trade-offs."""

    score_array = np.asarray(scores, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int32)
    if not (len(score_array) == len(target_array) == len(labels) == len(sources)):
        raise ValueError("scores, targets, labels, and sources must have equal length")
    candidates = tuple(thresholds) if thresholds is not None else threshold_candidates(scores)
    rows = [
        validation_at_threshold(score_array, target_array, labels, sources, threshold)
        for threshold in candidates
    ]
    best_f1 = _annotate_operating_point(_best(
        rows,
        lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
            -float(row["fpr"]),
            float(row["threshold"]),
        ),
    ))
    fpr_caps: dict[str, object] = {}
    for cap in (0.10, 0.05):
        qualifying = [row for row in rows if float(row["fpr"]) <= cap + 1e-15]
        selected = _annotate_operating_point(_best(qualifying, validation_rank))
        fpr_caps[f"fpr_at_most_{int(cap * 100)}pct"] = {
            "cap": cap,
            **selected,
        }

    recall_targets: dict[str, object] = {}
    for requested in (0.90, 0.95, 0.98):
        qualifying = [row for row in rows if float(row["recall"]) >= requested - 1e-15]
        key = f"recall_at_least_{int(requested * 100)}pct"
        if not qualifying:
            recall_targets[key] = {"requested_recall": requested, "feasible": False}
            continue
        selected = _best(
            qualifying,
            lambda row: (
                -float(row["fpr"]),
                float(row["precision"]),
                float(row["f1"]),
                float(row["threshold"]),
            ),
        )
        recall_targets[key] = {
            "requested_recall": requested,
            "feasible": True,
            **_annotate_operating_point(selected),
        }

    return {
        "best_f1": best_f1,
        "fpr_caps": fpr_caps,
        "recall_targets": recall_targets,
        "threshold_count": len(candidates),
        "threshold_sweep": [_annotate_operating_point(row) for row in rows],
    }


def accuracy(metrics: Mapping[str, object]) -> float:
    total = sum(int(metrics[key]) for key in ("tp", "fp", "tn", "fn"))
    return (
        (int(metrics["tp"]) + int(metrics["tn"])) / total
        if total
        else 0.0
    )


def select_average_candidates(
    checkpoint_rows: Sequence[Mapping[str, object]],
    *,
    minimum_count: int = 2,
    fallback_count: int = 3,
) -> dict[str, object]:
    """Apply the upstream low-FPR/high-recall/high-accuracy percentile rule.

    The upstream implementation averages candidates in the intersection of the
    10th FPR percentile and the 90th recall/accuracy percentiles.  A small run can
    have an empty intersection; in that case the top Validation-ranked available
    checkpoints are returned and the fallback is made explicit.
    """

    if not checkpoint_rows:
        raise ValueError("checkpoint_rows must not be empty")
    enriched: list[dict[str, object]] = []
    for row in checkpoint_rows:
        metrics = dict(row["selection_metrics"])  # type: ignore[arg-type]
        enriched.append({**dict(row), "accuracy": accuracy(metrics)})
    fpr_limit = float(np.percentile([row["selection_metrics"]["fpr"] for row in enriched], 10))  # type: ignore[index]
    recall_limit = float(np.percentile([row["selection_metrics"]["recall"] for row in enriched], 90))  # type: ignore[index]
    accuracy_limit = float(np.percentile([row["accuracy"] for row in enriched], 90))
    selected = [
        row
        for row in enriched
        if float(row["selection_metrics"]["fpr"]) <= fpr_limit + 1e-15  # type: ignore[index]
        and float(row["selection_metrics"]["recall"]) >= recall_limit - 1e-15  # type: ignore[index]
        and float(row["accuracy"]) >= accuracy_limit - 1e-15
    ]
    fallback = len(selected) < minimum_count
    if fallback:
        selected = sorted(
            enriched,
            key=lambda row: validation_rank(row["selection_metrics"]),  # type: ignore[arg-type]
            reverse=True,
        )[: min(fallback_count, len(enriched))]
    return {
        "method": "upstream_percentile_intersection",
        "percentiles": {
            "fpr_p10_max": fpr_limit,
            "recall_p90_min": recall_limit,
            "accuracy_p90_min": accuracy_limit,
        },
        "fallback_used": fallback,
        "fallback_reason": (
            "fewer_than_two_percentile_intersection_candidates"
            if fallback
            else None
        ),
        "selected_checkpoints": [str(row["checkpoint"]) for row in selected],
    }


def average_weight_sets(weight_sets: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    """Arithmetic mean of corresponding checkpoint tensors."""

    if not weight_sets:
        raise ValueError("weight_sets must not be empty")
    tensor_count = len(weight_sets[0])
    if any(len(values) != tensor_count for values in weight_sets):
        raise ValueError("checkpoint weight sets have different tensor counts")
    averaged: list[np.ndarray] = []
    for index in range(tensor_count):
        shapes = {tuple(values[index].shape) for values in weight_sets}
        if len(shapes) != 1:
            raise ValueError(f"checkpoint tensor {index} shapes differ")
        averaged.append(
            np.mean(
                np.stack([np.asarray(values[index]) for values in weight_sets]),
                axis=0,
                dtype=np.float64,
            ).astype(weight_sets[0][index].dtype)
        )
    return averaged
