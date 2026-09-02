"""Deterministic data and optimization policy for Model B v2 fast-track."""

from __future__ import annotations

import hashlib
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SPEECH_LABELS = ("positive", "negative", "hard_negative")
ALL_LABELS = (*SPEECH_LABELS, "ambient")
REQUIRED_SOURCES = ("kokoro", "voxcpm15")
CHECKPOINT_METRIC_FORMULA = (
    "Choose the Validation-only threshold with overall_fpr <= configured cap; "
    "then maximize lexicographically (worst_source_recall, overall_recall, "
    "-source_gap, f1, precision, -overall_fpr). Rank checkpoints by the same tuple."
)


def preserve_checkpoint_prefix(source_prefix: Path, destination_dir: Path) -> tuple[Path, ...]:
    """Copy a complete TensorFlow checkpoint prefix without changing its source.

    This protects the Validation best checkpoint from ordinary CheckpointManager
    retention when a long formal run resumes.  Existing identical copies are
    reused; conflicting destination files are rejected rather than overwritten.
    """

    source_prefix = Path(source_prefix)
    source_files = sorted(source_prefix.parent.glob(source_prefix.name + ".*"))
    required_index = source_prefix.with_suffix(".index")
    if required_index not in source_files or not any(".data-" in path.name for path in source_files):
        raise FileNotFoundError(f"Incomplete checkpoint prefix: {source_prefix}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in source_files:
        destination = destination_dir / source.name
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Preserved checkpoint conflicts with source: {destination}")
        else:
            shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)


def stable_key(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def positive_is_eligible(
    *,
    phrase_start_ms: float | None,
    phrase_end_ms: float | None,
    full_phrase_contained: bool,
    maximum_phrase_ms: float = 1950.0,
) -> bool:
    """Return whether a positive is safe for an exact two-second input."""

    if phrase_start_ms is None or phrase_end_ms is None:
        return False
    duration = float(phrase_end_ms) - float(phrase_start_ms)
    return duration > 0.0 and duration <= maximum_phrase_ms + 1e-6 and full_phrase_contained


def phase_for_step(schedule: Mapping[str, object], step: int) -> tuple[int, int, int]:
    """Return (phase number, one-based phase step, phase length)."""

    if step < 1:
        raise ValueError("step must be >= 1")
    lengths = [int(value) for value in schedule["phase_steps"]]  # type: ignore[index]
    cursor = 0
    for number, length in enumerate(lengths, start=1):
        if step <= cursor + length:
            return number, step - cursor, length
        cursor += length
    raise ValueError(f"step {step} exceeds planned total {cursor}")


def production_learning_rate(schedule: Mapping[str, object], step: int) -> float:
    """Three-phase LR: phase-1 warmup/hold/cosine, then 0.1x and 0.01x."""

    phase, local_step, phase_length = phase_for_step(schedule, step)
    base = float(schedule["base_learning_rate"])
    scales = [float(value) for value in schedule["phase_lr_scales"]]  # type: ignore[index]
    if phase != 1:
        return base * scales[phase - 1]

    warmup = int(round(phase_length * float(schedule["phase1_warmup_fraction"])))
    hold = int(round(phase_length * float(schedule["phase1_hold_fraction"])))
    warmup = max(1, warmup)
    hold = max(0, min(hold, phase_length - warmup))
    if local_step <= warmup:
        return base * scales[0] * local_step / warmup
    if local_step <= warmup + hold:
        return base * scales[0]

    decay_steps = phase_length - warmup - hold
    progress = (local_step - warmup - hold) / max(1, decay_steps)
    start = base * scales[0]
    end = base * scales[1]
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def negative_weight(schedule: Mapping[str, object], step: int) -> float:
    """Ramp negative weight from one to the configured maximum during phase 1."""

    phase, local_step, phase_length = phase_for_step(schedule, step)
    maximum = float(schedule["max_negative_weight"])
    if phase > 1:
        return maximum
    return 1.0 + (maximum - 1.0) * (local_step - 1) / max(1, phase_length - 1)


@dataclass(frozen=True, slots=True)
class SamplingRecord:
    index: int
    record_id: str
    label: str
    source: str
    speaker_id: str
    text: str | None


class HierarchicalBatchSampler:
    """Class -> source -> phrase (hard negatives) -> speaker -> record sampler.

    Source quotas are exact when the per-class count is divisible by the number of
    sources. Speaker, phrase, and record choices always take the least-exposed
    candidate, with a stable hash used only for ties.
    """

    def __init__(
        self,
        records: Mapping[str, Sequence[SamplingRecord]],
        batch_counts: Mapping[str, int],
        *,
        seed: int,
        required_sources: Sequence[str] = REQUIRED_SOURCES,
        required_hard_phrases: Sequence[str] = (),
    ) -> None:
        self.records = {label: tuple(rows) for label, rows in records.items()}
        self.batch_counts = {label: int(count) for label, count in batch_counts.items()}
        self.seed = int(seed)
        self.required_sources = tuple(required_sources)
        self.source_exposure: Counter[tuple[str, str]] = Counter()
        self.phrase_exposure: Counter[tuple[str, str]] = Counter()
        self.speaker_exposure: Counter[tuple[str, str, str]] = Counter()
        self.record_exposure: Counter[str] = Counter()

        missing_labels = set(ALL_LABELS) - set(self.records)
        if missing_labels:
            raise ValueError(f"missing sampler labels: {sorted(missing_labels)}")
        for label in SPEECH_LABELS:
            present = {row.source for row in self.records[label]}
            missing = set(self.required_sources) - present
            if missing:
                raise ValueError(f"{label} lacks required sources: {sorted(missing)}")

        all_hard_phrases = sorted({row.text for row in self.records["hard_negative"] if row.text})
        missing_phrases = set(required_hard_phrases) - set(all_hard_phrases)
        if missing_phrases:
            raise ValueError(f"hard-negative phrase families missing: {sorted(missing_phrases)}")
        self.hard_phrases = tuple(all_hard_phrases)
        for source in self.required_sources:
            available = {
                row.text
                for row in self.records["hard_negative"]
                if row.source == source and row.text
            }
            missing = set(self.hard_phrases) - available
            if missing:
                raise ValueError(
                    f"hard-negative source {source} lacks phrase families: {sorted(missing)}"
                )

    def _source_quotas(self, label: str, step: int) -> dict[str, int]:
        count = self.batch_counts[label]
        quotient, remainder = divmod(count, len(self.required_sources))
        quotas = {source: quotient for source in self.required_sources}
        if remainder:
            ordered = sorted(
                self.required_sources,
                key=lambda source: stable_key(self.seed, label, step, "source", source),
            )
            for source in ordered[:remainder]:
                quotas[source] += 1
        return quotas

    def _choose_record(
        self,
        candidates: Sequence[SamplingRecord],
        *,
        label: str,
        source: str,
        step: int,
        slot: int,
    ) -> SamplingRecord:
        speakers = sorted({row.speaker_id for row in candidates})
        speaker = min(
            speakers,
            key=lambda value: (
                self.speaker_exposure[(label, source, value)],
                stable_key(self.seed, label, source, step, slot, "speaker", value),
            ),
        )
        speaker_rows = [row for row in candidates if row.speaker_id == speaker]
        chosen = min(
            speaker_rows,
            key=lambda row: (
                self.record_exposure[row.record_id],
                stable_key(self.seed, label, source, step, slot, "record", row.record_id),
            ),
        )
        self.speaker_exposure[(label, source, speaker)] += 1
        self.record_exposure[chosen.record_id] += 1
        return chosen

    def sample(self, step: int) -> dict[str, list[int]]:
        selected: dict[str, list[int]] = {label: [] for label in ALL_LABELS}
        for label in ("positive", "negative"):
            for source, quota in self._source_quotas(label, step).items():
                pool = [row for row in self.records[label] if row.source == source]
                for slot in range(quota):
                    chosen = self._choose_record(
                        pool, label=label, source=source, step=step, slot=slot
                    )
                    selected[label].append(chosen.index)
                    self.source_exposure[(label, source)] += 1

        label = "hard_negative"
        for source, quota in self._source_quotas(label, step).items():
            for slot in range(quota):
                phrase = min(
                    self.hard_phrases,
                    key=lambda value: (
                        self.phrase_exposure[(source, value)],
                        stable_key(self.seed, label, source, step, slot, "phrase", value),
                    ),
                )
                pool = [
                    row
                    for row in self.records[label]
                    if row.source == source and row.text == phrase
                ]
                chosen = self._choose_record(
                    pool, label=label, source=source, step=step, slot=slot
                )
                selected[label].append(chosen.index)
                self.source_exposure[(label, source)] += 1
                self.phrase_exposure[(source, phrase)] += 1

        ambient = self.records["ambient"]
        for slot in range(self.batch_counts["ambient"]):
            chosen = self._choose_record(
                ambient, label="ambient", source="ambient", step=step, slot=slot
            )
            selected["ambient"].append(chosen.index)
        return selected

    def exposure_report(self) -> dict[str, object]:
        return {
            "source": {
                f"{label}:{source}": count
                for (label, source), count in sorted(self.source_exposure.items())
            },
            "hard_negative_phrase_by_source": {
                f"{source}:{phrase}": count
                for (source, phrase), count in sorted(self.phrase_exposure.items())
            },
            "speaker": {
                f"{label}:{source}:{speaker}": count
                for (label, source, speaker), count in sorted(self.speaker_exposure.items())
            },
            "unique_records_exposed": len(self.record_exposure),
            "total_record_exposures": int(sum(self.record_exposure.values())),
        }


def _binary_counts(score: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, int]:
    predicted = score >= threshold
    return {
        "tp": int(np.sum(predicted & (target == 1))),
        "fp": int(np.sum(predicted & (target == 0))),
        "tn": int(np.sum(~predicted & (target == 0))),
        "fn": int(np.sum(~predicted & (target == 1))),
    }


def validation_at_threshold(
    score: np.ndarray,
    target: np.ndarray,
    labels: Sequence[str],
    sources: Sequence[str],
    threshold: float,
) -> dict[str, object]:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.int32)
    label_array = np.asarray(labels, dtype=object)
    source_array = np.asarray(sources, dtype=object)
    counts = _binary_counts(score, target, threshold)
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    category_fpr: dict[str, float] = {}
    for label in ("negative", "hard_negative", "ambient"):
        mask = label_array == label
        category_fpr[label] = float(np.mean(score[mask] >= threshold)) if np.any(mask) else 0.0
    source_recall: dict[str, float] = {}
    for source in REQUIRED_SOURCES:
        mask = (target == 1) & (source_array == source)
        source_recall[source] = float(np.mean(score[mask] >= threshold)) if np.any(mask) else 0.0
    recalls = list(source_recall.values())
    result: dict[str, object] = {
        "threshold": float(threshold),
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "frr": 1.0 - recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "ordinary_negative_fpr": category_fpr["negative"],
        "hard_negative_fpr": category_fpr["hard_negative"],
        "ambient_fpr": category_fpr["ambient"],
        "source_recall": source_recall,
        "worst_source_recall": min(recalls),
        "source_gap": max(recalls) - min(recalls),
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score)),
    }
    return result


def validation_rank(metrics: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(metrics["worst_source_recall"]),
        float(metrics["recall"]),
        -float(metrics["source_gap"]),
        float(metrics["f1"]),
        float(metrics["precision"]),
        -float(metrics["fpr"]),
    )


def plain_validation_point(metrics: Mapping[str, object]) -> dict[str, object]:
    """Copy one Validation row into an independent JSON-native object."""

    source_recall = metrics["source_recall"]
    if not isinstance(source_recall, Mapping):
        raise TypeError("Validation source_recall must be a mapping")
    return {
        "threshold": float(metrics["threshold"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "frr": float(metrics["frr"]),
        "fpr": float(metrics["fpr"]),
        "ordinary_negative_fpr": float(metrics["ordinary_negative_fpr"]),
        "hard_negative_fpr": float(metrics["hard_negative_fpr"]),
        "ambient_fpr": float(metrics["ambient_fpr"]),
        "source_recall": {
            str(source): float(recall) for source, recall in source_recall.items()
        },
        "worst_source_recall": float(metrics["worst_source_recall"]),
        "source_gap": float(metrics["source_gap"]),
        "score_mean": float(metrics["score_mean"]),
        "score_std": float(metrics["score_std"]),
    }


def validation_improves_best(
    metrics: Mapping[str, object], best_rank: Sequence[float]
) -> bool:
    """Return whether a Validation result is eligible to replace the best model.

    A diagnostic fallback from an incomplete/custom threshold sweep must never
    become the formal best checkpoint when it violates the configured FPR cap.
    Metrics written before this flag was introduced remain compatible.
    """

    if metrics.get("fpr_cap_satisfied", True) is not True:
        return False
    if metrics.get("operating_point_degenerate", False) is True:
        return False
    return not best_rank or validation_rank(metrics) > tuple(float(v) for v in best_rank)


def select_validation_threshold(
    score: np.ndarray,
    target: np.ndarray,
    labels: Sequence[str],
    sources: Sequence[str],
    *,
    maximum_overall_fpr: float,
    thresholds: Sequence[float] | None = None,
) -> dict[str, object]:
    """Select a threshold and checkpoint statistics using Validation only."""

    score = np.asarray(score, dtype=np.float64)
    if score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("Validation scores must be non-empty and finite")
    if thresholds is None:
        # Keep the historical 0.01..0.99 reporting grid, but make the search
        # mathematically complete.  1.0 is an important sigmoid boundary, while
        # nextafter(max_score, +inf) guarantees a reject-all point even when a
        # float model emits exactly 1.0.
        candidates = tuple(float(value) for value in np.linspace(0.01, 0.99, 99))
        candidates += (1.0, float(np.nextafter(np.max(score), np.inf)))
        candidates = tuple(dict.fromkeys(candidates))
    else:
        candidates = tuple(float(value) for value in thresholds)
        if not candidates:
            raise ValueError("thresholds must not be empty")
    rows = [validation_at_threshold(score, target, labels, sources, value) for value in candidates]
    qualifying = [row for row in rows if float(row["fpr"]) <= maximum_overall_fpr]
    cap_satisfied = bool(qualifying)
    if cap_satisfied:
        selected_point = max(qualifying, key=validation_rank)
        selection_kind = "configured_fpr_cap"
    else:
        # This branch is reachable for deliberately supplied/custom incomplete
        # grids.  It is diagnostic only and is explicitly ineligible for best.
        selected_point = max(
            rows,
            key=lambda row: (
                -float(row["fpr"]),
                float(row["recall"]),
                float(row["worst_source_recall"]),
                float(row["f1"]),
                float(row["threshold"]),
            ),
        )
        selection_kind = "diagnostic_minimum_fpr_fallback"
    # ``selected_point`` is an item in ``rows``.  Never attach ``rows`` to that
    # same dict: selected -> sweep -> selected is a direct circular reference.
    best = plain_validation_point(selected_point)
    sweep = [plain_validation_point(row) for row in rows]
    best["fpr_cap_satisfied"] = cap_satisfied
    best["selection_kind"] = selection_kind
    best["best_available_fpr"] = min(float(row["fpr"]) for row in rows)
    # A no-positive/no-negative prediction point is mathematically useful as a
    # reject-all boundary, but it contains no evidence that the checkpoint is a
    # good detector and is therefore ineligible to replace the formal best.
    best["operating_point_degenerate"] = (
        int(best["tp"]) == 0 and int(best["fp"]) == 0
    )
    best["eligible_for_best"] = (
        cap_satisfied and not bool(best["operating_point_degenerate"])
    )
    best["maximum_overall_fpr"] = float(maximum_overall_fpr)
    best["checkpoint_metric_formula"] = CHECKPOINT_METRIC_FORMULA
    best["threshold_sweep"] = sweep
    return best
