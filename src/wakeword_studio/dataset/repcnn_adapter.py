"""DatasetManifest adapter for LiveKit Embedded Wakeword RepCNN training."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .manifest import DatasetManifest, DatasetRecord, sha256_file
from ..training.streaming_windows import StreamingWindow, plan_streaming_window


REPCNN_LABELS = ("positive", "negative", "hard_negative", "ambient")
REPCNN_ALLOWED_SPLITS = ("train", "validation")


@dataclass(frozen=True, slots=True)
class RepCNNSample:
    """A manifest record plus its deterministic 2-second classifier window."""

    record_id: str
    audio_path: Path
    label: str
    target: float
    split: str
    speaker_id: str
    source: str
    text: str | None
    snr_db: float | None
    phrase_start_ms: float | None
    phrase_end_ms: float | None
    window: StreamingWindow

    def metadata(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "audio_path": str(self.audio_path),
            "label": self.label,
            "target": self.target,
            "split": self.split,
            "speaker_id": self.speaker_id,
            "source": self.source,
            "text": self.text,
            "snr_db": self.snr_db,
            "phrase_start_ms": self.phrase_start_ms,
            "phrase_end_ms": self.phrase_end_ms,
            **self.window.to_dict(),
        }


class RepCNNDatasetAdapter:
    """Read qingxiaojia records in place without staging another WAV tree."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        expected_sha256: str,
        seed: int,
        clip_duration_ms: float = 2000.0,
        allowed_splits: Iterable[str] = REPCNN_ALLOWED_SPLITS,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if self.manifest_sha256 != expected_sha256.lower():
            raise RuntimeError(
                "DatasetManifest hash mismatch: "
                f"expected={expected_sha256.lower()} actual={self.manifest_sha256}"
            )
        self.seed = int(seed)
        self.clip_duration_ms = float(clip_duration_ms)
        self.allowed_splits = frozenset(allowed_splits)
        if not self.allowed_splits or not self.allowed_splits <= set(REPCNN_ALLOWED_SPLITS):
            raise ValueError("RepCNN preflight/training may use only train and validation")

        manifest = DatasetManifest.load(self.manifest_path)
        errors = manifest.validate(self.manifest_path, check_files=False)
        if errors:
            raise RuntimeError("Dataset metadata validation failed: " + "; ".join(errors[:10]))
        self.wake_word = manifest.wake_word
        self.dataset_root = Path(manifest.root).resolve()
        self.test_loaded = False
        self._samples: dict[tuple[str, str], tuple[RepCNNSample, ...]] = {}

        for split in sorted(self.allowed_splits):
            for label in REPCNN_LABELS:
                rows = [
                    row
                    for row in manifest.records
                    if row.split == split and row.label == label
                ]
                samples = tuple(self._adapt(row) for row in rows)
                if not samples:
                    raise RuntimeError(f"No DatasetManifest records for {split}/{label}")
                self._samples[(split, label)] = samples

    def _adapt(self, row: DatasetRecord) -> RepCNNSample:
        window = plan_streaming_window(
            record_id=row.record_id,
            label=row.label,
            duration_seconds=float(row.duration_seconds or 0.0),
            phrase_start_ms=row.acoustic.phrase_start_ms,
            phrase_end_ms=row.acoustic.phrase_end_ms,
            phrase_placement=row.acoustic.phrase_placement,
            window_ms=self.clip_duration_ms,
            seed=self.seed,
        )
        if not window.alignment_ok:
            raise RuntimeError(f"RepCNN window alignment failed: {row.record_id}")
        return RepCNNSample(
            record_id=row.record_id,
            audio_path=(self.dataset_root / row.audio_path).resolve(),
            label=row.label,
            target=1.0 if row.label == "positive" else 0.0,
            split=row.split,
            speaker_id=row.speaker.speaker_id,
            source=row.speaker.source,
            text=row.text,
            snr_db=row.acoustic.snr_db,
            phrase_start_ms=row.acoustic.phrase_start_ms,
            phrase_end_ms=row.acoustic.phrase_end_ms,
            window=window,
        )

    def samples(self, split: str, label: str) -> tuple[RepCNNSample, ...]:
        if split not in self.allowed_splits:
            raise ValueError(f"Split is not available through this adapter: {split}")
        return self._samples[(split, label)]

    def deterministic_sample(
        self, split: str, label: str, count: int, *, purpose: str
    ) -> list[RepCNNSample]:
        rows = self.samples(split, label)
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{self.seed}:{purpose}:{row.record_id}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ordered) < count:
            raise RuntimeError(
                f"Not enough {split}/{label} records: requested={count} available={len(ordered)}"
            )
        return ordered[:count]

    def source_label_counts(self, split: str) -> dict[str, dict[str, int]]:
        if split not in self.allowed_splits:
            raise ValueError(f"Split is not available through this adapter: {split}")
        result: dict[str, dict[str, int]] = {}
        for label in REPCNN_LABELS:
            result[label] = dict(
                sorted(Counter(row.source for row in self.samples(split, label)).items())
            )
        return result

    def counts(self) -> dict[str, int]:
        return {
            f"{split}:{label}": len(self.samples(split, label))
            for split in sorted(self.allowed_splits)
            for label in REPCNN_LABELS
        }
