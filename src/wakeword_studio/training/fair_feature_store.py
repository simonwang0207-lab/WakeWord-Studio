"""Frozen Train/Validation feature store shared by fair KWS experiments."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from wakeword_studio.training.repcnn_fasttrack import ALL_LABELS, SamplingRecord


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve portable config paths relative to the repository root."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@dataclass(slots=True)
class FeatureGroup:
    values: np.ndarray
    view_indices: np.ndarray
    samples: tuple["FeatureSample", ...]

    def __len__(self) -> int:
        return len(self.samples)

    def take(self, indices: list[int] | np.ndarray) -> np.ndarray:
        local = np.asarray(indices, dtype=np.int64)
        return np.asarray(self.values[self.view_indices[local]], dtype=np.float32)

    def batches(self, batch_size: int) -> Iterator[np.ndarray]:
        for offset in range(0, len(self), batch_size):
            yield self.take(np.arange(offset, min(len(self), offset + batch_size)))


@dataclass(frozen=True, slots=True)
class FrozenFeatureStore:
    groups: Mapping[tuple[str, str], FeatureGroup]
    input_shape: tuple[int, int]
    source_manifest_sha256: str
    view_manifest_sha256: str
    test_loaded: bool = False

    def sampling_records(self) -> dict[str, list[SamplingRecord]]:
        result: dict[str, list[SamplingRecord]] = {}
        for label in ALL_LABELS:
            result[label] = [
                SamplingRecord(
                    index=index,
                    record_id=sample.record_id,
                    label=sample.label,
                    source=sample.source,
                    speaker_id=sample.speaker_id,
                    text=sample.text,
                )
                for index, sample in enumerate(self.groups[("train", label)].samples)
            ]
        return result

    def counts(self) -> dict[str, int]:
        return {
            f"{split}:{label}": len(self.groups[(split, label)])
            for split in ("train", "validation")
            for label in ALL_LABELS
        }


@dataclass(frozen=True, slots=True)
class FeatureSample:
    record_id: str
    label: str
    target: float
    split: str
    speaker_id: str
    source: str
    text: str | None


def load_frozen_feature_store(
    config: Mapping[str, Any], *, project_root: Path
) -> FrozenFeatureStore:
    """Load the exact B2 Train/Validation view over the immutable v2 cache.

    The held-out Test split has no file name in the cache contract and is never
    enumerated by this loader.  The metadata view is hash-pinned independently
    from the source qingxiaojia_v2 manifest/cache.
    """

    data = config["data"]
    frontend = config["frontend"]
    view_manifest = resolve_project_path(project_root, str(data["view_manifest"]))
    cache_root = resolve_project_path(project_root, str(data["feature_cache"]))
    view_hash = str(data["view_manifest_sha256"]).lower()
    source_hash = str(data["source_manifest_sha256"]).lower()
    content = view_manifest.read_bytes()
    actual_view_hash = hashlib.sha256(content).hexdigest()
    if actual_view_hash != view_hash:
        raise RuntimeError(
            f"Fair-view manifest hash mismatch: expected={view_hash} actual={actual_view_hash}"
        )
    manifest = json.loads(content.decode("utf-8"))
    records = manifest.get("records", ())
    # Metadata outside the two allowed splits is never adapted into a sample.
    samples_by_group: dict[tuple[str, str], list[FeatureSample]] = {
        (split, label): []
        for split in ("train", "validation")
        for label in ALL_LABELS
    }
    for row in records:
        split = str(row.get("split"))
        label = str(row.get("label"))
        if split not in {"train", "validation"} or label not in ALL_LABELS:
            continue
        speaker = row.get("speaker", {})
        samples_by_group[(split, label)].append(
            FeatureSample(
                record_id=str(row["record_id"]),
                label=label,
                target=1.0 if label == "positive" else 0.0,
                split=split,
                speaker_id=str(speaker.get("speaker_id", "unknown")),
                source=str(speaker.get("source", "unknown")),
                text=str(row["text"]) if row.get("text") is not None else None,
            )
        )

    summary_path = cache_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("dataset_manifest_sha256", "")).lower() != source_hash:
        raise RuntimeError("Frozen feature cache does not match qingxiaojia_v2")
    if summary.get("test_loaded") is not False:
        raise RuntimeError("Feature cache is not proven Train/Validation-only")
    expected_shape = tuple(int(value) for value in frontend["input_shape"])
    if tuple(int(value) for value in summary.get("feature_shape", ())) != expected_shape:
        raise RuntimeError("Frozen frontend feature shape changed")

    groups: dict[tuple[str, str], FeatureGroup] = {}
    for split in ("train", "validation"):
        for label in ALL_LABELS:
            # Deliberately construct only these two explicit split names.  There
            # is no glob and therefore no accidental Test discovery.
            values_path = cache_root / f"{split}_{label}.npy"
            metadata_path = cache_root / f"{split}_{label}.jsonl"
            values = np.load(values_path, mmap_mode="r")
            metadata = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(metadata) != len(values):
                raise RuntimeError(f"Feature metadata length mismatch for {split}/{label}")
            if tuple(values.shape[1:]) != expected_shape:
                raise RuntimeError(f"Feature tensor shape changed for {split}/{label}")
            by_record = {str(row["record_id"]): int(row["feature_index"]) for row in metadata}
            samples = tuple(samples_by_group[(split, label)])
            if not samples:
                raise RuntimeError(f"No fair-view records for {split}/{label}")
            missing = [sample.record_id for sample in samples if sample.record_id not in by_record]
            if missing:
                raise RuntimeError(f"B2 view is absent from v2 feature cache: {missing[:5]}")
            indices = np.asarray([by_record[sample.record_id] for sample in samples], np.int64)
            groups[(split, label)] = FeatureGroup(values, indices, samples)

    return FrozenFeatureStore(
        groups=groups,
        input_shape=expected_shape,
        source_manifest_sha256=source_hash,
        view_manifest_sha256=view_hash,
        test_loaded=False,
    )
