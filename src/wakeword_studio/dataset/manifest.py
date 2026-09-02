"""Versioned, backend-neutral wake-word dataset manifest."""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ..audio import TARGET_CHANNELS, TARGET_SAMPLE_RATE_HZ, TARGET_SAMPLE_WIDTH_BYTES

Label = Literal["positive", "negative", "hard_negative", "ambient"]
Split = Literal["train", "validation", "test"]
AgeSource = Literal["verified", "reported", "unknown"]


@dataclass(slots=True)
class SpeakerMetadata:
    speaker_id: str
    source: str
    gender: str | None = None
    age_group: str | None = None
    age_source: AgeSource = "unknown"
    reference_speaker_id: str | None = None
    reference_age_group: str | None = None
    reference_age_group_source: str | None = None
    perceived_age_verified: bool = False


@dataclass(slots=True)
class AcousticMetadata:
    speaking_rate: float | None = None
    gain_db: float | None = None
    noise_id: str | None = None
    snr_db: float | None = None
    reverb_id: str | None = None
    acoustic_age_proxy: str | None = None
    leading_silence_seconds: float | None = None
    trailing_silence_seconds: float | None = None
    utterance_start_ms: float | None = None
    utterance_end_ms: float | None = None
    phrase_start_ms: float | None = None
    phrase_end_ms: float | None = None
    phrase_placement: str | None = None
    duration_bin: str | None = None
    window_alignment: str | None = None


@dataclass(slots=True)
class DatasetRecord:
    record_id: str
    audio_path: str
    label: Label
    split: Split
    text: str | None
    speaker: SpeakerMetadata
    acoustic: AcousticMetadata = field(default_factory=AcousticMetadata)
    sample_rate_hz: int | None = None
    original_sample_rate_hz: int | None = None
    duration_seconds: float | None = None
    sha256: str | None = None
    source_utterance_id: str | None = None
    source_group_id: str | None = None
    augmentation_id: str | None = None
    hard_negative_tier: int | None = None


@dataclass(slots=True)
class DatasetManifest:
    wake_word: str
    records: list[DatasetRecord]
    source_kind: Literal["generated", "imported", "mixed"]
    root: str
    schema: str = "wakeword-studio.dataset-manifest/v2"
    generator: dict[str, object] | None = None
    coverage_policy: dict[str, object] = field(
        default_factory=lambda: {
            "required_labels": ["positive", "negative", "hard_negative", "ambient"],
            "required_age_groups": ["child", "adult", "senior"],
            "age_rule": (
                "Only verified/reported speaker metadata counts as real age coverage; "
                "pitch/formant transforms are acoustic proxies only."
            ),
            "augmentation_dimensions": [
                "speaker",
                "speaking_rate",
                "volume",
                "noise",
                "reverb",
                "snr",
            ],
        }
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["records"] = [
            DatasetRecord(
                **{
                    **row,
                    "speaker": SpeakerMetadata(**row["speaker"]),
                    "acoustic": AcousticMetadata(**row.get("acoustic", {})),
                }
            )
            for row in raw["records"]
        ]
        return cls(**raw)

    def validate(self, manifest_path: Path | None = None, check_files: bool = True) -> list[str]:
        errors: list[str] = []
        ids: set[str] = set()
        utterance_splits: dict[str, set[str]] = {}
        source_group_splits: dict[str, set[str]] = {}
        root = Path(self.root)
        if manifest_path and not root.is_absolute():
            root = manifest_path.parent / root
        for row in self.records:
            if row.record_id in ids:
                errors.append(f"duplicate record_id: {row.record_id}")
            ids.add(row.record_id)
            if row.speaker.age_group and row.speaker.age_source == "unknown":
                errors.append(f"{row.record_id}: age_group present but age_source is unknown")
            if (
                row.speaker.reference_age_group
                and not row.speaker.reference_age_group_source
            ):
                errors.append(
                    f"{row.record_id}: reference_age_group present without its metadata source"
                )
            if row.speaker.perceived_age_verified:
                errors.append(
                    f"{row.record_id}: perceived age must remain unverified for this dataset"
                )
            if row.hard_negative_tier is not None:
                if row.label != "hard_negative":
                    errors.append(f"{row.record_id}: hard_negative_tier set for label {row.label}")
                elif row.hard_negative_tier not in {1, 2, 3}:
                    errors.append(f"{row.record_id}: invalid hard_negative_tier {row.hard_negative_tier}")
            if row.source_utterance_id:
                utterance_splits.setdefault(row.source_utterance_id, set()).add(row.split)
            if row.source_group_id:
                source_group_splits.setdefault(row.source_group_id, set()).add(row.split)
            audio_path = root / row.audio_path
            if check_files and not audio_path.is_file():
                errors.append(f"missing audio: {row.audio_path}")
            elif check_files:
                try:
                    with wave.open(str(audio_path), "rb") as handle:
                        actual_rate = handle.getframerate()
                        actual_channels = handle.getnchannels()
                        actual_width = handle.getsampwidth()
                except (EOFError, wave.Error) as exc:
                    errors.append(f"invalid WAV {row.audio_path}: {exc}")
                    continue
                if actual_rate != TARGET_SAMPLE_RATE_HZ:
                    errors.append(f"{row.record_id}: WAV sample rate is {actual_rate}, expected {TARGET_SAMPLE_RATE_HZ}")
                if row.sample_rate_hz != actual_rate:
                    errors.append(
                        f"{row.record_id}: manifest sample rate {row.sample_rate_hz} != WAV {actual_rate}"
                    )
                if actual_channels != TARGET_CHANNELS:
                    errors.append(f"{row.record_id}: WAV channels is {actual_channels}, expected {TARGET_CHANNELS}")
                if actual_width != TARGET_SAMPLE_WIDTH_BYTES:
                    errors.append(
                        f"{row.record_id}: WAV sample width is {actual_width}, expected {TARGET_SAMPLE_WIDTH_BYTES}"
                    )
        for utterance_id, splits in utterance_splits.items():
            if len(splits) > 1:
                errors.append(
                    f"source utterance leakage: {utterance_id} appears in {sorted(splits)}"
                )
        for group_id, splits in source_group_splits.items():
            if len(splits) > 1:
                errors.append(f"source group leakage: {group_id} appears in {sorted(splits)}")
        return errors

    def summary(self) -> dict[str, object]:
        labels: dict[str, int] = {}
        ages: dict[str, int] = {}
        real_age_records = 0
        proxy_records = 0
        splits: dict[str, int] = {}
        sources: dict[str, int] = {}
        for row in self.records:
            labels[row.label] = labels.get(row.label, 0) + 1
            age = row.speaker.age_group or "unknown"
            ages[age] = ages.get(age, 0) + 1
            if row.speaker.age_group and row.speaker.age_source in {"verified", "reported"}:
                real_age_records += 1
            if row.acoustic.acoustic_age_proxy:
                proxy_records += 1
            splits[row.split] = splits.get(row.split, 0) + 1
            sources[row.speaker.source] = sources.get(row.speaker.source, 0) + 1
        return {
            "records": len(self.records),
            "labels": labels,
            "age_groups": ages,
            "real_age_metadata_records": real_age_records,
            "acoustic_age_proxy_records": proxy_records,
            "splits": splits,
            "sources": sources,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
