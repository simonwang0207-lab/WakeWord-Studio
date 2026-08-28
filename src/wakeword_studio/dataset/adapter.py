"""One adapter for generated data and arbitrary existing folders."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..audio import standardize_wav
from .manifest import (
    AcousticMetadata,
    DatasetManifest,
    DatasetRecord,
    SpeakerMetadata,
    sha256_file,
)


class DatasetAdapter:
    AUDIO_SUFFIXES = {".wav"}
    LABEL_DIRS = {
        "positive": "positive",
        "negative": "negative",
        "hard_negative": "hard_negative",
        "hard-negative": "hard_negative",
    }
    SPLIT_DIRS = {"train": "train", "training": "train", "val": "validation", "validation": "validation", "test": "test", "testing": "test"}

    def import_folder(
        self,
        folder: Path,
        wake_word: str,
        standardized_root: Path | None = None,
    ) -> DatasetManifest:
        folder = folder.resolve()
        standardized_root = self._standardized_root(folder, standardized_root)
        sidecars = self._load_sidecars(folder)
        records: list[DatasetRecord] = []
        label_counts: dict[str, int] = {}
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in self.AUDIO_SUFFIXES)
        for index, path in enumerate(paths):
            relative = path.relative_to(folder)
            parts = [part.lower() for part in relative.parts[:-1]]
            label = next((self.LABEL_DIRS[p] for p in reversed(parts) if p in self.LABEL_DIRS), None)
            if label is None:
                continue
            split = next((self.SPLIT_DIRS[p] for p in parts if p in self.SPLIT_DIRS), None)
            if split is None:
                label_index = label_counts.get(label, 0)
                label_counts[label] = label_index + 1
                split = self._split_for_index(label_index)
            meta = sidecars.get(relative.as_posix(), {})
            age_group = meta.get("age_group") or None
            age_source = meta.get("age_source", "unknown") or "unknown"
            if age_source not in {"verified", "reported", "unknown"}:
                raise ValueError(f"Invalid age_source for {relative}: {age_source}")
            output_path = standardized_root / label / f"{index:06d}_{path.name}"
            audio_info = standardize_wav(path, output_path)
            records.append(
                DatasetRecord(
                    record_id=f"record-{index:06d}",
                    audio_path=output_path.relative_to(standardized_root).as_posix(),
                    label=label,
                    split=split,
                    text=meta.get("text") or (wake_word if label == "positive" else None),
                    speaker=SpeakerMetadata(
                        speaker_id=meta.get("speaker_id") or f"unknown-{index:06d}",
                        source=meta.get("speaker_source") or "folder_import",
                        gender=meta.get("gender") or None,
                        age_group=age_group,
                        age_source=age_source,
                    ),
                    acoustic=AcousticMetadata(
                        speaking_rate=self._float_or_none(meta.get("speaking_rate")),
                        gain_db=self._float_or_none(meta.get("gain_db")),
                        noise_id=meta.get("noise_id") or None,
                        snr_db=self._float_or_none(meta.get("snr_db")),
                        reverb_id=meta.get("reverb_id") or None,
                        acoustic_age_proxy=meta.get("acoustic_age_proxy") or None,
                    ),
                    sample_rate_hz=audio_info.sample_rate_hz,
                    original_sample_rate_hz=audio_info.original_sample_rate_hz,
                    duration_seconds=audio_info.duration_seconds,
                    sha256=sha256_file(output_path),
                )
            )
        if not records:
            raise ValueError(
                "No labeled WAV files found. Use positive/, negative/, and hard_negative/ directories."
            )
        return DatasetManifest(
            wake_word=wake_word,
            records=records,
            source_kind="imported",
            root=str(standardized_root),
        )

    def from_generator_manifest(
        self,
        generator_manifest: Path,
        standardized_root: Path | None = None,
        limit_per_label: int | None = None,
    ) -> DatasetManifest:
        raw = json.loads(generator_manifest.read_text(encoding="utf-8"))
        root = generator_manifest.parent.resolve()
        standardized_root = self._standardized_root(root, standardized_root)
        records: list[DatasetRecord] = []
        label_counts: dict[str, int] = {}
        source_records = raw.get("records", [])
        for index, item in enumerate(source_records):
            path = Path(item["path"])
            if path.is_absolute():
                try:
                    path = path.relative_to(root)
                except ValueError:
                    pass
            hard = bool(item.get("hard_negative", False))
            label = "hard_negative" if hard else item["label"]
            label_index = label_counts.get(label, 0)
            if limit_per_label is not None and label_index >= limit_per_label:
                continue
            label_counts[label] = label_index + 1
            speaker_id = str(item.get("voice") or item.get("speaker_id") or f"synthetic-{index}")
            source_path = root / path
            output_path = standardized_root / label / f"{label_index:06d}_{source_path.name}"
            audio_info = standardize_wav(source_path, output_path)
            records.append(
                DatasetRecord(
                    record_id=f"generated-{index:06d}",
                    audio_path=output_path.relative_to(standardized_root).as_posix(),
                    label=label,
                    split=self._split_for_index(label_index),
                    text=item.get("text"),
                    speaker=SpeakerMetadata(
                        speaker_id=speaker_id,
                        source=str(raw.get("generator", "synthetic_tts")),
                        # TTS voice IDs do not prove a human age.
                        age_group=None,
                        age_source="unknown",
                    ),
                    acoustic=AcousticMetadata(speaking_rate=self._float_or_none(item.get("speed"))),
                    sample_rate_hz=audio_info.sample_rate_hz,
                    original_sample_rate_hz=audio_info.original_sample_rate_hz,
                    duration_seconds=audio_info.duration_seconds,
                    sha256=sha256_file(output_path),
                )
            )
        return DatasetManifest(
            wake_word=raw["target"],
            records=records,
            source_kind="generated",
            root=str(standardized_root),
            generator={
                key: raw[key]
                for key in ("generator", "package_version", "model_repo", "model_license")
                if key in raw
            },
        )

    @staticmethod
    def _load_sidecars(folder: Path) -> dict[str, dict[str, str]]:
        csv_path = folder / "metadata.csv"
        jsonl_path = folder / "metadata.jsonl"
        if csv_path.is_file():
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return {row["audio_path"].replace("\\", "/"): row for row in csv.DictReader(handle)}
        if jsonl_path.is_file():
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return {row["audio_path"].replace("\\", "/"): row for row in rows}
        return {}

    @staticmethod
    def _standardized_root(source_root: Path, requested: Path | None) -> Path:
        if requested is None:
            requested = source_root.parent / f"{source_root.name}_standardized"
        root = requested.resolve()
        if root == source_root:
            raise ValueError("standardized_root must differ from the source dataset root")
        return root

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        return None if value in (None, "") else float(value)

    @staticmethod
    def _split_for_index(index: int) -> str:
        bucket = index % 10
        return "train" if bucket < 8 else "validation" if bucket == 8 else "test"
