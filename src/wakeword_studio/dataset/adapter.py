"""One adapter for generated data and arbitrary existing folders."""

from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import TARGET_PCM_SUBTYPE, TARGET_SAMPLE_RATE_HZ, load_audio_float32, standardize_wav
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
        "ambient": "ambient",
        "background": "ambient",
    }
    SPLIT_DIRS = {"train": "train", "training": "train", "val": "validation", "validation": "validation", "test": "test", "testing": "test"}
    AGE_DIRS = {
        "child": "child",
        "young": "young",
        "middle": "middle",
        "senior": "senior",
    }

    def __init__(self) -> None:
        self.last_import_errors: list[dict[str, str]] = []

    def import_folder(
        self,
        folder: Path,
        wake_word: str,
        standardized_root: Path | None = None,
        augment: bool = False,
    ) -> DatasetManifest:
        self.last_import_errors = []
        folder = folder.resolve()
        standardized_root = self._standardized_root(folder, standardized_root)
        sidecars = self._load_sidecars(folder)
        records: list[DatasetRecord] = []
        label_counts: dict[str, int] = {}
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in self.AUDIO_SUFFIXES)
        for index, path in enumerate(paths):
            relative = path.relative_to(folder)
            parts = [part.lower() for part in relative.parts[:-1]]
            meta = sidecars.get(relative.as_posix(), {})
            folder_age = next((self.AGE_DIRS[p] for p in parts if p in self.AGE_DIRS), None)
            age_group = meta.get("age_group") or folder_age or None
            metadata_label = str(meta.get("label") or "").lower()
            label = next((self.LABEL_DIRS[p] for p in reversed(parts) if p in self.LABEL_DIRS), None)
            if label is None and metadata_label in self.LABEL_DIRS:
                label = self.LABEL_DIRS[metadata_label]
            # The documented dataset/child|young|middle|senior layout represents
            # recordings of the configured wake phrase unless metadata says otherwise.
            if label is None and age_group:
                label = "positive"
            if label is None:
                continue
            split = next((self.SPLIT_DIRS[p] for p in parts if p in self.SPLIT_DIRS), None)
            if split is None:
                label_index = label_counts.get(label, 0)
                label_counts[label] = label_index + 1
                split = self._split_for_index(label_index)
            age_source = meta.get("age_source") or ("reported" if age_group else "unknown")
            if age_source not in {"verified", "reported", "unknown"}:
                raise ValueError(f"Invalid age_source for {relative}: {age_source}")
            output_path = standardized_root / label / f"{index:06d}_{path.name}"
            try:
                audio_info = standardize_wav(path, output_path)
            except Exception as exc:
                # One malformed recording must not discard all valid imports.
                # The source file is read-only and is never overwritten.
                self.last_import_errors.append({
                    "audio_path": relative.as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            record = DatasetRecord(
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
            records.append(record)
            if augment:
                records.append(self._augment_record(record, output_path, standardized_root, index))
        if not records:
            detail = f" Errors: {self.last_import_errors}" if self.last_import_errors else ""
            raise ValueError(
                "No valid labeled WAV files found. Use positive/, negative/, and hard_negative/ directories."
                + detail
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
                rows = list(csv.DictReader(handle))
                result = {}
                for row in rows:
                    key = row.get("audio_path") or row.get("file")
                    if not key:
                        raise ValueError("metadata.csv 必须包含 file 或 audio_path 列")
                    result[key.replace("\\", "/")] = row
                return result
        if jsonl_path.is_file():
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return {row["audio_path"].replace("\\", "/"): row for row in rows}
        return {}

    @staticmethod
    def _augment_record(
        source: DatasetRecord,
        source_path: Path,
        standardized_root: Path,
        index: int,
    ) -> DatasetRecord:
        """Create one deterministic noise/reverb/SNR variant without touching source WAV."""

        audio, _ = load_audio_float32(source_path)
        seed = int.from_bytes(hashlib.sha256(source.record_id.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        snr_db = float((10.0, 15.0, 20.0)[index % 3])
        delay = int(TARGET_SAMPLE_RATE_HZ * (0.035 + 0.015 * (index % 3)))
        reverbed = audio.copy()
        if len(audio) > delay:
            reverbed[delay:] += audio[:-delay] * 0.18
        noise = rng.normal(0.0, 1.0, len(reverbed)).astype(np.float32)
        signal_rms = float(np.sqrt(np.mean(reverbed * reverbed) + 1e-12))
        noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
        mixed = np.clip(
            reverbed + noise * signal_rms / (10 ** (snr_db / 20.0) * noise_rms),
            -1.0,
            1.0,
        )
        output_path = source_path.with_name(f"{source_path.stem}_aug.wav")
        sf.write(output_path, mixed, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE, format="WAV")
        info = sf.info(output_path)
        return DatasetRecord(
            record_id=f"{source.record_id}-aug",
            audio_path=output_path.relative_to(standardized_root).as_posix(),
            label=source.label,
            split=source.split,
            text=source.text,
            speaker=SpeakerMetadata(**source.speaker.__dict__) if hasattr(source.speaker, "__dict__") else SpeakerMetadata(
                speaker_id=source.speaker.speaker_id,
                source=source.speaker.source,
                gender=source.speaker.gender,
                age_group=source.speaker.age_group,
                age_source=source.speaker.age_source,
            ),
            acoustic=AcousticMetadata(
                noise_id=f"local_broadband_{index:06d}",
                snr_db=snr_db,
                reverb_id=f"local_early_reflection_{delay}_samples",
            ),
            sample_rate_hz=int(info.samplerate),
            original_sample_rate_hz=source.sample_rate_hz,
            duration_seconds=float(info.duration),
            sha256=sha256_file(output_path),
            source_utterance_id=source.record_id,
            source_group_id=f"local-{source.record_id}",
            augmentation_id=f"local-standard-{index:06d}",
        )

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
