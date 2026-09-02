from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from wakeword_studio.audio import TARGET_SAMPLE_RATE_HZ, TARGET_SAMPLE_WIDTH_BYTES
from wakeword_studio.dataset.formal_builder import FormalDatasetBuilder


def write_tone(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(3200, dtype=np.float32) / TARGET_SAMPLE_RATE_HZ
    samples = np.asarray(np.sin(2 * np.pi * frequency * time) * 2000, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(TARGET_SAMPLE_WIDTH_BYTES)
        handle.setframerate(TARGET_SAMPLE_RATE_HZ)
        handle.writeframes(samples.tobytes())


def test_formal_builder_exact_counts_canonical_audio_and_no_group_leakage(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_records = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label_index, label in enumerate(("positive", "negative", "hard_negative")):
            record_id = f"source-{split}-{label}"
            relative = Path(split) / label / f"{record_id}.wav"
            write_tone(source_root / relative, 180 + split_index * 40 + label_index * 20)
            source_records.append(
                {
                    "record_id": record_id,
                    "path": relative.as_posix(),
                    "label": label,
                    "text": "你好，青小甲" if label == "positive" else "测试文本",
                    "split": split,
                    "speaker_id": f"speaker-{split}",
                    "source_family": "test_tts",
                    "source_group_id": f"test_tts:speaker-{split}",
                    "source_utterance_id": record_id,
                    "gender": None,
                    "age_group": None,
                    "age_source": "unknown",
                    "speed": 1.0,
                    "hard_negative_tier": 1 if label == "hard_negative" else None,
                }
            )
    source_manifest = source_root / "source_manifest.json"
    source_manifest.write_text(
        json.dumps({"root": str(source_root), "records": source_records}, ensure_ascii=False),
        encoding="utf-8",
    )
    config = {
        "project": "test",
        "wake_word": "你好，青小甲",
        "seed": 7,
        "targets": {"positive": 8, "negative": 12, "hard_negative": 8, "ambient": 4},
        "split_ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
        "mean_duration_seconds": {
            "positive": 0.2,
            "negative": 0.2,
            "hard_negative": 0.2,
            "ambient": 0.2,
        },
        "noise_categories": ["clean", "room", "tv_speech", "music"],
        "snr_db": [20, 5],
        "age_proxies": [None, "higher_pitch_and_faster", "lower_pitch_and_slower"],
    }
    output_root = tmp_path / "formal"
    manifest = FormalDatasetBuilder(config, [source_manifest]).build(output_root)
    manifest_path = output_root / "DatasetManifest.json"

    assert not manifest.validate(manifest_path)
    assert manifest.summary()["labels"] == {
        "positive": 8,
        "negative": 12,
        "hard_negative": 8,
        "ambient": 4,
    }
    assert manifest.summary()["splits"] == {"train": 16, "validation": 8, "test": 8}
    groups: dict[str, set[str]] = {}
    for record in manifest.records:
        groups.setdefault(record.source_group_id or "", set()).add(record.split)
        with wave.open(str(output_root / record.audio_path), "rb") as handle:
            assert handle.getframerate() == TARGET_SAMPLE_RATE_HZ
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == TARGET_SAMPLE_WIDTH_BYTES
    assert all(len(splits) == 1 for splits in groups.values())
    assert not list(output_root.rglob("*.partial.wav"))

    before_hashes = {row.record_id: row.sha256 for row in manifest.records}
    hard_source = next(row for row in source_records if row["label"] == "hard_negative")
    write_tone(source_root / hard_source["path"], 777.0)
    hard_source["revision"] = 2
    source_manifest.write_text(
        json.dumps({"root": str(source_root), "records": source_records}, ensure_ascii=False),
        encoding="utf-8",
    )
    rebuilt = FormalDatasetBuilder(config, [source_manifest]).build(
        output_root, rebuild_labels={"hard_negative"}
    )
    after_hashes = {row.record_id: row.sha256 for row in rebuilt.records}
    assert any(
        after_hashes[record_id] != old_hash
        for record_id, old_hash in before_hashes.items()
        if "-hard_negative-" in record_id
    )
    assert all(
        after_hashes[record_id] == old_hash
        for record_id, old_hash in before_hashes.items()
        if "-hard_negative-" not in record_id
    )
    assert not rebuilt.validate(manifest_path)
