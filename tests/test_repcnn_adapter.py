from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wakeword_studio.dataset.repcnn_adapter import RepCNNDatasetAdapter


def write_manifest(tmp_path: Path) -> tuple[Path, str]:
    records = []
    for split in ("train", "validation", "test"):
        for label in ("positive", "negative", "hard_negative", "ambient"):
            positive = label == "positive"
            records.append(
                {
                    "record_id": f"{split}-{label}",
                    "audio_path": f"{split}/{label}.wav",
                    "label": label,
                    "split": split,
                    "text": "wake" if positive else None,
                    "speaker": {
                        "speaker_id": f"speaker-{label}",
                        "source": "speech" if label != "ambient" else "room",
                    },
                    "acoustic": {
                        "snr_db": 10.0,
                        "phrase_start_ms": 100.0 if positive else None,
                        "phrase_end_ms": 900.0 if positive else None,
                    },
                    "sample_rate_hz": 16000,
                    "duration_seconds": 1.5,
                }
            )
    path = tmp_path / "DatasetManifest.json"
    path.write_text(
        json.dumps(
            {
                "wake_word": "wake",
                "records": records,
                "source_kind": "mixed",
                "root": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_adapter_preserves_metadata_and_maps_all_negative_labels(tmp_path: Path) -> None:
    path, digest = write_manifest(tmp_path)
    adapter = RepCNNDatasetAdapter(path, expected_sha256=digest, seed=7)
    positive = adapter.samples("train", "positive")[0]
    assert positive.target == 1.0
    assert positive.speaker_id == "speaker-positive"
    assert positive.source == "speech"
    assert positive.text == "wake"
    assert positive.snr_db == 10.0
    assert positive.phrase_start_ms == 100.0
    assert positive.phrase_end_ms == 900.0
    for label in ("negative", "hard_negative", "ambient"):
        assert adapter.samples("train", label)[0].target == 0.0


def test_adapter_never_exposes_test(tmp_path: Path) -> None:
    path, digest = write_manifest(tmp_path)
    adapter = RepCNNDatasetAdapter(path, expected_sha256=digest, seed=7)
    assert adapter.test_loaded is False
    assert all(not key.startswith("test:") for key in adapter.counts())
    with pytest.raises(ValueError, match="not available"):
        adapter.samples("test", "positive")


def test_only_train_and_validation_are_allowed(tmp_path: Path) -> None:
    path, digest = write_manifest(tmp_path)
    with pytest.raises(ValueError, match="only train and validation"):
        RepCNNDatasetAdapter(
            path,
            expected_sha256=digest,
            seed=7,
            allowed_splits=("train", "test"),
        )


def test_manifest_hash_is_frozen(tmp_path: Path) -> None:
    path, digest = write_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        RepCNNDatasetAdapter(path, expected_sha256="0" * len(digest), seed=7)
