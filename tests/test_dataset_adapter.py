from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from wakeword_studio.audio import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE_HZ,
    TARGET_SAMPLE_WIDTH_BYTES,
)
from wakeword_studio.dataset.adapter import DatasetAdapter
from wakeword_studio.dataset.manifest import DatasetManifest
from wakeword_studio.frontends import load_inference_audio, load_training_audio


def write_wav(path: Path, sample_rate: int = TARGET_SAMPLE_RATE_HZ, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(960 * channels, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(TARGET_SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_import_folder_and_age_provenance(tmp_path: Path) -> None:
    write_wav(tmp_path / "positive" / "adult.wav")
    write_wav(tmp_path / "hard_negative" / "near.wav")
    with (tmp_path / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "speaker_id", "age_group", "age_source", "acoustic_age_proxy"])
        writer.writeheader()
        writer.writerow({"audio_path": "positive/adult.wav", "speaker_id": "s1", "age_group": "adult", "age_source": "reported", "acoustic_age_proxy": ""})
        writer.writerow({"audio_path": "hard_negative/near.wav", "speaker_id": "s2", "age_group": "", "age_source": "unknown", "acoustic_age_proxy": "pitch_shift_child_like"})
    manifest = DatasetAdapter().import_folder(tmp_path, "你好，青小甲")
    path = manifest.save(tmp_path / "DatasetManifest.json")
    loaded = DatasetManifest.load(path)
    assert not loaded.validate(path)
    assert loaded.summary()["real_age_metadata_records"] == 1
    assert loaded.summary()["acoustic_age_proxy_records"] == 1
    assert {row.label for row in loaded.records} == {"positive", "hard_negative"}
    assert all(row.sample_rate_hz == TARGET_SAMPLE_RATE_HZ for row in loaded.records)


def test_splits_are_stratified_per_label(tmp_path: Path) -> None:
    for label in ("positive", "negative"):
        for index in range(10):
            write_wav(tmp_path / label / f"{index:02d}.wav")
    manifest = DatasetAdapter().import_folder(tmp_path, "你好，青小甲")
    for label in ("positive", "negative"):
        splits = [row.split for row in manifest.records if row.label == label]
        assert splits.count("train") == 8
        assert splits.count("validation") == 1
        assert splits.count("test") == 1


def test_24k_input_is_saved_as_16k_mono_pcm16(tmp_path: Path) -> None:
    source = tmp_path / "source" / "positive" / "source_24k.wav"
    write_wav(source, sample_rate=24_000)
    output_root = tmp_path / "standardized"
    manifest = DatasetAdapter().import_folder(source.parents[1], "你好，青小甲", output_root)
    row = manifest.records[0]
    output = output_root / row.audio_path

    with wave.open(str(source), "rb") as original:
        assert original.getframerate() == 24_000
    with wave.open(str(output), "rb") as standardized:
        assert standardized.getframerate() == TARGET_SAMPLE_RATE_HZ
        assert standardized.getnchannels() == TARGET_CHANNELS
        assert standardized.getsampwidth() == TARGET_SAMPLE_WIDTH_BYTES
        assert row.sample_rate_hz == standardized.getframerate()
    assert row.original_sample_rate_hz == 24_000


def test_48k_stereo_input_is_saved_as_16k_mono(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_wav(source_root / "negative" / "source_48k_stereo.wav", sample_rate=48_000, channels=2)
    output_root = tmp_path / "standardized"
    manifest = DatasetAdapter().import_folder(source_root, "你好，青小甲", output_root)
    row = manifest.records[0]

    with wave.open(str(output_root / row.audio_path), "rb") as standardized:
        assert standardized.getframerate() == TARGET_SAMPLE_RATE_HZ
        assert standardized.getnchannels() == TARGET_CHANNELS
        assert standardized.getsampwidth() == TARGET_SAMPLE_WIDTH_BYTES
        assert row.sample_rate_hz == standardized.getframerate()
    assert row.original_sample_rate_hz == 48_000


def test_manifest_validation_rejects_header_metadata_mismatch(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_wav(source_root / "positive" / "sample.wav", sample_rate=24_000)
    manifest = DatasetAdapter().import_folder(source_root, "你好，青小甲", tmp_path / "standardized")
    manifest.records[0].sample_rate_hz = 24_000
    errors = manifest.validate(check_files=True)
    assert any("manifest sample rate" in error for error in errors)


def test_training_and_inference_frontends_share_audio_contract(tmp_path: Path) -> None:
    source = tmp_path / "source_48k_stereo.wav"
    write_wav(source, sample_rate=48_000, channels=2)
    training_audio = load_training_audio(source)
    inference_audio = load_inference_audio(source)
    np.testing.assert_array_equal(training_audio, inference_audio)
    assert training_audio.dtype == np.float32
    assert len(training_audio) == 320


def test_bad_wav_is_reported_without_discarding_valid_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_wav(source / "positive" / "valid.wav")
    bad = source / "positive" / "broken.wav"
    bad.write_bytes(b"not-a-wav")
    adapter = DatasetAdapter()
    manifest = adapter.import_folder(source, "你好，小智", tmp_path / "standardized")
    assert len(manifest.records) == 1
    assert manifest.records[0].text == "你好，小智"
    assert adapter.last_import_errors[0]["audio_path"] == "positive/broken.wav"
    assert bad.read_bytes() == b"not-a-wav"
