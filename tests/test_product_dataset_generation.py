from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from wakeword_studio.dataset.adapter import DatasetAdapter
from wakeword_studio.dataset.product_plan import build_product_plan
from wakeword_studio.launchers import GenerationRequest, build_generation_command
from wakeword_studio.providers import ProviderRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_product_scale_presets_match_real_formal_design() -> None:
    quick = build_product_plan("快速测试")
    assert quick.total == 12
    assert quick.split_targets == {"train": 4, "validation": 4, "test": 4}
    assert set(quick.targets) == {"positive", "hard_negative", "negative", "ambient"}
    formal = build_product_plan("正式训练")
    assert formal.targets == {
        "positive": 3800,
        "hard_negative": 3420,
        "negative": 6080,
        "ambient": 1900,
    }
    assert formal.split_targets == {"train": 12000, "validation": 1600, "test": 1600}
    assert formal.total == 15200
    assert formal.as_dict()["test_frozen_at_generation"] is True


def test_custom_product_plan_and_command_keep_exact_category_targets(tmp_path: Path) -> None:
    targets = {"positive": 17, "hard_negative": 13, "negative": 23, "ambient": 11}
    plan = build_product_plan("自定义", custom_targets=targets)
    assert plan.targets == targets
    assert sum(plan.split_targets.values()) == sum(targets.values())
    request = GenerationRequest(
        "你好，青小甲", sum(targets.values()), "Kokoro", True,
        tmp_path / "portable-output", scale_mode="自定义", custom_targets=targets,
    )
    command = build_generation_command(PROJECT_ROOT, request)
    import json
    serialized = command[command.index("--targets-json") + 1]
    assert json.loads(serialized) == targets


def _write_source(path: Path, rate: int = 22_050, channels: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(rate // 10, dtype=np.float32) / rate
    mono = 0.1 * np.sin(2 * np.pi * 440 * time)
    audio = np.column_stack((mono, mono * 0.8)) if channels == 2 else mono
    sf.write(path, audio, rate, subtype="PCM_24")


def test_local_folder_age_folders_are_standardized_and_augmented(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_source(source / "child" / "child.wav")
    _write_source(source / "senior" / "negative" / "senior.wav")
    output = tmp_path / "canonical"
    manifest = DatasetAdapter().import_folder(
        source, "新的唤醒词", standardized_root=output, augment=True
    )
    manifest_path = manifest.save(output / "DatasetManifest.json")
    assert manifest.wake_word == "新的唤醒词"
    assert len(manifest.records) == 4
    assert {row.label for row in manifest.records} == {"positive", "negative"}
    assert manifest.summary()["real_age_metadata_records"] == 4
    assert not manifest.validate(manifest_path)
    for record in manifest.records:
        assert record.speaker.age_group in {"child", "senior"}
        assert record.speaker.age_source == "reported"
        with wave.open(str(output / record.audio_path), "rb") as handle:
            assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (16000, 1, 2)
    augmented = [row for row in manifest.records if row.augmentation_id]
    assert len(augmented) == 2
    assert all(row.acoustic.snr_db in {10.0, 15.0, 20.0} for row in augmented)
    assert all(row.acoustic.reverb_id for row in augmented)


def test_local_folder_metadata_csv_accepts_file_age_group(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_source(source / "positive" / "one.wav", rate=16000, channels=1)
    with (source / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "age_group", "speaker_id"])
        writer.writeheader()
        writer.writerow({"file": "positive/one.wav", "age_group": "young", "speaker_id": "human-1"})
    output = tmp_path / "canonical"
    manifest = DatasetAdapter().import_folder(source, "你好，新词", standardized_root=output)
    record = manifest.records[0]
    assert record.speaker.age_group == "young"
    assert record.speaker.age_source == "reported"
    assert record.speaker.speaker_id == "human-1"


def test_provider_capabilities_and_registration_are_data_driven() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/demo/teacher_demo.yaml").read_text(encoding="utf-8"))
    registry = ProviderRegistry.from_config(PROJECT_ROOT, config)
    kokoro = registry.by_name("Kokoro")
    assert kokoro.capabilities.multi_speaker is True
    assert kokoro.capabilities.gender_metadata is True
    assert kokoro.capabilities.age_metadata is False
    local = registry.by_name("本地语音文件夹")
    assert local.capabilities.local_audio_import is True
    config["providers"]["provider_c"] = {
        "display_name": "Provider C",
        "kind": "local_folder",
        "capabilities": {"local_audio_import": True},
    }
    assert ProviderRegistry.from_config(PROJECT_ROOT, config).by_name("Provider C").id == "provider_c"


def test_kokoro_product_command_carries_explicit_quick_plan(tmp_path: Path) -> None:
    request = GenerationRequest(
        "你好，新词", 12, "Kokoro", True, tmp_path / "portable-output",
        scale_mode="快速测试",
    )
    command = build_generation_command(PROJECT_ROOT, request)
    assert command[command.index("--product-mode") + 1] == "quick"
    assert str(tmp_path / "portable-output") in command


def test_local_folder_command_uses_current_python_and_portable_paths(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    request = GenerationRequest(
        "你好，新词", 10, "本地语音文件夹", True, tmp_path / "output",
        scale_mode="快速测试", input_folder=source,
    )
    command = build_generation_command(PROJECT_ROOT, request)
    import sys
    assert Path(command[0]).resolve() == Path(sys.executable).resolve()
    assert str(source.resolve()) in command
    assert str((tmp_path / "output").resolve()) in command
