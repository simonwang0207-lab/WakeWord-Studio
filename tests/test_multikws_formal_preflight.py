from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wakeword_studio.training.multikws_sampler import DeterministicEpochSampler
from wakeword_studio.training.multikws_vocabulary import MultiKWSVocabulary


ROOT = Path(__file__).resolve().parents[1]
VOCABULARY_PATH = ROOT / "configs" / "multikws" / "teacher_six_keywords.json"
FORMAL_CONFIG = ROOT / "configs" / "multikws" / "teacher_six_formal_candidate.json"
QUICK_CONFIG = ROOT / "configs" / "multikws" / "teacher_six_quick.json"


def _generation_module():
    path = ROOT / "phase9" / "scripts" / "build_multikws_quick_dataset.py"
    spec = importlib.util.spec_from_file_location("phase9_multikws_generation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_plan_has_config_driven_identity_and_exact_4800_counts() -> None:
    module = _generation_module()
    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    vocabulary = MultiKWSVocabulary.load(VOCABULARY_PATH)
    effective = module.resolve_effective_config(config, vocabulary)
    jobs = module.build_jobs(vocabulary, effective)
    assert effective["dataset_id"] == "teacher_six_multikws_v1_formal"
    assert effective["profile"] == "formal_candidate"
    assert effective["experiment_stage"] == "first_formal_baseline"
    assert effective["production_quality"] is False
    assert len(jobs) == 4800
    assert {split: sum(job["split"] == split for job in jobs) for split in module.SPLITS} == {
        "train": 2400,
        "validation": 1200,
        "test": 1200,
    }
    for keyword in vocabulary.keywords:
        assert {split: sum(job["split"] == split and job["keyword_id"] == keyword.keyword_id for job in jobs) for split in module.SPLITS} == {
            "train": 200,
            "validation": 100,
            "test": 100,
        }
        for split, count in {"train": 100, "validation": 50, "test": 50}.items():
            assert sum(job["split"] == split and job["keyword_id"] == keyword.keyword_id
                       and job["speech_source"] == "kokoro" for job in jobs) == count
            assert sum(job["split"] == split and job["keyword_id"] == keyword.keyword_id
                       and job["speech_source"] == "voxcpm15" for job in jobs) == count
    assert sum(job["speech_source"] == "kokoro" for job in jobs) == 2100
    assert sum(job["speech_source"] == "voxcpm15" for job in jobs) == 2100
    assert sum(job["speech_source"] is None for job in jobs) == 600
    for kind, per_source in {"background_speech": 600, "hard_negative": 300}.items():
        assert sum(job["background_kind"] == kind and job["speech_source"] == "kokoro"
                   for job in jobs) == per_source
        assert sum(job["background_kind"] == kind and job["speech_source"] == "voxcpm15"
                   for job in jobs) == per_source
    assert effective["speech_source_mix"] == {"kokoro": 0.5, "voxcpm15": 0.5}
    for source in ("kokoro", "voxcpm15"):
        splits = effective["speaker_reference_splits"][source]
        assert not set(splits["train"]) & set(splits["validation"])
        assert not set(splits["train"]) & set(splits["test"])
        assert not set(splits["validation"]) & set(splits["test"])


def test_quick_identity_is_also_config_driven() -> None:
    module = _generation_module()
    config = json.loads(QUICK_CONFIG.read_text(encoding="utf-8"))
    effective = module.resolve_effective_config(
        config, MultiKWSVocabulary.load(VOCABULARY_PATH)
    )
    assert effective["dataset_id"] == "teacher_six_multikws_v1_quick"
    assert effective["profile"] == "quick_smoke"
    assert effective["experiment_stage"] == "regression_smoke"


def test_deterministic_epoch_sampler_exact_coverage_and_resume_order() -> None:
    sampler = DeterministicEpochSampler(2400, 32, 20260901, drop_last=False)
    assert sampler.steps_per_epoch == 75
    assert sampler.first_epoch_audit() == {
        "unique_samples": 2400,
        "missing_samples": 0,
        "duplicate_samples": 0,
    }
    first = np.concatenate([sampler.batch_indices(step) for step in range(75)])
    second = np.concatenate([sampler.batch_indices(step) for step in range(75, 150)])
    assert not np.array_equal(first, second)
    resumed = DeterministicEpochSampler(2400, 32, 20260901, drop_last=False)
    for absolute_step in (0, 74, 75, 99, 100, 1999):
        assert np.array_equal(sampler.batch_indices(absolute_step), resumed.batch_indices(absolute_step))


def test_configured_snr_and_interrupted_generation_resume(tmp_path: Path) -> None:
    module = _generation_module()
    config = json.loads(QUICK_CONFIG.read_text(encoding="utf-8"))
    config["dataset"]["dataset_id"] = "temporary_formal_like"
    config["dataset"]["manifest_profile"] = "formal_like_regression"
    config["dataset"]["experiment_stage"] = "regression_smoke"
    config["dataset"]["augmentation"]["snr_db_values"] = [7]
    config_path = tmp_path / "temporary_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    output = tmp_path / "generated"

    def fake_synthesizer(text: str, voice: str, speed: float) -> np.ndarray:
        del text, voice, speed
        return np.full(1600, 0.05, dtype=np.float32)

    with pytest.raises(KeyboardInterrupt):
        module.run_generation(
            config_path, output_root=output, synthesizer=fake_synthesizer, stop_after=3
        )
    interrupted = json.loads((output / "GENERATION_STATUS.json").read_text(encoding="utf-8"))
    assert interrupted["status"] == "INTERRUPTED"
    assert interrupted["completed_samples"] == 3
    assert not (output / "DatasetManifest.json").exists()

    info = module.run_generation(
        config_path, output_root=output, synthesizer=fake_synthesizer, resume=True
    )
    completed = json.loads((output / "GENERATION_STATUS.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "DatasetManifest.json").read_text(encoding="utf-8"))
    assert completed["status"] == "COMPLETED"
    assert completed["completed_samples"] == completed["planned_samples"] == 40
    assert completed["failed_samples"] == 0
    assert info["record_count"] == 40
    assert manifest["dataset_id"] == "temporary_formal_like"
    assert manifest["profile"] == "formal_like_regression"
    assert manifest["effective_config"]["augmentation"]["snr_db_values"] == [7]
    assert {record["acoustic"]["snr_db"] for record in manifest["records"]} == {7.0}
    ids = [record["sample_id"] for record in manifest["records"]]
    assert len(ids) == len(set(ids)) == 40
    assert manifest["test_frozen"] is True
    assert manifest["test_read_during_build"] is False
    stored_manifest_sha = manifest.pop("manifest_sha256")
    assert module.canonical_sha256(manifest) == stored_manifest_sha


def test_partial_generation_requires_explicit_resume(tmp_path: Path) -> None:
    module = _generation_module()
    config = json.loads(QUICK_CONFIG.read_text(encoding="utf-8"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "generated"

    def fake_synthesizer(_text: str, _voice: str, _speed: float) -> np.ndarray:
        return np.zeros(800, dtype=np.float32)

    with pytest.raises(KeyboardInterrupt):
        module.run_generation(config_path, output_root=output, synthesizer=fake_synthesizer, stop_after=1)
    with pytest.raises(RuntimeError, match="--resume"):
        module.run_generation(config_path, output_root=output, synthesizer=fake_synthesizer)


def test_multisource_manifest_records_balancing_and_metadata(tmp_path: Path) -> None:
    module = _generation_module()
    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    config["dataset"].update({
        "profile": "quick", "dataset_id": "multisource_unit_smoke",
        "manifest_profile": "multisource_unit_smoke",
        "experiment_stage": "regression_smoke",
        "positive_per_keyword": 6, "background_speech_count": 6,
        "hard_negative_count": 6, "ambient_count": 3,
        "split_weights": {"train": 1, "validation": 1, "test": 1},
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    def fake(job, _text, _voice, _speed):
        return np.full(800, 0.03, dtype=np.float32), {"provider_record": job["sample_id"]}

    module.run_generation(
        config_path, output_root=tmp_path / "dataset",
        synthesizers={"kokoro": fake, "voxcpm15": fake},
    )
    manifest = json.loads((tmp_path / "dataset" / "DatasetManifest.json").read_text(encoding="utf-8"))
    assert manifest["speech_sources"] == ["kokoro", "voxcpm15"]
    assert manifest["source_counts"] == {
        "kokoro": 24, "voxcpm15": 24, "procedural_ambient": 3,
    }
    assert manifest["age_metadata_verified"] is False
    for keyword in MultiKWSVocabulary.load(VOCABULARY_PATH).keywords:
        assert manifest["per_keyword_source_counts"][keyword.keyword_id] == {
            split: {"kokoro": 1, "voxcpm15": 1} for split in module.SPLITS
        }
