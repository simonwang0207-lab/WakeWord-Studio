from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wakeword_studio.training.multikws_vocabulary import MultiKWSVocabulary


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "multikws" / "teacher_six_formal_12k.json"
VOCABULARY = ROOT / "configs" / "multikws" / "teacher_six_keywords.json"


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_json_retries_transient_windows_replace_with_unique_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("multikws_atomic_json_retry", "phase9/scripts/build_multikws_quick_dataset.py")
    target = tmp_path / "GENERATION_STATUS.json"
    fixed_legacy_temp = tmp_path / "GENERATION_STATUS.json.tmp"
    target.write_text('{"generation": 1}', encoding="utf-8")
    fixed_legacy_temp.write_text("legacy residue", encoding="utf-8")
    real_replace = module.os.replace
    calls: list[tuple[Path, Path]] = []

    def briefly_locked(source, destination):
        calls.append((Path(source), Path(destination)))
        if len(calls) <= 3:
            raise PermissionError(13, "simulated Windows target lock")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", briefly_locked)
    module.atomic_json(
        target, {"generation": 2}, attempts=10,
        initial_delay_s=0.0, max_delay_s=0.0,
    )

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 2}
    assert len(calls) == 4
    assert all(source.name.startswith("GENERATION_STATUS.json.") for source, _ in calls)
    assert all(source.name.endswith(".tmp") for source, _ in calls)
    assert fixed_legacy_temp.read_text(encoding="utf-8") == "legacy residue"


def test_atomic_json_exhaustion_preserves_status_and_writes_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("multikws_atomic_json_fallback", "phase9/scripts/build_multikws_quick_dataset.py")
    target = tmp_path / "GENERATION_STATUS.json"
    target.write_text('{"generation": "last-valid"}', encoding="utf-8")
    real_replace = module.os.replace

    def locked_status_only(source, destination):
        if Path(destination) == target:
            raise PermissionError(13, "simulated persistent Windows target lock")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", locked_status_only)
    with pytest.raises(module.AtomicJsonWriteError):
        module.atomic_json(
            target, {"generation": "new"}, attempts=10,
            initial_delay_s=0.0, max_delay_s=0.0,
        )

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": "last-valid"}
    fallback = json.loads(
        (tmp_path / "GENERATION_STATUS_WRITE_ERROR.json").read_text(encoding="utf-8")
    )
    assert fallback["target"] == str(target)
    assert fallback["error_type"] == "PermissionError"


def test_status_reader_retries_and_closes_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("multikws_status_reader", "phase9/scripts/build_multikws_quick_dataset.py")
    target = tmp_path / "GENERATION_STATUS.json"
    target.write_text('{"status": "IN_PROGRESS"}', encoding="utf-8")
    real_open = Path.open
    attempts = 0

    def briefly_locked(self, *args, **kwargs):
        nonlocal attempts
        if self == target and attempts < 2:
            attempts += 1
            raise PermissionError(13, "simulated Windows read lock")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", briefly_locked)
    value = module.read_json_with_retry(
        target, attempts=10, initial_delay_s=0.0, max_delay_s=0.0,
    )
    renamed = target.with_name("status-renamed.json")
    target.replace(renamed)

    assert value == {"status": "IN_PROGRESS"}
    assert attempts == 2
    assert renamed.is_file()


def test_multikws_predict_in_batches_preserves_order_and_batch_limit() -> None:
    from wakeword_studio.training.multikws_trainer import predict_in_batches

    values = np.arange(1500 * 3, dtype=np.float32).reshape(1500, 3)
    observed_batch_sizes: list[int] = []

    class OrderedProbe:
        def __call__(self, batch, *, training):
            assert training is False
            observed_batch_sizes.append(len(batch))
            return np.stack((batch[:, 0], batch[:, 2]), axis=1)

    output = predict_in_batches(OrderedProbe(), values, batch_size=32)

    assert output.shape == (1500, 2)
    assert np.array_equal(output[:, 0], values[:, 0])
    assert np.array_equal(output[:, 1], values[:, 2])
    assert max(observed_batch_sizes) == 32
    assert observed_batch_sizes[-1] == 28
    assert len(observed_batch_sizes) == 47


def test_formal_12k_planner_exact_counts_balance_groups_and_schedule() -> None:
    module = _module("multikws_12k_builder", "phase9/scripts/build_multikws_12k_dataset.py")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    vocabulary = MultiKWSVocabulary.load(VOCABULARY)
    effective = module.resolve_12k_config(config, vocabulary)
    bases = module.build_base_jobs(vocabulary, effective)
    jobs = module.build_effective_jobs(bases, effective)

    assert len(bases) == 5400
    assert len(jobs) == 12000
    assert {split: sum(job["split"] == split for job in jobs) for split in module.SPLITS} == {
        "train": 9000, "validation": 1500, "test": 1500,
    }
    assert {source: sum(base["speech_source"] == source for base in bases)
            for source in ("kokoro", "voxcpm15")} == {"kokoro": 2700, "voxcpm15": 2700}
    for keyword in vocabulary.keywords:
        for split, per_source in {"train": 150, "validation": 75, "test": 75}.items():
            for source in ("kokoro", "voxcpm15"):
                assert sum(base["keyword_id"] == keyword.keyword_id and base["split"] == split
                           and base["speech_source"] == source for base in bases) == per_source
        assert {split: sum(job["keyword_id"] == keyword.keyword_id and job["split"] == split
                           for job in jobs) for split in module.SPLITS} == {
            "train": 900, "validation": 150, "test": 150,
        }

    group_splits: dict[str, set[str]] = {}
    for job in jobs:
        group_splits.setdefault(job["base_sample_id"], set()).add(job["split"])
    assert sum(len(splits) > 1 for splits in group_splits.values()) == 0
    assert effective["training_schedule"] == {
        "batch_size": 32, "steps_per_epoch": 282, "max_epochs": 30,
        "derived_max_steps": 8460, "validation_interval_steps": 282,
        "early_stopping_patience_validations": 6,
    }
    assert effective["ordinary_background_unique_text_count"] == 48
    assert effective["hard_negative_unique_text_count"] == 36


def test_group_safe_generation_resume_reuses_base_and_variants_are_distinct(tmp_path: Path) -> None:
    module = _module("multikws_12k_resume", "phase9/scripts/build_multikws_12k_dataset.py")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["dataset"].update({
        "dataset_id": "multikws_12k_group_resume_smoke",
        "manifest_profile": "group_resume_smoke",
        "experiment_stage": "pipeline_regression_smoke",
        "base_counts": {
            "wakeword_per_keyword": {"train": 2, "validation": 2, "test": 2},
            "ordinary_background": {"train": 2, "validation": 2, "test": 2},
            "hard_negative": {"train": 2, "validation": 2, "test": 2},
        },
        "ambient_effective_counts": {"train": 1, "validation": 1, "test": 1},
    })
    config["training"]["effective_train_samples"] = 49
    config_path = tmp_path / "smoke_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    output = tmp_path / "dataset"
    calls: list[str] = []

    def fake(job, _text, _voice, _speed):
        calls.append(job["base_sample_id"])
        frequency = 180 + job["index"] % 30 + (50 if job["speech_source"] == "voxcpm15" else 0)
        timeline = np.arange(8000, dtype=np.float32) / 16000.0
        return (0.08 * np.sin(2 * np.pi * frequency * timeline)).astype(np.float32), {
            "provider": job["speech_source"]
        }

    synth = {"kokoro": fake, "voxcpm15": fake}
    with pytest.raises(KeyboardInterrupt):
        module.run_generation(
            config_path, output_root=output, synthesizers=synth, stop_after_base=5
        )
    interrupted = json.loads((output / "GENERATION_STATUS.json").read_text(encoding="utf-8"))
    assert interrupted["completed_base_speech"] == 5
    assert interrupted["completed_effective_samples"] == 0

    info = module.run_generation(config_path, output_root=output, synthesizers=synth, resume=True)
    manifest = json.loads((output / "DatasetManifest.json").read_text(encoding="utf-8"))
    status = json.loads((output / "GENERATION_STATUS.json").read_text(encoding="utf-8"))
    assert len(calls) == len(set(calls)) == 48
    assert status["planned_base_speech"] == status["completed_base_speech"] == 48
    assert status["planned_effective_samples"] == status["completed_effective_samples"] == 83
    assert info["BASE_GROUP_SPLIT_LEAKAGE"] == manifest["base_group_split_leakage"] == 0
    assert manifest["base_source_counts"] == {"kokoro": 24, "voxcpm15": 24}
    groups: dict[str, list[dict[str, object]]] = {}
    for record in manifest["records"]:
        groups.setdefault(record["base_sample_id"], []).append(record)
    for base_id, siblings in groups.items():
        assert len({row["split"] for row in siblings}) == 1
        split = siblings[0]["split"]
        assert len(siblings) == (3 if split == "train" and not base_id.startswith("ambient-") else 1)
        if len(siblings) == 3:
            assert len({row["sha256"] for row in siblings}) == 3
            assert len({json.dumps(row["augmentation_parameters"], sort_keys=True)
                        for row in siblings}) == 3
    assert manifest["test_read_during_build"] is False


def test_formal_architecture_parameter_counts_and_int8_contract() -> None:
    tf = pytest.importorskip("tensorflow")
    from wakeword_studio.training.multikws_models import build_multikws_model

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    counts = {}
    for name in ("bcresnet", "convmixer"):
        model = build_multikws_model(name, (99, 40), 7, config["models"][name])
        counts[name] = int(model.count_params())
        assert model(tf.zeros((1, 99, 40))).shape.as_list() == [1, 7]
    assert counts == {"bcresnet": 23207, "convmixer": 27031}
    assert config["quantization"] == {
        "method": "full_int8_ptq", "representative_split": "train",
        "input_type": "int8", "output_type": "int8",
        "input_shape": [1, 99, 40], "output_shape": [1, 7],
    }


def test_training_ctrl_c_resume_restores_absolute_step_and_optimizer(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    from wakeword_studio.training.multikws_trainer import train_multikws

    features = ROOT / "phase9" / "artifacts" / "multisource_pipeline_smoke" / "train_validation_features.npz"
    if not features.is_file():
        pytest.skip("bounded Phase 9 feature smoke artifact is unavailable")
    run_dir = tmp_path / "resume_run"
    common = {
        "model_name": "bcresnet",
        "vocabulary_path": VOCABULARY,
        "feature_store_path": features,
        "run_dir": run_dir,
        "require_gpu": False,
        "seed": 20260901,
        "run_mode": "smoke",
        "validation_interval": 1,
        "architecture_config": {
            "channels": 4, "depth": 1, "subbands": 4,
            "temporal_dilations": [1], "dropout": 0.0, "activation": "relu",
        },
        "batch_size": 8,
    }
    with pytest.raises(KeyboardInterrupt):
        train_multikws(**common, smoke_steps=2, interrupt_after_step=1)
    interrupted = json.loads((run_dir / "TRAINING_STATE.json").read_text(encoding="utf-8"))
    assert interrupted["current_step"] == 1
    assert interrupted["next_absolute_step"] == 1
    assert interrupted["optimizer_state_checkpointed"] is True
    assert interrupted["interrupted"] is True

    report = train_multikws(**common, smoke_steps=2, resume=True)
    assert report["completed_steps"] == 2
    assert report["optimizer_iterations"] == 2
    assert report["sampler"]["resume_order_reproducible_from_absolute_step"] is True
    resumed = json.loads((run_dir / "TRAINING_STATE.json").read_text(encoding="utf-8"))
    assert resumed["current_step"] == 2
    assert resumed["next_absolute_step"] == 2
    assert resumed["interrupted"] is False
