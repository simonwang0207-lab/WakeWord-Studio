"""Group-safe base-to-variant builder for Phase 9 Formal Multi-KWS v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from build_multikws_quick_dataset import (  # noqa: E402
    AtomicJsonWriteError,
    SPLITS,
    TARGET_SAMPLE_RATE_HZ,
    VoxCPMWorkerClient,
    _kokoro_synthesizer,
    allocate_weighted,
    atomic_json,
    augment_audio,
    canonical_sha256,
    procedural_noise,
    read_json_with_retry,
    resolve_augmentation,
    sha256_file,
    utc_now,
)
from wakeword_studio.training.multikws_vocabulary import MultiKWSVocabulary  # noqa: E402


def stable_seed(*parts: object) -> int:
    value = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _disjoint(values: dict[str, list[str]]) -> bool:
    return all(
        not set(values[left]) & set(values[right])
        for left in SPLITS for right in SPLITS if left < right
    )


def resolve_12k_config(config: dict[str, Any], vocabulary: MultiKWSVocabulary) -> dict[str, Any]:
    dataset = dict(config["dataset"])
    if dataset.get("production_quality") is not False:
        raise ValueError("Formal experimental baseline must set production_quality=false")
    base_counts = {
        kind: {split: int(value[split]) for split in SPLITS}
        for kind, value in dataset["base_counts"].items()
    }
    if set(base_counts) != {"wakeword_per_keyword", "ordinary_background", "hard_negative"}:
        raise ValueError("base_counts must define wakeword_per_keyword/ordinary_background/hard_negative")
    variants = {split: int(dataset["variants_per_base"][split]) for split in SPLITS}
    if variants != {"train": 3, "validation": 1, "test": 1}:
        raise ValueError("Formal v2 requires variants_per_base train=3 validation=1 test=1")
    ambient = {split: int(dataset["ambient_effective_counts"][split]) for split in SPLITS}
    source_mix = {str(name): float(value) for name, value in dataset["speech_source_mix"].items()}
    sources = {str(name): dict(value) for name, value in dataset["speech_sources"].items()}
    if set(source_mix) != set(sources) or set(source_mix) != {"kokoro", "voxcpm15"}:
        raise ValueError("Formal v2 speech sources must be kokoro and voxcpm15")
    speaker_splits: dict[str, dict[str, list[str]]] = {}
    for source_name, source in sources.items():
        key = "reference_speaker_splits" if source_name == "voxcpm15" else "speaker_splits"
        values = {split: [str(item) for item in source[key][split]] for split in SPLITS}
        if any(not values[split] for split in SPLITS) or not _disjoint(values):
            raise ValueError(f"{source_name} speaker/reference splits must be nonempty and disjoint")
        speaker_splits[source_name] = values
    ordinary = [str(value) for value in dataset["ordinary_background_phrases"]]
    hard = [str(value) for value in dataset["hard_negative_phrases"]]
    wake_texts = {keyword.display_name for keyword in vocabulary.keywords}
    if len(set(ordinary)) != len(ordinary) or len(set(hard)) != len(hard):
        raise ValueError("Phrase banks must not contain duplicates")
    if len(set(hard)) < 18:
        raise ValueError("hard-negative phrase inventory must contain at least 18 unique texts")
    if wake_texts & (set(ordinary) | set(hard)):
        raise ValueError("Formal wake words must never be background/hard-negative texts")
    effective_counts = {
        split: (
            len(vocabulary.keywords) * base_counts["wakeword_per_keyword"][split] * variants[split]
            + base_counts["ordinary_background"][split] * variants[split]
            + base_counts["hard_negative"][split] * variants[split]
            + ambient[split]
        ) for split in SPLITS
    }
    planned_base_speech = (
        len(vocabulary.keywords) * sum(base_counts["wakeword_per_keyword"].values())
        + sum(base_counts["ordinary_background"].values())
        + sum(base_counts["hard_negative"].values())
    )
    training = dict(config["training"])
    if int(training["effective_train_samples"]) != effective_counts["train"]:
        raise ValueError("training.effective_train_samples does not match the dataset plan")
    batch_size = int(training["batch_size"])
    steps_per_epoch = math.ceil(effective_counts["train"] / batch_size)
    schedule = {
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "max_epochs": int(training["max_epochs"]),
        "derived_max_steps": steps_per_epoch * int(training["max_epochs"]),
        "validation_interval_steps": steps_per_epoch * int(training["validation_every_epochs"]),
        "early_stopping_patience_validations": int(training["early_stopping_patience_epochs"]),
    }
    return {
        "dataset_id": str(dataset["dataset_id"]),
        "profile": str(dataset["manifest_profile"]),
        "experiment_stage": str(dataset["experiment_stage"]),
        "production_quality": False,
        "seed": int(config["seed"]),
        "vocabulary_path": str(config["vocabulary"]),
        "vocabulary_id": vocabulary.vocabulary_id,
        "num_classes": vocabulary.num_classes,
        "class_names": list(vocabulary.class_names),
        "base_counts": base_counts,
        "ambient_effective_counts": ambient,
        "variants_per_base": variants,
        "effective_counts": effective_counts,
        "planned_effective_samples": sum(effective_counts.values()),
        "planned_base_speech": planned_base_speech,
        "speech_source_mix": source_mix,
        "speech_sources": sources,
        "speaker_reference_splits": speaker_splits,
        "ordinary_background_phrases": ordinary,
        "hard_negative_phrases": hard,
        "ordinary_background_unique_text_count": len(set(ordinary)),
        "hard_negative_unique_text_count": len(set(hard)),
        "train_augmentation": resolve_augmentation({"augmentation": dataset["train_augmentation"]}),
        "evaluation_augmentation": resolve_augmentation({"augmentation": dataset["evaluation_augmentation"]}),
        "audio": {
            "sample_rate_hz": int(dataset["sample_rate_hz"]),
            "channels": int(dataset["channels"]),
            "subtype": str(dataset["subtype"]),
        },
        "age_metadata_policy": str(dataset["age_metadata_policy"]),
        "training_schedule": schedule,
    }


def _source_assignments(count: int, mix: dict[str, float]) -> list[str]:
    counts = allocate_weighted(count, mix)
    remaining = dict(counts)
    output: list[str] = []
    while len(output) < count:
        for name in mix:
            if remaining[name]:
                output.append(name)
                remaining[name] -= 1
    return output


def build_base_jobs(vocabulary: MultiKWSVocabulary, effective: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def add_group(split: str, keyword_id: str, class_index: int, kind: str,
                  count: int, texts: list[str]) -> None:
        source_seen = {name: 0 for name in effective["speech_source_mix"]}
        for ordinal, source in enumerate(_source_assignments(count, effective["speech_source_mix"])):
            speaker_pool = effective["speaker_reference_splits"][source][split]
            speaker = speaker_pool[source_seen[source] % len(speaker_pool)]
            source_seen[source] += 1
            base_sample_id = f"base-{split}-{kind}-{keyword_id}-{ordinal:06d}"
            jobs.append({
                "index": len(jobs), "base_sample_id": base_sample_id,
                "sample_id": base_sample_id, "split": split,
                "keyword_id": keyword_id, "class_index": class_index,
                "background_kind": None if class_index else kind,
                "text": texts[ordinal % len(texts)], "speech_source": source,
                "speaker_id": speaker,
                "reference_speaker_id": speaker if source == "voxcpm15" else None,
                "relative_path": (Path("_base_audio") / split / keyword_id / f"{base_sample_id}.wav").as_posix(),
            })

    for split in SPLITS:
        for keyword in vocabulary.keywords:
            add_group(
                split, keyword.keyword_id, keyword.class_index, "positive",
                effective["base_counts"]["wakeword_per_keyword"][split],
                [keyword.display_name],
            )
        add_group(
            split, "background", 0, "ordinary_background",
            effective["base_counts"]["ordinary_background"][split],
            effective["ordinary_background_phrases"],
        )
        add_group(
            split, "background", 0, "hard_negative",
            effective["base_counts"]["hard_negative"][split],
            effective["hard_negative_phrases"],
        )
    return jobs


def build_effective_jobs(base_jobs: list[dict[str, Any]], effective: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for base in base_jobs:
        for variant_id in range(effective["variants_per_base"][base["split"]]):
            sample_id = f"{base['base_sample_id']}-v{variant_id}"
            jobs.append({
                **base, "index": len(jobs), "sample_id": sample_id,
                "variant_id": variant_id,
                "relative_path": (Path(base["split"]) / base["keyword_id"] / f"{sample_id}.wav").as_posix(),
            })
    for split in SPLITS:
        for ordinal in range(effective["ambient_effective_counts"][split]):
            base_id = f"ambient-{split}-{ordinal:06d}"
            jobs.append({
                "index": len(jobs), "sample_id": f"{base_id}-v0", "base_sample_id": base_id,
                "variant_id": 0, "split": split, "keyword_id": "background",
                "class_index": 0, "background_kind": "ambient", "text": None,
                "speech_source": None, "speaker_id": None, "reference_speaker_id": None,
                "relative_path": (Path(split) / "background" / f"{base_id}-v0.wav").as_posix(),
            })
    return jobs


def planner_report(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    vocabulary = MultiKWSVocabulary.load((PROJECT_ROOT / config["vocabulary"]).resolve())
    effective = resolve_12k_config(config, vocabulary)
    base_jobs = build_base_jobs(vocabulary, effective)
    jobs = build_effective_jobs(base_jobs, effective)
    base_source_counts = {
        name: sum(job["speech_source"] == name for job in base_jobs)
        for name in effective["speech_source_mix"]
    }
    return {
        "dataset_id": effective["dataset_id"],
        "planned_base_speech": len(base_jobs),
        "base_source_counts": base_source_counts,
        "ambient_total": sum(effective["ambient_effective_counts"].values()),
        "effective_counts": {split: sum(job["split"] == split for job in jobs) for split in SPLITS},
        "planned_effective_samples": len(jobs),
        "training_schedule": effective["training_schedule"],
        "ordinary_background_unique_text_count": effective["ordinary_background_unique_text_count"],
        "hard_negative_unique_text_count": effective["hard_negative_unique_text_count"],
    }


def _load_records(
    path: Path, root: Path, jobs: list[dict[str, Any]], *, base: bool,
) -> dict[str, dict[str, Any]]:
    """Recover complete deterministic records without opening frozen WAV data."""

    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    id_key = "base_sample_id" if base else "sample_id"
    expected_paths = {str(job[id_key]): str(job["relative_path"]) for job in jobs}
    required = {
        "record_id", id_key, "path", "split", "class_index", "keyword_id",
        "speech_source", "sample_rate_hz", "channels", "subtype", "sha256",
    }
    if base:
        required.add("speaker_id")
    else:
        required.update({"base_sample_id", "variant_id", "augmentation_parameters"})
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A process interruption can leave only its final JSONL line
                # incomplete. Its deterministic ID will be rebuilt normally.
                continue
            record_id = str(record.get("record_id", ""))
            if record_id not in expected_paths or not required.issubset(record):
                continue
            if str(record.get(id_key, "")) != record_id:
                continue
            if str(record.get("path", "")) != expected_paths[record_id]:
                continue
            audio_path = root / expected_paths[record_id]
            try:
                complete_file = audio_path.is_file() and audio_path.stat().st_size > 44
            except OSError:
                complete_file = False
            if complete_file:
                records[record_id] = record
    return records


def _speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if factor == 1.0 or len(signal) < 2:
        return signal
    output_length = max(1, int(round(len(signal) / factor)))
    positions = np.linspace(0.0, len(signal) - 1, output_length)
    return np.interp(positions, np.arange(len(signal)), signal).astype(np.float32)


def _variant_audio(raw: np.ndarray, job: dict[str, Any], effective: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    profile_name = "train" if job["split"] == "train" else "evaluation"
    profile = effective[f"{profile_name}_augmentation"]
    derived_seed = stable_seed(effective["seed"], job["base_sample_id"], job["variant_id"], profile_name)
    speed = float(profile["speed_factors"][derived_seed % len(profile["speed_factors"])])
    shifted = _speed_perturb(raw, speed)
    audio, parameters = augment_audio(
        shifted, profile, index=derived_seed % 2_000_000_000, seed=effective["seed"]
    )
    parameters.update({
        "profile": profile_name, "augmentation_seed": derived_seed,
        "speed_factor": speed,
    })
    return audio, parameters


def run_generation(
    config_path: Path, *, output_root: Path | None = None, resume: bool = False,
    synthesizers: dict[str, Callable[..., tuple[np.ndarray, dict[str, Any]]]] | None = None,
    stop_after_base: int | None = None, stop_after_effective: int | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config_path)
    vocabulary = MultiKWSVocabulary.load((PROJECT_ROOT / config["vocabulary"]).resolve())
    effective = resolve_12k_config(config, vocabulary)
    base_jobs = build_base_jobs(vocabulary, effective)
    effective_jobs = build_effective_jobs(base_jobs, effective)
    root = (output_root or PROJECT_ROOT / config["dataset"]["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "GENERATION_STATUS.json"
    base_partial_path = root / "PARTIAL_BASE_RECORDS.jsonl"
    effective_partial_path = root / "PARTIAL_EFFECTIVE_RECORDS.jsonl"
    manifest_path = root / "DatasetManifest.json"
    existing = read_json_with_retry(status_path) if status_path.exists() else None
    if existing and existing.get("config_sha256") != config_sha256:
        raise RuntimeError("Existing generation state belongs to a different config")
    if existing and existing.get("status") == "COMPLETED":
        return json.loads((root / "DATASET_INFO.json").read_text(encoding="utf-8"))
    if existing and not resume:
        raise RuntimeError("Partial generation exists; rerun with --resume")
    base_records = _load_records(base_partial_path, root, base_jobs, base=True)
    records = _load_records(effective_partial_path, root, effective_jobs, base=False)
    created_at = existing.get("created_at", utc_now()) if existing else utc_now()

    def write_status(state: str, failed: int = 0) -> None:
        atomic_json(status_path, {
            "schema": "wakeword-studio.multikws-generation-status/v2",
            "status": state, "dataset_id": effective["dataset_id"],
            "planned_base_speech": len(base_jobs),
            "completed_base_speech": len(base_records),
            "planned_effective_samples": len(effective_jobs),
            "completed_effective_samples": len(records),
            "failed_samples": failed, "created_at": created_at,
            "last_updated": utc_now(), "config_sha256": config_sha256,
            "resume_supported": True, "base_tts_reused_on_resume": True,
            "test_frozen": True,
        })

    provider_functions = dict(synthesizers or {})
    provider_devices: dict[str, str] = {
        name: "injected_test_synthesizer" for name in provider_functions
    }
    clients: list[VoxCPMWorkerClient] = []
    missing_sources = {
        str(job["speech_source"]) for job in base_jobs
        if job["base_sample_id"] not in base_records
    }
    for source in sorted(missing_sources):
        if source in provider_functions:
            continue
        if source == "kokoro":
            provider_functions[source], provider_devices[source] = _kokoro_synthesizer(effective)
        elif source == "voxcpm15":
            client = VoxCPMWorkerClient(effective, root / ".source_cache")
            clients.append(client)
            provider_functions[source] = client.synthesize
            provider_devices[source] = client.device
        else:
            raise ValueError(f"Unsupported speech source: {source}")

    write_status("IN_PROGRESS")
    new_bases = 0
    new_effective = 0
    try:
        with base_partial_path.open("a", encoding="utf-8") as partial:
            for job in base_jobs:
                if job["base_sample_id"] in base_records:
                    continue
                if stop_after_base is not None and new_bases >= stop_after_base:
                    raise KeyboardInterrupt("intentional base-generation interruption")
                source = str(job["speech_source"])
                raw, source_metadata = provider_functions[source](
                    job, str(job["text"]), str(job["speaker_id"]), 1.0
                )
                destination = root / job["relative_path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, np.asarray(raw, np.float32), TARGET_SAMPLE_RATE_HZ,
                         subtype=effective["audio"]["subtype"])
                record = {
                    "record_id": job["base_sample_id"], "base_sample_id": job["base_sample_id"],
                    "path": job["relative_path"], "split": job["split"],
                    "class_index": job["class_index"], "keyword_id": job["keyword_id"],
                    "background_kind": job["background_kind"], "text": job["text"],
                    "speech_source": source, "speaker_id": job["speaker_id"],
                    "reference_speaker_id": job["reference_speaker_id"],
                    "source_metadata": source_metadata, "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                    "channels": 1, "subtype": effective["audio"]["subtype"],
                    "sha256": sha256_file(destination),
                }
                partial.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial.flush(); os.fsync(partial.fileno())
                base_records[job["base_sample_id"]] = record
                new_bases += 1; write_status("IN_PROGRESS")

        with effective_partial_path.open("a", encoding="utf-8") as partial:
            for job in effective_jobs:
                if job["sample_id"] in records:
                    continue
                if stop_after_effective is not None and new_effective >= stop_after_effective:
                    raise KeyboardInterrupt("intentional effective-generation interruption")
                if job["speech_source"] is None:
                    seed = stable_seed(effective["seed"], job["base_sample_id"], "ambient")
                    raw = procedural_noise("office", 2 * TARGET_SAMPLE_RATE_HZ,
                                           np.random.default_rng(seed)) * 0.04
                    source_record: dict[str, Any] | None = None
                else:
                    source_record = base_records[job["base_sample_id"]]
                    raw, sample_rate = sf.read(root / source_record["path"], dtype="float32",
                                               always_2d=False)
                    if int(sample_rate) != TARGET_SAMPLE_RATE_HZ:
                        raise RuntimeError("Base cache must be 16 kHz")
                audio, parameters = _variant_audio(np.asarray(raw, np.float32), job, effective)
                destination = root / job["relative_path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, audio, TARGET_SAMPLE_RATE_HZ,
                         subtype=effective["audio"]["subtype"])
                speech_source = "procedural_ambient" if source_record is None else source_record["speech_source"]
                record = {
                    "record_id": job["sample_id"], "sample_id": job["sample_id"],
                    "base_sample_id": job["base_sample_id"], "variant_id": job["variant_id"],
                    "path": job["relative_path"], "split": job["split"],
                    "class_index": job["class_index"], "keyword_id": job["keyword_id"],
                    "background_kind": job["background_kind"], "text": job["text"],
                    "speech_source": speech_source,
                    "speaker": {
                        "speaker_id": job["speaker_id"] or "none", "source": speech_source,
                        "gender": None, "age_group": None, "age_verified": False,
                    },
                    "reference_speaker_id": job["reference_speaker_id"],
                    "augmentation_parameters": parameters,
                    "sample_rate_hz": TARGET_SAMPLE_RATE_HZ, "channels": 1,
                    "subtype": effective["audio"]["subtype"], "sha256": sha256_file(destination),
                }
                partial.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial.flush(); os.fsync(partial.fileno())
                records[job["sample_id"]] = record
                new_effective += 1; write_status("IN_PROGRESS")
    except KeyboardInterrupt:
        write_status("INTERRUPTED")
        raise
    except AtomicJsonWriteError:
        # atomic_json already exhausted its retry budget and wrote an independent
        # diagnostic. Do not recursively attempt the same status write again.
        raise
    except Exception:
        write_status("INTERRUPTED", failed=1)
        raise
    finally:
        for client in clients:
            client.close()

    ordered_bases = [base_records[job["base_sample_id"]] for job in base_jobs]
    ordered_records = [records[job["sample_id"]] for job in effective_jobs]
    group_splits: dict[str, set[str]] = {}
    group_variants: dict[str, set[int]] = {}
    for record in ordered_records:
        group_splits.setdefault(record["base_sample_id"], set()).add(record["split"])
        group_variants.setdefault(record["base_sample_id"], set()).add(int(record["variant_id"]))
    leakage = sum(len(splits) > 1 for splits in group_splits.values())
    source_counts = {
        name: sum(record["speech_source"] == name for record in ordered_records)
        for name in [*effective["speech_source_mix"], "procedural_ambient"]
    }
    base_source_counts = {
        name: sum(record["speech_source"] == name for record in ordered_bases)
        for name in effective["speech_source_mix"]
    }
    dataset_sha = hashlib.sha256(
        "".join(sorted(record["sha256"] for record in ordered_records)).encode("ascii")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema": "wakeword-studio.multikws-dataset/v3",
        "dataset_id": effective["dataset_id"], "profile": effective["profile"],
        "experiment_stage": effective["experiment_stage"], "production_quality": False,
        "created_at": created_at, "root": str(root), "requested_config": config,
        "effective_config": effective, "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha, "base_records": ordered_bases,
        "records": ordered_records, "base_source_counts": base_source_counts,
        "source_counts": source_counts,
        "split_counts": {split: sum(record["split"] == split for record in ordered_records) for split in SPLITS},
        "class_counts": {name: sum(record["keyword_id"] == name for record in ordered_records)
                         for name in vocabulary.class_names},
        "speech_sources": list(effective["speech_source_mix"]),
        "speaker_reference_splits": effective["speaker_reference_splits"],
        "kokoro_speaker_disjoint": _disjoint(effective["speaker_reference_splits"]["kokoro"]),
        "voxcpm_reference_speaker_disjoint": _disjoint(effective["speaker_reference_splits"]["voxcpm15"]),
        "age_metadata_verified": False,
        "base_group_split_leakage": leakage,
        "variants_per_base": effective["variants_per_base"],
        "ordinary_background_unique_text_count": effective["ordinary_background_unique_text_count"],
        "hard_negative_unique_text_count": effective["hard_negative_unique_text_count"],
        "test_frozen": True, "test_read_during_build": False,
        "generator": {"providers": list(effective["speech_source_mix"]), "devices": provider_devices},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    info = {
        "dataset_id": effective["dataset_id"], "record_count": len(ordered_records),
        "planned_base_speech": len(ordered_bases), "split_counts": manifest["split_counts"],
        "base_source_counts": base_source_counts, "source_counts": source_counts,
        "BASE_GROUP_SPLIT_LEAKAGE": leakage, "TEST_READ": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path), "dataset_sha256": dataset_sha,
    }
    atomic_json(root / "DATASET_INFO.json", info)
    write_status("COMPLETED")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(planner_report(args.config), ensure_ascii=False))
        return
    if args.status:
        config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
        root = (args.output_root or PROJECT_ROOT / config["dataset"]["output_root"]).resolve()
        print(json.dumps(read_json_with_retry(root / "GENERATION_STATUS.json"), ensure_ascii=False, indent=2))
        return
    print(json.dumps(run_generation(args.config, output_root=args.output_root, resume=args.resume), ensure_ascii=False))


if __name__ == "__main__":
    main()
