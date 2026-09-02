"""Configuration-driven, resumable Multi-KWS dataset generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.audio import (  # noqa: E402
    KOKORO_OUTPUT_SAMPLE_RATE_HZ,
    TARGET_SAMPLE_RATE_HZ,
    resample_audio,
)
from wakeword_studio.dataset.multikws_dataset import resolve_dataset_plan  # noqa: E402
from wakeword_studio.training.multikws_vocabulary import MultiKWSVocabulary  # noqa: E402


SPLITS = ("train", "validation", "test")
SUPPORTED_NOISE_TYPES = {
    "office", "fan_ac", "keyboard", "tv_speech", "babble",
    "street", "car", "classroom", "cafe", "device_mic",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


_TRANSIENT_WINDOWS_REPLACE_ERRORS = {5, 32}


class AtomicJsonWriteError(OSError):
    """Raised after a JSON target could not be atomically replaced."""


def _is_transient_replace_error(error: OSError) -> bool:
    return isinstance(error, PermissionError) or getattr(error, "winerror", None) in (
        _TRANSIENT_WINDOWS_REPLACE_ERRORS
    )


def _unique_json_temp(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _write_fsynced_json(path: Path, value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_with_retry(
    temporary: Path, path: Path, *, attempts: int, initial_delay_s: float,
    max_delay_s: float,
) -> None:
    delay = initial_delay_s
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if not _is_transient_replace_error(error) or attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(max_delay_s, max(initial_delay_s, delay * 2))


def _write_status_failure_log(target: Path, temporary: Path, error: OSError) -> None:
    fallback = target.with_name("GENERATION_STATUS_WRITE_ERROR.json")
    fallback_temp = _unique_json_temp(fallback)
    payload = {
        "schema": "wakeword-studio.generation-status-write-error/v1",
        "target": str(target),
        "preserved_temporary": str(temporary),
        "error_type": type(error).__name__,
        "error": str(error),
        "winerror": getattr(error, "winerror", None),
        "recorded_at": utc_now(),
    }
    try:
        _write_fsynced_json(fallback_temp, payload)
        _replace_with_retry(
            fallback_temp, fallback, attempts=12, initial_delay_s=0.05,
            max_delay_s=0.4,
        )
    except OSError:
        # Never risk the last valid status merely to publish diagnostics. The
        # unique fallback temp remains available for forensic inspection.
        return


def atomic_json(
    path: Path, value: Any, *, attempts: int = 12,
    initial_delay_s: float = 0.05, max_delay_s: float = 0.4,
) -> None:
    """Atomically write JSON with bounded retries for transient Windows locks."""

    if attempts < 10:
        raise ValueError("atomic_json requires at least 10 replace attempts")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_json_temp(path)
    _write_fsynced_json(temporary, value)
    try:
        _replace_with_retry(
            temporary, path, attempts=attempts,
            initial_delay_s=initial_delay_s, max_delay_s=max_delay_s,
        )
    except OSError as error:
        if path.name == "GENERATION_STATUS.json":
            _write_status_failure_log(path, temporary, error)
        raise AtomicJsonWriteError(
            f"Atomic JSON replace failed for {path}; last valid target was preserved"
        ) from error


def read_json_with_retry(
    path: Path, *, attempts: int = 12, initial_delay_s: float = 0.05,
    max_delay_s: float = 0.4,
) -> Any:
    """Read and close a JSON file, retrying transient Windows open failures."""

    if attempts < 1:
        raise ValueError("read_json_with_retry attempts must be positive")
    delay = initial_delay_s
    for attempt in range(attempts):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except OSError as error:
            transient = isinstance(error, FileNotFoundError) or _is_transient_replace_error(error)
            if not transient or attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(max_delay_s, max(initial_delay_s, delay * 2))
    raise AssertionError("unreachable")


def allocate_split(total: int, weights: dict[str, int]) -> dict[str, int]:
    if total < 1 or set(weights) != set(SPLITS) or any(int(weights[name]) < 1 for name in SPLITS):
        raise ValueError("split_weights must contain positive train/validation/test weights")
    denominator = sum(int(weights[name]) for name in SPLITS)
    raw = {name: total * int(weights[name]) / denominator for name in SPLITS}
    result = {name: int(np.floor(raw[name])) for name in SPLITS}
    remainder = total - sum(result.values())
    order = sorted(SPLITS, key=lambda name: (-(raw[name] - result[name]), SPLITS.index(name)))
    for name in order[:remainder]:
        result[name] += 1
    return result


def allocate_weighted(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total < 1 or not weights or any(float(value) <= 0 for value in weights.values()):
        raise ValueError("source weights must be positive")
    names = list(weights)
    denominator = sum(float(weights[name]) for name in names)
    raw = {name: total * float(weights[name]) / denominator for name in names}
    result = {name: int(np.floor(raw[name])) for name in names}
    remainder = total - sum(result.values())
    order = sorted(names, key=lambda name: (-(raw[name] - result[name]), names.index(name)))
    for name in order[:remainder]:
        result[name] += 1
    return result


def resolve_augmentation(dataset_config: dict[str, Any]) -> dict[str, Any]:
    requested = dict(dataset_config.get("augmentation", {}))
    defaults: dict[str, Any] = {
        "speed_factors": [1.0], "gain_db_range": [0.0, 0.0],
        "leading_silence_ms_range": [0, 0], "trailing_silence_ms_range": [0, 0],
        "reverb_probability": 0.0, "far_field_probability": 0.0,
        "reverb_delay_ms": 55, "reverb_decay": 0.17, "far_field_gain": 0.55,
        "snr_db_values": [20.0], "noise_types": ["office"],
    }
    effective = {**defaults, **requested}
    for name in ("speed_factors", "snr_db_values", "noise_types"):
        if not effective[name]:
            raise ValueError(f"augmentation.{name} must not be empty")
    for name in ("gain_db_range", "leading_silence_ms_range", "trailing_silence_ms_range"):
        values = effective[name]
        if len(values) != 2 or float(values[0]) > float(values[1]):
            raise ValueError(f"augmentation.{name} must be an ordered pair")
    for name in ("reverb_probability", "far_field_probability"):
        if not 0.0 <= float(effective[name]) <= 1.0:
            raise ValueError(f"augmentation.{name} must be in [0,1]")
    unknown = set(str(value) for value in effective["noise_types"]) - SUPPORTED_NOISE_TYPES
    if unknown:
        raise ValueError(f"Unsupported noise types: {sorted(unknown)}")
    return effective


def resolve_effective_config(config: dict[str, Any], vocabulary: MultiKWSVocabulary) -> dict[str, Any]:
    dataset = dict(config["dataset"])
    plan = resolve_dataset_plan({**dataset, "seed": config["seed"]})
    required = (
        "dataset_id", "manifest_profile", "experiment_stage", "split_weights",
        "speech_source_mix", "speech_sources", "background_phrases", "hard_negative_phrases",
    )
    missing = [name for name in required if name not in dataset]
    if missing:
        raise ValueError(f"Missing dataset config fields: {missing}")
    source_mix = {str(name): float(value) for name, value in dataset["speech_source_mix"].items()}
    sources = {str(name): dict(value) for name, value in dataset["speech_sources"].items()}
    if set(source_mix) != set(sources):
        raise ValueError("speech_source_mix and speech_sources must have identical keys")
    speaker_splits: dict[str, dict[str, list[str]]] = {}
    for source_name, source in sources.items():
        split_key = "reference_speaker_splits" if source_name == "voxcpm15" else "speaker_splits"
        if split_key not in source:
            raise ValueError(f"{source_name} requires {split_key}")
        split_values = {name: [str(value) for value in source[split_key][name]] for name in SPLITS}
        if any(not split_values[name] for name in SPLITS):
            raise ValueError(f"Every {source_name} split requires at least one speaker/reference")
        if any(set(split_values[left]) & set(split_values[right]) for left in SPLITS for right in SPLITS if left < right):
            raise ValueError(f"{source_name} speaker/reference splits must be disjoint")
        speaker_splits[source_name] = split_values
    weights = {name: int(dataset["split_weights"][name]) for name in SPLITS}
    allocations = {
        "positive_per_keyword": allocate_split(plan.positive_per_keyword, weights),
        "background_speech": allocate_split(plan.background_speech_count, weights),
        "hard_negative": allocate_split(plan.hard_negative_count, weights),
        "ambient": allocate_split(plan.ambient_count, weights),
    }
    return {
        "dataset_id": str(dataset["dataset_id"]), "profile": str(dataset["manifest_profile"]),
        "scale_profile": plan.profile, "experiment_stage": str(dataset["experiment_stage"]),
        "production_quality": False, "seed": int(config["seed"]),
        "vocabulary_path": str(config["vocabulary"]), "vocabulary_id": vocabulary.vocabulary_id,
        "num_classes": vocabulary.num_classes, "class_names": list(vocabulary.class_names),
        "split_weights": weights, "split_allocations": allocations,
        "speech_source_mix": source_mix, "speech_sources": sources,
        "speaker_reference_splits": speaker_splits,
        "background_phrases": [str(value) for value in dataset["background_phrases"]],
        "hard_negative_phrases": [str(value) for value in dataset["hard_negative_phrases"]],
        "augmentation": resolve_augmentation(dataset),
        "audio": {"sample_rate_hz": int(dataset["sample_rate_hz"]), "channels": int(dataset["channels"]), "subtype": str(dataset["subtype"])},
        "age_metadata_policy": str(dataset["age_metadata_policy"]),
    }


def build_jobs(vocabulary: MultiKWSVocabulary, effective: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    index = 0

    def source_assignments(count: int) -> list[str]:
        counts = allocate_weighted(count, effective["speech_source_mix"])
        remaining = dict(counts)
        result: list[str] = []
        while len(result) < count:
            for name in effective["speech_source_mix"]:
                if remaining[name] > 0:
                    result.append(name)
                    remaining[name] -= 1
        return result

    def add_job(split: str, keyword_id: str, class_index: int, kind: str,
                ordinal: int, text: str | None, speech_source: str | None,
                source_ordinal: int = 0) -> None:
        nonlocal index
        speakers = [] if speech_source is None else effective["speaker_reference_splits"][speech_source][split]
        speaker = None if speech_source is None else speakers[source_ordinal % len(speakers)]
        sample_id = f"{split}-{kind}-{keyword_id}-{ordinal:06d}"
        jobs.append({
            "index": index, "sample_id": sample_id, "split": split,
            "keyword_id": keyword_id, "class_index": class_index,
            "background_kind": None if class_index else kind, "text": text,
            "speech_source": speech_source, "speaker_id": speaker,
            "reference_speaker_id": speaker if speech_source == "voxcpm15" else None,
            "relative_path": (Path(split) / keyword_id / f"{sample_id}.wav").as_posix(),
        })
        index += 1

    for split in SPLITS:
        count = effective["split_allocations"]["positive_per_keyword"][split]
        for keyword in vocabulary.keywords:
            source_seen = {name: 0 for name in effective["speech_source_mix"]}
            for ordinal, source in enumerate(source_assignments(count)):
                add_job(split, keyword.keyword_id, keyword.class_index, "positive", ordinal,
                        keyword.display_name, source, source_seen[source])
                source_seen[source] += 1
        for kind, phrases_key in (("background_speech", "background_phrases"), ("hard_negative", "hard_negative_phrases")):
            phrases = effective[phrases_key]
            source_seen = {name: 0 for name in effective["speech_source_mix"]}
            count = effective["split_allocations"][kind][split]
            for ordinal, source in enumerate(source_assignments(count)):
                add_job(split, "background", 0, kind, ordinal, phrases[ordinal % len(phrases)],
                        source, source_seen[source])
                source_seen[source] += 1
        for ordinal in range(effective["split_allocations"]["ambient"][split]):
            add_job(split, "background", 0, "ambient", ordinal, None, None)
    return jobs


def procedural_noise(category: str, length: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0, 1, length).astype(np.float32)
    timeline = np.arange(length) / TARGET_SAMPLE_RATE_HZ
    if category in {"fan_ac", "car"}:
        frequency = 55 if category == "fan_ac" else 90
        return (0.7 * np.sin(2 * np.pi * frequency * timeline) + 0.2 * white).astype(np.float32)
    if category in {"street", "office", "classroom", "cafe", "tv_speech", "babble"}:
        return np.convolve(white, np.ones(40, np.float32) / 40, mode="same").astype(np.float32)
    if category == "keyboard":
        output = 0.05 * white
        for position in rng.integers(0, max(1, length - 80), max(1, length // 4000)):
            output[int(position): int(position) + 80] += np.exp(-np.arange(80) / 15)
        return output.astype(np.float32)
    if category == "device_mic":
        return np.clip(np.round(white * 12) / 12, -1, 1).astype(np.float32)
    return white


def augment_audio(audio: np.ndarray, augmentation: dict[str, Any], *, index: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed + index * 7919)
    gain_db = float(rng.uniform(*[float(value) for value in augmentation["gain_db_range"]]))
    signal = np.asarray(audio, np.float32) * 10 ** (gain_db / 20)
    leading_ms = float(rng.uniform(*[float(value) for value in augmentation["leading_silence_ms_range"]]))
    trailing_ms = float(rng.uniform(*[float(value) for value in augmentation["trailing_silence_ms_range"]]))
    leading = int(round(leading_ms * TARGET_SAMPLE_RATE_HZ / 1000))
    trailing = int(round(trailing_ms * TARGET_SAMPLE_RATE_HZ / 1000))
    signal = np.pad(signal, (leading, trailing))
    snr_db = float(augmentation["snr_db_values"][index % len(augmentation["snr_db_values"])])
    noise_type = str(augmentation["noise_types"][index % len(augmentation["noise_types"])])
    noise = procedural_noise(noise_type, len(signal), rng)
    signal_rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    mixed = signal + noise * (signal_rms / (10 ** (snr_db / 20) * noise_rms))
    reverb = bool(rng.random() < float(augmentation["reverb_probability"]))
    far_field = bool(rng.random() < float(augmentation["far_field_probability"]))
    delay = int(round(float(augmentation["reverb_delay_ms"]) * TARGET_SAMPLE_RATE_HZ / 1000))
    if reverb and delay and len(mixed) > delay:
        mixed[delay:] += mixed[:-delay] * float(augmentation["reverb_decay"])
    if far_field:
        mixed *= float(augmentation["far_field_gain"])
    return np.clip(mixed, -1, 1).astype(np.float32), {
        "gain_db": gain_db, "snr_db": snr_db, "noise_category": noise_type,
        "reverb": "synthetic_early_reflection" if reverb else None,
        "far_field": far_field, "leading_silence_ms": leading_ms,
        "trailing_silence_ms": trailing_ms,
    }


def _load_partial_records(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        audio_path = root / record["path"]
        if audio_path.is_file() and sha256_file(audio_path) == record["sha256"]:
            records[str(record["record_id"])] = record
    return records


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _kokoro_synthesizer(effective: dict[str, Any]) -> tuple[Callable[..., tuple[np.ndarray, dict[str, Any]]], str]:
    import torch
    from kokoro import KModel, KPipeline

    tts = effective["speech_sources"]["kokoro"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(effective["seed"]))
    model = KModel(repo_id=str(tts["repo_id"])).to(device).eval()
    pipeline = KPipeline(lang_code=str(tts["lang_code"]), repo_id=str(tts["repo_id"]), model=model)

    def synthesize(job: dict[str, Any], text: str, voice: str, speed: float) -> tuple[np.ndarray, dict[str, Any]]:
        del job
        audio = np.asarray(next(pipeline(text, voice=voice, speed=speed)).audio, np.float32)
        return resample_audio(audio, KOKORO_OUTPUT_SAMPLE_RATE_HZ), {}

    return synthesize, device


class VoxCPMWorkerClient:
    def __init__(self, effective: dict[str, Any], temporary_root: Path):
        source = effective["speech_sources"]["voxcpm15"]
        python = _resolve_project_path(str(source["python_executable"]))
        repo_src = _resolve_project_path(str(source["repo_src"]))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repo_src) + os.pathsep + environment.get("PYTHONPATH", "")
        command = [
            str(python), str(PROJECT_ROOT / "phase9/scripts/voxcpm15_jsonl_worker.py"),
            "--model-dir", str(_resolve_project_path(str(source["model_dir"]))),
            "--reference-manifest", str(_resolve_project_path(str(source["reference_manifest"]))),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1, env=environment,
        )
        self.source = source
        self.temporary_root = temporary_root
        ready = self._read()
        if ready.get("status") != "READY":
            raise RuntimeError(f"VoxCPM worker failed to become ready: {ready}")
        self.device = f"cuda:{ready.get('gpu', 'unknown')}"

    def _read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"VoxCPM worker exited with code {self.process.poll()}")
        return json.loads(line)

    def synthesize(self, job: dict[str, Any], text: str, voice: str,
                   speed: float) -> tuple[np.ndarray, dict[str, Any]]:
        del speed
        temporary = self.temporary_root / f"{job['sample_id']}.wav"
        request = {
            "sample_id": job["sample_id"], "text": text,
            "reference_speaker_id": voice,
            "seed_material": f"{job['sample_id']}:{job['index']}",
            "output_path": str(temporary),
            "cfg_value": self.source["cfg_value"],
            "inference_timesteps": self.source["inference_timesteps"],
            "max_len": self.source["max_len"],
        }
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self._read()
        if response.get("status") != "OK":
            raise RuntimeError(f"VoxCPM worker request failed: {response}")
        expected_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if response.get("input_text_sha256") != expected_text_sha256:
            raise RuntimeError("VoxCPM worker text encoding/hash mismatch")
        audio, sample_rate = sf.read(temporary, dtype="float32", always_2d=False)
        temporary.unlink(missing_ok=True)
        if np.asarray(audio).ndim != 1:
            audio = np.mean(audio, axis=-1)
        return resample_audio(np.asarray(audio, np.float32), int(sample_rate)), response

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write('{"command":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        except Exception:
            self.process.terminate()
            self.process.wait(timeout=10)


def run_generation(
    config_path: Path, *, output_root: Path | None = None, resume: bool = False,
    synthesizer: Callable[[str, str, float], np.ndarray] | None = None,
    synthesizers: dict[str, Callable[..., tuple[np.ndarray, dict[str, Any]]]] | None = None,
    stop_after: int | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config_path)
    vocabulary = MultiKWSVocabulary.load((PROJECT_ROOT / config["vocabulary"]).resolve())
    effective = resolve_effective_config(config, vocabulary)
    jobs = build_jobs(vocabulary, effective)
    root = (output_root or PROJECT_ROOT / config["dataset"]["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "GENERATION_STATUS.json"
    manifest_path = root / "DatasetManifest.json"
    partial_path = root / "PARTIAL_RECORDS.jsonl"
    existing_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    if existing_status and existing_status.get("config_sha256") != config_sha256:
        raise RuntimeError("Existing generation state belongs to a different config")
    if existing_status and existing_status.get("status") == "COMPLETED":
        if not manifest_path.is_file():
            raise RuntimeError("COMPLETED state is missing DatasetManifest.json")
        return json.loads((root / "DATASET_INFO.json").read_text(encoding="utf-8"))
    if existing_status and not resume:
        raise RuntimeError("Partial generation exists; rerun with --resume")
    records = _load_partial_records(partial_path, root)
    created_at = existing_status.get("created_at", utc_now()) if existing_status else utc_now()

    def write_status(state: str, failed: int = 0) -> None:
        atomic_json(status_path, {
            "schema": "wakeword-studio.multikws-generation-status/v1", "status": state,
            "dataset_id": effective["dataset_id"], "planned_samples": len(jobs),
            "completed_samples": len(records), "failed_samples": failed,
            "created_at": created_at, "last_updated": utc_now(),
            "config_sha256": config_sha256, "resume_supported": True, "test_frozen": True,
        })

    write_status("IN_PROGRESS")
    missing_sources = {
        str(job["speech_source"]) for job in jobs
        if job["sample_id"] not in records and job["speech_source"] is not None
    }
    provider_functions = dict(synthesizers or {})
    provider_devices: dict[str, str] = {}
    clients: list[VoxCPMWorkerClient] = []
    if synthesizer is not None:
        def legacy(job: dict[str, Any], text: str, voice: str,
                   speed: float) -> tuple[np.ndarray, dict[str, Any]]:
            del job
            return synthesizer(text, voice, speed), {}
        provider_functions.update({name: legacy for name in missing_sources})
        provider_devices.update({name: "injected_test_synthesizer" for name in missing_sources})
    for source_name in sorted(missing_sources):
        if source_name in provider_functions:
            provider_devices.setdefault(source_name, "injected_test_synthesizer")
        elif source_name == "kokoro":
            provider_functions[source_name], provider_devices[source_name] = _kokoro_synthesizer(effective)
        elif source_name == "voxcpm15":
            client = VoxCPMWorkerClient(effective, root / ".source_cache")
            clients.append(client)
            provider_functions[source_name] = client.synthesize
            provider_devices[source_name] = client.device
        else:
            raise ValueError(f"Unsupported speech source: {source_name}")
    generated = 0
    try:
        with partial_path.open("a", encoding="utf-8") as partial:
            for job in jobs:
                if job["sample_id"] in records:
                    continue
                if stop_after is not None and generated >= stop_after:
                    raise KeyboardInterrupt("intentional regression interruption")
                speed = float(effective["augmentation"]["speed_factors"][job["index"] % len(effective["augmentation"]["speed_factors"])])
                if job["text"] is None:
                    rng = np.random.default_rng(effective["seed"] + job["index"])
                    raw = rng.normal(0, 0.04, 2 * TARGET_SAMPLE_RATE_HZ).astype(np.float32)
                else:
                    source_name = str(job["speech_source"])
                    assert job["speaker_id"] is not None
                    raw, source_metadata = provider_functions[source_name](
                        job, str(job["text"]), str(job["speaker_id"]), speed
                    )
                if job["text"] is None:
                    source_metadata = {}
                audio, acoustic = augment_audio(raw, effective["augmentation"], index=job["index"], seed=effective["seed"])
                destination = root / job["relative_path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, audio, TARGET_SAMPLE_RATE_HZ, subtype=effective["audio"]["subtype"])
                source = "procedural_ambient" if job["text"] is None else str(job["speech_source"])
                record = {
                    "record_id": job["sample_id"], "sample_id": job["sample_id"],
                    "path": job["relative_path"], "split": job["split"],
                    "class_index": job["class_index"], "keyword_id": job["keyword_id"],
                    "background_kind": job["background_kind"], "text": job["text"],
                    "speech_source": source,
                    "speaker": {"speaker_id": job["speaker_id"] or "none", "source": source, "gender": None, "age_group": None, "age_verified": False},
                    "reference_speaker_id": job["reference_speaker_id"],
                    "source_metadata": source_metadata,
                    "speaking_rate": speed if job["text"] is not None else None,
                    "acoustic": acoustic, "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                    "channels": 1, "subtype": effective["audio"]["subtype"],
                    "sha256": sha256_file(destination),
                }
                partial.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial.flush()
                os.fsync(partial.fileno())
                records[job["sample_id"]] = record
                generated += 1
                write_status("IN_PROGRESS")
    except KeyboardInterrupt:
        write_status("INTERRUPTED")
        raise
    except AtomicJsonWriteError:
        raise
    except Exception:
        write_status("INTERRUPTED", failed=1)
        raise
    finally:
        for client in clients:
            client.close()

    ordered_records = [records[job["sample_id"]] for job in jobs]
    if len(ordered_records) != len(jobs):
        write_status("INTERRUPTED", failed=1)
        raise RuntimeError("Generation ended without all planned samples")
    dataset_sha256 = hashlib.sha256("".join(sorted(record["sha256"] for record in ordered_records)).encode("ascii")).hexdigest()
    speech_sources = list(effective["speech_source_mix"])
    source_counts = {
        name: sum(record["speech_source"] == name for record in ordered_records)
        for name in [*speech_sources, "procedural_ambient"]
    }
    per_split_source_counts = {
        split: {name: sum(record["split"] == split and record["speech_source"] == name
                          for record in ordered_records)
                for name in [*speech_sources, "procedural_ambient"]}
        for split in SPLITS
    }
    per_keyword_source_counts = {
        keyword.keyword_id: {
            split: {name: sum(record["keyword_id"] == keyword.keyword_id and
                              record["split"] == split and record["speech_source"] == name
                              for record in ordered_records)
                    for name in speech_sources}
            for split in SPLITS
        }
        for keyword in vocabulary.keywords
    }
    manifest: dict[str, Any] = {
        "schema": "wakeword-studio.multikws-dataset/v2",
        "dataset_id": effective["dataset_id"], "profile": effective["profile"],
        "experiment_stage": effective["experiment_stage"], "production_quality": False,
        "root": str(root), "created_at": created_at,
        "requested_config": config, "effective_config": effective,
        "config_sha256": config_sha256, "dataset_sha256": dataset_sha256,
        "split_counts": {name: sum(record["split"] == name for record in ordered_records) for name in SPLITS},
        "class_counts": {name: sum(record["keyword_id"] == name for record in ordered_records) for name in vocabulary.class_names},
        "vocabulary": {"path": effective["vocabulary_path"], "vocabulary_id": vocabulary.vocabulary_id, "num_classes": vocabulary.num_classes, "class_names": list(vocabulary.class_names)},
        "records": ordered_records, "speaker_disjoint": True,
        "speech_sources": speech_sources, "source_counts": source_counts,
        "per_keyword_source_counts": per_keyword_source_counts,
        "per_split_source_counts": per_split_source_counts,
        "speaker_reference_splits": effective["speaker_reference_splits"],
        "kokoro_speaker_disjoint": all(
            not set(effective["speaker_reference_splits"]["kokoro"][left])
            & set(effective["speaker_reference_splits"]["kokoro"][right])
            for left in SPLITS for right in SPLITS if left < right
        ) if "kokoro" in speech_sources else None,
        "voxcpm_reference_speaker_disjoint": all(
            not set(effective["speaker_reference_splits"]["voxcpm15"][left])
            & set(effective["speaker_reference_splits"]["voxcpm15"][right])
            for left in SPLITS for right in SPLITS if left < right
        ) if "voxcpm15" in speech_sources else None,
        "age_metadata_verified": False, "test_frozen": True, "test_read_during_build": False,
        "generator": {"providers": speech_sources, "devices": provider_devices},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    info = {
        "manifest_sha256": manifest["manifest_sha256"], "manifest_file_sha256": sha256_file(manifest_path),
        "dataset_sha256": dataset_sha256, "record_count": len(ordered_records),
        "split_counts": {name: sum(record["split"] == name for record in ordered_records) for name in SPLITS},
        "class_counts": {name: sum(record["keyword_id"] == name for record in ordered_records) for name in vocabulary.class_names},
        "source_counts": source_counts,
        "TEST_READ": False,
    }
    atomic_json(root / "DATASET_INFO.json", info)
    write_status("COMPLETED")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_generation(args.config, output_root=args.output_root, resume=args.resume), ensure_ascii=False))


if __name__ == "__main__":
    main()
