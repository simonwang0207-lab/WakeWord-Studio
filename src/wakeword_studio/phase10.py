"""Phase 10 product contracts: vocabulary expansion, jobs and local evaluation."""

from __future__ import annotations

import copy
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pypinyin import Style, lazy_pinyin

from .json_utils import atomic_write_json, normalize_json_value
from .training.multikws_vocabulary import normalize_keyword


ADD_KEYWORD_REQUIRES_RETRAIN = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_keyword_id(display_name: str, existing_ids: Iterable[str] = ()) -> str:
    """Create a deterministic ASCII ID; omit the conventional ``你好`` prefix."""

    normalized = normalize_keyword(display_name)
    stem = normalized[2:] if normalized.startswith("你好") and len(normalized) > 2 else normalized
    slug = "".join(lazy_pinyin(stem, style=Style.NORMAL, errors=lambda chars: [f"u{ord(c):x}" for c in chars]))
    slug = re.sub(r"[^a-z0-9]+", "", slug.lower()) or "keyword"
    occupied = set(existing_ids)
    if slug not in occupied:
        return slug
    suffix = 2
    while f"{slug}{suffix}" in occupied:
        suffix += 1
    return f"{slug}{suffix}"


@dataclass(frozen=True, slots=True)
class VocabularyEntry:
    class_id: int
    keyword_id: str
    display_name: str
    enabled: bool
    created_at: str


@dataclass(frozen=True, slots=True)
class VocabularyManifest:
    schema_version: str
    vocabulary_id: str
    created_at: str
    parent_vocabulary_id: str | None
    classes: tuple[VocabularyEntry, ...]

    @classmethod
    def from_legacy(cls, path: Path) -> "VocabularyManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        stamp = str(raw.get("created_at", "phase9-frozen"))
        items = [raw["background"], *raw["keywords"]]
        return cls(
            schema_version="wakeword-studio.vocabulary-manifest/v1",
            vocabulary_id=str(raw["vocabulary_id"]),
            created_at=stamp,
            parent_vocabulary_id=raw.get("parent_vocabulary_id"),
            classes=tuple(VocabularyEntry(
                class_id=int(item["class_index"]), keyword_id=str(item["keyword_id"]),
                display_name=str(item["display_name"]), enabled=bool(item.get("enabled", True)),
                created_at=str(item.get("created_at", stamp)),
            ) for item in items),
        )

    def expand(self, display_name: str, *, created_at: str | None = None) -> "VocabularyManifest":
        normalized = normalize_keyword(display_name)
        if not normalized:
            raise ValueError("提示词不能为空")
        if normalized in {normalize_keyword(item.display_name) for item in self.classes}:
            raise ValueError("提示词已经存在")
        keyword_id = stable_keyword_id(display_name, (item.keyword_id for item in self.classes))
        stamp = created_at or utc_now()
        return VocabularyManifest(
            schema_version=self.schema_version,
            vocabulary_id=f"{self.vocabulary_id}_plus_{keyword_id}_v1",
            created_at=stamp,
            parent_vocabulary_id=self.vocabulary_id,
            classes=(*self.classes, VocabularyEntry(len(self.classes), keyword_id, display_name.strip(), True, stamp)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpansionPlan:
    schema: str
    display_name: str
    keyword_id: str
    old_num_classes: int
    new_num_classes: int
    source_dataset_id: str
    new_dataset_id: str
    source_dataset_immutable: bool
    replay_existing_classes: bool
    background_included: bool
    hard_negatives: tuple[dict[str, str], ...]
    positive_samples: int
    hard_negative_samples: int
    augmentation: str
    speech_sources: tuple[str, ...]
    model_architecture: str
    input_mode: str
    user_wav_directory: str | None
    audio_contract: str
    confusion_aware_groups: tuple[tuple[str, ...], ...]
    requires_retrain: bool = True
    starts_long_job: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def confusion_aware_hard_negatives(
    display_name: str,
    existing_wakewords: Iterable[str],
) -> tuple[dict[str, str], ...]:
    """Generate generic structural near misses without labeling wake words negative."""

    text = normalize_keyword(display_name)
    protected = {normalize_keyword(item) for item in existing_wakewords} | {text}
    candidates: list[tuple[str, str]] = []
    if text.startswith("你好"):
        body = text[2:]
        candidates.extend([
            (body, "missing_greeting"),
            ("你好" + body[:-1], "missing_final_character"),
            ("你好呀" + body, "extra_prefix_syllable"),
            ("你好" + body + "呀", "extra_final_syllable"),
            ("请问" + body, "non_wake_similar_phrase"),
            (body + "你好", "reversed_order"),
            ("您好" + body, "greeting_variant"),
            ("你好请" + body, "inserted_syllable"),
            ("你好" + body + "吗", "question_suffix"),
            ("你好" + body + "呢", "particle_suffix"),
            ("我说你好" + body, "embedded_non_wake_phrase"),
        ])
        if len(body) >= 2:
            candidates.append(("你好" + body[1:], "missing_first_name_character"))
            candidates.append(("你好" + body[:-1] + body[-1] * 2, "repeated_final_syllable"))
            candidates.append(("你好" + body[0] * 2 + body[1:], "repeated_first_syllable"))
            candidates.append(("你好" + body[::-1], "reordered_name_syllables"))
            for index in range(len(body)):
                candidates.append(("你好" + body[:index] + body[index + 1 :], "single_character_omission"))
            homophones = {
                "zhi": ("知", "只", "志"), "rui": ("睿", "锐", "瑞"),
                "jia": ("佳", "家", "甲"), "dou": ("豆", "逗"),
                "dian": ("点", "典"), "duo": ("多", "朵"),
            }
            final_pinyin = lazy_pinyin(body[-1], style=Style.NORMAL, errors="ignore")
            for replacement in homophones.get(final_pinyin[0] if final_pinyin else "", ()):
                candidates.append(("你好" + body[:-1] + replacement, "pronunciation_neighbor"))
    else:
        candidates.extend([(text[:-1], "missing_final_character"), (text + "呀", "extra_final_syllable")])
    unique: dict[str, str] = {}
    for phrase, reason in candidates:
        normalized = normalize_keyword(phrase)
        if normalized and normalized not in protected:
            unique.setdefault(phrase, reason)
    return tuple({"text": phrase, "reason": reason} for phrase, reason in unique.items())


def build_keyword_expansion_plan(
    vocabulary_path: Path,
    display_name: str,
    *,
    source_dataset_id: str = "teacher_six_multikws_v2_formal_12k",
    positive_samples: int = 600,
    hard_negative_samples: int = 300,
    augmentation: str = "standard",
    speech_sources: tuple[str, ...] = ("kokoro", "voxcpm15"),
    model_architecture: str = "convmixer",
    input_mode: str = "auto_generate",
    user_wav_directory: str | None = None,
) -> dict[str, Any]:
    if input_mode not in {"auto_generate", "user_wav", "mixed"}:
        raise ValueError("input_mode must be auto_generate, user_wav, or mixed")
    if input_mode in {"user_wav", "mixed"} and not str(user_wav_directory or "").strip():
        raise ValueError("选择用户 WAV 后必须填写只读源目录")
    current = VocabularyManifest.from_legacy(vocabulary_path)
    expanded = current.expand(display_name)
    new = expanded.classes[-1]
    existing_names = [item.display_name for item in current.classes[1:]]
    plan = ExpansionPlan(
        schema="wakeword-studio.keyword-expansion-plan/v1",
        display_name=new.display_name,
        keyword_id=new.keyword_id,
        old_num_classes=len(current.classes),
        new_num_classes=len(expanded.classes),
        source_dataset_id=source_dataset_id,
        new_dataset_id=f"teacher_six_plus_{new.keyword_id}_v1",
        source_dataset_immutable=True,
        replay_existing_classes=True,
        background_included=True,
        hard_negatives=confusion_aware_hard_negatives(display_name, existing_names),
        positive_samples=int(positive_samples),
        hard_negative_samples=int(hard_negative_samples),
        augmentation=augmentation,
        speech_sources=tuple(speech_sources),
        model_architecture=model_architecture,
        input_mode=input_mode,
        user_wav_directory=str(user_wav_directory).strip() if user_wav_directory else None,
        audio_contract="16 kHz / mono / PCM16; source files are never overwritten",
        confusion_aware_groups=(("doudou", "diandian", "duoduo"),),
    )
    return {
        "ok": True,
        "vocabulary": expanded.to_dict(),
        "plan": plan.to_dict(),
        "ADD_KEYWORD_REQUIRES_RETRAIN": True,
        "USER_ACTION_REQUIRED": True,
    }


class JobState(str, Enum):
    PENDING = "PENDING"
    DATA_PREPARING = "DATA_PREPARING"
    READY_TO_TRAIN = "READY_TO_TRAIN"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    QUANTIZING = "QUANTIZING"
    EVALUATING = "EVALUATING"
    READY_CANDIDATE = "READY_CANDIDATE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class MultiKWSJob:
    job_id: str
    state: str
    run_dir: str
    progress: float = 0.0
    current_epoch: int | None = None
    current_step: int | None = None
    best_validation: float | None = None
    eta_seconds: float | None = None
    latest_checkpoint: str | None = None
    error_message: str | None = None
    resume_supported: bool = True
    checkpoint_preserved_on_cancel: bool = True

    @classmethod
    def pending(cls, runs_root: Path, keyword_id: str) -> "MultiKWSJob":
        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        return cls(job_id, JobState.PENDING.value, str(runs_root / keyword_id / job_id))

    def cancel(self) -> None:
        self.state = JobState.CANCELLED.value

    def fail(self, error: str) -> None:
        self.state, self.error_message = JobState.FAILED.value, error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def materialize_job_preflight(
    project_root: Path,
    job: MultiKWSJob,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Persist only planning/config artifacts; never start generation/training."""

    root = Path(project_root).resolve()
    run_dir = Path(job.run_dir).resolve()
    if not run_dir.is_relative_to((root / "runs/multikws/user_expansions").resolve()):
        raise ValueError("Expansion run_dir escaped the dedicated Phase 10 root")
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(preflight["plan"])
    expanded = dict(preflight["vocabulary"])
    new_class = dict(expanded["classes"][-1])
    base_vocab_path = root / "configs/multikws/teacher_six_keywords.json"
    base_vocab = json.loads(base_vocab_path.read_text(encoding="utf-8"))
    stamp = str(new_class["created_at"])
    appended = {
        "keyword_id": str(new_class["keyword_id"]), "display_name": str(new_class["display_name"]),
        "normalized_text": normalize_keyword(str(new_class["display_name"])),
        "class_index": int(new_class["class_id"]), "aliases": [], "enabled": True,
        "created_at": stamp, "dataset": {"language": "zh-CN"},
    }
    full_vocab = copy.deepcopy(base_vocab)
    full_vocab.update({
        "schema": "wakeword-studio.multikws-vocabulary/v1",
        "vocabulary_id": str(expanded["vocabulary_id"]),
        "version": int(base_vocab.get("version", 1)) + 1,
        "created_at": stamp,
        "parent_vocabulary_id": str(base_vocab["vocabulary_id"]),
        "add_keyword_requires_retrain": True,
    })
    full_vocab["keywords"].append(appended)
    new_only_vocab = {
        "schema": "wakeword-studio.multikws-vocabulary/v1",
        "vocabulary_id": f"{appended['keyword_id']}_generation_only",
        "version": 1,
        "background": copy.deepcopy(base_vocab["background"]),
        "keywords": [{**appended, "class_index": 1}],
        "add_keyword_requires_retrain": True,
    }
    full_vocab_path = run_dir / "VocabularyManifest.json"
    new_only_vocab_path = run_dir / "NEW_ONLY_VOCABULARY.json"
    atomic_write_json(full_vocab_path, full_vocab)
    atomic_write_json(new_only_vocab_path, new_only_vocab)

    template = json.loads((root / "configs/multikws/teacher_six_formal_12k.json").read_text(encoding="utf-8"))

    def effective_base_counts(total: int) -> dict[str, int]:
        validation = max(1, round(total * 0.10))
        test = max(1, round(total * 0.10))
        train_effective = max(1, total - validation - test)
        return {"train": max(1, round(train_effective / 3)), "validation": validation, "test": test}

    positive_base = effective_base_counts(int(plan["positive_samples"]))
    hard_base = effective_base_counts(int(plan["hard_negative_samples"]))
    new_config = copy.deepcopy(template)
    new_config["experiment_id"] = f"{plan['new_dataset_id']}_new_only_generation"
    new_config["vocabulary"] = new_only_vocab_path.relative_to(root).as_posix()
    new_config["dataset"]["dataset_id"] = f"{plan['new_dataset_id']}_new_only"
    new_config["dataset"]["output_root"] = f"datasets/projects/{plan['new_dataset_id']}_new_only"
    new_config["dataset"]["experiment_stage"] = "phase10_new_keyword_generation"
    new_config["dataset"]["base_counts"] = {
        "wakeword_per_keyword": positive_base,
        # The Phase 9 builder requires every speech category to have a
        # positive count. Keep the minimum one-per-split ordinary replay.
        "ordinary_background": {"train": 1, "validation": 1, "test": 1},
        "hard_negative": hard_base,
    }
    new_config["dataset"]["ambient_effective_counts"] = {"train": 0, "validation": 0, "test": 0}
    hard_phrases = [str(item["text"]) for item in plan["hard_negatives"]]
    if len(set(hard_phrases)) < 18:
        raise RuntimeError("Expansion hard-negative bank must contain at least 18 unique phrases")
    new_config["dataset"]["hard_negative_phrases"] = hard_phrases
    new_train_effective = positive_base["train"] * 3 + hard_base["train"] * 3 + 3
    new_config["training"]["effective_train_samples"] = new_train_effective
    new_config_path = run_dir / "NEW_ONLY_GENERATION_CONFIG.json"
    atomic_write_json(new_config_path, new_config)

    final_config = copy.deepcopy(template)
    final_config["experiment_id"] = str(plan["new_dataset_id"])
    final_config["vocabulary"] = full_vocab_path.relative_to(root).as_posix()
    final_config["dataset"]["dataset_id"] = str(plan["new_dataset_id"])
    final_config["dataset"]["output_root"] = f"datasets/projects/{plan['new_dataset_id']}"
    final_config["dataset"]["experiment_stage"] = "phase10_vocabulary_expansion"
    final_config["training"]["effective_train_samples"] = 9000 + new_train_effective
    final_config_path = run_dir / "TRAINING_CONFIG.json"
    atomic_write_json(final_config_path, final_config)
    atomic_write_json(run_dir / "EXPANSION_PLAN.json", normalize_json_value(preflight))
    atomic_write_json(run_dir / "JOB_STATUS.json", job.to_dict())

    new_root = f"datasets/projects/{plan['new_dataset_id']}_new_only"
    merged_root = f"datasets/projects/{plan['new_dataset_id']}"
    features = f"datasets/features/{plan['new_dataset_id']}_train_validation.npz"
    model = str(plan["model_architecture"])
    commands = {
        "generate_new_only_powershell": (
            f".\\.envs\\kokoro\\Scripts\\python.exe .\\phase9\\scripts\\build_multikws_12k_dataset.py "
            f"--config \"{new_config_path.relative_to(root)}\" --output-root \"{new_root}\""
        ),
        "merge_replay_powershell": (
            f"python .\\phase10\\scripts\\merge_keyword_expansion_dataset.py --old-manifest "
            f"\"datasets/projects/teacher_six_multikws_v2_formal_12k/DatasetManifest.json\" --new-manifest "
            f"\"{new_root}/DatasetManifest.json\" --vocabulary \"{full_vocab_path.relative_to(root)}\" "
            f"--output-root \"{merged_root}\""
        ),
        "extract_train_validation_powershell": (
            f".\\.envs\\livekit\\Scripts\\python.exe .\\phase9\\scripts\\extract_multikws_features.py "
            f"--manifest \"{merged_root}/DatasetManifest.json\" --output \"{features}\""
        ),
        "formal_training_wsl": (
            "wsl.exe -d Ubuntu -- bash -lc \"cd /mnt/f/ZJU_intership/task/4/WakeWord-Studio && "
            f"python3 phase9/scripts/run_multikws_training.py --config '{final_config_path.relative_to(root).as_posix()}' "
            f"--model {model} --features '{features}' --run-dir 'runs/multikws/user_expansions/{plan['keyword_id']}/{job.job_id}/formal_{model}'\""
        ),
    }
    atomic_write_json(run_dir / "USER_ACTION_COMMANDS.json", commands)
    return {
        "run_dir": str(run_dir.relative_to(root)), "commands": commands,
        "new_only_config": str(new_config_path.relative_to(root)),
        "training_config": str(final_config_path.relative_to(root)),
        "long_job_started": False,
    }


class RuntimeFeedbackStore:
    def __init__(self, path: Path, *, save_audio: bool = False):
        self.path = Path(path)
        self.save_audio = bool(save_audio)

    def append(self, event: dict[str, Any], verdict: str, ground_truth: str | None = None) -> dict[str, Any]:
        row = {
            "schema": "wakeword-studio.runtime-feedback/v1", "timestamp": utc_now(),
            **event, "verdict": verdict, "ground_truth": ground_truth,
            "audio_saved": bool(self.save_audio and event.get("audio_segment_path")),
        }
        if not self.save_audio:
            row["audio_segment_path"] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


@dataclass(slots=True)
class MicAcceptanceSession:
    model_id: str
    vocabulary_id: str
    target_attempts_per_keyword: int = 10
    results: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, expected_keyword_id: str, result: str) -> None:
        if result not in {"correct", "wrong_keyword", "rejected"}:
            raise ValueError("result must be correct, wrong_keyword, or rejected")
        bucket = self.results.setdefault(expected_keyword_id, {"attempts": 0, "correct": 0, "wrong_keyword": 0, "rejected": 0})
        bucket["attempts"] += 1
        bucket[result] += 1

    def report(self) -> dict[str, Any]:
        rows = {}
        for keyword_id, counts in self.results.items():
            rows[keyword_id] = {**counts, "hit_rate": counts["correct"] / counts["attempts"] if counts["attempts"] else 0.0}
        return {
            "schema": "wakeword-studio.real-mic-acceptance/v1", "report_type": "REAL_MIC_ACCEPTANCE",
            "created_at": utc_now(), "model_id": self.model_id, "vocabulary_id": self.vocabulary_id,
            "target_attempts_per_keyword": self.target_attempts_per_keyword, "per_keyword": rows,
            "is_held_out_test": False,
        }

    def save(self, path: Path) -> Path:
        atomic_write_json(path, self.report())
        return path


@dataclass(slots=True)
class FalseWakeSession:
    started_monotonic: float = field(default_factory=time.monotonic)
    false_wake_count: int = 0

    def report(self, now: float | None = None) -> dict[str, Any]:
        duration = max(0.0, (time.monotonic() if now is None else now) - self.started_monotonic)
        return {
            "schema": "wakeword-studio.false-wake-session/v1", "metric": "FALSE_WAKES_PER_HOUR",
            "false_wake_count": self.false_wake_count, "observed_duration_seconds": duration,
            "false_wakes_per_hour": self.false_wake_count * 3600.0 / duration if duration else 0.0,
            "distinct_from_background_far": True,
        }
