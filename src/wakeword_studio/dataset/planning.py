"""Dry-run planning and conservative resource estimates for formal datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..audio import TARGET_CHANNELS, TARGET_SAMPLE_RATE_HZ, TARGET_SAMPLE_WIDTH_BYTES
from .hard_negatives import HardNegativeGenerator


@dataclass(frozen=True, slots=True)
class DatasetPlanEstimate:
    report: dict[str, object]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def build_estimate(config: dict[str, object]) -> DatasetPlanEstimate:
    targets = {key: int(value) for key, value in dict(config["targets"]).items()}
    required_labels = {"positive", "negative", "hard_negative", "ambient"}
    if set(targets) != required_labels:
        raise ValueError(f"targets must contain exactly {sorted(required_labels)}")

    ratios = {key: float(value) for key, value in dict(config["split_ratios"]).items()}
    if set(ratios) != {"train", "validation", "test"} or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split_ratios must define train/validation/test and sum to 1")

    source_groups = list(config["source_groups"])
    group_ids = [str(group["id"]) for group in source_groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("source group IDs must be unique")
    groups_by_split = {
        split: [str(group["id"]) for group in source_groups if group["split"] == split]
        for split in ratios
    }
    if any(not groups for groups in groups_by_split.values()):
        raise ValueError("every split must own at least one exclusive source group")
    group_splits = {str(group["id"]): str(group["split"]) for group in source_groups}
    if len(group_splits) != len(source_groups):
        raise ValueError("a source group may belong to only one split")

    duration_assumptions = {
        key: float(value) for key, value in dict(config["mean_duration_seconds"]).items()
    }
    split_by_label = {label: split_counts(count, ratios) for label, count in targets.items()}
    total_samples = sum(targets.values())
    duration_by_label = {
        label: targets[label] * duration_assumptions[label] for label in targets
    }
    total_duration_seconds = sum(duration_by_label.values())
    pcm_bytes = int(
        total_duration_seconds
        * TARGET_SAMPLE_RATE_HZ
        * TARGET_CHANNELS
        * TARGET_SAMPLE_WIDTH_BYTES
        + total_samples * 44
    )

    runtime = dict(config["runtime_estimate"])
    tts_base_utterances = int(runtime["tts_base_utterances"])
    tts_minutes = tts_base_utterances / float(runtime["tts_utterances_per_minute"])
    augmentation_minutes = total_samples / float(runtime["augmentations_per_minute"])
    overhead_minutes = float(runtime.get("fixed_overhead_minutes", 0.0))
    generation_minutes = tts_minutes + augmentation_minutes + overhead_minutes
    artifact_multiplier = float(runtime.get("artifact_size_multiplier", 1.0))

    hard_negatives = HardNegativeGenerator().generate()
    tier_counts = {
        str(tier): sum(item.tier == tier for item in hard_negatives) for tier in (1, 2, 3)
    }
    verified_age_groups = sorted(
        {
            str(group["age_group"])
            for group in source_groups
            if group.get("age_source") in {"verified", "reported"} and group.get("age_group")
        }
    )
    warnings: list[str] = []
    if not verified_age_groups:
        warnings.append(
            "No verified/reported age groups are available from the TTS sources; "
            "acoustic age proxies must not be reported as real age coverage."
        )
    test_families = sorted(
        {str(group["family"]) for group in source_groups if group["split"] == "test"}
    )

    return DatasetPlanEstimate(
        report={
            "schema": "wakeword-studio.dataset-build-estimate/v1",
            "project": config["project"],
            "wake_word": config["wake_word"],
            "output_root": config["output_root"],
            "audio_contract": {
                "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                "channels": TARGET_CHANNELS,
                "sample_width_bytes": TARGET_SAMPLE_WIDTH_BYTES,
            },
            "targets": targets,
            "total_samples": total_samples,
            "split_counts_by_label": split_by_label,
            "source_groups_by_split": groups_by_split,
            "speaker_source_group_exclusive": True,
            "test_source_families": test_families,
            "hard_negative_curriculum": {
                "unique_phrases": len(hard_negatives),
                "tier_counts": tier_counts,
                "phrases": [
                    {"text": item.text, "tier": item.tier, "reason": item.reason}
                    for item in hard_negatives
                ],
            },
            "noise_categories": list(config["noise_categories"]),
            "snr_db": list(config["snr_db"]),
            "age_proxies": list(config["age_proxies"]),
            "verified_age_groups": verified_age_groups,
            "estimated_audio_duration_seconds": round(total_duration_seconds, 1),
            "estimated_audio_duration_hours": round(total_duration_seconds / 3600.0, 3),
            "estimated_standardized_wav_mib": round(pcm_bytes / (1024**2), 1),
            "estimated_total_artifacts_gib": round(pcm_bytes * artifact_multiplier / (1024**3), 3),
            "estimated_generation_minutes": round(generation_minutes, 1),
            "runtime_assumptions": runtime,
            "warnings": warnings,
        }
    )


def load_and_estimate(path: Path) -> DatasetPlanEstimate:
    return build_estimate(json.loads(path.read_text(encoding="utf-8")))
