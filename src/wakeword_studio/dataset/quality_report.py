"""Dataset quality statistics and deterministic listening-list generation."""

from __future__ import annotations

import collections
import random
from pathlib import Path

from .manifest import DatasetManifest


def _noise_category(noise_id: str | None) -> str:
    if not noise_id:
        return "none"
    if noise_id == "clean":
        return "clean"
    if noise_id.startswith("tv_speech:"):
        return "tv_speech"
    if noise_id.startswith("procedural_"):
        return noise_id.removeprefix("procedural_").split(":", 1)[0]
    return noise_id.split(":", 1)[0]


def build_quality_report(manifest_path: Path, listening_per_label: int = 6) -> str:
    manifest_path = manifest_path.resolve()
    manifest = DatasetManifest.load(manifest_path)
    root = Path(manifest.root)
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    errors = manifest.validate(manifest_path)

    labels = collections.Counter(row.label for row in manifest.records)
    splits = collections.Counter(row.split for row in manifest.records)
    sources = collections.Counter(row.speaker.source for row in manifest.records)
    source_splits = collections.Counter(
        f"{row.split}:{row.speaker.source}" for row in manifest.records
    )
    speakers = collections.Counter(
        f"{row.speaker.source}:{row.speaker.speaker_id}" for row in manifest.records
    )
    age_metadata = collections.Counter(
        f"{row.speaker.age_source}:{row.speaker.age_group or 'unknown'}"
        for row in manifest.records
    )
    age_proxies = collections.Counter(
        row.acoustic.acoustic_age_proxy or "none" for row in manifest.records
    )
    noise = collections.Counter(_noise_category(row.acoustic.noise_id) for row in manifest.records)
    snr = collections.Counter(
        "none" if row.acoustic.snr_db is None else f"{row.acoustic.snr_db:g}"
        for row in manifest.records
    )
    hard_tiers = collections.Counter(
        str(row.hard_negative_tier) for row in manifest.records if row.label == "hard_negative"
    )
    split_labels = {
        split: collections.Counter(row.label for row in manifest.records if row.split == split)
        for split in ("train", "validation", "test")
    }
    durations = [float(row.duration_seconds or 0.0) for row in manifest.records]
    duration_outliers = [
        row for row in manifest.records if (row.duration_seconds or 0.0) < 0.3 or (row.duration_seconds or 0.0) > 5.0
    ]
    total_bytes = sum((root / row.audio_path).stat().st_size for row in manifest.records)
    train_speech_families = sorted(
        {
            row.speaker.source
            for row in manifest.records
            if row.split == "train" and row.label != "ambient"
        }
    )

    rng = random.Random(20260829)
    listening: list[object] = []
    for label in ("positive", "negative", "hard_negative", "ambient"):
        candidates = [row for row in manifest.records if row.label == label]
        listening.extend(rng.sample(candidates, min(listening_per_label, len(candidates))))

    def counter_table(title: str, counter: collections.Counter[str]) -> list[str]:
        lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
        lines.extend(f"| {key} | {value} |" for key, value in sorted(counter.items()))
        lines.append("")
        return lines

    lines = [
        "# Qingxiaojia v1 Dataset Quality Report",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Total samples: **{len(manifest.records):,}**",
        f"- Total duration: **{sum(durations) / 3600.0:.3f} hours**",
        f"- Dataset WAV bytes: **{total_bytes / (1024**2):.1f} MiB**",
        "- Canonical contract: **16,000 Hz / mono / PCM16**",
        f"- Manifest/WAV validation errors: **{len(errors)}**",
        f"- Duration outliers (<0.3 s or >5.0 s): **{len(duration_outliers)}**",
        "",
        "Age caveat: no TTS voice supplies verified or reported age metadata. Acoustic",
        "pitch/speed proxies are counted separately and must not be described as real child",
        "or senior voices.",
        "",
        "Source-holdout caveat: MeloTTS has only one Chinese speaker and is kept entirely",
        "in test to prevent speaker leakage. This provides unseen-family evaluation, but",
        f"the current training speech families are only: `{', '.join(train_speech_families)}`.",
        "",
    ]
    lines += counter_table("Labels", labels)
    lines += counter_table("Splits", splits)
    lines += [
        "## Split × label",
        "",
        "| Split | Positive | Negative | Hard negative | Ambient |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        counts = split_labels[split]
        lines.append(
            f"| {split} | {counts['positive']} | {counts['negative']} | "
            f"{counts['hard_negative']} | {counts['ambient']} |"
        )
    lines.append("")
    lines += counter_table("Source distribution", sources)
    lines += counter_table("Source × split distribution", source_splits)
    lines += counter_table("Speaker distribution", speakers)
    lines += counter_table("Age metadata distribution", age_metadata)
    lines += counter_table("Acoustic age proxy distribution", age_proxies)
    lines += counter_table("Noise distribution", noise)
    lines += counter_table("SNR distribution (dB)", snr)
    lines += counter_table("Hard-negative tier distribution", hard_tiers)
    lines += [
        "## Duration",
        "",
        f"- Minimum: {min(durations):.3f} s",
        f"- Mean: {sum(durations) / len(durations):.3f} s",
        f"- Maximum: {max(durations):.3f} s",
        "",
        "## Deterministic listening list",
        "",
        "The following paths were sampled with seed `20260829`:",
        "",
    ]
    for row in listening:
        lines.append(
            f"- `{row.audio_path}` — split={row.split}, label={row.label}, "
            f"source={row.speaker.source}, speaker={row.speaker.speaker_id}, "
            f"text={row.text!r}"
        )
    lines += ["", "## Validation details", ""]
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- No missing files, audio-header violations, or source/group split leakage detected.")
    lines.append("")
    return "\n".join(lines)
