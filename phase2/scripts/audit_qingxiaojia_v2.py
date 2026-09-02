"""Full qingxiaojia_v2 contract, leakage, distribution, and listening audit."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from wakeword_studio.dataset.manifest import DatasetManifest, sha256_file


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def counter_dict(values) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(value) for value in values).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--listening-dir", type=Path, required=True)
    args = parser.parse_args()

    print("V2 FULL AUDIT START", flush=True)
    manifest_path = args.manifest.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = DatasetManifest.load(manifest_path)
    root = Path(manifest.root).resolve()
    errors = manifest.validate(manifest_path)
    expected_counts = config["counts"]
    actual_counts = {
        split: {
            label: sum(
                row.split == split and row.label == label for row in manifest.records
            )
            for label in ("positive", "negative", "hard_negative", "ambient")
        }
        for split in ("train", "validation", "test")
    }
    if actual_counts != expected_counts:
        errors.append(f"split/label counts mismatch: {actual_counts} != {expected_counts}")

    speaker_splits: dict[str, set[str]] = {}
    group_splits: dict[str, set[str]] = {}
    utterance_splits: dict[str, set[str]] = {}
    hash_records: dict[str, list[object]] = {}
    path_records: dict[str, list[str]] = {}
    hash_mismatches: list[str] = []
    total_bytes = 0
    for index, row in enumerate(manifest.records, start=1):
        speaker_key = f"{row.speaker.source}:{row.speaker.speaker_id}"
        if row.label != "ambient":
            speaker_splits.setdefault(speaker_key, set()).add(row.split)
        if row.source_group_id:
            group_splits.setdefault(row.source_group_id, set()).add(row.split)
        if row.source_utterance_id:
            utterance_splits.setdefault(row.source_utterance_id, set()).add(row.split)
        if row.sha256:
            hash_records.setdefault(row.sha256, []).append(row)
        path_records.setdefault(row.audio_path, []).append(row.record_id)
        path = root / row.audio_path
        total_bytes += path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_hash != row.sha256:
            hash_mismatches.append(row.record_id)
        if index % 500 == 0 or index == len(manifest.records):
            print(f"V2 AUDIT HEARTBEAT files={index}/{len(manifest.records)}", flush=True)

    speaker_leakage = {
        key: sorted(value) for key, value in speaker_splits.items() if len(value) > 1
    }
    group_leakage = {
        key: sorted(value) for key, value in group_splits.items() if len(value) > 1
    }
    utterance_leakage = {
        key: sorted(value) for key, value in utterance_splits.items() if len(value) > 1
    }
    duplicate_hashes = {
        digest: [row.record_id for row in rows]
        for digest, rows in hash_records.items()
        if len(rows) > 1
    }
    cross_split_duplicate_hashes = {
        digest: ids
        for digest, ids in duplicate_hashes.items()
        if len({next(row.split for row in hash_records[digest] if row.record_id == item) for item in ids})
        > 1
    }
    duplicate_paths = {key: value for key, value in path_records.items() if len(value) > 1}
    data_roots = [root / split for split in ("train", "validation", "test")]
    partial_files = [
        path.as_posix()
        for data_root in data_roots
        for path in data_root.rglob("*.partial*")
    ]
    wav_count = sum(1 for data_root in data_roots for _ in data_root.rglob("*.wav"))
    v1_path = Path(config["preserve"]["v1_manifest"]).resolve()
    v1_hash = sha256_file(v1_path)
    v1_preserved = v1_hash == config["preserve"]["v1_manifest_sha256"]

    if speaker_leakage:
        errors.append(f"speaker leakage: {speaker_leakage}")
    if group_leakage:
        errors.append(f"source-group leakage: {group_leakage}")
    if utterance_leakage:
        errors.append(f"source-utterance leakage: {utterance_leakage}")
    if duplicate_hashes:
        errors.append(f"duplicate audio hashes: {len(duplicate_hashes)}")
    if duplicate_paths:
        errors.append(f"duplicate audio paths: {duplicate_paths}")
    if hash_mismatches:
        errors.append(f"manifest SHA-256 mismatches: {len(hash_mismatches)}")
    if partial_files:
        errors.append(f"partial files remain: {partial_files[:20]}")
    if wav_count != len(manifest.records):
        errors.append(f"WAV count {wav_count} != manifest records {len(manifest.records)}")
    if not v1_preserved:
        errors.append(f"v1 manifest hash changed: {v1_hash}")

    external = set(config["external_v1_test_speakers"])
    external_overlap = sorted(external & set(speaker_splits))
    if external_overlap:
        errors.append(f"v1 external speaker admitted to v2: {external_overlap}")

    durations = {
        split: stats(
            [float(row.duration_seconds) for row in manifest.records if row.split == split]
        )
        for split in ("train", "validation", "test")
    }
    leading = {
        split: stats(
            [
                float(row.acoustic.leading_silence_seconds)
                for row in manifest.records
                if row.split == split and row.acoustic.leading_silence_seconds is not None
            ]
        )
        for split in ("train", "validation", "test")
    }
    trailing = {
        split: stats(
            [
                float(row.acoustic.trailing_silence_seconds)
                for row in manifest.records
                if row.split == split and row.acoustic.trailing_silence_seconds is not None
            ]
        )
        for split in ("train", "validation", "test")
    }
    duration_bins = {
        split: counter_dict(
            row.acoustic.duration_bin for row in manifest.records if row.split == split
        )
        for split in ("train", "validation", "test")
    }
    placement = {
        split: counter_dict(
            row.acoustic.phrase_placement
            for row in manifest.records
            if row.split == split and row.label != "ambient"
        )
        for split in ("train", "validation", "test")
    }
    noise = {
        split: counter_dict(
            "clean"
            if row.acoustic.noise_id == "clean"
            else str(row.acoustic.noise_id).split(":", 1)[0]
            for row in manifest.records
            if row.split == split
        )
        for split in ("train", "validation", "test")
    }
    snr = {
        split: counter_dict(
            "none" if row.acoustic.snr_db is None else f"{row.acoustic.snr_db:g}"
            for row in manifest.records
            if row.split == split
        )
        for split in ("train", "validation", "test")
    }
    family_counts = counter_dict(row.speaker.source for row in manifest.records)
    split_family_counts = {
        split: counter_dict(
            row.speaker.source for row in manifest.records if row.split == split
        )
        for split in ("train", "validation", "test")
    }
    split_speakers = {
        split: {
            family: sorted(
                {
                    row.speaker.speaker_id
                    for row in manifest.records
                    if row.split == split and row.speaker.source == family
                }
            )
            for family in ("kokoro", "voxcpm15")
        }
        for split in ("train", "validation", "test")
    }
    hard_text = counter_dict(
        row.text for row in manifest.records if row.label == "hard_negative"
    )
    age_metadata = counter_dict(
        f"{row.speaker.reference_age_group_source}:{row.speaker.reference_age_group}"
        for row in manifest.records
        if row.speaker.source == "voxcpm15"
    )
    positive_alignment_missing = [
        row.record_id
        for row in manifest.records
        if row.label == "positive"
        and (row.acoustic.phrase_start_ms is None or row.acoustic.phrase_end_ms is None)
    ]
    if positive_alignment_missing:
        errors.append(f"positive phrase alignment missing: {len(positive_alignment_missing)}")

    listening_dir = args.listening_dir.resolve()
    listening_dir.mkdir(parents=True, exist_ok=True)
    for old in listening_dir.glob("*.wav"):
        old.unlink()
    selected: list[object] = []
    for split in ("train", "validation", "test"):
        for family in ("kokoro", "voxcpm15"):
            candidates = [
                row
                for row in manifest.records
                if row.split == split and row.label == "positive" and row.speaker.source == family
            ]
            if candidates:
                selected.append(candidates[len(candidates) // 2])
    for family in ("kokoro", "voxcpm15"):
        selected.append(
            next(
                row
                for row in manifest.records
                if row.label == "negative" and row.speaker.source == family
            )
        )
        for text in ("你好，小甲", "你好，青甲"):
            selected.append(
                next(
                    row
                    for row in manifest.records
                    if row.label == "hard_negative"
                    and row.speaker.source == family
                    and row.text == text
                )
            )
    for split in ("train", "validation", "test"):
        selected.append(
            next(
                row
                for row in manifest.records
                if row.split == split and row.label == "ambient"
            )
        )

    listening_records: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        destination = listening_dir / (
            f"{index:02d}_{row.split}_{row.label}_{row.speaker.source}_"
            f"{row.speaker.speaker_id}.wav"
        )
        shutil.copy2(root / row.audio_path, destination)
        listening_records.append(
            {
                "path": destination.name,
                "record_id": row.record_id,
                "split": row.split,
                "label": row.label,
                "source": row.speaker.source,
                "speaker_id": row.speaker.speaker_id,
                "text": row.text,
                "reference_age_group": row.speaker.reference_age_group,
                "reference_age_group_source": row.speaker.reference_age_group_source,
                "perceived_age_verified": row.speaker.perceived_age_verified,
                "duration_seconds": row.duration_seconds,
                "leading_silence_seconds": row.acoustic.leading_silence_seconds,
                "trailing_silence_seconds": row.acoustic.trailing_silence_seconds,
                "phrase_placement": row.acoustic.phrase_placement,
                "noise_id": row.acoustic.noise_id,
                "snr_db": row.acoustic.snr_db,
            }
        )
    (listening_dir / "metadata.json").write_text(
        json.dumps(listening_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "schema": "wakeword-studio.qingxiaojia-v2-quality-audit/v1",
        "status": "PASSED_AWAITING_HUMAN_LISTENING" if not errors else "FAILED",
        "training_authorized": False,
        "manifest": str(manifest_path),
        "records": len(manifest.records),
        "audio_hours": round(
            sum(float(row.duration_seconds) for row in manifest.records) / 3600, 6
        ),
        "dataset_bytes": total_bytes,
        "dataset_mib": round(total_bytes / 1048576, 3),
        "counts": actual_counts,
        "family_counts": family_counts,
        "split_family_counts": split_family_counts,
        "split_speakers": split_speakers,
        "melotts_role": "v1_external_test_only_unseen_tts_family",
        "duration_seconds": durations,
        "duration_bins": duration_bins,
        "leading_silence_seconds": leading,
        "trailing_silence_seconds": trailing,
        "phrase_placement": placement,
        "noise": noise,
        "snr_db": snr,
        "hard_negative_text": hard_text,
        "reference_age_metadata": age_metadata,
        "perceived_age_verified": False,
        "speaker_leakage": speaker_leakage,
        "source_group_leakage": group_leakage,
        "source_utterance_leakage": utterance_leakage,
        "duplicate_hashes": len(duplicate_hashes),
        "cross_split_duplicate_hashes": len(cross_split_duplicate_hashes),
        "hash_mismatches": hash_mismatches,
        "partial_files": partial_files,
        "wav_count": wav_count,
        "v1_manifest_sha256": v1_hash,
        "v1_preserved": v1_preserved,
        "external_v1_speaker_overlap": external_overlap,
        "listening_files": listening_records,
        "errors": errors,
    }
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "quality_audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# qingxiaojia_v2 Dataset Quality Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- Records: **{report['records']:,}**",
        f"- Audio: **{report['audio_hours']:.3f} h**",
        f"- WAV size: **{report['dataset_mib']:.1f} MiB**",
        f"- Validation errors: **{len(errors)}**",
        f"- Speaker/source-group/source-utterance leakage: **{len(speaker_leakage)}/{len(group_leakage)}/{len(utterance_leakage)}**",
        f"- Duplicate hashes: **{len(duplicate_hashes)}**",
        f"- v1 external benchmark preserved: **{v1_preserved}**",
        "- Training authorized: **False**",
        "",
        "## Speaker allocation",
        "",
        "| Split | Kokoro | VoxCPM1.5/AISHELL-3 reference |",
        "|---|---|---|",
    ]
    for split in ("train", "validation", "test"):
        lines.append(
            f"| {split} | {', '.join(split_speakers[split]['kokoro'])} | "
            f"{', '.join(split_speakers[split]['voxcpm15'])} |"
        )
    lines += ["", "## Duration and padding", ""]
    for split in ("train", "validation", "test"):
        lines.append(
            f"- {split}: duration={durations[split]}, leading={leading[split]}, "
            f"trailing={trailing[split]}, placement={placement[split]}"
        )
    lines += [
        "",
        "AISHELL-3 A/B/C/D values are retained as verified demographic metadata only.",
        "`perceived_age_verified` is false for every record.",
        "",
        f"Listening directory: `{listening_dir}`",
        "",
    ]
    if errors:
        lines += ["## Errors", ""] + [f"- {error}" for error in errors]
    (report_dir / "quality_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"V2 FULL AUDIT COMPLETE status={report['status']} errors={len(errors)} "
        f"report={json_path} listening={listening_dir}",
        flush=True,
    )
    if errors:
        raise SystemExit("\n".join(errors[:100]))


if __name__ == "__main__":
    main()
