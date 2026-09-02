"""Build manifest-aligned microfrontend features for qingxiaojia_v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from mmap_ninja.ragged import RaggedMmap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microwakeword.audio.audio_utils import generate_features_for_clip
from wakeword_studio.dataset.manifest import DatasetManifest
from wakeword_studio.frontends import load_training_audio
from wakeword_studio.training.streaming_windows import (
    extract_streaming_window,
    plan_streaming_window,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def feature_mode(split: str, label: str) -> str:
    mode = {"train": "training", "validation": "validation"}[split]
    if label == "ambient" and split == "validation":
        return "validation_ambient"
    return mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["dataset_manifest"]).resolve()
    expected_hash = str(config["dataset_manifest_sha256"]).lower()
    actual_hash = sha256_file(manifest_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"DatasetManifest hash mismatch: expected={expected_hash} actual={actual_hash}"
        )

    manifest = DatasetManifest.load(manifest_path)
    if manifest.wake_word != config["wake_word"]:
        raise RuntimeError("Dataset wake word does not match training config")
    errors = manifest.validate(manifest_path, check_files=False)
    if errors:
        raise RuntimeError("Dataset metadata validation failed: " + "; ".join(errors[:10]))

    generation = config["feature_generation"]
    included_splits = set(generation["included_splits"])
    if "test" in included_splits:
        raise RuntimeError("Preflight feature generation must not include held-out Test")
    window_ms = float(config["frontend"]["clip_duration_ms"])
    sample_rate_hz = int(config["frontend"]["sample_rate_hz"])
    seed = int(config["streaming_window_policy"]["seed"])
    dataset_root = Path(manifest.root).resolve()
    features_root = Path(config["features_root"]).resolve()
    audit_csv = Path(config["streaming_window_policy"]["audit_csv"]).resolve()
    audit_json = audit_csv.with_suffix(".json")

    rows = [row for row in manifest.records if row.split in included_splits]
    plans: dict[str, object] = {}
    maximum_phrase_span_ms = 0.0
    for row in rows:
        plan = plan_streaming_window(
            record_id=row.record_id,
            label=row.label,
            duration_seconds=float(row.duration_seconds or 0.0),
            phrase_start_ms=row.acoustic.phrase_start_ms,
            phrase_end_ms=row.acoustic.phrase_end_ms,
            phrase_placement=row.acoustic.phrase_placement,
            window_ms=window_ms,
            seed=seed,
        )
        if not plan.alignment_ok:
            raise RuntimeError(f"{row.record_id}: streaming-window alignment failed")
        plans[row.record_id] = plan
        if plan.phrase_start_ms is not None and plan.phrase_end_ms is not None:
            maximum_phrase_span_ms = max(
                maximum_phrase_span_ms, plan.phrase_end_ms - plan.phrase_start_ms
            )

    selected: list[object] = []
    for label in ("positive", "negative", "hard_negative"):
        candidates = sorted(
            (row for row in rows if row.split == "train" and row.label == label),
            key=lambda row: hashlib.sha256(f"{seed}:{row.record_id}:audit".encode()).hexdigest(),
        )
        selected.extend(candidates[:10])
    if len(selected) != 30:
        raise RuntimeError("Alignment audit requires 10 rows from each requested class")

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "wav_path",
        "split",
        "label",
        "source",
        "speaker_id",
        "text",
        "window_start_ms",
        "window_end_ms",
        "phrase_start_ms",
        "phrase_end_ms",
        "effective_phrase_start_ms",
        "effective_phrase_end_ms",
        "phrase_placement",
        "leading_padding_ms",
        "trailing_padding_ms",
        "full_phrase_contained",
        "overlength_terminal_decision_window",
        "alignment_ok",
    ]
    with audit_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            plan = plans[row.record_id]
            writer.writerow(
                {
                    "record_id": row.record_id,
                    "wav_path": str(dataset_root / row.audio_path),
                    "split": row.split,
                    "label": row.label,
                    "source": row.speaker.source,
                    "speaker_id": row.speaker.speaker_id,
                    "text": row.text,
                    "window_start_ms": round(plan.window_start_ms, 3),
                    "window_end_ms": round(plan.window_end_ms, 3),
                    "phrase_start_ms": plan.phrase_start_ms,
                    "phrase_end_ms": plan.phrase_end_ms,
                    "effective_phrase_start_ms": plan.effective_phrase_start_ms,
                    "effective_phrase_end_ms": plan.effective_phrase_end_ms,
                    "phrase_placement": plan.phrase_placement,
                    "leading_padding_ms": round(plan.leading_padding_ms, 3),
                    "trailing_padding_ms": round(plan.trailing_padding_ms, 3),
                    "full_phrase_contained": plan.full_phrase_contained,
                    "overlength_terminal_decision_window": (
                        plan.overlength_terminal_decision_window
                    ),
                    "alignment_ok": plan.alignment_ok,
                }
            )

    audit_summary = {
        "schema": "wakeword-studio.streaming-window-alignment-audit/v1",
        "created_at": utc_now(),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": actual_hash,
        "window_ms": window_ms,
        "included_splits": sorted(included_splits),
        "full_records_checked": len(rows),
        "full_positive_checked": sum(row.label == "positive" for row in rows),
        "full_negative_class_checked": sum(row.label != "positive" for row in rows),
        "maximum_phrase_span_ms": round(maximum_phrase_span_ms, 3),
        "alignment_failures": 0,
        "positive_full_phrase_contained": sum(
            row.label == "positive" and plans[row.record_id].full_phrase_contained
            for row in rows
        ),
        "positive_overlength_terminal_decision_windows": sum(
            row.label == "positive"
            and plans[row.record_id].overlength_terminal_decision_window
            for row in rows
        ),
        "negative_records_with_phrase_interval": 0,
        "manual_audit_rows": dict(Counter(row.label for row in selected)),
        "audit_csv": str(audit_csv),
        "held_out_test_loaded": False,
    }
    atomic_json(audit_json, audit_summary)
    print("STREAMING ALIGNMENT AUDIT COMPLETE", flush=True)
    print(json.dumps(audit_summary, ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        return

    features_root.mkdir(parents=True, exist_ok=True)
    status_path = features_root / "FEATURE_STATUS.json"
    shard_size = int(generation["feature_shard_size"])
    grouped: dict[tuple[str, str], list[object]] = defaultdict(list)
    for row in rows:
        grouped[(row.label, feature_mode(row.split, row.label))].append(row)

    total = len(rows)
    completed = 0
    started = time.perf_counter()
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "start_time": utc_now(),
        "last_update": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_hash,
        "features_root": str(features_root),
        "total_records": total,
        "completed_records": 0,
        "held_out_test_loaded": False,
    }
    atomic_json(status_path, status)

    try:
        for (label, mode), group_rows in sorted(grouped.items()):
            group_rows = sorted(group_rows, key=lambda row: row.record_id)
            for shard_index, offset in enumerate(range(0, len(group_rows), shard_size)):
                shard_rows = group_rows[offset : offset + shard_size]
                mode_root = features_root / label / mode
                destination = mode_root / f"shard_{shard_index:03d}_mmap"
                metadata_path = mode_root / f"shard_{shard_index:03d}_records.jsonl"
                metadata = []
                for index, row in enumerate(shard_rows):
                    plan = plans[row.record_id]
                    metadata.append(
                        {
                            "feature_index": index,
                            "record_id": row.record_id,
                            "audio_path": row.audio_path,
                            "label": row.label,
                            "split": row.split,
                            "text": row.text,
                            "source": row.speaker.source,
                            "speaker_id": row.speaker.speaker_id,
                            "noise_id": row.acoustic.noise_id,
                            "snr_db": row.acoustic.snr_db,
                            "hard_negative_tier": row.hard_negative_tier,
                            **plan.to_dict(),
                        }
                    )
                if destination.exists():
                    if not metadata_path.exists():
                        raise RuntimeError(f"Existing feature shard lacks metadata: {destination}")
                    completed += len(shard_rows)
                    print(
                        f"features={completed}/{total} label={label} mode={mode} "
                        f"shard={shard_index} reused=true",
                        flush=True,
                    )
                    continue

                mode_root.mkdir(parents=True, exist_ok=True)
                partial = mode_root / f"shard_{shard_index:03d}.partial-{os.getpid()}"

                def generate():
                    for local_index, row in enumerate(shard_rows, start=1):
                        audio = load_training_audio(dataset_root / row.audio_path)
                        window = extract_streaming_window(
                            audio,
                            plans[row.record_id],
                            sample_rate_hz=sample_rate_hz,
                            window_ms=window_ms,
                        )
                        if len(window) != int(round(sample_rate_hz * window_ms / 1000.0)):
                            raise RuntimeError(f"{row.record_id}: feature window length mismatch")
                        yield generate_features_for_clip(
                            window, step_ms=int(config["frontend"]["window_step_ms"])
                        )
                        if local_index % 25 == 0 or local_index == len(shard_rows):
                            elapsed = time.perf_counter() - started
                            print(
                                f"FEATURE_HEARTBEAT label={label} mode={mode} "
                                f"shard={shard_index} item={local_index}/{len(shard_rows)} "
                                f"elapsed_seconds={elapsed:.1f}",
                                flush=True,
                            )

                RaggedMmap.from_generator(
                    out_dir=partial,
                    sample_generator=generate(),
                    batch_size=32,
                    verbose=False,
                )
                partial.replace(destination)
                metadata_path.write_text(
                    "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata),
                    encoding="utf-8",
                )
                completed += len(shard_rows)
                status.update(
                    {
                        "completed_records": completed,
                        "last_update": utc_now(),
                        "current_label": label,
                        "current_mode": mode,
                    }
                )
                atomic_json(status_path, status)
                print(
                    f"features={completed}/{total} label={label} mode={mode} "
                    f"shard={shard_index} reused=false",
                    flush=True,
                )

        summary = {
            "schema": "wakeword-studio.microwakeword-features/v2",
            "created_at": utc_now(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": actual_hash,
            "features_root": str(features_root),
            "frontend": config["frontend"],
            "streaming_window_policy": config["streaming_window_policy"],
            "included_splits": sorted(included_splits),
            "held_out_test_loaded": False,
            "records": total,
            "counts": dict(sorted(Counter(f"{row.split}:{row.label}" for row in rows).items())),
            "alignment_audit_sha256": sha256_file(audit_csv),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        atomic_json(features_root / "summary.json", summary)
        status.update(
            {
                "status": "COMPLETED",
                "completed_records": total,
                "last_update": utc_now(),
                "end_time": utc_now(),
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
        atomic_json(status_path, status)
        print("FEATURE GENERATION COMPLETE", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        status.update(
            {
                "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                "last_update": utc_now(),
                "error": repr(exc),
                "completed_records": completed,
            }
        )
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
