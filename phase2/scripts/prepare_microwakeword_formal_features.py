"""Build resumable microWakeWord microfrontend features from DatasetManifest splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.audio_utils import generate_features_for_clip
from wakeword_studio.dataset.manifest import DatasetManifest
from wakeword_studio.frontends import load_training_audio


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
    modes = {"train": "training", "validation": "validation", "test": "testing"}
    mode = modes[split]
    if label == "ambient" and split != "train":
        return f"{mode}_ambient"
    return mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
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
    errors = manifest.validate(manifest_path)
    if errors:
        raise RuntimeError("Dataset validation failed: " + "; ".join(errors[:10]))

    dataset_root = Path(manifest.root).resolve()
    features_root = Path(config["features_root"]).resolve()
    features_root.mkdir(parents=True, exist_ok=True)
    status_path = features_root / "FEATURE_STATUS.json"
    shard_size = int(config["benchmark"]["feature_shard_size"])
    grouped: dict[tuple[str, str], list[object]] = defaultdict(list)
    for row in manifest.records:
        grouped[(row.label, feature_mode(row.split, row.label))].append(row)

    total = len(manifest.records)
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
    }
    atomic_json(status_path, status)

    try:
        for (label, mode), records in sorted(grouped.items()):
            records = sorted(records, key=lambda row: row.record_id)
            for shard_index, offset in enumerate(range(0, len(records), shard_size)):
                shard_records = records[offset : offset + shard_size]
                mode_root = features_root / label / mode
                destination = mode_root / f"shard_{shard_index:03d}_mmap"
                metadata_path = mode_root / f"shard_{shard_index:03d}_records.jsonl"
                metadata = [
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
                    }
                    for index, row in enumerate(shard_records)
                ]
                if destination.exists():
                    if not metadata_path.exists():
                        metadata_path.write_text(
                            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
                            encoding="utf-8",
                        )
                    completed += len(shard_records)
                    print(
                        f"features={completed}/{total} label={label} mode={mode} "
                        f"shard={shard_index} reused=true",
                        flush=True,
                    )
                    continue

                mode_root.mkdir(parents=True, exist_ok=True)
                partial = mode_root / f"shard_{shard_index:03d}.partial-{os.getpid()}"

                def generate():
                    for local_index, row in enumerate(shard_records, start=1):
                        audio = load_training_audio(dataset_root / row.audio_path)
                        yield generate_features_for_clip(
                            audio, step_ms=int(config["frontend"]["window_step_ms"])
                        )
                        if local_index % 25 == 0 or local_index == len(shard_records):
                            elapsed = time.perf_counter() - started
                            print(
                                f"FEATURE_HEARTBEAT label={label} mode={mode} "
                                f"shard={shard_index} item={local_index}/{len(shard_records)} "
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
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
                    encoding="utf-8",
                )
                completed += len(shard_records)
                status["completed_records"] = completed
                status["last_update"] = utc_now()
                status["current_label"] = label
                status["current_mode"] = mode
                atomic_json(status_path, status)
                print(
                    f"features={completed}/{total} label={label} mode={mode} "
                    f"shard={shard_index} reused=false",
                    flush=True,
                )

        summary = {
            "schema": "wakeword-studio.microwakeword-features/v1",
            "created_at": utc_now(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": actual_hash,
            "features_root": str(features_root),
            "frontend": config["frontend"],
            "records": total,
            "counts": dict(sorted(Counter(f"{row.split}:{row.label}" for row in manifest.records).items())),
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
