"""Create an immutable replay dataset without decoding frozen Test audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def link_or_copy(source: Path, destination: Path) -> str:
    """Resume safely; never overwrite a pre-existing destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing replay target differs: {destination}")
        return "reused"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def merge(
    old_manifest_path: Path,
    new_manifest_path: Path,
    vocabulary_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    old_manifest_path, new_manifest_path = old_manifest_path.resolve(), new_manifest_path.resolve()
    output_root = output_root.resolve()
    output_manifest = output_root / "DatasetManifest.json"
    if output_manifest.exists():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            return existing
        raise RuntimeError("Partial output manifest exists; inspect it before resuming")
    old = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    new = json.loads(new_manifest_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocabulary_path.resolve().read_text(encoding="utf-8"))
    new_keyword = vocabulary["keywords"][-1]
    new_keyword_id, new_class_index = str(new_keyword["keyword_id"]), int(new_keyword["class_index"])
    if new_class_index <= max(int(item["class_index"]) for item in vocabulary["keywords"][:-1]):
        raise ValueError("New keyword class must be appended after all replay classes")
    records: list[dict[str, Any]] = []
    methods = {"hardlink": 0, "copy": 0, "reused": 0}

    def append_rows(manifest: dict[str, Any], source_root: Path, prefix: str, remap_new: bool) -> None:
        for row in manifest["records"]:
            source = source_root / str(row["path"])
            if not source.is_file() or source.stat().st_size <= 44:
                raise FileNotFoundError(source)
            item = dict(row)
            original_keyword = str(item["keyword_id"])
            if remap_new and original_keyword != "background":
                item["keyword_id"] = new_keyword_id
                item["class_index"] = new_class_index
            item["record_id"] = f"{prefix}-{item['record_id']}"
            item["sample_id"] = item["record_id"]
            relative = Path(prefix) / str(item["path"])
            item["path"] = relative.as_posix()
            methods[link_or_copy(source, output_root / relative)] += 1
            records.append(item)

    # Path/metadata/stat operations are allowed for frozen Test. No WAV is decoded.
    append_rows(old, old_manifest_path.parent, "replay", False)
    append_rows(new, new_manifest_path.parent, "new_keyword", True)
    ids = [str(item["record_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Merged record IDs are not unique")
    split_counts = {name: sum(item["split"] == name for item in records) for name in ("train", "validation", "test")}
    class_counts = {
        "background": sum(item["keyword_id"] == "background" for item in records),
        **{str(item["keyword_id"]): sum(row["keyword_id"] == item["keyword_id"] for row in records)
           for item in vocabulary["keywords"]},
    }
    manifest: dict[str, Any] = {
        "schema": "wakeword-studio.multikws-dataset/v3",
        "status": "COMPLETED", "dataset_id": output_root.name,
        "profile": "phase10_replay_expansion", "experiment_stage": "phase10_vocabulary_expansion",
        "production_quality": False, "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(output_root), "parent_dataset_id": str(old["dataset_id"]),
        "parent_dataset_immutable": True, "new_keyword_id": new_keyword_id,
        "vocabulary": str(vocabulary_path.resolve()), "records": records,
        "split_counts": split_counts, "class_counts": class_counts,
        "replay_record_count": len(old["records"]), "new_record_count": len(new["records"]),
        "link_methods": methods, "test_frozen": True, "test_read_during_merge": False,
        "TEST_READ": False,
    }
    manifest["dataset_sha256"] = hashlib.sha256(
        "".join(sorted(str(item["sha256"]) for item in records)).encode("ascii")
    ).hexdigest()
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_manifest.with_name(output_manifest.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output_manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.old_manifest, args.new_manifest, args.vocabulary, args.output_root)
    print(json.dumps({key: result[key] for key in (
        "dataset_id", "split_counts", "class_counts", "replay_record_count",
        "new_record_count", "TEST_READ",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
