"""Metadata-only QA for a completed Multi-KWS dataset; never opens Test WAV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    status = json.loads((root / "GENERATION_STATUS.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "DatasetManifest.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"Generation is not complete: {status['status']}")
    status_v2 = status.get("schema") == "wakeword-studio.multikws-generation-status/v2"
    if status_v2:
        if (status["completed_base_speech"] != status["planned_base_speech"] or
                status["completed_effective_samples"] != status["planned_effective_samples"] or
                status["failed_samples"] != 0):
            raise RuntimeError("Base/effective generation counts are incomplete")
    elif status["completed_samples"] != status["planned_samples"] or status["failed_samples"] != 0:
        raise RuntimeError("Generation counts are incomplete")
    if file_sha256(args.config.resolve()) != manifest["config_sha256"]:
        raise RuntimeError("Config SHA256 mismatch")
    stored_manifest_sha = manifest.pop("manifest_sha256")
    if canonical_sha256(manifest) != stored_manifest_sha:
        raise RuntimeError("Canonical manifest SHA256 mismatch")
    manifest["manifest_sha256"] = stored_manifest_sha
    records = manifest["records"]
    ids = [str(record["sample_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate deterministic sample_id")
    planned_records = status["planned_effective_samples"] if status_v2 else status["planned_samples"]
    if len(records) != planned_records:
        raise RuntimeError("Manifest record count mismatch")
    dataset_sha = hashlib.sha256(
        "".join(sorted(str(record["sha256"]) for record in records)).encode("ascii")
    ).hexdigest()
    if dataset_sha != manifest["dataset_sha256"]:
        raise RuntimeError("Dataset recorded-hash aggregate mismatch")
    missing_paths = [record["path"] for record in records if not (root / record["path"]).is_file()]
    if missing_paths:
        raise RuntimeError(f"Missing WAV paths: {missing_paths[:3]}")
    actual_splits = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    if actual_splits != manifest["split_counts"]:
        raise RuntimeError("Split count mismatch")
    if manifest.get("test_frozen") is not True:
        raise RuntimeError("Test split is not frozen")
    base_group_split_leakage = None
    if status_v2:
        group_splits: dict[str, set[str]] = {}
        group_variants: dict[str, list[int]] = {}
        for record in records:
            base_id = str(record["base_sample_id"])
            group_splits.setdefault(base_id, set()).add(str(record["split"]))
            group_variants.setdefault(base_id, []).append(int(record["variant_id"]))
        base_group_split_leakage = sum(len(splits) > 1 for splits in group_splits.values())
        if base_group_split_leakage != 0 or manifest.get("base_group_split_leakage") != 0:
            raise RuntimeError("Base augmentation sibling split leakage detected")
        expected_variants = manifest["variants_per_base"]
        for base_id, variants in group_variants.items():
            split = next(iter(group_splits[base_id]))
            expected = 1 if base_id.startswith("ambient-") else int(expected_variants[split])
            if sorted(variants) != list(range(expected)):
                raise RuntimeError(f"Variant coverage mismatch for {base_id}")
        base_records = manifest.get("base_records", [])
        if len(base_records) != status["planned_base_speech"]:
            raise RuntimeError("Base record count mismatch")
        base_ids = [str(record["base_sample_id"]) for record in base_records]
        if len(base_ids) != len(set(base_ids)):
            raise RuntimeError("Duplicate base_sample_id")
    print(json.dumps({
        "QA_PASS": True,
        "dataset_id": manifest["dataset_id"],
        "record_count": len(records),
        "split_counts": actual_splits,
        "manifest_sha256": stored_manifest_sha,
        "dataset_sha256": dataset_sha,
        "planned_base_speech": status.get("planned_base_speech"),
        "BASE_GROUP_SPLIT_LEAKAGE": base_group_split_leakage,
        "TEST_READ": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
