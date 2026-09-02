"""Freeze v2 source/speaker allocation without modifying v1 source audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v1-kokoro-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = args.v1_kokoro_manifest.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    allocation = config["speaker_allocation"]
    speaker_split = {
        speaker: split
        for split, families in allocation.items()
        for speaker in families["kokoro"]
    }
    if len(speaker_split) != sum(len(x["kokoro"]) for x in allocation.values()):
        raise RuntimeError("Kokoro speaker appears in more than one v2 split")

    records: list[dict[str, object]] = []
    for row in source["records"]:
        speaker = str(row["speaker_id"])
        if speaker not in speaker_split:
            continue
        copied = dict(row)
        copied["split"] = speaker_split[speaker]
        copied["source_group_id"] = f"kokoro:{speaker}"
        copied["reference_speaker_id"] = None
        copied["reference_age_group"] = None
        copied["reference_age_group_source"] = None
        copied["perceived_age_verified"] = False
        records.append(copied)

    present = {str(row["speaker_id"]) for row in records}
    if present != set(speaker_split):
        raise RuntimeError(
            f"Missing allocated Kokoro speakers: {sorted(set(speaker_split) - present)}"
        )
    external = set(config["external_v1_test_speakers"])
    overlap = {f"kokoro:{speaker}" for speaker in present} & external
    if overlap:
        raise RuntimeError(f"External v1 Test speaker admitted to v2: {sorted(overlap)}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "wakeword-studio.source-manifest/v2",
        "target": config["wake_word"],
        "generator": "reuse_immutable_v1_kokoro_sources_with_v2_split_map",
        "model_repo": source.get("model_repo"),
        "root": source["root"],
        "upstream_manifest": str(source_path),
        "upstream_manifest_sha256": sha256_file(source_path),
        "records": records,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"V2 KOKORO SOURCE MAP READY records={len(records)} speakers={len(present)} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
