"""Copy a compact, deterministic pre-training review set into one folder."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from wakeword_studio.dataset.manifest import DatasetManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kokoro-source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = DatasetManifest.load(args.manifest)
    dataset_root = Path(manifest.root).resolve()
    source_raw = json.loads(args.kokoro_source_manifest.read_text(encoding="utf-8"))
    source_root = Path(source_raw["root"]).resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.jsonl"
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            managed_file = (output_root / str(row["audio_path"])).resolve()
            if managed_file.parent != output_root:
                raise RuntimeError(f"Refusing to remove unmanaged review path: {managed_file}")
            managed_file.unlink(missing_ok=True)
        metadata_path.unlink()
    rng = random.Random(20260829)

    selections: list[dict[str, object]] = []
    hard_sources = [
        row
        for row in source_raw["records"]
        if row["split"] == "validation" and row["label"] == "hard_negative"
    ]
    for row in sorted(hard_sources, key=lambda item: (item["hard_negative_tier"], item["text"])):
        selections.append(
            {
                "kind": "clean_hard_negative",
                "source": source_root / row["path"],
                "label": row["label"],
                "text": row["text"],
                "speaker": row["speaker_id"],
                "tier": row["hard_negative_tier"],
                "noise_id": "clean_source",
            }
        )

    formal_by_label = {
        label: [row for row in manifest.records if row.label == label]
        for label in ("positive", "negative", "ambient")
    }
    positive_anchors = [
        next(
            row
            for row in formal_by_label["positive"]
            if row.speaker.source == "melotts" and row.acoustic.noise_id == "clean"
        ),
        next(
            row
            for row in formal_by_label["positive"]
            if row.speaker.source == "melotts"
            and (row.acoustic.noise_id or "").startswith("tv_speech:")
        ),
    ]
    remaining_positive = [
        row
        for row in formal_by_label["positive"]
        if row.speaker.source == "kokoro" and row.acoustic.noise_id != "clean"
    ]
    formal_records = (
        positive_anchors
        + rng.sample(remaining_positive, 4)
        + rng.sample(formal_by_label["negative"], 4)
        + rng.sample(formal_by_label["ambient"], 4)
    )
    for row in formal_records:
        selections.append(
            {
                "kind": "formal_augmented",
                "source": dataset_root / row.audio_path,
                "label": row.label,
                "text": row.text,
                "speaker": row.speaker.speaker_id,
                "tier": row.hard_negative_tier,
                "noise_id": row.acoustic.noise_id,
            }
        )

    metadata: list[dict[str, object]] = []
    for index, item in enumerate(selections, start=1):
        source = Path(item.pop("source"))
        destination = output_root / f"{index:02d}_{item['kind']}_{item['label']}_{source.name}"
        if not destination.exists():
            shutil.copy2(source, destination)
        metadata.append(
            {
                "order": index,
                "audio_path": destination.name,
                "original_path": str(source),
                **item,
            }
        )
    metadata_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
        encoding="utf-8",
    )
    print(f"review_files={len(metadata)}")
    print(f"review_root={output_root}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
