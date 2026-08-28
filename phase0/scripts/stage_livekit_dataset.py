"""Stage the human-approved Kokoro WAVs into LiveKit's official split layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _stage(source: Path, destinations: list[tuple[Path, int]]) -> list[dict[str, str]]:
    wavs = sorted(source.glob("*.wav"))
    expected = sum(count for _, count in destinations)
    if len(wavs) < expected:
        raise RuntimeError(f"Need {expected} WAVs in {source}, found {len(wavs)}")

    manifest: list[dict[str, str]] = []
    offset = 0
    for destination, count in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for local_index, wav in enumerate(wavs[offset : offset + count]):
            out = destination / f"clip_{local_index:06d}.wav"
            shutil.copy2(wav, out)
            manifest.append({"source": str(wav), "destination": str(out)})
        offset += count
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    rows += _stage(
        args.dataset / "positive",
        [(args.output / "positive_train", 40), (args.output / "positive_test", 10)],
    )
    rows += _stage(
        args.dataset / "negative",
        [(args.output / "negative_train", 40), (args.output / "negative_test", 10)],
    )
    (args.output / "staging_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Staged {len(rows)} WAVs into {args.output}")


if __name__ == "__main__":
    main()
