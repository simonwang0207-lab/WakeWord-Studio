"""Build the resumable canonical qingxiaojia_v1 dataset from source manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wakeword_studio.dataset.formal_builder import FormalDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--rebuild-label",
        action="append",
        choices=("positive", "negative", "hard_negative", "ambient"),
        default=[],
        help="Overwrite only this label while retaining the rest of a completed build",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = FormalDatasetBuilder(config, args.source_manifest).build(
        args.output_root,
        rebuild_labels=set(args.rebuild_label),
    )
    errors = manifest.validate(args.output_root / "DatasetManifest.json")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(manifest.summary(), ensure_ascii=False, indent=2), flush=True)
    print(f"manifest={(args.output_root / 'DatasetManifest.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
