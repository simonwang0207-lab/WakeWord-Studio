"""Build qingxiaojia_v2 from frozen Kokoro and VoxCPM source manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wakeword_studio.dataset.v2_builder import V2DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = V2DatasetBuilder(config, args.source_manifest).build(
        args.output_root, limit=args.limit
    )
    name = "DatasetManifest.pilot.json" if args.limit else "DatasetManifest.json"
    errors = manifest.validate(args.output_root / name)
    if errors:
        raise SystemExit("\n".join(errors[:100]))
    print(json.dumps(manifest.summary(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
