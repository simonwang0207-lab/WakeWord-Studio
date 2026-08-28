"""Create one DatasetManifest from generated or externally supplied audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wakeword_studio.dataset.adapter import DatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    imported = sub.add_parser("import-folder", help="Import positive/negative/hard_negative dirs")
    imported.add_argument("--folder", type=Path, required=True)
    imported.add_argument("--wake-word", required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--standardized-root", type=Path)
    generated = sub.add_parser("from-generator", help="Normalize an existing generator manifest")
    generated.add_argument("--generator-manifest", type=Path, required=True)
    generated.add_argument("--output", type=Path, required=True)
    generated.add_argument("--standardized-root", type=Path)
    generated.add_argument("--limit-per-label", type=int)
    args = parser.parse_args()

    adapter = DatasetAdapter()
    if args.command == "import-folder":
        root = args.standardized_root or args.output.parent
        manifest = adapter.import_folder(args.folder, args.wake_word, root)
    else:
        root = args.standardized_root or args.output.parent
        manifest = adapter.from_generator_manifest(
            args.generator_manifest,
            root,
            limit_per_label=args.limit_per_label,
        )
    output = manifest.save(args.output)
    errors = manifest.validate(output)
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(manifest.summary(), ensure_ascii=False, indent=2))
    print(f"manifest={output.resolve()}")


if __name__ == "__main__":
    main()
