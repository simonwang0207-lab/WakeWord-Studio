"""Import an age-labelled local folder into the canonical product manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wakeword_studio.dataset.adapter import DatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake-word", required=True)
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--augmentation", choices=("standard", "none"), default="standard")
    args = parser.parse_args()
    manifest = DatasetAdapter().import_folder(
        args.input_folder,
        args.wake_word,
        standardized_root=args.output_root,
        augment=args.augmentation == "standard",
    )
    output = manifest.save(args.output_root / "DatasetManifest.json")
    errors = manifest.validate(output)
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(manifest.summary(), ensure_ascii=False), flush=True)
    print(f"manifest={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
