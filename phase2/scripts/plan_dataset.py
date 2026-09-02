"""Validate a formal dataset config and write a no-audio dry-run estimate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wakeword_studio.dataset.planning import load_and_estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    estimate = load_and_estimate(args.config)
    estimate.save(args.output)
    print(json.dumps(estimate.report, ensure_ascii=False, indent=2))
    print(f"estimate={args.output.resolve()}")


if __name__ == "__main__":
    main()

