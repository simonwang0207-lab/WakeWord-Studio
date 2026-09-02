"""Write the qingxiaojia_v1 Markdown quality report."""

from __future__ import annotations

import argparse
from pathlib import Path

from wakeword_studio.dataset.quality_report import build_quality_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_quality_report(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()

