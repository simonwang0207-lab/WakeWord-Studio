"""Launch the modern local WakeWord Studio dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from wakeword_studio.webapp import serve


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs/demo/teacher_demo.yaml",
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
