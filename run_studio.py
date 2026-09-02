"""Default launcher for the modern browser UI; desktop and legacy modes remain available."""

from __future__ import annotations

import sys


def main() -> None:
    legacy_flags = {"--legacy", "--ui-smoke", "--model-a-listen-smoke"}
    if legacy_flags.intersection(sys.argv[1:]):
        if "--legacy" in sys.argv:
            sys.argv.remove("--legacy")
        from phase5.scripts.wakeword_studio_demo import main as legacy_main
        legacy_main()
        return
    if "--desktop" in sys.argv:
        sys.argv.remove("--desktop")
        from run_studio_desktop import main as desktop_main
        desktop_main()
        return
    if "--web" in sys.argv:
        sys.argv.remove("--web")
    from run_studio_modern import main as modern_main
    modern_main()


if __name__ == "__main__":
    main()
