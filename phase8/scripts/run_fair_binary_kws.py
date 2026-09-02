"""Single entry point for RepCNN / BC-ResNet / ConvMixer fair experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.training.fair_feature_store import load_frozen_feature_store  # noqa: E402
from wakeword_studio.training.fair_trainer import run_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--allow-formal-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke_steps:
        if args.allow_formal_training or args.resume:
            parser.error("Smoke cannot use --allow-formal-training or --resume")
        if not 1 <= args.smoke_steps <= 20:
            parser.error("--smoke-steps is limited to 1..20")
    elif not args.allow_formal_training:
        parser.error("Formal training is gated; pass --allow-formal-training")
    return args


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    config_path = args.config.resolve()
    if config_path.suffix.lower() == ".json":
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "YAML config requires PyYAML; use the equivalent .json config in minimal environments"
            ) from error
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    store = load_frozen_feature_store(config, project_root=PROJECT_ROOT)
    if store.test_loaded:
        raise RuntimeError("TEST_READ=true is prohibited")
    run_training(
        tf=tf,
        config=config,
        store=store,
        run_dir=args.run_dir.resolve(),
        smoke_steps=args.smoke_steps,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
