"""CLI for one bounded Multi-KWS architecture smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.training.multikws_trainer import train_multikws  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("bcresnet", "convmixer"), required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=2)
    args = parser.parse_args()
    report = train_multikws(
        model_name=args.model,
        vocabulary_path=args.vocabulary.resolve(),
        feature_store_path=args.features.resolve(),
        run_dir=args.run_dir.resolve(),
        smoke_steps=args.smoke_steps,
    )
    print(
        json.dumps(
            {
                "model": report["model_name"],
                "GPU_DETECTED": report["GPU_DETECTED"],
                "GPU_OP_EXECUTED": report["GPU_OP_EXECUTED"],
                "output_shape": report["int8_export"]["output_shape"],
                "TEST_READ": report["TEST_READ"],
            }
        )
    )


if __name__ == "__main__":
    main()
