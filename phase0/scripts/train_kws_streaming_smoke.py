"""Run Google's official data loader and training loop for two smoke steps."""

from __future__ import annotations

import argparse
from pathlib import Path

from kws_smoke_common import make_flags
from kws_streaming.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.train_dir, args.train_dir / "restore", args.train_dir / "logs", args.train_dir / "train"):
        path.mkdir(parents=True, exist_ok=True)
    flags = make_flags(args.data_dir, args.train_dir)
    train.train(flags)
    print(f"Training finished; best weights prefix: {args.train_dir / 'best_weights'}")


if __name__ == "__main__":
    main()
