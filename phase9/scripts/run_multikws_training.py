"""Resumable formal Multi-KWS training CLI. It is not invoked by smoke automation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.training.multikws_trainer import train_multikws  # noqa: E402


def derive_epoch_schedule(config: dict[str, object]) -> dict[str, int]:
    training = dict(config["training"])  # type: ignore[arg-type]
    batch_size = int(training["batch_size"])
    train_samples = int(training["effective_train_samples"])
    steps_per_epoch = math.ceil(train_samples / batch_size)
    max_epochs = int(training["max_epochs"])
    validation_every_epochs = int(training["validation_every_epochs"])
    return {
        "batch_size": batch_size,
        "effective_train_samples": train_samples,
        "steps_per_epoch": steps_per_epoch,
        "max_epochs": max_epochs,
        "derived_max_steps": steps_per_epoch * max_epochs,
        "validation_interval_steps": steps_per_epoch * validation_every_epochs,
        "early_stopping_patience_validations": int(training["early_stopping_patience_epochs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("bcresnet", "convmixer"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    training = config["training"]
    sampler = training["sampler"]
    if "max_epochs" in training:
        schedule = derive_epoch_schedule(config)
        maximum_steps = int(args.max_steps or schedule["derived_max_steps"])
        validation_interval = int(args.validation_interval or schedule["validation_interval_steps"])
        patience = int(args.early_stopping_patience or schedule["early_stopping_patience_validations"])
    else:
        schedule = None
        maximum_steps = int(args.max_steps or training["max_steps"])
        validation_interval = int(args.validation_interval or training["validation_interval"])
        patience = int(args.early_stopping_patience or training["early_stopping_patience"])
    report = train_multikws(
        model_name=args.model,
        vocabulary_path=(PROJECT_ROOT / config["vocabulary"]).resolve(),
        feature_store_path=args.features.resolve(),
        run_dir=args.run_dir.resolve(),
        seed=int(config["seed"]),
        smoke_steps=maximum_steps,
        run_mode="formal",
        validation_interval=validation_interval,
        early_stopping_patience=patience,
        resume=args.resume,
        require_gpu=True,
        architecture_config=config["models"][args.model],
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        sampler_kind=str(sampler["kind"]),
        drop_last=bool(sampler["drop_last"]),
        max_epochs=None if schedule is None else schedule["max_epochs"],
        validation_every_epochs=None if schedule is None else int(training["validation_every_epochs"]),
        evaluation_batch_size=int(args.evaluation_batch_size),
    )
    print(json.dumps({
        "model": report["model_name"],
        "completed_steps": report["completed_steps"],
        "best_checkpoint_path": report["best_checkpoint_path"],
        "GPU_DETECTED": report["GPU_DETECTED"],
        "TEST_READ": report["TEST_READ"],
        "steps_per_epoch": report["sampler"]["steps_per_epoch"],
        "maximum_steps": report["maximum_steps"],
    }))


if __name__ == "__main__":
    main()
