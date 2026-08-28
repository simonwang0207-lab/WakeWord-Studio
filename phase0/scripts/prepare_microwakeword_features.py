"""Create microfrontend RaggedMmap features and a tiny MixedNet config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.audio_utils import generate_features_for_clip
from wakeword_studio.frontends import load_training_audio


TARGET = "\u4f60\u597d\uff0c\u9752\u5c0f\u7532"


def load_16k(path: Path) -> np.ndarray:
    return load_training_audio(path)


def augment(audio: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gain = rng.uniform(0.72, 1.08)
    noise_level = rng.uniform(0.001, 0.008)
    shifted = np.roll(audio, int(rng.integers(-800, 801)))
    return np.clip(shifted * gain + rng.normal(0, noise_level, shifted.shape), -1, 1).astype(
        np.float32
    )


def split_paths(paths: list[Path]) -> dict[str, list[Path]]:
    return {
        "training": [p for i, p in enumerate(paths) if i % 10 < 8],
        "testing": [p for i, p in enumerate(paths) if i % 10 == 8],
        "validation": [p for i, p in enumerate(paths) if i % 10 == 9],
    }


def feature_generator(paths: list[Path], add_augmented_copy: bool):
    for index, path in enumerate(paths):
        audio = load_16k(path)
        yield generate_features_for_clip(audio, step_ms=10)
        if add_augmented_copy:
            yield generate_features_for_clip(augment(audio, 20260828 + index), step_ms=10)


def build_class_features(source_dir: Path, destination: Path) -> dict[str, int]:
    paths = sorted(source_dir.glob("*.wav"))
    splits = split_paths(paths)
    counts = {}
    for split, split_paths_list in splits.items():
        out_dir = destination / split / f"{split}_mmap"
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        augmented = split == "training"
        RaggedMmap.from_generator(
            out_dir=out_dir,
            sample_generator=feature_generator(split_paths_list, augmented),
            batch_size=16,
            verbose=True,
        )
        counts[split] = len(split_paths_list) * (2 if augmented else 1)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["target"] != TARGET:
        raise ValueError("Dataset target text does not match the required wake phrase")
    if manifest["positive_count"] != 50 or manifest["negative_count"] != 50:
        raise ValueError("Expected exactly 50 positive and 50 negative WAV files")

    positive_counts = build_class_features(
        args.dataset_root / "positive", args.features_root / "positive"
    )
    negative_counts = build_class_features(
        args.dataset_root / "negative", args.features_root / "negative"
    )

    config = {
        "window_step_ms": 10,
        "train_dir": str(args.train_dir.resolve()),
        "features": [
            {
                "features_dir": str((args.features_root / "positive").resolve()),
                "sampling_weight": 1.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str((args.features_root / "negative").resolve()),
                "sampling_weight": 1.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
        ],
        "training_steps": [60],
        "positive_class_weight": [1.0],
        "negative_class_weight": [1.0],
        "learning_rates": [0.001],
        "batch_size": 8,
        "time_mask_max_size": [3],
        "time_mask_count": [1],
        "freq_mask_max_size": [3],
        "freq_mask_count": [1],
        "eval_step_interval": 20,
        "clip_duration_ms": 3000,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "accuracy",
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = {
        "target": TARGET,
        "source_positive_wavs": manifest["positive_count"],
        "source_negative_wavs": manifest["negative_count"],
        "source_hard_negatives": manifest["hard_negative_count"],
        "positive_feature_counts": positive_counts,
        "negative_feature_counts": negative_counts,
        "microfrontend": "pymicro-features 2.0.2",
        "feature_bins": 40,
        "feature_step_ms": 10,
    }
    summary_path = args.features_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))
    print(f"config={args.config.resolve()}")


if __name__ == "__main__":
    main()
