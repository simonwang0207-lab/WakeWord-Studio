"""Adapt approved WAVs to kws_streaming's explicit split/label directory layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def one_second(path: Path, positive: bool) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if sr != 16_000:
        raise ValueError(f"Expected 16 kHz: {path}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    target = 16_000
    if len(audio) >= target:
        # Positive clips were aligned to the end by LiveKit's official augment stage.
        start = len(audio) - target if positive else (len(audio) - target) // 2
        return audio[start : start + target]
    pad_left = target - len(audio) if positive else (target - len(audio)) // 2
    return np.pad(audio, (pad_left, target - len(audio) - pad_left))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--livekit-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = {
        "training": ("train", 40),
        "validation": ("test", 5),
        "testing": ("test", 5),
    }
    manifest = []
    test_offsets = {"validation": 0, "testing": 5}
    for split, (source_suffix, count) in mapping.items():
        for label, source_prefix, positive in (
            ("qingxiaojia", "positive", True),
            ("other", "negative", False),
        ):
            source_dir = args.livekit_model_dir / f"{source_prefix}_{source_suffix}"
            wavs = sorted(source_dir.glob("clip_*.wav"))
            offset = test_offsets.get(split, 0)
            selected = wavs[offset : offset + count]
            if len(selected) != count:
                raise RuntimeError(f"Need {count} files for {split}/{label}, found {len(selected)}")
            out_dir = args.output / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(selected):
                out = out_dir / f"clip_{index:06d}.wav"
                sf.write(out, one_second(source, positive), 16_000, subtype="PCM_16")
                manifest.append({"source": str(source), "destination": str(out), "label": label})

    # Google's loader supports a root-level background directory and mixes it during training.
    noise_dir = args.output / "_background_noise_"
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260828)
    sf.write(noise_dir / "white_noise.wav", rng.normal(0, 0.08, 160_000), 16_000, subtype="PCM_16")
    (args.output / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Staged {len(manifest)} one-second WAVs plus background noise")


if __name__ == "__main__":
    main()
