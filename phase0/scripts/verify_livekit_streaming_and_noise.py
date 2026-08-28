"""Verify RepCNN score cadence, multi-model API, and configured-SNR background mixing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.embedded_wakeword.data.augment import AudioAugmentor
from livekit.embedded_wakeword.inference.model import StreamingWakeWordModel, WakeWordModel


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if sr != 16_000:
        raise ValueError(f"Expected 16 kHz, got {sr}")
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def measured_snr(clean: np.ndarray, mixed: np.ndarray) -> float:
    noise = mixed - clean
    return float(10 * np.log10((np.mean(clean**2) + 1e-12) / (np.mean(noise**2) + 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audio = load_audio(args.wav)
    # Give the 99-frame classifier enough history and several subsequent hops.
    stream_audio = np.concatenate([np.zeros(16_000, np.float32), audio, np.zeros(8_000, np.float32)])
    streaming = StreamingWakeWordModel(args.model)
    scores: list[dict[str, float | int]] = []
    for offset in range(0, len(stream_audio), 320):
        score = streaming.predict_streaming(stream_audio[offset : offset + 320])
        if score is not None:
            scores.append({"sample_offset": offset, "score": float(score)})

    parallel = WakeWordModel()
    for name in ("qingxiaojia", "xiaoan", "xiaorui"):
        parallel.load_model(args.model, model_name=name)
    parallel_scores = parallel.predict(audio)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    noise_dir = args.work_dir / "backgrounds"
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260828)
    # Long enough to avoid tiling correlations in the tested clip.
    noise = rng.normal(0.0, 0.1, len(audio) + 1600).astype(np.float32)
    sf.write(noise_dir / "white_noise.wav", noise, 16_000, subtype="FLOAT")
    augmentor = AudioAugmentor(background_paths=[noise_dir], rir_paths=[])

    snr_trials = []
    for target_snr in (5.0, 10.0, 15.0):
        # Fix the range endpoints so the implementation must use this exact target.
        mixed = augmentor.mix_with_background(audio, snr_db_range=(target_snr, target_snr))
        snr_trials.append({
            "target_snr_db": target_snr,
            "measured_snr_db": measured_snr(audio, mixed),
        })

    result = {
        "streaming_input_samples": int(len(stream_audio)),
        "streaming_score_count": len(scores),
        "streaming_score_min": min(row["score"] for row in scores),
        "streaming_score_max": max(row["score"] for row in scores),
        "streaming_scores": scores,
        "parallel_binary_api_scores": parallel_scores,
        "snr_trials": snr_trials,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "streaming_scores"}, indent=2))


if __name__ == "__main__":
    main()
