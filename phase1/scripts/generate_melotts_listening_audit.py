"""Generate a tiny, manually reviewed MeloTTS listening set.

This script is deliberately capped at six positives and five hard negatives.
It is an audit utility, not a dataset-generation entry point.
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from melo.api import TTS

from wakeword_studio.audio import standardize_wav


MODEL_REPO = "myshell-ai/MeloTTS-Chinese"
MODEL_LICENSE = "MIT"
SPEAKER_ID = "ZH"

INITIAL_SAMPLES = (
    ("positive_01", "positive", "你好，青小甲", 1.00),
    ("positive_02", "positive", "你好，青小甲", 0.90),
    ("positive_03", "positive", "你好，青小甲", 0.95),
    ("positive_04", "positive", "你好，青小甲", 1.05),
    ("positive_05", "positive", "你好，青小甲", 1.10),
    ("positive_06", "positive", "你好，青小甲", 1.15),
    ("hard_negative_01", "hard_negative", "你好，青小佳", 1.00),
    ("hard_negative_02", "hard_negative", "你好，请小甲", 1.00),
    ("hard_negative_03", "hard_negative", "你好，青小杰", 1.00),
    ("hard_negative_04", "hard_negative", "你好，小青甲", 1.00),
    ("hard_negative_05", "hard_negative", "你好，青小", 1.00),
)

RELISTEN_V2_SAMPLES = (
    ("positive_period", "positive", "你好，青小甲。", 1.00),
    ("positive_exclamation", "positive", "你好，青小甲！", 1.00),
    ("positive_ellipsis", "positive", "你好，青小甲……", 1.00),
    ("positive_trailing_comma", "positive", "你好，青小甲，", 1.00),
    ("hard_negative_qingxiaojia", "hard_negative", "你好，青小佳。", 1.00),
    ("hard_negative_qingxiaojia3", "hard_negative", "你好，请小甲。", 1.00),
    ("hard_negative_xiaoqingjia", "hard_negative", "你好，小青甲。", 1.00),
)


def inspect_wav(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as handle:
        sample_rate_hz = handle.getframerate()
        channels = handle.getnchannels()
        sample_width_bytes = handle.getsampwidth()
        frames = handle.getnframes()
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    return {
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "duration_seconds": round(frames / sample_rate_hz, 4),
        "peak": round(float(np.max(np.abs(samples))), 6),
        "rms": round(float(np.sqrt(np.mean(np.square(samples)))), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("initial", "relisten-v2"),
        default="initial",
        help="Select the original audit or the tiny terminal-punctuation re-listening set.",
    )
    args = parser.parse_args()
    samples = INITIAL_SAMPLES if args.profile == "initial" else RELISTEN_V2_SAMPLES

    output_root = args.output_root.resolve()
    raw_root = output_root / "raw"
    listen_root = output_root / "listen"
    raw_root.mkdir(parents=True, exist_ok=True)
    listen_root.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading MeloTTS ZH model", flush=True)
    model = TTS(language="ZH", device="cpu")
    speaker_ids = dict(model.hps.data.spk2id)
    if SPEAKER_ID not in speaker_ids:
        raise RuntimeError(f"Expected speaker {SPEAKER_ID!r}; found {speaker_ids}")
    print(f"[1/3] Speakers: {speaker_ids}", flush=True)

    records: list[dict[str, object]] = []
    for index, (sample_id, label, text, speed) in enumerate(samples, start=1):
        raw_path = raw_root / f"{sample_id}_source.wav"
        listen_path = listen_root / label / f"{sample_id}.wav"

        if raw_path.exists():
            print(f"[2/3] {index:02d}/{len(samples)} reuse {raw_path.name}", flush=True)
        else:
            print(
                f"[2/3] {index:02d}/{len(samples)} synthesize "
                f"label={label} speed={speed:.2f} text={text}",
                flush=True,
            )
            model.tts_to_file(
                text,
                speaker_ids[SPEAKER_ID],
                str(raw_path),
                speed=speed,
            )

        if listen_path.exists():
            print(f"[2/3] {index:02d}/{len(samples)} reuse {listen_path.name}", flush=True)
            audio_info = None
        else:
            audio_info = asdict(standardize_wav(raw_path, listen_path))

        audit = inspect_wav(listen_path)
        if (
            audit["sample_rate_hz"] != 16_000
            or audit["channels"] != 1
            or audit["sample_width_bytes"] != 2
        ):
            raise RuntimeError(f"Canonical audio audit failed for {listen_path}: {audit}")

        records.append(
            {
                "sample_id": sample_id,
                "label": label,
                "text": text,
                "tts_family": "MeloTTS",
                "model_repo": MODEL_REPO,
                "model_license": MODEL_LICENSE,
                "speaker_id": SPEAKER_ID,
                "speaker_count_in_upstream_zh_model": 1,
                "speaker_age_years": None,
                "speaker_age_verified": False,
                "speaker_age_notes": "Upstream does not provide age; no age label inferred.",
                "speed": speed,
                "source_wav": str(raw_path),
                "listening_wav": str(listen_path),
                "conversion": audio_info,
                "audit": audit,
            }
        )
        print(f"[2/3] {index:02d}/{len(samples)} ready {listen_path}", flush=True)

    manifest_path = output_root / "listening_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"[3/3] manifest={manifest_path}", flush=True)
    print(f"[3/3] positives={sum(r['label'] == 'positive' for r in records)}", flush=True)
    print(
        f"[3/3] hard_negatives={sum(r['label'] == 'hard_negative' for r in records)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
