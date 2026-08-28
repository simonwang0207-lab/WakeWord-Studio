"""Generate a tiny, auditable Chinese TTS batch for the Phase 0 listening gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline

from wakeword_studio.audio import KOKORO_OUTPUT_SAMPLE_RATE_HZ

REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
SAMPLE_RATE = KOKORO_OUTPUT_SAMPLE_RATE_HZ
# Use Unicode escapes so the target survives every Windows shell/code-page boundary.
TEXT = "\u4f60\u597d\uff0c\u9752\u5c0f\u7532"
VOICES_AND_SPEEDS = [
    ("zf_001", 0.92),
    ("zf_008", 0.97),
    ("zf_026", 1.02),
    ("zf_059", 1.07),
    ("zm_010", 0.92),
    ("zm_025", 0.97),
    ("zm_050", 1.02),
    ("zm_089", 1.07),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260828)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id=REPO_ID).to(device).eval()
    pipeline = KPipeline(lang_code="z", repo_id=REPO_ID, model=model)

    records = []
    for index, (voice, speed) in enumerate(VOICES_AND_SPEEDS, start=1):
        result = next(pipeline(TEXT, voice=voice, speed=speed))
        audio = np.asarray(result.audio, dtype=np.float32)
        output_path = args.output_dir / f"qingxiaojia_{index:02d}_{voice}_s{speed:.2f}.wav"
        sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
        records.append(
            {
                "path": str(output_path.resolve()),
                "text": TEXT,
                "voice": voice,
                "speed": speed,
                "sample_rate_hz": SAMPLE_RATE,
                "frames": int(audio.size),
                "duration_seconds": round(audio.size / SAMPLE_RATE, 6),
                "peak_abs": round(float(np.max(np.abs(audio))), 6),
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
                "graphemes": result.graphemes,
                "phonemes": result.phonemes,
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False))

    manifest = {
        "generator": "kokoro",
        "package_version": "0.9.4",
        "model_repo": REPO_ID,
        "model_license": "Apache-2.0",
        "model_parameters": 82_000_000,
        "device": device,
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
