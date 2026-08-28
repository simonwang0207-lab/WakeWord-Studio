"""Generate the fixed Chinese acknowledgement asset with cached Kokoro."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline

from wakeword_studio.audio import (
    KOKORO_OUTPUT_SAMPLE_RATE_HZ,
    TARGET_PCM_SUBTYPE,
    TARGET_SAMPLE_RATE_HZ,
    resample_audio,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh").to("cpu").eval()
    pipeline = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=model)
    torch.manual_seed(20260828)
    result = next(pipeline("我醒来了", voice="zf_001", speed=1.0))
    audio_24k = np.asarray(result.audio, np.float32)
    audio = resample_audio(audio_24k, KOKORO_OUTPUT_SAMPLE_RATE_HZ)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE)
    print(f"output={args.output.resolve()} duration={len(audio) / TARGET_SAMPLE_RATE_HZ:.3f}s")


if __name__ == "__main__":
    main()
