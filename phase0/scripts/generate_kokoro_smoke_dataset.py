"""Generate a deliberately tiny positive/negative Chinese KWS dataset."""

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
TARGET = "\u4f60\u597d\uff0c\u9752\u5c0f\u7532"
NEGATIVE_PHRASES = [
    "\u4f60\u597d\uff0c\u5c0f\u5b89",  # 你好，小安
    "\u4f60\u597d\uff0c\u5c0f\u745e",  # 你好，小瑞
    "\u9752\u5c0f\u7532",
    "\u4f60\u597d\uff0c\u9752\u5c0f\u4f73",
    "\u4f60\u597d\uff0c\u9752\u5c0f\u5bb6",
    "\u4f60\u597d\uff0c\u9752\u5c0f\u96c5",
    "\u4f60\u597d\uff0c\u8f7b\u5c0f\u7532",
    "\u4f60\u597d\uff0c\u5c0f\u7532",
    "\u60a8\u597d\uff0c\u9752\u5c0f\u7532",
    "\u4f60\u597d\uff0c\u9752\u7532",
    "\u4f60\u597d",
    "\u4eca\u5929\u5929\u6c14\u4e0d\u9519",
    "\u8bf7\u6253\u5f00\u5ba2\u5385\u7684\u706f",
    "\u64ad\u653e\u4e00\u9996\u97f3\u4e50",
    "\u73b0\u5728\u51e0\u70b9\u4e86",
    "\u660e\u5929\u4f1a\u4e0b\u96e8\u5417",
    "\u8bf7\u628a\u7a97\u5e18\u5173\u4e0a",
    "\u5e2e\u6211\u8bbe\u7f6e\u4e00\u4e2a\u95f9\u949f",
    "\u97f3\u91cf\u8c03\u5c0f\u4e00\u70b9",
    "\u6211\u60f3\u542c\u4eca\u65e5\u65b0\u95fb",
    "\u9752\u5c71\u4f9d\u65e7\u5728",
    "\u5c0f\u7532\u4eca\u5929\u4e0d\u5728",
    "\u4f60\u597d\uff0c\u8bf7\u95ee\u6709\u4ec0\u4e48\u4e8b",
    "\u8f7b\u58f0\u8bf4\u4e00\u53e5\u4f60\u597d",
    "\u9752\u5c0f\u7532\u7684\u6545\u4e8b",
]
VOICES = [
    "zf_001", "zf_002", "zf_003", "zf_004", "zf_005",
    "zf_006", "zf_007", "zf_008", "zf_017", "zf_018",
    "zf_019", "zf_021", "zf_022", "zf_023", "zf_024",
    "zf_026", "zf_027", "zf_028", "zf_032", "zf_036",
    "zf_038", "zf_039", "zf_040", "zf_042", "zf_043",
    "zm_009", "zm_010", "zm_011", "zm_012", "zm_013",
    "zm_014", "zm_015", "zm_016", "zm_020", "zm_025",
    "zm_029", "zm_030", "zm_031", "zm_033", "zm_034",
    "zm_035", "zm_037", "zm_041", "zm_045", "zm_050",
    "zm_052", "zm_053", "zm_054", "zm_055", "zm_056",
]
SPEEDS = [0.90, 0.95, 1.00, 1.05, 1.10]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def synthesize(pipeline: KPipeline, text: str, voice: str, speed: float, path: Path) -> dict:
    result = next(pipeline(text, voice=voice, speed=speed))
    audio = np.asarray(result.audio, dtype=np.float32)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return {
        "path": str(path.resolve()),
        "text": text,
        "voice": voice,
        "speed": speed,
        "sample_rate_hz": SAMPLE_RATE,
        "frames": int(audio.size),
        "duration_seconds": round(audio.size / SAMPLE_RATE, 6),
        "peak_abs": round(float(np.max(np.abs(audio))), 6),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "graphemes": result.graphemes,
        "phonemes": result.phonemes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    positive_dir = args.output_root / "positive"
    negative_dir = args.output_root / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260828)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id=REPO_ID).to(device).eval()
    pipeline = KPipeline(lang_code="z", repo_id=REPO_ID, model=model)

    records = []
    for index, voice in enumerate(VOICES):
        speed = SPEEDS[index % len(SPEEDS)]
        pos_path = positive_dir / f"positive_{index:03d}_{voice}_s{speed:.2f}.wav"
        pos = synthesize(pipeline, TARGET, voice, speed, pos_path)
        pos["label"] = "positive"
        records.append(pos)

        neg_text = NEGATIVE_PHRASES[index % len(NEGATIVE_PHRASES)]
        neg_path = negative_dir / f"negative_{index:03d}_{voice}_s{speed:.2f}.wav"
        neg = synthesize(pipeline, neg_text, voice, speed, neg_path)
        neg["label"] = "negative"
        neg["hard_negative"] = index % len(NEGATIVE_PHRASES) < 11
        records.append(neg)
        print(f"completed_pair={index + 1}/{len(VOICES)} voice={voice}", flush=True)

    manifest = {
        "generator": "kokoro",
        "package_version": "0.9.4",
        "model_repo": REPO_ID,
        "model_license": "Apache-2.0",
        "model_parameters": 82_000_000,
        "device": device,
        "target": TARGET,
        "positive_count": len(VOICES),
        "negative_count": len(VOICES),
        "hard_negative_count": sum(r.get("hard_negative", False) for r in records),
        "records": records,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
