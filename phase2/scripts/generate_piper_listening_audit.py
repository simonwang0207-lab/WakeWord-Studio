"""Generate a tiny, listening-gated Piper Chinese source audit.

This script intentionally generates only eight fixed utterances.  Its output is
not a formal dataset and must not be imported until a human listening gate has
passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from wakeword_studio.audio import (
    TARGET_PCM_SUBTYPE,
    TARGET_SAMPLE_RATE_HZ,
    load_audio_float32,
)
from wakeword_studio.dataset.manifest import sha256_file


@dataclass(frozen=True, slots=True)
class AuditItem:
    item_id: str
    label: str
    text: str
    length_scale: float
    hard_negative_tier: int | None = None


AUDIT_ITEMS = (
    AuditItem("positive_01_normal", "positive", "你好，青小甲。", 1.00),
    AuditItem("positive_02_faster", "positive", "你好，青小甲。", 0.90),
    AuditItem("positive_03_slower", "positive", "你好，青小甲。", 1.10),
    AuditItem("positive_04_careful", "positive", "你好，青小甲。", 1.20),
    AuditItem("hard_01_xiaojia", "hard_negative", "你好，小甲。", 1.00, 1),
    AuditItem("hard_02_qingjia", "hard_negative", "你好，青甲。", 1.00, 1),
    AuditItem("hard_03_qingxiaojia", "hard_negative", "你好，青小佳。", 1.00, 1),
    AuditItem("hard_04_qingxiaojia", "hard_negative", "你好，请小甲。", 1.00, 2),
)


def trailing_silence_seconds(path: Path, threshold_dbfs: float = -45.0) -> float:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    non_silent = np.flatnonzero(np.abs(mono) > threshold)
    if not len(non_silent):
        return float(len(mono) / sample_rate)
    return float((len(mono) - int(non_silent[-1]) - 1) / sample_rate)


def wav_contract(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
    return {
        "sample_rate_hz": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "duration_seconds": round(frames / rate, 6),
    }


def is_valid_wav(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnchannels() > 0 and handle.getframerate() > 0 and handle.getnframes() > 0
    except (EOFError, wave.Error):
        return False


def write_canonical_with_minimum_tail(
    source: Path,
    destination: Path,
    minimum_tail_seconds: float = 0.35,
    threshold_dbfs: float = -45.0,
) -> None:
    audio, _ = load_audio_float32(source)
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    non_silent = np.flatnonzero(np.abs(audio) > threshold)
    existing_tail = len(audio) if not len(non_silent) else len(audio) - int(non_silent[-1]) - 1
    required_tail = round(minimum_tail_seconds * TARGET_SAMPLE_RATE_HZ)
    if existing_tail < required_tail:
        audio = np.pad(audio, (0, required_tail - existing_tail))
    partial = destination.with_name(f"{destination.stem}.partial.wav")
    sf.write(partial, audio, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE, format="WAV")
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--piper-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(1, len(AUDIT_ITEMS) + 1), default=8)
    args = parser.parse_args()

    piper_python = args.piper_python.resolve()
    model = args.model.resolve()
    config = args.config.resolve()
    output_root = args.output_root.resolve()
    for required in (piper_python, model, config):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    selected = AUDIT_ITEMS[: args.limit]
    for index, item in enumerate(selected, start=1):
        raw_path = output_root / "raw_22050hz" / item.label / f"{item.item_id}.wav"
        listen_path = output_root / "listen" / item.label / f"{item.item_id}.wav"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        listen_path.parent.mkdir(parents=True, exist_ok=True)

        if raw_path.exists() and not is_valid_wav(raw_path):
            raw_path.unlink()
        if not raw_path.exists():
            command = [
                str(piper_python),
                "-m",
                "piper",
                "--model",
                str(model),
                "--config",
                str(config),
                "--output-file",
                str(raw_path),
                "--length-scale",
                str(item.length_scale),
                "--sentence-silence",
                "0.45",
            ]
            completed = subprocess.run(
                command,
                input=f"{item.text}\n",
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Piper failed for {item.item_id} (exit {completed.returncode}):\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )

        if not listen_path.exists() or trailing_silence_seconds(listen_path) < 0.30:
            write_canonical_with_minimum_tail(raw_path, listen_path)

        contract = wav_contract(listen_path)
        if (
            contract["sample_rate_hz"] != 16_000
            or contract["channels"] != 1
            or contract["sample_width_bytes"] != 2
        ):
            raise RuntimeError(f"Canonical contract failed: {listen_path}")
        if not math.isfinite(float(contract["duration_seconds"])):
            raise RuntimeError(f"Invalid duration: {listen_path}")

        records.append(
            {
                "item_id": item.item_id,
                "label": item.label,
                "text": item.text.rstrip("。！"),
                "synthesis_text": item.text,
                "length_scale": item.length_scale,
                "hard_negative_tier": item.hard_negative_tier,
                "speaker_id": "chaowen",
                "source_type": "tts",
                "source_family": "piper_chaowen",
                "age_group": None,
                "age_verified": False,
                "age_source": "unknown",
                "gender_if_available": None,
                "listen_path": listen_path.relative_to(output_root).as_posix(),
                "raw_path": raw_path.relative_to(output_root).as_posix(),
                **contract,
                "trailing_silence_seconds_at_minus_45_dbfs": round(
                    trailing_silence_seconds(listen_path), 6
                ),
                "sha256": sha256_file(listen_path),
            }
        )
        print(
            f"PIPER_AUDIT {index}/{len(selected)} id={item.item_id} "
            f"duration={contract['duration_seconds']}s",
            flush=True,
        )

    manifest = {
        "schema": "wakeword-studio.tts-listening-gate/v1",
        "status": "AWAITING_HUMAN_LISTENING",
        "formal_dataset_eligible": False,
        "generator": "piper-tts[zh]==1.4.2",
        "voice": "zh_CN-chaowen-medium",
        "voice_model_bytes": model.stat().st_size,
        "voice_model_sha256": sha256_file(model),
        "license_review": {
            "dataset_claim": "CC0",
            "model_card_note": "Finetuned from Xiao Ya voice (non-commercial dataset)",
            "decision": "research-only candidate; commercial clearance not established",
            "model_card_url": (
                "https://huggingface.co/rhasspy/piper-voices/blob/main/"
                "zh/zh_CN/chaowen/medium/MODEL_CARD"
            ),
        },
        "records": records,
    }
    manifest_path = output_root / "listening_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PIPER_AUDIT_COMPLETE manifest={manifest_path} records={len(records)}", flush=True)


if __name__ == "__main__":
    main()
