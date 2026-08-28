"""Input a wake word and automatically synthesize a small, diverse DatasetManifest.

Real age coverage is deliberately never inferred from TTS voice IDs. Environmental
augmentations are recorded separately from speaker metadata.
"""

from __future__ import annotations

import argparse
import hashlib
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
from wakeword_studio.dataset.manifest import (
    AcousticMetadata,
    DatasetManifest,
    DatasetRecord,
    SpeakerMetadata,
)

REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
VOICES = ["zf_001", "zf_003", "zf_006", "zf_017", "zf_021", "zm_009", "zm_013", "zm_020", "zm_031", "zm_041", "zm_053", "zm_056"]
DEFAULT_HARD_NEGATIVES = ["你好，小安", "你好，小瑞", "你好，青小佳", "你好，青小家", "青小甲", "你好，小甲", "您好，青小甲", "你好，青甲"]
DEFAULT_NEGATIVES = ["今天天气不错", "请打开客厅的灯", "播放一首音乐", "现在几点了"]
SANITY_SPEEDS = [0.88, 0.93, 0.98, 1.03, 1.08, 0.91, 0.96, 1.01, 1.06, 1.12]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesize(pipeline: KPipeline, text: str, voice: str, speed: float) -> np.ndarray:
    result = next(pipeline(text, voice=voice, speed=speed))
    audio_24k = np.asarray(result.audio, dtype=np.float32)
    return resample_audio(audio_24k, KOKORO_OUTPUT_SAMPLE_RATE_HZ)


def environment_augment(audio: np.ndarray, rng: np.random.Generator, index: int) -> tuple[np.ndarray, AcousticMetadata]:
    gain_db = float(rng.uniform(-5.0, 3.0))
    snr_db = float(rng.choice([5.0, 10.0, 15.0, 20.0]))
    reverb_seconds = float(rng.uniform(0.04, 0.18))
    # Sparse early reflections keep the generator dependency-light.
    reverbed = audio.copy()
    for fraction, level in ((0.25, 0.35), (0.55, 0.18), (1.0, 0.08)):
        delay = max(1, int(TARGET_SAMPLE_RATE_HZ * reverb_seconds * fraction))
        reverbed[delay:] += audio[:-delay] * level
    gained = reverbed * 10 ** (gain_db / 20.0)
    rng_noise = rng.normal(0.0, 1.0, len(gained)).astype(np.float32)
    signal_rms = np.sqrt(np.mean(gained * gained) + 1e-12)
    noise_rms = np.sqrt(np.mean(rng_noise * rng_noise) + 1e-12)
    noise_scale = signal_rms / (10 ** (snr_db / 20.0) * noise_rms)
    mixed = np.clip(gained + rng_noise * noise_scale, -1.0, 1.0).astype(np.float32)
    return mixed, AcousticMetadata(
        gain_db=gain_db,
        noise_id=f"synthetic_broadband_{index:03d}",
        snr_db=snr_db,
        reverb_id=f"synthetic_decay_{reverb_seconds:.3f}s",
        acoustic_age_proxy=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake-word", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.per_label <= len(VOICES):
        raise ValueError(f"--per-label must be between 2 and {len(VOICES)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260829)
    rng = np.random.default_rng(20260829)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id=REPO_ID).to(device).eval()
    pipeline = KPipeline(lang_code="z", repo_id=REPO_ID, model=model)
    records: list[DatasetRecord] = []
    for index, voice in enumerate(VOICES[: args.per_label]):
        speed = SANITY_SPEEDS[index]
        examples = (
            ("positive", args.wake_word),
            ("negative", DEFAULT_NEGATIVES[index % len(DEFAULT_NEGATIVES)]),
            ("hard_negative", DEFAULT_HARD_NEGATIVES[index % len(DEFAULT_HARD_NEGATIVES)]),
        )
        for item_label, item_text in examples:
            audio = synthesize(pipeline, item_text, voice, speed)
            audio, acoustic = environment_augment(audio, rng, len(records))
            acoustic.speaking_rate = speed
            directory = args.output_root / item_label
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{item_label}_{index:03d}_{voice}.wav"
            sf.write(path, audio, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE)
            relative = path.relative_to(args.output_root).as_posix()
            records.append(
                DatasetRecord(
                    record_id=f"auto-{len(records):06d}",
                    audio_path=relative,
                    label=item_label,
                    split="train" if index < args.per_label - 2 else "validation" if index == args.per_label - 2 else "test",
                    text=item_text,
                    speaker=SpeakerMetadata(
                        speaker_id=voice,
                        source="kokoro_tts",
                        # Unknown on purpose: voice name/gender does not establish age.
                        gender="female" if voice.startswith("zf") else "male",
                        age_group=None,
                        age_source="unknown",
                    ),
                    acoustic=acoustic,
                    sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
                    original_sample_rate_hz=KOKORO_OUTPUT_SAMPLE_RATE_HZ,
                    duration_seconds=len(audio) / TARGET_SAMPLE_RATE_HZ,
                    sha256=hash_file(path),
                )
            )
        print(f"completed_voice={index + 1}/{args.per_label} records={len(records)}", flush=True)
    manifest = DatasetManifest(
        wake_word=args.wake_word,
        records=records,
        source_kind="generated",
        root=str(args.output_root.resolve()),
        generator={"name": "kokoro", "model_repo": REPO_ID, "device": device, "seed": 20260829},
    )
    output = manifest.save(args.output_root / "DatasetManifest.json")
    print(f"manifest={output.resolve()}")
    print(manifest.summary())


if __name__ == "__main__":
    main()
