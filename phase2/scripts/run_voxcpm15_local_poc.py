"""Run the bounded, local-only VoxCPM1.5 Phase 2D listening POC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import soundfile as sf
import torch
import torchaudio


SCHEDULE = (
    ("01", "SSB0393", "positive", "你好，青小甲"),
    ("02", "SSB0273", "positive", "你好，青小甲"),
    ("03", "SSB0632", "positive", "你好，青小甲"),
    ("04", "SSB0710", "positive", "你好，青小甲"),
    ("05", "SSB0197", "positive", "你好，青小甲"),
    ("06", "SSB0434", "positive", "你好，青小甲"),
    ("07", "SSB0737", "positive", "你好，青小甲"),
    ("08", "SSB0393", "hard_negative", "你好，小甲"),
    ("09", "SSB0710", "hard_negative", "你好，青甲"),
    ("10", "SSB0737", "hard_negative", "你好，小甲"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def soundfile_load(
    uri,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format=None,
    buffer_size: int = 4096,
    backend=None,
):
    """PCM WAV-only replacement for torchaudio 2.10's TorchCodec loader."""
    del normalize, format, buffer_size, backend
    frames = -1 if num_frames is None or num_frames < 0 else num_frames
    data, sample_rate = sf.read(
        uri,
        start=frame_offset,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(data))
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, sample_rate


def validate_wav(path: Path, expected_rate: int | None = None) -> dict[str, object]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError(f"Invalid generated WAV: {path}")
    if expected_rate is not None and sample_rate != expected_rate:
        raise RuntimeError(f"Unexpected sample rate {sample_rate}, expected {expected_rate}: {path}")
    return {
        "sample_rate_hz": sample_rate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": round(info.frames / sample_rate, 6),
        "peak_absolute": round(float(np.max(np.abs(audio))), 8),
        "finite": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=1)
    args = parser.parse_args()

    print("VOXCPM15 POC START", flush=True)
    model_dir = args.model_dir.resolve()
    reference_manifest = args.reference_manifest.resolve()
    output_dir = args.output_dir.resolve()
    required = (
        "model.safetensors",
        "audiovae.pth",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    for filename in required:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)
    print(f"LOCAL MODEL FILES FOUND path={model_dir}", flush=True)

    manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
    records = {record["speaker_id"]: record for record in manifest["records"]}
    reference_root = reference_manifest.parent
    print(f"REFERENCE MANIFEST LOADED speakers={len(records)}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(
        f"CUDA READY torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)} free_mib={free_bytes / 1048576:.1f} "
        f"total_mib={total_bytes / 1048576:.1f}",
        flush=True,
    )

    # torchaudio 2.10 delegates even PCM WAVs to TorchCodec.  Its Windows DLL
    # is not required for this POC, so use libsndfile for the one load API that
    # VoxCPM1.5 calls.  The repository and system FFmpeg remain untouched.
    torchaudio.load = soundfile_load
    print("PCM WAV LOADER ADAPTER READY", flush=True)

    from voxcpm import VoxCPM

    print("LOCAL MODEL LOAD START", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    load_started = time.perf_counter()
    model = VoxCPM.from_pretrained(
        str(model_dir),
        load_denoiser=False,
        local_files_only=True,
        optimize=False,
        device="cuda",
    )
    load_seconds = time.perf_counter() - load_started
    sample_rate = int(model.tts_model.sample_rate)
    print(
        f"LOCAL MODEL LOADED seconds={load_seconds:.3f} sample_rate={sample_rate} "
        f"allocated_mib={torch.cuda.memory_allocated(0) / 1048576:.1f} "
        f"reserved_mib={torch.cuda.memory_reserved(0) / 1048576:.1f}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for index, (ordinal, speaker_id, label, text) in enumerate(SCHEDULE[: args.limit], start=1):
        reference = records[speaker_id]
        reference_path = (reference_root / reference["reference_path"]).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        filename = (
            f"{ordinal}_{speaker_id}_age-{reference['age_group']}_"
            f"{reference['gender_if_available']}_{label}.wav"
        )
        destination = output_dir / filename
        print(
            f"GENERATE START {index}/{args.limit} speaker={speaker_id} "
            f"age={reference['age_group']} gender={reference['gender_if_available']} label={label}",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        wav = model.generate(
            text=text,
            prompt_wav_path=str(reference_path),
            prompt_text=reference["text"],
            cfg_value=2.0,
            inference_timesteps=10,
            max_len=512,
            normalize=False,
            denoise=False,
            retry_badcase=False,
            seed=4200 + index,
        )
        elapsed = time.perf_counter() - started
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size == 0 or not np.isfinite(wav).all():
            raise RuntimeError(f"Generation returned invalid audio for {speaker_id}")
        partial = destination.with_suffix(".partial.wav")
        sf.write(partial, wav, sample_rate, subtype="PCM_16")
        partial.replace(destination)
        audio_info = validate_wav(destination, sample_rate)
        result = {
            "item_id": ordinal,
            "path": destination.relative_to(output_dir.parent).as_posix(),
            "label": label,
            "text": text,
            "reference_speaker": speaker_id,
            "reference_path": reference_path.as_posix(),
            "reference_text": reference["text"],
            "gender": reference["gender_if_available"],
            "age_group": reference["age_group"],
            "age_range_years": reference["age_range_years"],
            "age_group_source": reference["age_group_source"],
            "accent": reference["accent"],
            "generation_seconds": round(elapsed, 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1048576, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1048576, 1),
            **audio_info,
        }
        results.append(result)
        print(
            f"GENERATE COMPLETE {index}/{args.limit} seconds={elapsed:.3f} "
            f"duration={audio_info['duration_seconds']} peak_allocated_mib={result['peak_allocated_mib']} "
            f"path={destination}",
            flush=True,
        )

    report = {
        "schema": "wakeword-studio.voxcpm15-listening-poc/v1",
        "status": "SMOKE_READY_FOR_VALIDATION" if args.limit == 1 else "LISTENING_REQUIRED",
        "formal_dataset_eligible": False,
        "local_only": True,
        "model_path": model_dir.as_posix(),
        "model_load_seconds": round(load_seconds, 3),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "sample_rate_hz": sample_rate,
        "torchcodec_workaround": "process-local torchaudio.load -> soundfile PCM WAV adapter",
        "records": results,
    }
    report_path = output_dir.parent / ("smoke_report.json" if args.limit == 1 else "listening_manifest.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"VOXCPM15 POC COMPLETE report={report_path} records={len(results)}", flush=True)


if __name__ == "__main__":
    main()
