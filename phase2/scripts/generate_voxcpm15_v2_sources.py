"""Generate a bounded formal VoxCPM1.5 source pool for qingxiaojia_v2."""

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


POSITIVE = "你好，青小甲"
ORDINARY_NEGATIVES = (
    "今天天气不错",
    "请打开客厅的灯",
    "播放一首音乐",
    "现在几点了",
    "记得出门的时候带钥匙",
    "我想听今天的新闻",
    "周末我们一起去公园吧",
    "请把电视声音调小",
)
HARD_NEGATIVES = (
    ("你好，星小甲", 1),
    ("你好，请小甲", 1),
    ("你好，金小甲", 1),
    ("你好，青小佳", 1),
    ("你好，青小杰", 1),
    ("你好，青小架", 1),
    ("你好吗，青小甲", 2),
    ("你好，小甲", 2),
    ("你好，青甲", 2),
    ("青小甲", 2),
    ("你好，小安", 3),
    ("你好，小瑞", 3),
)


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


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


def terminal(text: str) -> str:
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else f"{text}。"


def schedule() -> list[tuple[str, str, str, int | None]]:
    rows: list[tuple[str, str, str, int | None]] = []
    for index in range(5):
        rows.append((f"positive-{index:02d}", "positive", POSITIVE, None))
    for index, text in enumerate(ORDINARY_NEGATIVES):
        rows.append((f"negative-{index:02d}", "negative", text, None))
    for index, (text, tier) in enumerate(HARD_NEGATIVES):
        rows.append((f"hard-negative-{index:02d}", "hard_negative", text, tier))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    print("VOXCPM15 V2 SOURCES START", flush=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_dir = args.model_dir.resolve()
    reference_manifest_path = args.reference_manifest.resolve()
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    reference_root = reference_manifest_path.parent
    references = {row["speaker_id"]: row for row in reference_manifest["records"]}
    speaker_split = {
        speaker: split
        for split, families in config["speaker_allocation"].items()
        for speaker in families["voxcpm15"]
    }
    if set(speaker_split) != set(references):
        raise RuntimeError(
            f"VoxCPM allocation/reference mismatch: allocation={sorted(speaker_split)}, "
            f"references={sorted(references)}"
        )
    required = (
        "model.safetensors",
        "audiovae.pth",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local model files: {missing}")
    print(f"LOCAL INPUTS READY speakers={len(references)} model={model_dir}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(
        f"CUDA READY gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"cuda={torch.version.cuda} free_mib={free_bytes / 1048576:.1f} "
        f"total_mib={total_bytes / 1048576:.1f}",
        flush=True,
    )
    torchaudio.load = soundfile_load
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
        f"LOCAL MODEL LOADED seconds={load_seconds:.3f} sample_rate={sample_rate}",
        flush=True,
    )

    output_root = args.output_root.resolve()
    jobs = [
        (speaker, speaker_split[speaker], item_id, label, text, tier)
        for speaker in sorted(speaker_split)
        for item_id, label, text, tier in schedule()
    ]
    total = min(len(jobs), args.limit) if args.limit else len(jobs)
    records: list[dict[str, object]] = []
    started_all = time.perf_counter()
    for job_index, (speaker, split, item_id, label, text, tier) in enumerate(
        jobs[:total], start=1
    ):
        reference = references[speaker]
        reference_path = (reference_root / reference["reference_path"]).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        record_id = f"voxcpm15-{speaker}-{item_id}"
        relative = Path(split) / speaker / label / f"{item_id}.wav"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        elapsed = 0.0
        if not destination.exists():
            one_started = time.perf_counter()
            wav = model.generate(
                text=terminal(text),
                prompt_wav_path=str(reference_path),
                prompt_text=reference["text"],
                cfg_value=2.0,
                inference_timesteps=10,
                max_len=512,
                normalize=False,
                denoise=False,
                retry_badcase=False,
                seed=stable_seed(f"{config['seed']}:{record_id}"),
            )
            elapsed = time.perf_counter() - one_started
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if not len(wav) or not np.isfinite(wav).all():
                raise RuntimeError(f"Invalid VoxCPM output: {record_id}")
            partial = destination.with_name(f"{destination.stem}.partial.wav")
            sf.write(partial, wav, sample_rate, subtype="PCM_16")
            partial.replace(destination)
        info = sf.info(destination)
        if info.channels != 1 or info.subtype != "PCM_16" or info.frames <= 0:
            raise RuntimeError(f"Invalid source WAV contract: {destination} {info}")
        records.append(
            {
                "record_id": record_id,
                "path": relative.as_posix(),
                "label": label,
                "text": text,
                "synthesis_text": terminal(text),
                "split": split,
                "speaker_id": speaker,
                "source_family": "voxcpm15",
                "source_group_id": f"voxcpm15:{speaker}",
                "source_utterance_id": record_id,
                "gender": reference["gender_if_available"],
                "age_group": reference["age_group"],
                "age_source": "verified",
                "reference_speaker_id": speaker,
                "reference_age_group": reference["age_group"],
                "reference_age_group_source": "verified_dataset_metadata",
                "perceived_age_verified": False,
                "accent": reference["accent"],
                "speed": 1.0,
                "hard_negative_tier": tier,
                "sample_rate_hz": int(info.samplerate),
                "duration_seconds": round(float(info.duration), 6),
                "sha256": sha256_file(destination),
                "generation_seconds": round(elapsed, 3),
            }
        )
        if job_index % 10 == 0 or job_index == total:
            partial_manifest = output_root / "source_manifest.partial.json"
            partial_manifest.write_text(
                json.dumps({"records": records}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"VOXCPM SOURCE HEARTBEAT records={job_index}/{total} "
                f"speaker={speaker} label={label} last_seconds={elapsed:.3f}",
                flush=True,
            )

    report = {
        "schema": "wakeword-studio.source-manifest/v2",
        "target": config["wake_word"],
        "generator": "VoxCPM1.5 local formal source pool",
        "model_path": str(model_dir),
        "model_license": "Apache-2.0",
        "reference_dataset": "AISHELL-3",
        "reference_dataset_license": "Apache-2.0",
        "root": str(output_root),
        "model_load_seconds": round(load_seconds, 3),
        "generation_total_seconds": round(time.perf_counter() - started_all, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1048576, 1),
        "records": records,
    }
    name = "source_manifest.smoke.json" if args.limit else "source_manifest.json"
    manifest_path = output_root / name
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.limit:
        (output_root / "source_manifest.partial.json").unlink(missing_ok=True)
    print(
        f"VOXCPM15 V2 SOURCES COMPLETE records={len(records)} seconds="
        f"{report['generation_total_seconds']} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
