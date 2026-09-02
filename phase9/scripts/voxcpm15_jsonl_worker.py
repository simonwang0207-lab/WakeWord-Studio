"""Persistent local VoxCPM1.5 JSONL worker used by the Phase 9 generator."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
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


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def soundfile_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                   channels_first=True, format=None, buffer_size=4096, backend=None):
    del normalize, format, buffer_size, backend
    frames = -1 if num_frames is None or num_frames < 0 else num_frames
    data, sample_rate = sf.read(uri, start=frame_offset, frames=frames,
                                dtype="float32", always_2d=True)
    tensor = torch.from_numpy(np.ascontiguousarray(data))
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, sample_rate


def terminal(text: str) -> str:
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else f"{text}。"


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    # Popen sends UTF-8 JSONL.  Windows otherwise decodes redirected stdin with
    # the active ANSI code page, which can create surrogate characters.
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("VoxCPM formal source generation requires CUDA")
    torchaudio.load = soundfile_load
    with contextlib.redirect_stdout(sys.stderr):
        from voxcpm import VoxCPM

    reference_manifest_path = args.reference_manifest.resolve()
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    references = {row["speaker_id"]: row for row in reference_manifest["records"]}
    with contextlib.redirect_stdout(sys.stderr):
        model = VoxCPM.from_pretrained(
            str(args.model_dir.resolve()), load_denoiser=False, local_files_only=True,
            optimize=False, device="cuda",
        )
    wrapped_tokenizer = model.tts_model.text_tokenizer
    fallback_state = {"count": 0}

    def robust_text_tokenizer(text: str) -> list[int]:
        try:
            return wrapped_tokenizer(text)
        except ValueError as error:
            if "TextEncodeInput" not in str(error):
                raise
            fallback_state["count"] += 1
            # Phase 9 phrases and AISHELL-3 prompts are Chinese characters plus
            # punctuation.  Character IDs are the intended result of VoxCPM's
            # mask_multichar_chinese_tokens wrapper, without its failing
            # encode_batch path on repeated-character targets such as 豆豆.
            tokens = [character for character in str(text) if not character.isspace()]
            return wrapped_tokenizer.tokenizer.convert_tokens_to_ids(tokens)

    model.tts_model.text_tokenizer = robust_text_tokenizer
    sample_rate = int(model.tts_model.sample_rate)
    emit({"status": "READY", "sample_rate_hz": sample_rate,
          "gpu": torch.cuda.get_device_name(0), "reference_speakers": sorted(references)})

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close":
            emit({"status": "CLOSED"})
            return
        speaker = str(request["reference_speaker_id"])
        input_text = str(request["text"])
        reference = references[speaker]
        reference_path = (reference_manifest_path.parent / reference["reference_path"]).resolve()
        started = time.perf_counter()
        fallback_before = fallback_state["count"]
        with contextlib.redirect_stdout(sys.stderr):
            wav = model.generate(
                text=terminal(input_text),
                prompt_wav_path=str(reference_path),
                prompt_text=str(reference["text"]),
                cfg_value=float(request["cfg_value"]),
                inference_timesteps=int(request["inference_timesteps"]),
                max_len=int(request["max_len"]), normalize=False, denoise=False,
                retry_badcase=False, seed=stable_seed(str(request["seed_material"])),
            )
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if not len(wav) or not np.isfinite(wav).all():
            raise RuntimeError(f"Invalid VoxCPM output: {request['sample_id']}")
        destination = Path(request["output_path"]).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, wav, sample_rate, subtype="PCM_16")
        emit({
            "status": "OK", "sample_id": request["sample_id"],
            "output_path": str(destination), "sample_rate_hz": sample_rate,
            "generation_seconds": round(time.perf_counter() - started, 3),
            "reference_speaker_id": speaker,
            "reference_source": "AISHELL-3",
            "reference_gender": reference.get("gender_if_available"),
            "reference_age_group": reference.get("age_group"),
            "reference_age_group_source": reference.get("age_group_source"),
            "tokenizer_single_encode_fallback": fallback_state["count"] > fallback_before,
            "input_text_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        })


if __name__ == "__main__":
    main()
