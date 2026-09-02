"""Generate resumable, canonical Kokoro source utterances for qingxiaojia_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from wakeword_studio.dataset.manifest import sha256_file
from wakeword_studio.dataset.source_catalog import source_utterance_specs


MODEL_REPO = "hexgrad/Kokoro-82M-v1.1-zh"


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    groups = [group for group in config["source_groups"] if group["family"] == "kokoro"]
    specs = source_utterance_specs(config["wake_word"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/2] loading Kokoro model device={device} groups={len(groups)}", flush=True)
    model = KModel(repo_id=MODEL_REPO).to(device).eval()
    pipeline = KPipeline(lang_code="z", repo_id=MODEL_REPO, model=model)

    records: list[dict[str, object]] = []
    total = len(groups) * len(specs)
    completed = 0
    for group_index, group in enumerate(groups, start=1):
        voice = str(group["speaker_id"])
        for spec in specs:
            record_id = f"kokoro-{voice}-{spec.utterance_id}"
            relative = Path(str(group["split"])) / voice / spec.label / f"{spec.utterance_id}.wav"
            path = output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                torch.manual_seed(stable_seed(f"{config['seed']}:{record_id}"))
                result = next(pipeline(spec.synthesis_text, voice=voice, speed=spec.speed))
                audio = resample_audio(
                    np.asarray(result.audio, dtype=np.float32),
                    KOKORO_OUTPUT_SAMPLE_RATE_HZ,
                )
                partial = path.with_name(f"{path.stem}.partial.wav")
                sf.write(partial, audio, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE)
                partial.replace(path)
            info = sf.info(path)
            records.append(
                {
                    "record_id": record_id,
                    "path": relative.as_posix(),
                    "label": spec.label,
                    "text": spec.text,
                    "synthesis_text": spec.synthesis_text,
                    "split": group["split"],
                    "speaker_id": voice,
                    "source_family": "kokoro",
                    "source_group_id": group["id"],
                    "source_utterance_id": record_id,
                    "gender": group.get("gender"),
                    "age_group": group.get("age_group"),
                    "age_source": group.get("age_source", "unknown"),
                    "speed": spec.speed,
                    "hard_negative_tier": spec.hard_negative_tier,
                    "sample_rate_hz": int(info.samplerate),
                    "original_sample_rate_hz": KOKORO_OUTPUT_SAMPLE_RATE_HZ,
                    "duration_seconds": round(float(info.duration), 6),
                    "sha256": sha256_file(path),
                }
            )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[2/2] sources={completed}/{total} voice={voice}", flush=True)
        partial_manifest = output_root / "source_manifest.partial.json"
        partial_manifest.write_text(
            json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[2/2] completed_group={group_index}/{len(groups)} voice={voice}", flush=True)

    manifest = {
        "schema": "wakeword-studio.source-manifest/v1",
        "target": config["wake_word"],
        "generator": "kokoro",
        "model_repo": MODEL_REPO,
        "root": str(output_root),
        "records": records,
    }
    manifest_path = output_root / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "source_manifest.partial.json").unlink(missing_ok=True)
    print(f"manifest={manifest_path} records={len(records)}", flush=True)


if __name__ == "__main__":
    main()

