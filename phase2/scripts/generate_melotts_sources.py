"""Generate resumable, canonical MeloTTS source utterances for qingxiaojia_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from wakeword_studio.audio import TARGET_PCM_SUBTYPE, TARGET_SAMPLE_RATE_HZ, resample_audio
from wakeword_studio.dataset.manifest import sha256_file
from wakeword_studio.dataset.source_catalog import source_utterance_specs


MODEL_REPO = "myshell-ai/MeloTTS-Chinese"
MODEL_LICENSE = "MIT"


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing-only",
        action="store_true",
        help="Refresh the manifest with matching existing records without loading MeloTTS",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    groups = [group for group in config["source_groups"] if group["family"] == "melotts"]
    if len(groups) != 1:
        raise ValueError(f"Expected one MeloTTS source group, found {len(groups)}")
    group = groups[0]
    specs = source_utterance_specs(config["wake_word"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing_only:
        manifest_path = output_root / "source_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("--reuse-existing-only requires an existing source_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        desired_ids = {
            f"melotts-{group['speaker_id']}-{spec.utterance_id}" for spec in specs
        }
        records = [row for row in previous["records"] if row["record_id"] in desired_ids]
        manifest = {**previous, "records": records}
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"manifest={manifest_path} records={len(records)} reused_existing_only=true",
            flush=True,
        )
        return

    import torch
    from melo.api import TTS

    print(f"[1/2] loading MeloTTS model group={group['id']}", flush=True)
    model = TTS(language="ZH", device="cpu")
    speaker_ids = dict(model.hps.data.spk2id)
    speaker = str(group["speaker_id"])
    if speaker not in speaker_ids:
        raise RuntimeError(f"Expected speaker {speaker}; found {speaker_ids}")
    source_rate = int(model.hps.data.sampling_rate)

    records: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        record_id = f"melotts-{speaker}-{spec.utterance_id}"
        relative = Path(str(group["split"])) / speaker / spec.label / f"{spec.utterance_id}.wav"
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            torch.manual_seed(stable_seed(f"{config['seed']}:{record_id}"))
            audio = model.tts_to_file(
                spec.synthesis_text,
                speaker_ids[speaker],
                output_path=None,
                speed=spec.speed,
                quiet=True,
            )
            audio_16k = resample_audio(np.asarray(audio, dtype=np.float32), source_rate)
            partial = path.with_name(f"{path.stem}.partial.wav")
            sf.write(partial, audio_16k, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE)
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
                "speaker_id": speaker,
                "source_family": "melotts",
                "source_group_id": group["id"],
                "source_utterance_id": record_id,
                "gender": group.get("gender"),
                "age_group": None,
                "age_source": "unknown",
                "speed": spec.speed,
                "hard_negative_tier": spec.hard_negative_tier,
                "sample_rate_hz": int(info.samplerate),
                "original_sample_rate_hz": source_rate,
                "duration_seconds": round(float(info.duration), 6),
                "sha256": sha256_file(path),
            }
        )
        if index % 10 == 0 or index == len(specs):
            (output_root / "source_manifest.partial.json").write_text(
                json.dumps({"records": records}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[2/2] sources={index}/{len(specs)}", flush=True)

    manifest = {
        "schema": "wakeword-studio.source-manifest/v1",
        "target": config["wake_word"],
        "generator": "melotts",
        "model_repo": MODEL_REPO,
        "model_license": MODEL_LICENSE,
        "root": str(output_root),
        "records": records,
    }
    manifest_path = output_root / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "source_manifest.partial.json").unlink(missing_ok=True)
    print(f"manifest={manifest_path} records={len(records)}", flush=True)


if __name__ == "__main__":
    main()
