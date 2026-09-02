"""Download a tiny, metadata-grounded AISHELL-3 voice-cloning reference set.

This is a reference-only POC artifact.  It never downloads the 18/19 GB
OpenSLR archive and never adds audio to a formal DatasetManifest.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from wakeword_studio.dataset.manifest import sha256_file


REPO_ID = "AISHELL/AISHELL-3"
REPO_TYPE = "dataset"
LICENSE = "Apache-2.0"
AGE_RANGES = {"A": "<14", "B": "14-25", "C": "26-40", "D": ">41"}


@dataclass(frozen=True, slots=True)
class SpeakerChoice:
    speaker_id: str
    age_code: str
    gender: str
    accent: str


SPEAKERS = (
    SpeakerChoice("SSB0393", "A", "female", "north"),
    SpeakerChoice("SSB0273", "B", "male", "north"),
    SpeakerChoice("SSB0632", "B", "female", "south"),
    SpeakerChoice("SSB0710", "C", "male", "north"),
    SpeakerChoice("SSB0197", "C", "female", "south"),
    SpeakerChoice("SSB0434", "D", "male", "north"),
    SpeakerChoice("SSB0737", "D", "female", "north"),
)


def cjk_text(value: str) -> str:
    return "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))


def wav_info(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "sample_rate_hz": rate,
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "duration_seconds": round(frames / rate, 6),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    source_root = output_root / "official_snapshot"
    references_root = output_root / "references"
    source_root.mkdir(parents=True, exist_ok=True)
    references_root.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_info = api.repo_info(REPO_ID, repo_type=REPO_TYPE)
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE, revision=repo_info.sha)
    available = set(files)

    metadata_paths: dict[str, Path] = {}
    for filename in ("spk-info.txt", "test/content.txt"):
        metadata_paths[filename] = Path(
            hf_hub_download(
                REPO_ID,
                filename,
                repo_type=REPO_TYPE,
                revision=repo_info.sha,
                local_dir=source_root,
            )
        )

    official_speakers: dict[str, tuple[str, str, str]] = {}
    for line in metadata_paths["spk-info.txt"].read_text(encoding="utf-8").splitlines():
        if line.startswith("SSB"):
            speaker_id, age_code, gender, accent = line.split()
            official_speakers[speaker_id] = (age_code, gender, accent)

    transcripts: dict[str, dict[str, str]] = {}
    for line in metadata_paths["test/content.txt"].read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        filename, annotated = line.split("\t", 1)
        transcripts[filename] = {
            "annotated_transcript": annotated,
            "text": cjk_text(annotated),
        }

    records: list[dict[str, object]] = []
    for index, choice in enumerate(SPEAKERS, start=1):
        actual = official_speakers.get(choice.speaker_id)
        expected = (choice.age_code, choice.gender, choice.accent)
        if actual != expected:
            raise RuntimeError(
                f"Official metadata changed for {choice.speaker_id}: expected={expected}, actual={actual}"
            )
        candidates = sorted(
            name
            for name in available
            if name.startswith(f"test/wav/{choice.speaker_id}/") and name.endswith(".wav")
        )
        candidates = [name for name in candidates if Path(name).name in transcripts]
        if not candidates:
            raise RuntimeError(f"No mirrored test WAV with transcript for {choice.speaker_id}")
        selected = max(
            candidates,
            key=lambda name: (len(transcripts[Path(name).name]["text"]), Path(name).name),
        )
        source_path = Path(
            hf_hub_download(
                REPO_ID,
                selected,
                repo_type=REPO_TYPE,
                revision=repo_info.sha,
                local_dir=source_root,
            )
        )
        destination = references_root / f"{choice.speaker_id}_{Path(selected).name}"
        if not destination.exists() or sha256_file(destination) != sha256_file(source_path):
            partial = destination.with_suffix(".partial.wav")
            shutil.copy2(source_path, partial)
            partial.replace(destination)
        transcript = transcripts[Path(selected).name]
        record = {
            "speaker_id": choice.speaker_id,
            "source_type": "public_corpus_reference",
            "source_family": "aishell3",
            "source_repo": REPO_ID,
            "source_revision": repo_info.sha,
            "source_path": selected,
            "reference_path": destination.relative_to(output_root).as_posix(),
            "license": LICENSE,
            "age_group": choice.age_code,
            "age_range_years": AGE_RANGES[choice.age_code],
            "age_group_source": "verified_dataset_metadata",
            "age_verified_by_project": False,
            "gender_if_available": choice.gender,
            "accent": choice.accent,
            "text": transcript["text"],
            "annotated_transcript": transcript["annotated_transcript"],
            **wav_info(destination),
            "sha256": sha256_file(destination),
        }
        records.append(record)
        print(
            f"AISHELL3_REFERENCE {index}/{len(SPEAKERS)} speaker={choice.speaker_id} "
            f"age={choice.age_code} gender={choice.gender} duration={record['duration_seconds']}s",
            flush=True,
        )

    manifest = {
        "schema": "wakeword-studio.voice-reference-poc/v1",
        "status": "REFERENCES_READY_VOXCPM_DOWNLOAD_APPROVAL_PENDING",
        "formal_dataset_eligible": False,
        "purpose": "VoxCPM1.5 voice-cloning POC references only",
        "source_repo": REPO_ID,
        "source_revision": repo_info.sha,
        "source_license": LICENSE,
        "openslr_full_archive_required": False,
        "age_code_definition": AGE_RANGES,
        "records": records,
    }
    manifest_path = output_root / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AISHELL3_REFERENCE_COMPLETE manifest={manifest_path} records={len(records)}", flush=True)


if __name__ == "__main__":
    main()
