"""Extract train/validation microfrontend tensors without opening Test audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_window(audio: np.ndarray, sample_rate: int, samples: int = 32_000) -> np.ndarray:
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate}")
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("Expected mono audio")
    if len(signal) > samples:
        start = (len(signal) - samples) // 2
        signal = signal[start : start + samples]
    elif len(signal) < samples:
        missing = samples - len(signal)
        signal = np.pad(signal, (missing // 2, missing - missing // 2))
    return signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    frontend = MicroFrontend(
        sample_rate=16_000, window_size_ms=30, window_step_ms=20, num_channels=40
    )
    features: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
    labels: dict[str, list[int]] = {"train": [], "validation": []}
    metadata: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    opened_test_audio = False

    for record in manifest["records"]:
        split = str(record["split"])
        if split == "test":
            continue
        if split not in features:
            raise ValueError(f"Unexpected split: {split}")
        audio, sample_rate = sf.read(root / str(record["path"]), dtype="float32", always_2d=False)
        feature = np.asarray(frontend(fixed_window(audio, sample_rate))[0], dtype=np.float32)
        if feature.shape != (99, 40):
            raise RuntimeError(f"Unexpected microfrontend shape: {feature.shape}")
        features[split].append(feature)
        labels[split].append(int(record["class_index"]))
        speaker = record.get("speaker", {})
        metadata[split].append(
            {
                "record_id": record["record_id"],
                "keyword_id": record["keyword_id"],
                "source": speaker.get("source", "unknown"),
                "speaker_id": speaker.get("speaker_id", "unknown"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x_train=np.asarray(features["train"], np.float32),
        y_train=np.asarray(labels["train"], np.int32),
        x_validation=np.asarray(features["validation"], np.float32),
        y_validation=np.asarray(labels["validation"], np.int32),
    )
    report = {
        "schema": "wakeword-studio.multikws-features/v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "feature_store": str(args.output.resolve()),
        "feature_store_sha256": sha256(args.output),
        "input_shape": [99, 40],
        "counts": {split: len(rows) for split, rows in features.items()},
        "metadata": metadata,
        "TEST_READ": opened_test_audio,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("counts", "input_shape", "TEST_READ")}))


if __name__ == "__main__":
    main()
