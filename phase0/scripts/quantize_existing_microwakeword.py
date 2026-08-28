"""Quantize an already-exported microWakeWord streaming SavedModel."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

from microwakeword.data import FeatureHandler
from microwakeword.utils import convert_saved_model_to_tflite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.train_dir / "training_config.yaml"
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    data_processor = FeatureHandler(config)
    source = args.train_dir / "stream_state_internal"
    destination = args.train_dir / "tflite_stream_state_internal_quant"
    filename = "stream_state_internal_quant.tflite"
    convert_saved_model_to_tflite(
        config=config,
        audio_processor=data_processor,
        path_to_model=str(source),
        folder=str(destination),
        fname=filename,
        quantize=True,
    )
    model_path = destination / filename
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    print(f"path={model_path.resolve()}")
    print(f"bytes={model_path.stat().st_size}")
    print(f"kib={model_path.stat().st_size / 1024:.6f}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
