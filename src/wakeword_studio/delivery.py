"""Model inspection and explicit, unvalidated ESP32-S3 package preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelInfo:
    path: str
    bytes: int
    kib: float
    sha256: str
    full_int8: bool
    input_shape: list[int]
    input_dtype: str
    input_quantization: tuple[float, int]
    output_shape: list[int]
    output_dtype: str
    output_quantization: tuple[float, int]


def inspect_tflite_model(path: Path) -> ModelInfo:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("Delivery demo supports one-input/one-output TFLite models")
    input_detail = inputs[0]
    output_detail = outputs[0]
    input_quantization = input_detail["quantization"]
    output_quantization = output_detail["quantization"]
    content = path.read_bytes()
    input_dtype = np.dtype(input_detail["dtype"]).name
    output_dtype = np.dtype(output_detail["dtype"]).name
    return ModelInfo(
        path=str(path),
        bytes=len(content),
        kib=len(content) / 1024.0,
        sha256=hashlib.sha256(content).hexdigest(),
        full_int8=input_dtype in {"int8", "uint8"} and output_dtype in {"int8", "uint8"},
        input_shape=[int(value) for value in input_detail["shape"]],
        input_dtype=input_dtype,
        input_quantization=(float(input_quantization[0]), int(input_quantization[1])),
        output_shape=[int(value) for value in output_detail["shape"]],
        output_dtype=output_dtype,
        output_quantization=(float(output_quantization[0]), int(output_quantization[1])),
    )


def prepare_esp32s3_package(model_path: Path, output_dir: Path) -> dict[str, object]:
    """Copy a model plus metadata; this deliberately makes no hardware claim."""

    info = inspect_tflite_model(model_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / Path(info.path).name
    if destination.resolve() != Path(info.path).resolve():
        shutil.copy2(info.path, destination)
    report = {
        "schema": "wakeword-studio.esp32s3-export-package/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {**asdict(info), "packaged_path": str(destination)},
        "hardware_validation": False,
        "status": "ARTIFACT_PREPARED_NOT_HARDWARE_VALIDATED",
        "integration_target": "firmware/repcnn_esp32s3",
    }
    (output_dir / "model_info.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "ESP32-S3 deployment interface artifact only.\n"
        "No physical ESP32-S3 inference, latency, memory, or wake acceptance test has been run.\n",
        encoding="utf-8",
    )
    return report
