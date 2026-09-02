from __future__ import annotations

import tomllib
from pathlib import Path

from wakeword_studio.tflite_runtime import (
    create_tflite_interpreter,
    interpreter_implementation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_extra_does_not_install_tensorflow() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert "numpy>=1.24,<2" in project["project"]["dependencies"]
    assert not any(item.startswith("tensorflow") for item in extras["runtime"])
    assert any(item.startswith("ai-edge-litert") for item in extras["runtime"])
    assert any(item.startswith("pymicro-features") for item in extras["runtime"])
    assert not any(item.startswith("livekit-wakeword") for item in extras["runtime"])
    assert any(item.startswith("tensorflow") for item in extras["training"])
    assert any(item.startswith("livekit-wakeword") for item in extras["training"])


def test_release_model_loads_with_available_lightweight_interpreter() -> None:
    model_path = (
        PROJECT_ROOT
        / "artifacts/models/teacher_six/teacher_six_convmixer_full_int8.tflite"
    )
    interpreter = create_tflite_interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    assert interpreter.get_input_details()[0]["shape"].tolist() == [1, 99, 40]
    assert interpreter.get_output_details()[0]["shape"].tolist() == [1, 7]
    assert interpreter_implementation() in {"ai_edge_litert", "tensorflow"}
