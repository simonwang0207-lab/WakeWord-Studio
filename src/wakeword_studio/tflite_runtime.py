"""Small TFLite interpreter adapter shared by runtime-only code paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def interpreter_implementation() -> str:
    """Return the available interpreter without importing full TensorFlow first."""

    try:
        from ai_edge_litert.interpreter import Interpreter  # noqa: F401

        return "ai_edge_litert"
    except ImportError:
        try:
            from tensorflow.lite import Interpreter  # noqa: F401

            return "tensorflow"
        except ImportError as exc:
            raise RuntimeError(
                "TFLite runtime is not installed. Install the project with "
                "`python -m pip install -e \".[runtime,demo]\"`."
            ) from exc


def create_tflite_interpreter(
    *,
    model_path: str | Path | None = None,
    model_content: bytes | None = None,
) -> Any:
    """Create an interpreter from LiteRT, with TensorFlow as a compatibility fallback."""

    implementation = interpreter_implementation()
    if implementation == "ai_edge_litert":
        from ai_edge_litert.interpreter import Interpreter
    else:
        from tensorflow.lite import Interpreter

    if (model_path is None) == (model_content is None):
        raise ValueError("Provide exactly one of model_path or model_content")
    if model_path is not None:
        return Interpreter(model_path=str(model_path))
    return Interpreter(model_content=model_content)
