"""JSON helpers that normalize NumPy values without changing calculations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def normalize_json_value(value: Any) -> Any:
    """Recursively convert NumPy containers/scalars to JSON-native values."""

    if isinstance(value, np.ndarray):
        return normalize_json_value(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {
            normalize_json_value(key): normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    return value


def json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize after recursive NumPy normalization."""

    return json.dumps(normalize_json_value(value), **kwargs)


def validate_plain_json_tree(value: Any) -> None:
    """Reject cycles and non-native JSON values before an atomic status write."""

    active: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not np.isfinite(item):
                raise ValueError(f"Non-finite JSON float at {path}")
            return
        if type(item) not in (dict, list):
            raise TypeError(
                f"Non-plain JSON value at {path}: {type(item).__name__}"
            )
        identity = id(item)
        if identity in active:
            raise ValueError(f"Circular JSON reference detected at {path}")
        active.add(identity)
        try:
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise TypeError(
                            f"Non-string JSON object key at {path}: {type(key).__name__}"
                        )
                    visit(child, f"{path}.{key}")
            else:
                for index, child in enumerate(item):
                    visit(child, f"{path}[{index}]")
        finally:
            active.remove(identity)

    visit(value, "$")


def plain_json_dumps(value: Any, **kwargs: Any) -> str:
    """Strict serializer for training state; no coercion and no ``default=str``."""

    validate_plain_json_tree(value)
    return json.dumps(value, allow_nan=False, **kwargs)


def atomic_write_json(path: Path, value: Any) -> None:
    """Validate/serialize completely before touching the destination path."""

    serialized = plain_json_dumps(value, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(serialized, encoding="utf-8")
    partial.replace(path)
