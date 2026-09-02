from __future__ import annotations

import json

import numpy as np

import pytest

from wakeword_studio.json_utils import (
    atomic_write_json,
    json_dumps,
    normalize_json_value,
    plain_json_dumps,
)


def test_recursive_numpy_json_normalization() -> None:
    value = {
        "integer": np.int64(7),
        "float32": np.float32(0.25),
        "float64": np.float64(0.5),
        "boolean": np.bool_(True),
        "array": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "nested": [np.int32(9), (np.float32(1.5), {"ok": np.bool_(False)})],
    }

    normalized = normalize_json_value(value)
    assert type(normalized["integer"]) is int
    assert type(normalized["float32"]) is float
    assert type(normalized["float64"]) is float
    assert type(normalized["boolean"]) is bool
    assert normalized["array"] == [[1, 2], [3, 4]]
    assert isinstance(normalized["nested"][1], list)
    assert type(normalized["nested"][1][1]["ok"]) is bool
    json.dumps(normalized)
    json.loads(json_dumps(value))


def test_real_validation_status_shape_is_json_safe() -> None:
    value = {
        "status": "RUNNING",
        "current_step": np.int64(2500),
        "last_validation": {
            "threshold": np.float64(0.15262039005756378),
            "all_identical": np.bool_(False),
            "false_accepts_by_group": {
                "negative": np.int64(81),
                "hard_negative": np.int64(42),
                "ambient": np.int64(0),
            },
            "score_trace": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        },
    }

    restored = json.loads(json_dumps(value))
    assert restored["current_step"] == 2500
    assert restored["last_validation"]["false_accepts_by_group"] == {
        "negative": 81,
        "hard_negative": 42,
        "ambient": 0,
    }
    assert len(restored["last_validation"]["score_trace"]) == 3


def test_strict_status_serializer_rejects_cycle_and_non_plain_values(tmp_path) -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="Circular JSON reference"):
        plain_json_dumps(cyclic)
    with pytest.raises(TypeError, match="Non-plain JSON value"):
        atomic_write_json(tmp_path / "bad.json", {"value": np.int64(1)})
    assert not (tmp_path / "bad.json").exists()
