from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from wakeword_studio.backends.microwakeword import MicroWakeWordBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/demo/teacher_demo.yaml"
MODEL_KEY = "Model A — MicroWakeWord Tiny"


def _model_config() -> dict[str, object]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return config["models"][MODEL_KEY]


def test_model_a_deployment_binding_is_frozen() -> None:
    model = _model_config()
    path = Path(str(model["path"]))
    assert model["backend"] == "microwakeword"
    assert float(model["threshold"]) == pytest.approx(0.3671875)
    assert path.is_file()
    assert path.stat().st_size == 52840


def test_model_a_tflite_loads_and_infers_without_training_package() -> None:
    pytest.importorskip("tensorflow")
    pytest.importorskip("pymicro_features")
    model = _model_config()
    backend = MicroWakeWordBackend(keyword="你好，青小甲")
    backend.load(Path(str(model["path"])))

    scores = backend.stream_scores(np.zeros(2 * 16000, dtype=np.int16))

    assert list(scores) == ["你好，青小甲"]
    assert 0.0 <= scores["你好，青小甲"] <= 1.0
    backend.reset_stream()


def test_model_b_deployment_binding_is_unchanged() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model = config["models"]["Model B — RepCNN Performance"]
    assert float(model["threshold"]) == pytest.approx(0.21875)
    assert model["deployment"]["sha256"] == (
        "6acfecf7cc8651c1fba52809eee1d89abbcffa0a48bd46662b2e58ac23ce064f"
    )
