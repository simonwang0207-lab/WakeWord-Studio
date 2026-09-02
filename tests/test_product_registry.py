from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wakeword_studio.providers import ProviderRegistry
from wakeword_studio.registry import ModelRegistry, resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/demo/teacher_demo.yaml").read_text(encoding="utf-8")
    )


def test_model_a_and_b_are_loaded_through_one_registry() -> None:
    registry = ModelRegistry.from_config(PROJECT_ROOT, _config())
    assert registry.display_names[:2] == ("Model A — microWakeWord Tiny", "Model B — RepCNN")
    assert {"BC-ResNet Binary Formal", "ConvMixer Binary Formal"} <= set(registry.display_names)

    model_a = registry.by_id("model_a")
    assert model_a.threshold == pytest.approx(0.3671875)
    assert model_a.runtime_mode == "native_streaming"
    assert model_a.model_size_kib == pytest.approx(51.6015625)

    model_b = registry.by_id("model_b")
    assert model_b.threshold == pytest.approx(0.21875)
    assert model_b.runtime_mode == "rolling_window"
    assert model_b.window_seconds == pytest.approx(2.0)
    assert model_b.hop_seconds == pytest.approx(0.20)
    assert model_b.smoothing == "raw"


def test_registered_paths_relocate_with_project_root(tmp_path: Path) -> None:
    registry = ModelRegistry.from_config(tmp_path, _config())
    for model in registry.all():
        assert model.model_path.is_relative_to(tmp_path)
    assert resolve_project_path(tmp_path, "assets/i_am_awake.wav") == (
        tmp_path / "assets/i_am_awake.wav"
    ).resolve()


def test_provider_registry_exposes_registered_generation_providers() -> None:
    registry = ProviderRegistry.from_config(PROJECT_ROOT, _config())
    available = registry.available(generation_only=True)
    assert [provider.name for provider in available] == ["Kokoro", "本地语音文件夹"]
    metadata = available[0].metadata()
    assert metadata["available"] is True
    assert metadata["age_coverage_claimed"] is False
    assert metadata["capabilities"]["multi_speaker"] is True


def test_adding_model_c_requires_only_registration() -> None:
    config = _config()
    config["models"]["Model C"] = {
        "id": "model_c",
        "display_name": "Model C — Example",
        "backend": "repcnn",
        "path": "models/model_c.tflite",
        "threshold": 0.5,
        "runtime_mode": "rolling_window",
        "window_seconds": 2.0,
        "hop_seconds": 0.2,
        "smoothing": "raw",
        "deployment": {"kib": 120.0},
    }
    registry = ModelRegistry.from_config(PROJECT_ROOT, config)
    assert registry.by_id("model_c").display_name == "Model C — Example"
