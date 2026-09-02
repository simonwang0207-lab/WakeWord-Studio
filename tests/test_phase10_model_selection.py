"""Phase 10 registry -> runtime -> API -> UI model-selection contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wakeword_studio.registry import ActiveModelStore
from wakeword_studio.webapp import StudioController


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HISTORICAL_MODELS = {
    "model_a",
    "model_b",
    "bcresnet_binary_formal",
    "convmixer_binary_formal",
    "teacher_six_bcresnet",
    "teacher_six_convmixer",
}


@pytest.fixture()
def isolated_controller(tmp_path: Path) -> StudioController:
    controller = StudioController(PROJECT_ROOT, PROJECT_ROOT / "configs/demo/teacher_demo.yaml")
    controller.active_models = ActiveModelStore(
        tmp_path / "active_model.json",
        controller.models,
        "teacher_six_convmixer",
    )
    controller.model = controller.models.by_id(controller.active_models.active_model_id)
    controller.preload_runtime_backend = True
    controller.loaded_backend = SimpleNamespace(model_id=controller.model.id)
    controller.loaded_backend_model_id = controller.model.id
    controller.loaded_backend_keyword = controller.keyword
    controller._prepare_runtime_backend = lambda model: SimpleNamespace(model_id=model.id)  # type: ignore[method-assign]
    yield controller
    controller.stop_live()
    controller.playback.close(wait=False)


def test_default_registry_and_web_api_state_are_convmixer() -> None:
    controller = StudioController(PROJECT_ROOT, PROJECT_ROOT / "configs/demo/teacher_demo.yaml")
    try:
        boot = controller.bootstrap()
        assert EXPECTED_HISTORICAL_MODELS <= {item["id"] for item in boot["models"]}
        assert boot["active_model_id"] == "teacher_six_convmixer"
        assert boot["state"]["active_model_id"] == "teacher_six_convmixer"
        assert boot["state"]["model_id"] == "teacher_six_convmixer"
        assert next(item for item in boot["models"] if item["active"])["id"] == "teacher_six_convmixer"
    finally:
        controller.playback.close(wait=False)


def test_switch_chain_keeps_registry_runtime_and_api_consistent(
    isolated_controller: StudioController,
) -> None:
    for model_id in (
        "teacher_six_bcresnet",
        "model_a",
        "model_b",
        "teacher_six_convmixer",
    ):
        isolated_controller.activate_model(model_id)
        boot = isolated_controller.bootstrap()
        assert isolated_controller.active_models.active_model_id == model_id
        assert isolated_controller.model.id == model_id
        assert isolated_controller.loaded_backend_model_id == model_id
        assert boot["active_model_id"] == model_id
        assert boot["state"]["active_model_id"] == model_id
        assert boot["state"]["model_id"] == model_id
        assert boot["state"]["runtime_backend_model_id"] == model_id


def test_failed_switch_preserves_previous_active_backend(
    isolated_controller: StudioController,
) -> None:
    previous = isolated_controller.active_models.active_model_id
    previous_backend = isolated_controller.loaded_backend

    def fail_for_bc(model):  # noqa: ANN001
        if model.id == "teacher_six_bcresnet":
            raise RuntimeError("synthetic load failure")
        return SimpleNamespace(model_id=model.id)

    isolated_controller._prepare_runtime_backend = fail_for_bc  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="synthetic load failure"):
        isolated_controller.activate_model("teacher_six_bcresnet")
    assert isolated_controller.active_models.active_model_id == previous
    assert isolated_controller.model.id == previous
    assert isolated_controller.loaded_backend is previous_backend


def test_active_store_write_failure_does_not_change_memory_state(
    isolated_controller: StudioController,
) -> None:
    previous = isolated_controller.active_models.active_model_id

    def fail_write(state):  # noqa: ANN001
        raise PermissionError("synthetic Windows write denial")

    isolated_controller.active_models._write = fail_write  # type: ignore[method-assign]
    with pytest.raises(PermissionError, match="synthetic Windows write denial"):
        isolated_controller.active_models.activate("teacher_six_bcresnet")
    assert isolated_controller.active_models.active_model_id == previous


def test_teacher_six_metadata_is_model_specific_and_frozen() -> None:
    controller = StudioController(PROJECT_ROOT, PROJECT_ROOT / "configs/demo/teacher_demo.yaml")
    try:
        models = {item["id"]: item for item in controller.bootstrap()["models"]}
        conv = models["teacher_six_convmixer"]
        bc = models["teacher_six_bcresnet"]
        assert (conv["threshold"], conv["margin_threshold"]) == (0.4, 0.0)
        assert (bc["threshold"], bc["margin_threshold"]) == (0.0, 0.2)
        for item in (conv, bc):
            assert item["task_type"] == "multi_kws"
            assert item["full_int8"] is True
            assert item["input_shape"] == [1, 99, 40]
            assert item["output_shape"] == [1, 7]
            assert item["num_classes"] == 7
            assert len(item["classes"]) == 7
            assert len(item["sha256"]) == 64
    finally:
        controller.playback.close(wait=False)


def test_live_ui_uses_dynamic_grouped_registry_picker() -> None:
    html = (PROJECT_ROOT / "phase7/webui/index.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "phase7/webui/app.js").read_text(encoding="utf-8")
    assert 'id="live-model-select"' in html
    assert "picker.onchange=()=>activateLiveModel(picker.value)" in javascript
    assert "boot.active_model_id" in javascript
    assert "['Multi-KWS',multi],['Binary',binary],['Imported',imported]" in javascript
    assert "selectedModel='model_b'" not in javascript
    for element_id in (
        "runtime-background-score",
        "runtime-detection",
        "runtime-rejection",
        "runtime-final-keyword",
        "runtime-architecture",
        "runtime-class-count",
    ):
        assert f'id="{element_id}"' in html
