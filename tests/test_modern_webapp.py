from __future__ import annotations

import hashlib
import base64
import importlib.util
import re
from types import SimpleNamespace
from pathlib import Path

import pytest

from wakeword_studio.webapp import StudioController


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


@pytest.fixture()
def controller() -> StudioController:
    value = StudioController(PROJECT_ROOT, PROJECT_ROOT / "configs/demo/teacher_demo.yaml")
    yield value
    value.stop_live()
    value.playback.close(wait=False)


def test_bootstrap_is_registry_driven_and_formal_plan_is_real(controller: StudioController) -> None:
    boot = controller.bootstrap()
    ids = [item["id"] for item in boot["models"]]
    assert ids[:2] == ["model_a", "model_b"]
    assert {"bcresnet_binary_formal", "convmixer_binary_formal", "teacher_six_bcresnet", "teacher_six_convmixer"} <= set(ids)
    assert [item["threshold"] for item in boot["models"][:2]] == [0.3671875, 0.21875]
    assert boot["active_model_id"] == "teacher_six_convmixer"
    assert boot["ADD_KEYWORD_REQUIRES_RETRAIN"] is True
    assert boot["plans"]["正式训练"]["total"] == 15200
    assert boot["plans"]["正式训练"]["split_targets"] == {
        "train": 12000, "validation": 1600, "test": 1600,
    }
    assert boot["state"]["status"] == "STOPPED"


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="model loading runs in .envs/livekit")
def test_model_b_can_start_and_stop_without_changing_frozen_artifact(controller: StudioController) -> None:
    model = controller.models.by_id("model_b")
    controller.active_models._state["active_model_id"] = "model_b"
    before = hashlib.sha256(model.model_path.read_bytes()).hexdigest()
    started = controller.start_live("model_b")
    assert started["running"] is True
    assert started["threshold"] == pytest.approx(0.21875)
    assert started["status"] == "IDLE"
    stopped = controller.stop_live()
    assert stopped["status"] == "STOPPED"
    assert hashlib.sha256(model.model_path.read_bytes()).hexdigest() == before


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="model inspection runs in .envs/livekit")
def test_model_inspection_reports_frozen_final_b2(controller: StudioController) -> None:
    info = controller.inspect_model("model_b")
    assert info["kib"] == pytest.approx(110.171875)
    assert info["full_int8"] is True
    assert info["sha256"] == "6acfecf7cc8651c1fba52809eee1d89abbcffa0a48bd46662b2e58ac23ce064f"


def test_default_launcher_selects_modern_ui_and_keeps_optional_desktop_mode() -> None:
    source = (PROJECT_ROOT / "run_studio.py").read_text(encoding="utf-8")
    assert "from run_studio_desktop import main as desktop_main" in source
    assert "from run_studio_modern import main as modern_main" in source
    assert "--desktop" in source
    assert "--legacy" in source


def test_frontend_javascript_references_existing_dom_ids() -> None:
    html = (PROJECT_ROOT / "phase7/webui/index.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "phase7/webui/app.js").read_text(encoding="utf-8")
    referenced = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", javascript))
    declared = set(re.findall(r'id="([^"]+)"', html))
    assert referenced - declared == set()


def test_frontend_exposes_custom_targets_history_filters_and_model_import() -> None:
    html = (PROJECT_ROOT / "phase7/webui/index.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "phase7/webui/app.js").read_text(encoding="utf-8")
    for element_id in ("custom-positive", "custom-hard", "custom-negative", "custom-ambient"):
        assert f'id="{element_id}"' in html
    assert 'data-filter="WAKE"' in html
    assert 'data-filter="REJECT"' in html
    assert "historyFilter=button.dataset.filter" in javascript
    assert "/api/model/import" in javascript


def test_user_tflite_import_is_registered_without_becoming_trainable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "teacher_demo.yaml"
    config_path.write_text(
        (PROJECT_ROOT / "configs/demo/teacher_demo.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "wakeword_studio.webapp.inspect_tflite_model",
        lambda path: SimpleNamespace(
            bytes=2048, kib=2.0, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            full_int8=True, input_shape=[1, 99, 40], output_shape=[1, 1],
            input_dtype="int8", output_dtype="int8",
        ),
    )
    temporary = StudioController(tmp_path, config_path)
    try:
        result = temporary.import_model({
            "name": "Model C Classroom", "backend": "repcnn", "threshold": 0.42,
            "data_base64": base64.b64encode(b"TFL3" + b"\0" * 2044).decode("ascii"),
        })
        imported = temporary.models.by_id(result["model_id"])
        assert imported.display_name == "Model C Classroom"
        assert imported.threshold == pytest.approx(0.42)
        assert imported.trainer == {}
        assert imported.model_path.is_file()
        assert (tmp_path / "configs/demo/user_models.json").is_file()
    finally:
        temporary.playback.close(wait=False)
