"""Validated command builders and managed subprocesses for the teacher UI."""

from __future__ import annotations

import subprocess
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import yaml

from .providers import ProviderRegistry
from .registry import ModelRegistry, resolve_project_path
from .dataset.product_plan import build_product_plan


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    wake_phrase: str
    sample_count: int
    source: str
    noise_augmentation: bool
    output_folder: Path
    scale_mode: str = "legacy"
    custom_total: int | None = None
    input_folder: Path | None = None
    custom_targets: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    dataset_folder: Path
    model_type: str
    wake_phrase: str
    output_run_directory: Path


def build_generation_command(project_root: Path, request: GenerationRequest) -> list[str]:
    if not request.wake_phrase.strip():
        raise ValueError("Wake phrase is required")
    config = yaml.safe_load(
        (project_root / "configs/demo/teacher_demo.yaml").read_text(encoding="utf-8")
    )
    providers = ProviderRegistry.from_config(project_root, config)
    try:
        provider = providers.by_name(request.source)
    except KeyError as exc:
        raise ValueError(f"数据源尚未注册：{request.source}") from exc
    if not provider.available():
        raise ValueError(f"数据源当前不可用：{request.source}")
    python = provider.python_executable()
    if not provider.worker:
        raise ValueError(f"数据源没有配置生成 worker：{request.source}")
    script = resolve_project_path(project_root, provider.worker)
    output_folder = resolve_project_path(project_root, str(request.output_folder))
    if not script.is_file():
        raise FileNotFoundError(f"数据生成脚本不存在：{script}")
    if provider.kind == "local_folder":
        input_folder = None if request.input_folder is None else resolve_project_path(project_root, str(request.input_folder))
        if input_folder is None or not input_folder.is_dir():
            raise ValueError("本地语音文件夹不存在")
        return [
            str(python), str(script), "--wake-word", request.wake_phrase.strip(),
            "--input-folder", str(input_folder),
            "--output-root", str(output_folder),
            "--augmentation", "standard" if request.noise_augmentation else "none",
        ]
    if provider.kind != "tts":
        raise ValueError(f"数据源不支持生成：{request.source}")
    if request.scale_mode == "legacy":
        if not 2 <= int(request.sample_count) <= 12:
            raise ValueError("Kokoro voice slot count must be between 2 and 12")
        target_args = ["--per-label", str(int(request.sample_count))]
    else:
        mode_ids = {"快速测试": "quick", "小规模实验": "small", "正式训练": "formal", "自定义": "custom"}
        build_product_plan(request.scale_mode, custom_total=request.custom_total, custom_targets=request.custom_targets)
        if request.scale_mode == "自定义" and request.custom_targets:
            target_args = ["--targets-json", json.dumps(request.custom_targets, ensure_ascii=False)]
        else:
            target_args = ["--product-mode", mode_ids[request.scale_mode]]
        if request.scale_mode == "自定义" and not request.custom_targets:
            target_args += ["--custom-total", str(int(request.custom_total or request.sample_count))]
    command = [
        str(python),
        str(script),
        "--wake-word",
        request.wake_phrase.strip(),
        "--output-root",
        str(output_folder),
        *target_args,
        "--noise-augmentation",
        "standard" if request.noise_augmentation else "none",
    ]
    return command


def build_training_command(project_root: Path, request: TrainingRequest) -> list[str]:
    if request.wake_phrase.strip() != "你好，青小甲":
        raise ValueError("Frozen training launchers currently support wake phrase 你好，青小甲")
    dataset = resolve_project_path(project_root, str(request.dataset_folder))
    if not (dataset / "DatasetManifest.json").is_file():
        raise FileNotFoundError(dataset / "DatasetManifest.json")
    config = yaml.safe_load(
        (project_root / "configs/demo/teacher_demo.yaml").read_text(encoding="utf-8")
    )
    registry = ModelRegistry.from_config(project_root, config)
    try:
        model = registry.resolve(request.model_type)
    except KeyError as exc:
        raise ValueError(f"未注册的模型：{request.model_type}") from exc
    if not model.trainer:
        raise ValueError(f"模型没有配置训练入口：{model.display_name}")
    expected = resolve_project_path(project_root, model.trainer["dataset"])
    if dataset != expected:
        raise ValueError(f"{model.display_name} requires {expected.name}（需要指定训练数据集）")
    python = model.training_python(project_root)
    script = resolve_project_path(project_root, model.trainer["script"])
    training_config = resolve_project_path(project_root, model.trainer["config"])
    if not python.is_file():
        raise FileNotFoundError(f"训练 Python 不存在：{python}")
    if not script.is_file() or not training_config.is_file():
        raise FileNotFoundError("训练脚本或配置文件不存在")
    return [
        str(python),
        str(script),
        "--config",
        str(training_config),
        "--run-dir",
        str(resolve_project_path(project_root, str(request.output_run_directory))),
        "--allow-formal-training",
    ]


class ManagedProcess:
    """Own exactly one UI-launched child; never adopts or stops external training."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(
        self,
        command: Sequence[str],
        on_line: Callable[[str], None],
        *,
        cwd: Path | None = None,
    ) -> int:
        if self.running:
            raise RuntimeError("A managed process is already running")
        self.process = subprocess.Popen(
            list(command),
            cwd=str(cwd.resolve()) if cwd else None,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stdout is not None
        for line in self.process.stdout:
            on_line(line.rstrip())
        return int(self.process.wait())

    def stop(self) -> None:
        if self.running:
            assert self.process is not None
            self.process.terminate()
