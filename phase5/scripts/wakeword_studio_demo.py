"""WakeWord Studio 中文教师演示界面。"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from wakeword_studio.delivery import inspect_tflite_model, prepare_esp32s3_package  # noqa: E402
from wakeword_studio.launchers import (  # noqa: E402
    GenerationRequest,
    ManagedProcess,
    TrainingRequest,
    build_generation_command,
    build_training_command,
)
from wakeword_studio.runtime.detection_logic import DetectionConfig, DetectionLogic  # noqa: E402
from wakeword_studio.runtime.engine import StreamingWakeWordEngine  # noqa: E402
from wakeword_studio.runtime.live_audio import MicrophoneCapture, list_input_devices  # noqa: E402
from wakeword_studio.runtime.playback import WakePlaybackQueue  # noqa: E402
from wakeword_studio.providers import ProviderRegistry  # noqa: E402
from wakeword_studio.dataset.product_plan import build_product_plan  # noqa: E402
from wakeword_studio.ui.theme import apply_modern_theme  # noqa: E402
from wakeword_studio.registry import ModelRegistry, resolve_project_path  # noqa: E402

MODEL_A = "Model A — microWakeWord Tiny"
MODEL_B = "Model B — RepCNN"
STATE_TEXT = {
    "IDLE": "● 等待语音", "SOUND": "● 检测到声音", "SPEECH": "● 检测到语音",
    "EVALUATING": "● 正在识别", "TAIL": "● 尾部识别", "WAKE": "✓ 唤醒成功", "COOLDOWN": "◷ 冷却中", "STOPPED": "■ 已停止",
}


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    """One completed KWS episode, detached from mutable backend state."""

    detected_at: datetime
    keyword: str
    result: str
    raw_max_score: float
    decision_max_score: float
    threshold: float
    energy_result: str
    vad_result: str
    speech_result: str
    l1_result: str
    l2_result: str
    l3_result: str
    l4_result: str
    l5_result: str
    rejection_reason: str
    duration_seconds: float
    inference_windows: int
    peak_window: int | None


def request_final_wake_playback(
    playback: WakePlaybackQueue,
    snapshot: DetectionSnapshot,
    awake_wav: Path,
) -> bool:
    """Bind playback only to the one latched edge of a completed WAKE episode."""

    if snapshot.result != "WAKE":
        return False
    return playback.request(
        awake_wav,
        episode_id=snapshot.detected_at.isoformat(timespec="milliseconds"),
    )


class DetectionEpisodeTracker:
    """Latch WAKE/REJECT results while discarding background IGNORE traffic."""

    NO_SCORE_REASONS = {"", "NO_NEW_SCORE"}

    def __init__(self, *, max_history: int = 10, timeout_seconds: float = 5.0) -> None:
        self.max_history = int(max_history)
        self.timeout_seconds = float(timeout_seconds)
        self.latest: DetectionSnapshot | None = None
        self.history: deque[DetectionSnapshot] = deque(maxlen=self.max_history)
        self._active = False
        self._suppressed_until_inactive = False
        self._reset_episode_fields()

    def _reset_episode_fields(self) -> None:
        self._started_at = 0.0
        self._raw_max = 0.0
        self._decision_max = 0.0
        self._threshold = 0.0
        self._energy_passed = False
        self._vad_passed = False
        self._speech_passed = False
        self._l1_executed = False
        self._l1_passed = False
        self._l2_executed = False
        self._l2_passed = False
        self._cooldown_blocked = False
        self._l3_executed = False
        self._l5_executed = False
        self._l5_passed = False
        self._last_rejection_reason = ""
        self._inference_windows = 0
        self._peak_window: int | None = None

    def cancel_active(self) -> None:
        """Cancel an unfinished episode on manual stop without recording REJECT."""

        self._active = False
        self._suppressed_until_inactive = False
        self._reset_episode_fields()

    def clear_latest(self) -> None:
        self.latest = None

    def clear_history(self) -> None:
        self.history.clear()

    def update(
        self,
        state,  # noqa: ANN001
        *,
        keyword: str,
        now: float | None = None,
        wall_time: datetime | None = None,
    ) -> DetectionSnapshot | None:
        """Consume one RuntimeLog and return exactly one completed snapshot."""

        now = time.monotonic() if now is None else float(now)
        kws_active = bool(getattr(state, "kws_active", False))
        if self._suppressed_until_inactive:
            if not kws_active:
                self._suppressed_until_inactive = False
            return None
        if not self._active:
            if not kws_active:
                return None  # IGNORE: the three-stage gate never activated KWS.
            self._active = True
            self._reset_episode_fields()
            self._started_at = now

        self._accumulate(state)
        if bool(getattr(state, "final_wake_event", False)):
            return self._finalize("WAKE", keyword, now, wall_time, suppress=kws_active)
        if now - self._started_at >= self.timeout_seconds:
            self._last_rejection_reason = "REJECT_TIMEOUT"
            return self._finalize("REJECT", keyword, now, wall_time, suppress=kws_active)
        if not kws_active:
            if not self._last_rejection_reason:
                self._last_rejection_reason = "NO_VALID_MODEL_SCORE"
            return self._finalize("REJECT", keyword, now, wall_time, suppress=False)
        return None

    def _accumulate(self, state) -> None:  # noqa: ANN001
        energy = float(getattr(state, "energy", 0.0) or 0.0)
        adaptive_threshold = float(getattr(state, "adaptive_threshold", 0.0) or 0.0)
        self._energy_passed = self._energy_passed or energy >= adaptive_threshold
        self._vad_passed = self._vad_passed or bool(getattr(state, "vad", False))
        self._speech_passed = self._speech_passed or int(getattr(state, "speech_frame_count", 0) or 0) >= 3
        self._threshold = float(getattr(state, "wake_threshold", self._threshold) or self._threshold)

        reason = str(getattr(state, "rejection_reason", "") or "")
        new_score = reason not in self.NO_SCORE_REASONS
        raw = float(getattr(state, "raw_wake_score", 0.0) or 0.0)
        decision = float(getattr(state, "decision_wake_score", raw) or 0.0)
        if new_score:
            self._inference_windows += 1
            self._l1_executed = True
            self._l2_executed = True
            self._l3_executed = True
            if decision > self._decision_max or self._peak_window is None:
                self._peak_window = self._inference_windows
            self._raw_max = max(self._raw_max, raw)
            self._decision_max = max(self._decision_max, decision)
        l1_text = str(getattr(state, "l1_status", ""))
        self._l1_passed = self._l1_passed or l1_text.endswith(":True")
        self._l2_passed = self._l2_passed or str(getattr(state, "l2_status", "False")).lower() == "true"
        cooldown = float(getattr(state, "cooldown", 0.0) or 0.0)
        self._cooldown_blocked = self._cooldown_blocked or cooldown > 0.0 or reason in {"COOLDOWN", "L3_COOLDOWN_ACTIVE"}
        l5 = str(getattr(state, "l5_status", "waiting") or "waiting")
        self._l5_executed = self._l5_executed or l5 != "waiting"
        self._l5_passed = self._l5_passed or l5 == "passed"
        if reason not in self.NO_SCORE_REASONS | {"FINAL_WAKE_EVENT"}:
            self._last_rejection_reason = reason

    def _finalize(
        self,
        result: str,
        keyword: str,
        now: float,
        wall_time: datetime | None,
        *,
        suppress: bool,
    ) -> DetectionSnapshot:
        snapshot = DetectionSnapshot(
            detected_at=wall_time or datetime.now(),
            keyword=keyword,
            result=result,
            raw_max_score=self._raw_max,
            decision_max_score=self._decision_max,
            threshold=self._threshold,
            energy_result="通过" if self._energy_passed else "未通过",
            vad_result="通过" if self._vad_passed else "未通过",
            speech_result="3/3" if self._speech_passed else "未达到",
            l1_result="通过" if self._l1_passed else "未通过" if self._l1_executed else "未执行",
            l2_result="通过" if self._l2_passed else "未通过" if self._l2_executed else "未执行",
            l3_result="冷却阻止" if self._cooldown_blocked else "通过" if self._l3_executed else "未执行",
            l4_result="单关键词无需竞争",
            l5_result="通过" if self._l5_passed or result == "WAKE" else "未通过" if self._l5_executed else "未执行",
            rejection_reason="" if result == "WAKE" else self._last_rejection_reason,
            duration_seconds=max(0.0, now - self._started_at),
            inference_windows=self._inference_windows,
            peak_window=self._peak_window,
        )
        self.latest = snapshot
        self.history.append(snapshot)
        self._active = False
        self._suppressed_until_inactive = suppress
        self._reset_episode_fields()
        return snapshot


class WakeWordStudioApp:
    """只编排展示层，不接管外部训练进程。"""

    def __init__(self, root, config_path: Path):  # noqa: ANN001
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.model_registry = ModelRegistry.from_config(PROJECT_ROOT, self.config)
        self.provider_registry = ProviderRegistry.from_config(PROJECT_ROOT, self.config)
        root.title("WakeWord Studio — 中文离线唤醒词工作台")
        root.geometry("1200x780")
        root.minsize(1040, 700)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.playback = WakePlaybackQueue(
            lambda line: self.log_queue.put(("playback", line))
        )
        self.capture: MicrophoneCapture | None = None
        self.engine: StreamingWakeWordEngine | None = None
        self.generation_process, self.training_process = ManagedProcess(), ManagedProcess()
        self.episode_tracker = DetectionEpisodeTracker(max_history=10, timeout_seconds=5.0)
        self._closing = False
        self._configure_style()
        self._build()
        self._refresh_microphones()
        self._apply_model_defaults()
        self._refresh_training_status()
        root.after(30, self._poll)
        root.after(2000, self._refresh_training_status_periodic)
        root.after(1000, self._refresh_snapshot_age)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_style(self) -> None:
        from tkinter import font
        apply_modern_theme(self.root, self.ttk, font)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = self.ttk.Frame(self.root, padding=(14, 9, 14, 5))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        self.ttk.Label(header, text="WakeWord Studio — 中文离线唤醒词工作台", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(header, text="① 生成数据  →  ② 训练模型  →  ③ 实时唤醒  →  ④ 导出部署", style="Flow.TLabel").grid(row=0, column=1, sticky="e", padx=(20, 0))
        self.notebook = self.ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 10))
        self.generation_tab = self.ttk.Frame(self.notebook, padding=8)
        self.training_tab = self.ttk.Frame(self.notebook, padding=8)
        self.live_tab = self.ttk.Frame(self.notebook, padding=8)
        self.export_tab = self.ttk.Frame(self.notebook, padding=8)
        for tab, title in ((self.generation_tab, "数据集生成"), (self.training_tab, "模型训练"), (self.live_tab, "实时唤醒"), (self.export_tab, "模型与部署")):
            self.notebook.add(tab, text=title)
        self._build_generation()
        self._build_training()
        self._build_live()
        self._build_export()

    def _help(self, parent, text: str) -> None:  # noqa: ANN001
        self.ttk.Label(parent, text=text, style="Help.TLabel", wraplength=1120, justify="left").grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 7))

    def _row(self, parent, row: int, label: str, variable, *, browse=None, values=None, button_text="选择", width=58):  # noqa: ANN001
        try:
            label_style = "Card.TLabel" if str(parent.cget("style")) == "Card.TFrame" else "TLabel"
        except Exception:
            label_style = "TLabel"
        self.ttk.Label(parent, text=label, style=label_style).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        widget = (self.ttk.Entry(parent, textvariable=variable, width=width) if values is None else
                  self.ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width - 3))
        widget.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        if browse:
            self.ttk.Button(parent, text=button_text, command=browse, style="Secondary.TButton").grid(row=row, column=2, padx=6, pady=4)
        return widget

    def _build_generation(self) -> None:
        from tkinter import scrolledtext
        tab = self.generation_tab
        tab.columnconfigure(0, weight=1); tab.rowconfigure(6, weight=1)
        heading = self.ttk.Frame(tab)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.ttk.Label(heading, text="创建唤醒词数据集", style="PageTitle.TLabel").pack(anchor="w")
        self.ttk.Label(heading, text="按明确的样本总量和冻结 split 创建可复现数据；训练不会读取 Test。", style="Help.TLabel").pack(anchor="w")

        form = self.ttk.Frame(tab, style="Card.TFrame", padding=12)
        form.grid(row=1, column=0, sticky="ew"); form.columnconfigure(1, weight=1)
        self.ttk.Label(form, text="唤醒词与生成规模", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.gen_phrase = self.tk.StringVar(value=self.config["wake_phrase"])
        self.gen_count = self.tk.StringVar(value="10")
        self.gen_scale_mode = self.tk.StringVar(value="快速测试")
        available_providers = [item.name for item in self.provider_registry.available(generation_only=True)]
        self.gen_source = self.tk.StringVar(value=available_providers[0] if available_providers else "无可用数据源")
        self.gen_noise = self.tk.BooleanVar(value=True)
        self.gen_output = self.tk.StringVar(value=str(PROJECT_ROOT / "outputs/teacher_generated"))
        self.gen_input = self.tk.StringVar(value="")
        self._row(form, 1, "唤醒词：", self.gen_phrase)
        self.gen_scale_widget = self._row(form, 2, "生成规模：", self.gen_scale_mode, values=["快速测试", "小规模实验", "正式训练", "自定义"])
        self.gen_scale_widget.bind("<<ComboboxSelected>>", lambda _e: self._apply_generation_scale())
        self.gen_count_widget = self._row(form, 3, "预计样本总量：", self.gen_count)
        source = self._row(form, 4, "语音来源：", self.gen_source, values=available_providers or ["无可用数据源"])
        source.bind("<<ComboboxSelected>>", lambda _e: self._apply_provider_capability())
        self.gen_provider_capability_var = self.tk.StringVar(value="")
        self.ttk.Label(form, textvariable=self.gen_provider_capability_var, style="CardHelp.TLabel").grid(row=4, column=2, sticky="w", padx=6)
        self.gen_input_label = self.ttk.Label(form, text="本地素材目录：", style="Card.TLabel")
        self.gen_input_entry = self.ttk.Entry(form, textvariable=self.gen_input)
        self.gen_input_button = self.ttk.Button(form, text="选择目录", command=lambda: self._browse_dir(self.gen_input), style="Secondary.TButton")
        self.gen_input_label.grid(row=5, column=0, sticky="w", padx=6, pady=4)
        self.gen_input_entry.grid(row=5, column=1, sticky="ew", padx=6, pady=4)
        self.gen_input_button.grid(row=5, column=2, padx=6, pady=4)
        self._row(form, 6, "输出目录：", self.gen_output, browse=lambda: self._browse_dir(self.gen_output), button_text="选择目录")
        middle = self.ttk.Frame(tab); middle.grid(row=2, column=0, sticky="ew", pady=4)
        middle.columnconfigure(0, weight=1); middle.columnconfigure(1, weight=1)
        diversity = self.ttk.Frame(middle, style="Card.TFrame", padding=12); diversity.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.ttk.Label(diversity, text="数据多样性", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        for row, text in enumerate(("✓ 多说话人", "✓ 困难负样本", "✓ 背景噪声 / 混响 / 随机 SNR", "○ 年龄覆盖取决于真实导入 metadata"), start=1):
            self.ttk.Label(diversity, text=text, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=3)
        self.ttk.Checkbutton(diversity, text="启用统一增强", variable=self.gen_noise).grid(row=5, column=0, sticky="w", pady=(8, 0))

        estimate = self.ttk.Frame(middle, style="Card.TFrame", padding=12); estimate.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.ttk.Label(estimate, text="预计生成统计", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.gen_estimate_vars = {name: self.tk.StringVar(value="—") for name in ("positive", "hard_negative", "negative", "ambient", "splits", "total")}
        estimate_rows = (("正样本", "positive"), ("困难负样本", "hard_negative"), ("普通负样本", "negative"), ("环境负样本", "ambient"), ("Train / Validation / Test", "splits"), ("预计总量", "total"))
        for row, (label, key) in enumerate(estimate_rows, start=1):
            self.ttk.Label(estimate, text=f"{label}：", style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=2)
            self.ttk.Label(estimate, textvariable=self.gen_estimate_vars[key], style="CardValue.Card.TLabel").grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
        self._apply_generation_scale()
        self._apply_provider_capability()

        success = self.ttk.Frame(tab, style="Card.TFrame", padding=8)
        success.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.ttk.Label(success, text="✓ 数据集生成状态", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.gen_status_var = self.tk.StringVar(value="未开始")
        self.gen_progress_var = self.tk.StringVar(value="等待生成器提供统计")
        self.gen_positive_var = self.tk.StringVar(value="等待生成器提供统计")
        self.gen_negative_var = self.tk.StringVar(value="等待生成器提供统计")
        self.ttk.Label(success, textvariable=self.gen_status_var, style="CardValue.Card.TLabel").grid(row=0, column=1, sticky="w", padx=12)
        self.ttk.Label(success, textvariable=self.gen_progress_var, style="CardHelp.TLabel").grid(row=0, column=2, sticky="w", padx=12)
        self.gen_success_detail_var = self.tk.StringVar(value="完成后在此显示标签统计、总时长和输出目录。")
        self.ttk.Label(success, textvariable=self.gen_success_detail_var, style="CardHelp.TLabel", wraplength=1000).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))
        controls = self.ttk.Frame(tab); controls.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.ttk.Button(controls, text="开始生成", command=self.start_generation, style="Primary.TButton").pack(side="left")
        self.ttk.Button(controls, text="生成前检查", command=self.generation_dry_run, style="Secondary.TButton").pack(side="left", padx=8)
        self.ttk.Button(controls, text="打开目录", command=lambda: self._open_path(Path(self.gen_output.get())), style="Secondary.TButton").pack(side="left")
        self.ttk.Button(controls, text="停止", command=self.stop_generation).pack(side="right")
        self.gen_log_button = self.ttk.Button(tab, text="展开运行日志", command=self._toggle_generation_log, style="Secondary.TButton")
        self.gen_log_button.grid(row=5, column=0, sticky="w")
        self.generation_log = scrolledtext.ScrolledText(tab, height=1, state="disabled", wrap="word", relief="flat")
        self.generation_log.grid(row=6, column=0, sticky="nsew", pady=(4, 0))
        self.generation_log.grid_remove()

    def _toggle_generation_log(self) -> None:
        if self.generation_log.winfo_ismapped():
            self.generation_log.grid_remove()
            self.gen_log_button.configure(text="展开运行日志")
        else:
            self.generation_log.grid()
            self.gen_log_button.configure(text="收起运行日志")

    def _build_training(self) -> None:
        from tkinter import scrolledtext
        tab = self.training_tab
        tab.columnconfigure(0, weight=1); tab.rowconfigure(5, weight=1)
        heading = self.ttk.Frame(tab); heading.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.ttk.Label(heading, text="训练唤醒词模型", style="PageTitle.TLabel").pack(anchor="w")
        self.ttk.Label(heading, text="选择数据集、模型与输出位置；训练只用 Train / Validation，Test 保持冻结。", style="Help.TLabel").pack(anchor="w", pady=(3, 0))
        form = self.ttk.Frame(tab, style="Card.TFrame", padding=16); form.grid(row=1, column=0, sticky="ew"); form.columnconfigure(1, weight=1)
        self.ttk.Label(form, text="训练设置", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        default_training_model = self.model_registry.by_id("model_b")
        self.train_dataset = self.tk.StringVar(value=str(default_training_model.trainer["dataset"]))
        self.train_model = self.tk.StringVar(value=MODEL_B)
        self.train_phrase = self.tk.StringVar(value=self.config["wake_phrase"])
        self.train_output = self.tk.StringVar(value="runs/teacher_ui/manual_run")
        self._row(form, 1, "数据集：", self.train_dataset, browse=lambda: self._browse_dir(self.train_dataset), button_text="选择文件夹")
        self.ttk.Label(form, text="选择模型：", style="Card.TLabel").grid(row=2, column=0, sticky="nw", padx=6, pady=6)
        model_cards = self.ttk.Frame(form, style="Card.TFrame"); model_cards.grid(row=2, column=1, columnspan=2, sticky="ew", padx=6)
        for column, model in enumerate(self.model_registry.all()):
            model_cards.columnconfigure(column, weight=1)
            card = self.ttk.Frame(model_cards, style="Card.TFrame", padding=8)
            card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))
            self.ttk.Radiobutton(card, text=model.display_name, variable=self.train_model, value=model.display_name, command=self._apply_training_default).pack(anchor="w")
            self.ttk.Label(card, text=f"约 {model.model_size_kib:.0f} KiB · {model.description}", style="CardHelp.TLabel").pack(anchor="w", padx=22)
        self._row(form, 3, "输出目录：", self.train_output, browse=lambda: self._browse_dir(self.train_output), button_text="选择目录")
        status = self.ttk.Frame(tab, style="Card.TFrame", padding=16); status.grid(row=2, column=0, sticky="ew", pady=12)
        self.ttk.Label(status, text="训练状态", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))
        for col in range(6): status.columnconfigure(col, weight=1 if col % 2 else 0)
        self.train_status_var = self.tk.StringVar(value="未开始")
        self.train_step_var = self.tk.StringVar(value="0 / 0")
        self.train_phase_var = self.tk.StringVar(value="尚无阶段信息")
        self.train_loss_var = self.tk.StringVar(value="尚无数据")
        self.train_lr_var = self.tk.StringVar(value="尚无数据")
        self.train_elapsed_var = self.tk.StringVar(value="尚无数据")
        self.train_eta_var = self.tk.StringVar(value="尚无数据")
        self.train_recall_var = self.tk.StringVar(value="尚无评估结果")
        self.train_fpr_var = self.tk.StringVar(value="尚无评估结果")
        self.train_worst_recall_var = self.tk.StringVar(value="尚无评估结果")
        self.train_checkpoint_var = self.tk.StringVar(value="尚无 checkpoint")
        self.train_best_threshold_var = self.tk.StringVar(value="尚无评估结果")
        self.train_status_source_var = self.tk.StringVar(value="等待发现 TRAINING_STATUS.json")
        fields = (("训练状态：", self.train_status_var), ("进度：", self.train_step_var), ("当前阶段：", self.train_phase_var),
                  ("当前 Loss：", self.train_loss_var), ("当前学习率：", self.train_lr_var), ("已训练时间：", self.train_elapsed_var),
                  ("Validation Recall：", self.train_recall_var), ("Validation FPR：", self.train_fpr_var), ("最差数据源 Recall：", self.train_worst_recall_var),
                  ("当前 checkpoint：", self.train_checkpoint_var), ("当前最佳阈值：", self.train_best_threshold_var), ("预计剩余：", self.train_eta_var))
        visible_fields = (("训练状态：", self.train_status_var), ("进度：", self.train_step_var), ("当前 Loss：", self.train_loss_var),
                          ("Validation Recall：", self.train_recall_var), ("Validation FPR：", self.train_fpr_var), ("预计剩余：", self.train_eta_var))
        for i, (label, variable) in enumerate(visible_fields):
            row, pair = divmod(i, 3); col = pair * 2
            self.ttk.Label(status, text=label).grid(row=row, column=col, sticky="w", padx=(5, 2), pady=3)
            self.ttk.Label(status, textvariable=variable, style="CardValue.TLabel").grid(row=row, column=col + 1, sticky="w", padx=(2, 14), pady=3)
        self.train_progress = self.ttk.Progressbar(status, maximum=100); self.train_progress.grid(row=3, column=0, columnspan=6, sticky="ew", padx=5, pady=(10, 3))
        controls = self.ttk.Frame(tab); controls.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        self.ttk.Button(controls, text="训练配置检查", command=self.training_dry_run).pack(side="left", padx=(0, 6))
        self.ttk.Button(controls, text="开始训练", command=self.start_training, style="Primary.TButton").pack(side="left", padx=6)
        self.ttk.Button(controls, text="停止本界面启动的训练", command=self.stop_training).pack(side="left", padx=6)
        self.ttk.Button(controls, text="刷新状态", command=self._refresh_training_status).pack(side="left", padx=6)
        self.ttk.Button(controls, text="清空日志", command=lambda: self._clear(self.training_log)).pack(side="left", padx=6)
        self.ttk.Label(tab, text="此页面不会接管或停止在 PowerShell 中启动的外部训练。", style="Help.TLabel").grid(row=4, column=0, sticky="w", padx=3, pady=(0, 4))
        self.training_log = scrolledtext.ScrolledText(tab, height=5, state="disabled", wrap="word"); self.training_log.grid(row=5, column=0, sticky="nsew")

    def _build_live(self) -> None:
        from tkinter import scrolledtext
        tab = self.live_tab
        tab.columnconfigure(0, weight=1); tab.rowconfigure(5, weight=1)
        heading = self.ttk.Frame(tab); heading.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.ttk.Label(heading, text="实时唤醒", style="PageTitle.TLabel").pack(side="left")
        self.ttk.Label(heading, text="  麦克风 → 三级前处理 → 唤醒模型 → L1–L5 → IGNORE / REJECT / WAKE", style="Help.TLabel").pack(side="left", padx=12)
        top = self.ttk.Frame(tab); top.grid(row=1, column=0, sticky="ew"); top.columnconfigure(0, weight=3); top.columnconfigure(1, weight=2)
        settings = self.ttk.LabelFrame(top, text="唤醒设置", padding=6); settings.grid(row=0, column=0, sticky="nsew", padx=(0, 5)); settings.columnconfigure(1, weight=1); settings.columnconfigure(3, weight=1)
        self.microphone_var = self.tk.StringVar()
        self.live_phrase_var = self.tk.StringVar(value=self.config["wake_phrase"])
        self.model_var = self.tk.StringVar(value=MODEL_B)
        self.model_path_var = self.tk.StringVar()
        self.threshold_var = self.tk.StringVar()
        self.hop_var = self.tk.StringVar(value="0.20")
        self.backend_var = self.tk.StringVar(value="-")
        self.state_var = self.tk.StringVar(value=STATE_TEXT["STOPPED"])
        self.score_var = self.tk.StringVar(value="—")
        self.raw_score_var = self.tk.StringVar(value="—")
        self.smoothing_var = self.tk.StringVar(value="raw")
        self.inference_state_var = self.tk.StringVar(value="未激活")
        self.rolling_help_var = self.tk.StringVar(value="每 0.20 秒使用最近 2 秒语音进行一次唤醒判断。")
        self.score_progress_var = self.tk.DoubleVar(value=0.0)
        self.ttk.Label(settings, text="麦克风：").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.microphone_combo = self.ttk.Combobox(settings, textvariable=self.microphone_var, values=[], state="readonly")
        self.microphone_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=4)
        self.ttk.Button(settings, text="刷新设备", command=self._refresh_microphones).grid(row=0, column=4, padx=6)
        self.ttk.Label(settings, text="唤醒词：").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ttk.Entry(settings, textvariable=self.live_phrase_var).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        self.ttk.Label(settings, text="模型：").grid(row=1, column=2, sticky="w", padx=6, pady=4)
        combo = self.ttk.Combobox(settings, textvariable=self.model_var, values=self.model_registry.display_names, state="readonly")
        combo.grid(row=1, column=3, columnspan=2, sticky="ew", padx=6, pady=4); combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_model_defaults())
        self.ttk.Label(settings, text="模型文件：").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ttk.Entry(settings, textvariable=self.model_path_var).grid(row=2, column=1, columnspan=3, sticky="ew", padx=6, pady=4)
        model_actions = self.ttk.Frame(settings); model_actions.grid(row=2, column=4, padx=6)
        self.ttk.Button(model_actions, text="选择模型", command=self._browse_model).pack(side="left")
        self.ttk.Button(model_actions, text="恢复注册模型", command=self._load_latest_model_config).pack(side="left", padx=(5, 0))
        self.ttk.Label(settings, text="识别阈值：").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.ttk.Entry(settings, textvariable=self.threshold_var).grid(row=3, column=1, sticky="ew", padx=6, pady=4)
        self.ttk.Label(settings, text="推理滑窗步长：").grid(row=3, column=2, sticky="w", padx=6, pady=4)
        self.ttk.Entry(settings, textvariable=self.hop_var).grid(row=3, column=3, sticky="ew", padx=6, pady=4)
        self.ttk.Label(settings, text="秒", style="Help.TLabel").grid(row=3, column=4, sticky="w")
        self.ttk.Label(settings, textvariable=self.rolling_help_var, style="Help.TLabel", wraplength=650).grid(row=4, column=0, columnspan=5, sticky="w", padx=6)
        status = self.ttk.LabelFrame(top, text="实时状态", padding=10); status.grid(row=0, column=1, sticky="nsew", padx=(5, 0)); status.columnconfigure(1, weight=1)
        self.state_label = self.ttk.Label(status, textvariable=self.state_var, style="Stopped.Status.TLabel"); self.state_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        score_line = self.ttk.Frame(status); score_line.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.ttk.Label(score_line, text="原始：").pack(side="left"); self.ttk.Label(score_line, textvariable=self.raw_score_var).pack(side="left")
        self.ttk.Label(score_line, text="判定：").pack(side="left", padx=(16, 0)); self.ttk.Label(score_line, textvariable=self.score_var, style="CardValue.TLabel").pack(side="left")
        self.ttk.Progressbar(status, variable=self.score_progress_var, maximum=1.0).grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        score_meta = self.ttk.Frame(status); score_meta.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.ttk.Label(score_meta, text="当前识别阈值：").pack(side="left"); self.ttk.Label(score_meta, textvariable=self.threshold_var).pack(side="left")
        self.ttk.Label(score_meta, text="当前 smoothing：").pack(side="left", padx=(18, 0)); self.ttk.Label(score_meta, textvariable=self.smoothing_var).pack(side="left")
        self.ttk.Label(status, text="模型推理：").grid(row=4, column=0, sticky="nw"); self.ttk.Label(status, textvariable=self.inference_state_var, wraplength=300, justify="right").grid(row=4, column=1, sticky="e")
        logic = self.ttk.Frame(tab); logic.grid(row=2, column=0, sticky="ew", pady=7)
        logic.columnconfigure(0, weight=2); logic.columnconfigure(1, weight=2); logic.columnconfigure(2, weight=3)
        pre = self.ttk.LabelFrame(logic, text="三级前置语音检测", padding=6); pre.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.energy_state_var = self.tk.StringVar(value="等待检测"); self.vad_state_var = self.tk.StringVar(value="等待检测"); self.speech_frames_var = self.tk.StringVar(value="等待检测")
        pre_rows = (("第一级", "能量 + 自适应阈值", self.energy_state_var), ("第二级", "WebRTC VAD", self.vad_state_var), ("第三级", "连续语音帧确认", self.speech_frames_var))
        for row, (level, name, var) in enumerate(pre_rows):
            self.ttk.Label(pre, text=level).grid(row=row, column=0, sticky="w", padx=4, pady=4); self.ttk.Label(pre, text=name).grid(row=row, column=1, sticky="w", padx=8)
            self.ttk.Label(pre, textvariable=var, style="CardValue.TLabel").grid(row=row, column=2, sticky="w", padx=8)
        post = self.ttk.LabelFrame(logic, text="五层防误触发 DetectionLogic", padding=6); post.grid(row=0, column=1, sticky="nsew", padx=(5, 0)); post.columnconfigure(1, weight=1)
        self.detection_vars = [self.tk.StringVar(value="等待") for _ in range(5)]
        post_rows = (("L1 连续确认", self.detection_vars[0]), ("L2 峰值/背景比", self.detection_vars[1]), ("L3 冷却检查", self.detection_vars[2]), ("L4 多关键词", self.detection_vars[3]), ("L5 前后静音", self.detection_vars[4]))
        for index, (name, var) in enumerate(post_rows):
            row, pair = divmod(index, 2); column = pair * 2
            self.ttk.Label(post, text=f"{name}：").grid(row=row, column=column, sticky="w", padx=(4, 1), pady=3)
            self.ttk.Label(post, textvariable=var, style="CardValue.TLabel").grid(row=row, column=column + 1, sticky="w", padx=(1, 8), pady=3)
        self.final_decision_var = self.tk.StringVar(value="未触发"); self.reject_reason_var = self.tk.StringVar(value="等待检测")
        final_line = self.ttk.Frame(post); final_line.grid(row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 1))
        self.ttk.Label(final_line, text="实时判定：").pack(side="left"); self.ttk.Label(final_line, textvariable=self.final_decision_var, style="CardValue.TLabel").pack(side="left", padx=(3, 12))
        self.ttk.Label(final_line, textvariable=self.reject_reason_var, wraplength=240).pack(side="left", padx=3)

        snapshot = self.ttk.LabelFrame(logic, text="最近一次检测结果（唤醒候选）", padding=6)
        snapshot.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        for column in range(8): snapshot.columnconfigure(column, weight=1 if column % 2 else 0)
        self.snapshot_time_var = self.tk.StringVar(value="尚无检测结果")
        self.snapshot_keyword_var = self.tk.StringVar(value="—")
        self.snapshot_result_var = self.tk.StringVar(value="—")
        self.snapshot_raw_var = self.tk.StringVar(value="—")
        self.snapshot_decision_var = self.tk.StringVar(value="—")
        self.snapshot_threshold_var = self.tk.StringVar(value="—")
        self.snapshot_preprocess_var = self.tk.StringVar(value="一级能量 — ｜二级 VAD — ｜三级连续帧 —")
        self.snapshot_detection_var = self.tk.StringVar(value="L1 — ｜L2 — ｜L3 — ｜L4 — ｜L5 —")
        self.snapshot_reason_var = self.tk.StringVar(value="—")
        self.snapshot_metrics_var = self.tk.StringVar(value="持续时间 —  |  推理窗口 —  |  最高分窗口 —")
        first_row = (("检测时间：", self.snapshot_time_var), ("最终结果：", self.snapshot_result_var))
        for pair, (label, variable) in enumerate(first_row):
            column = pair * 2
            self.ttk.Label(snapshot, text=label).grid(row=0, column=column, sticky="w", padx=(4, 2), pady=2)
            self.ttk.Label(snapshot, textvariable=variable, style="CardValue.TLabel").grid(row=0, column=column + 1, sticky="w", padx=(2, 12), pady=2)
        self.ttk.Label(snapshot, text="唤醒词：").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.ttk.Label(snapshot, textvariable=self.snapshot_keyword_var, style="CardValue.TLabel").grid(row=1, column=1, columnspan=5, sticky="w", padx=2, pady=2)
        score_row = (("原始最高分：", self.snapshot_raw_var), ("判定最高分：", self.snapshot_decision_var), ("识别阈值：", self.snapshot_threshold_var))
        for pair, (label, variable) in enumerate(score_row):
            column = pair * 2
            self.ttk.Label(snapshot, text=label).grid(row=2, column=column, sticky="w", padx=(4, 2), pady=2)
            self.ttk.Label(snapshot, textvariable=variable).grid(row=2, column=column + 1, sticky="w", padx=(2, 8), pady=2)
        self.ttk.Label(snapshot, text="三级：").grid(row=3, column=0, sticky="nw", padx=4, pady=2)
        self.ttk.Label(snapshot, textvariable=self.snapshot_preprocess_var, wraplength=450, justify="left").grid(row=3, column=1, columnspan=7, sticky="w", padx=2, pady=2)
        self.ttk.Label(snapshot, text="五层：").grid(row=4, column=0, sticky="nw", padx=4, pady=2)
        self.ttk.Label(snapshot, textvariable=self.snapshot_detection_var, wraplength=450, justify="left").grid(row=4, column=1, columnspan=7, sticky="w", padx=2, pady=2)
        self.ttk.Label(snapshot, text="原因：").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        self.ttk.Label(snapshot, textvariable=self.snapshot_reason_var, wraplength=420).grid(row=5, column=1, columnspan=7, sticky="w", padx=2, pady=2)
        self.ttk.Label(snapshot, textvariable=self.snapshot_metrics_var, style="Help.TLabel", wraplength=500).grid(row=6, column=0, columnspan=8, sticky="w", padx=4, pady=2)

        history = self.ttk.LabelFrame(tab, text="检测历史（仅记录 WAKE / REJECT，最多 10 条）", padding=4)
        history.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        history.columnconfigure(0, weight=1)
        columns = ("time", "keyword", "score", "threshold", "result", "reason")
        self.history_tree = self.ttk.Treeview(history, columns=columns, show="headings", height=2)
        headings = (("time", "时间", 80), ("keyword", "唤醒词", 150), ("score", "最高分", 80), ("threshold", "阈值", 80), ("result", "结果", 70), ("reason", "拒绝原因", 300))
        for name, title, width in headings:
            self.history_tree.heading(name, text=title)
            self.history_tree.column(name, width=width, minwidth=60, anchor="center" if name != "reason" else "w")
        self.history_tree.grid(row=0, column=0, sticky="ew")
        scrollbar = self.ttk.Scrollbar(history, orient="vertical", command=self.history_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns"); self.history_tree.configure(yscrollcommand=scrollbar.set)

        controls = self.ttk.Frame(tab); controls.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        self.ttk.Button(controls, text="▶ 开始监听", command=self.start_listening, style="Primary.TButton").pack(side="left", padx=(0, 6)); self.ttk.Button(controls, text="■ 停止监听", command=self.stop_listening).pack(side="left", padx=6)
        self.ttk.Button(controls, text="清除最近结果", command=self._clear_latest_result).pack(side="left", padx=6)
        self.ttk.Button(controls, text="清空检测历史", command=self._clear_detection_history).pack(side="left", padx=6)
        self.ttk.Button(controls, text="清空运行日志", command=lambda: self._clear(self.live_log)).pack(side="left", padx=6)
        self.ttk.Label(controls, text="运行日志", style="CardValue.TLabel").pack(side="left", padx=(18, 0))
        self.live_log = scrolledtext.ScrolledText(tab, height=1, state="disabled", wrap="word"); self.live_log.grid(row=5, column=0, sticky="nsew")

    def _build_export(self) -> None:
        tab = self.export_tab
        tab.columnconfigure(0, weight=1); tab.rowconfigure(3, weight=1)
        heading = self.ttk.Frame(tab); heading.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.ttk.Label(heading, text="模型与部署", style="PageTitle.TLabel").pack(anchor="w")
        self.ttk.Label(heading, text="选择已注册的 INT8 模型并生成部署文件；模型卡默认只展示必要信息。", style="Help.TLabel").pack(anchor="w", pady=(3, 0))
        form = self.ttk.Frame(tab, style="Card.TFrame", padding=16); form.grid(row=1, column=0, sticky="ew"); form.columnconfigure(1, weight=1)
        self._row(form, 0, "模型文件：", self.model_path_var, browse=self._browse_model, button_text="选择模型")
        self.ttk.Button(form, text="查看模型信息", command=self.inspect_model, style="Primary.TButton").grid(row=1, column=0, padx=6, pady=7, sticky="w")
        cards = self.ttk.Frame(form); cards.grid(row=1, column=1, columnspan=5, sticky="ew", padx=6, pady=4)
        for column, model in enumerate(self.model_registry.all()):
            card = self.ttk.Frame(cards, style="Card.TFrame", padding=8)
            card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0))
            self.ttk.Label(card, text=model.display_name, style="CardTitle.TLabel").pack(anchor="w")
            self.ttk.Label(card, text=f"INT8 · {model.model_size_kib:.1f} KiB · {model.description} · 可生成部署文件", style="CardHelp.TLabel").pack(anchor="w", pady=(3, 0))
        self.model_info_vars = {key: self.tk.StringVar(value="等待检查") for key in ("model", "format", "quant", "size", "input", "output", "sha", "validation")}
        for i, (label, key) in enumerate((("模型：", "model"), ("模型格式：", "format"), ("量化方式：", "quant"), ("模型大小：", "size"), ("输入：", "input"), ("输出：", "output"))):
            row, pair = divmod(i, 3); col = pair * 2
            self.ttk.Label(form, text=label).grid(row=row + 2, column=col, sticky="w", padx=(6, 2), pady=4); self.ttk.Label(form, textvariable=self.model_info_vars[key], style="CardValue.TLabel").grid(row=row + 2, column=col + 1, sticky="w", padx=(2, 18))
        self.ttk.Label(form, text="模型 SHA256：").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.ttk.Entry(form, textvariable=self.model_info_vars["sha"], state="readonly").grid(row=4, column=1, columnspan=4, sticky="ew", padx=6)
        self.ttk.Button(form, text="复制 SHA256", command=self._copy_sha).grid(row=4, column=5, padx=6)
        self.ttk.Label(form, text="Validation Recall：").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        self.ttk.Label(form, textvariable=self.model_info_vars["validation"], style="CardValue.TLabel").grid(row=5, column=1, columnspan=5, sticky="w", padx=6, pady=4)
        self.export_detail_widgets = [widget for row in range(2, 6) for widget in form.grid_slaves(row=row)]
        for widget in self.export_detail_widgets:
            widget.grid_remove()
        self.export_details_visible = False
        self.ttk.Button(form, text="详细信息", command=self._toggle_export_details, style="Secondary.TButton").grid(row=1, column=5, padx=6)
        deploy = self.ttk.LabelFrame(tab, text="ESP32-S3 部署", padding=10); deploy.grid(row=2, column=0, sticky="ew", pady=9); deploy.columnconfigure(1, weight=1)
        self.deploy_int8_var = self.tk.StringVar(value="○ 等待模型检查"); self.deploy_size_var = self.tk.StringVar(value="○ 等待模型检查")
        self.deploy_hardware_var = self.tk.StringVar(value="● 尚未进行真实 ESP32-S3 板端测试")
        self.export_path_var = self.tk.StringVar(value="尚未生成"); self.export_contents_var = self.tk.StringVar(value="生成后显示实际文件清单")
        rows = (("目标设备：", "ESP32-S3"), ("部署状态：", self.deploy_int8_var), ("", self.deploy_size_var), ("", self.deploy_hardware_var), ("部署包路径：", self.export_path_var), ("包含文件：", self.export_contents_var))
        for row, (label, value) in enumerate(rows):
            self.ttk.Label(deploy, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=4)
            (self.ttk.Label(deploy, text=value, style="CardValue.TLabel") if isinstance(value, str) else self.ttk.Label(deploy, textvariable=value, wraplength=900)).grid(row=row, column=1, sticky="w", padx=5, pady=4)
        self.ttk.Button(deploy, text="生成 ESP32-S3 部署包", command=self.prepare_export, style="Primary.TButton").grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=(9, 3))
        deploy_actions = self.ttk.Frame(deploy); deploy_actions.grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=3)
        self.ttk.Button(deploy_actions, text="打开模型目录", command=lambda: self._open_path(resolve_project_path(PROJECT_ROOT, self.model_path_var.get()).parent)).pack(side="left")
        self.ttk.Button(deploy_actions, text="打开 ESP32-S3 工程目录", command=lambda: self._open_path(PROJECT_ROOT / "firmware/repcnn_esp32s3")).pack(side="left", padx=6)
        self.ttk.Label(tab, text="部署包用于固件集成；生成文件不等于已经过真实板端验证。", style="Help.TLabel").grid(row=3, column=0, sticky="nw", padx=4)

    def _toggle_export_details(self) -> None:
        self.export_details_visible = not self.export_details_visible
        for widget in self.export_detail_widgets:
            if self.export_details_visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _refresh_microphones(self) -> None:
        try:
            values = [item.display_name for item in list_input_devices()]
        except Exception as exc:
            values = [f"设备不可用 — {exc}"]
        self.microphone_combo.configure(values=values)
        if values and self.microphone_var.get() not in values: self.microphone_var.set(values[0])

    def _current_model(self):  # noqa: ANN201
        return self.model_registry.by_display_name(self.model_var.get())

    def _apply_generation_scale(self) -> None:
        mode = self.gen_scale_mode.get()
        custom = None
        if mode == "自定义":
            try:
                custom = int(self.gen_count.get())
            except ValueError:
                custom = 1000
                self.gen_count.set(str(custom))
        plan = build_product_plan(mode, custom_total=custom)
        if mode != "自定义":
            self.gen_count.set(str(plan.total))
        for key, value in plan.targets.items():
            self.gen_estimate_vars[key].set(f"{value:,} 条")
        split = plan.split_targets
        self.gen_estimate_vars["splits"].set(
            f"{split['train']:,} / {split['validation']:,} / {split['test']:,}"
        )
        self.gen_estimate_vars["total"].set(f"{plan.total:,} 条")

    def _apply_provider_capability(self) -> None:
        try:
            provider = self.provider_registry.by_name(self.gen_source.get())
        except KeyError:
            return
        capabilities = provider.capabilities
        if capabilities.age_metadata:
            text = "支持按 metadata 或目录读取真实年龄分组"
        elif capabilities.multi_speaker:
            text = "多说话人；无可靠年龄标签"
        else:
            text = "单一来源；无可靠年龄标签"
        self.gen_provider_capability_var.set(text)
        widgets = (self.gen_input_label, self.gen_input_entry, self.gen_input_button)
        if capabilities.local_audio_import:
            for widget in widgets:
                widget.grid()
            self.gen_scale_widget.configure(state="disabled")
            self.gen_count_widget.configure(state="disabled")
            for variable in self.gen_estimate_vars.values():
                variable.set("导入扫描后统计")
        else:
            for widget in widgets:
                widget.grid_remove()
            self.gen_scale_widget.configure(state="readonly")
            self.gen_count_widget.configure(state="normal")
            self._apply_generation_scale()

    def _apply_model_defaults(self) -> None:
        model = self._current_model()
        deployment = model.deployment
        try:
            self.model_path_var.set(model.model_path.relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            self.model_path_var.set(str(model.model_path))
        self.threshold_var.set(str(model.threshold))
        self.backend_var.set(model.display_name)
        if model.runtime_mode == "rolling_window":
            self.hop_var.set(f"{float(model.hop_seconds or 0.20):.2f}")
            self.smoothing_var.set(model.smoothing)
            self.rolling_help_var.set(
                f"{float(model.window_seconds or 2.0):.1f} 秒滚动窗口，每 "
                f"{float(model.hop_seconds or 0.20):.2f} 秒判断一次；smoothing 为 {model.smoothing}。"
            )
        else:
            self.hop_var.set("原生流式推理")
            self.smoothing_var.set("原生流式")
            self.rolling_help_var.set("Model A 使用原生流式推理，不使用固定 rolling window。")
        recall = deployment.get("validation_recall")
        target = float(deployment.get("validation_target_recall", 0.98))
        target_met = bool(deployment.get("validation_target_met", False))
        validation_text = "未配置展示指标" if recall is None else (
            f"{float(recall) * 100:.1f}%（{'已达到' if target_met else '未达到'} {target * 100:.0f}%）"
        )
        preset = {
            "model": model.display_name,
            "format": "TensorFlow Lite",
            "quant": str(deployment.get("format", "Full INT8")),
            "size": f"{model.model_size_kib:.2f} KiB",
            "input": f"{deployment.get('input_shape', [])} {str(deployment.get('input_dtype', '')).upper()}",
            "output": f"{deployment.get('output_shape', [])} {str(deployment.get('output_dtype', '')).upper()}",
            "sha": str(deployment.get("sha256", "等待检查")),
            "validation": validation_text,
        }
        for key, value in preset.items():
            self.model_info_vars[key].set(value)
        self.deploy_int8_var.set("✓ Full INT8 TFLite，可导出")
        self.deploy_size_var.set(
            f"✓ {model.model_size_kib:.2f} KiB；目标设备：{' / '.join(model.supported_platforms)}"
        )
        self.inference_state_var.set("未激活")

    def _load_latest_model_config(self) -> None:
        """Reload only the registered finalized model; never discover training checkpoints."""

        from tkinter import messagebox
        try:
            self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            self.model_registry = ModelRegistry.from_config(PROJECT_ROOT, self.config)
            self.provider_registry = ProviderRegistry.from_config(PROJECT_ROOT, self.config)
            self._apply_model_defaults()
            self._append(self.live_log, f"已恢复注册模型 | {self._current_model().display_name}")
        except Exception as exc:
            messagebox.showerror("加载最新模型失败", str(exc))

    def _apply_training_default(self) -> None:
        model = self.model_registry.by_display_name(self.train_model.get())
        self.train_dataset.set(str(resolve_project_path(PROJECT_ROOT, model.trainer["dataset"])))

    def _browse_model(self) -> None:
        from tkinter import filedialog
        value = filedialog.askopenfilename(title="选择 TFLite 模型", filetypes=[("TFLite 模型", "*.tflite"), ("所有文件", "*.*")])
        if value: self.model_path_var.set(value)

    def _browse_dir(self, variable) -> None:  # noqa: ANN001
        from tkinter import filedialog
        value = filedialog.askdirectory(title="选择目录")
        if value: variable.set(value)

    def _open_path(self, path: Path) -> None:
        from tkinter import messagebox

        resolved = path.resolve()
        if not resolved.exists():
            messagebox.showerror("无法打开目录", f"目录不存在：\n{resolved}")
            return
        try:
            os.startfile(resolved)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("无法打开目录", f"系统无法打开该目录。\n详细原因：{exc}")

    def _append(self, widget, line: str) -> None:  # noqa: ANN001
        widget.configure(state="normal"); widget.insert("end", line + "\n"); widget.see("end"); widget.configure(state="disabled")

    def _clear(self, widget) -> None:  # noqa: ANN001
        widget.configure(state="normal"); widget.delete("1.0", "end"); widget.configure(state="disabled")

    @staticmethod
    def _snapshot_time_text(snapshot: DetectionSnapshot) -> str:
        age_seconds = max(0, int((datetime.now() - snapshot.detected_at).total_seconds()))
        if age_seconds < 60:
            age = "刚刚"
        elif age_seconds < 3600:
            age = f"{age_seconds // 60} 分钟前"
        else:
            age = f"{age_seconds // 3600} 小时前"
        return f"{snapshot.detected_at:%H:%M:%S}（{age}）"

    def _latch_snapshot(self, snapshot: DetectionSnapshot) -> None:
        result_text = "唤醒成功" if snapshot.result == "WAKE" else "未唤醒（REJECT）"
        reason_text = "无" if snapshot.result == "WAKE" else self._translate_rejection_reason(snapshot.rejection_reason)
        self.snapshot_time_var.set(self._snapshot_time_text(snapshot))
        self.snapshot_keyword_var.set(snapshot.keyword)
        self.snapshot_result_var.set(result_text)
        self.snapshot_raw_var.set(f"{snapshot.raw_max_score:.4f}")
        self.snapshot_decision_var.set(f"{snapshot.decision_max_score:.4f}")
        self.snapshot_threshold_var.set(f"{snapshot.threshold:.4f}")
        self.snapshot_preprocess_var.set(
            f"一级能量 {snapshot.energy_result} ｜二级 VAD {snapshot.vad_result} ｜三级连续帧 {snapshot.speech_result}"
        )
        self.snapshot_detection_var.set(
            f"L1 {snapshot.l1_result} ｜L2 {snapshot.l2_result} ｜L3 {snapshot.l3_result} ｜"
            f"L4 {snapshot.l4_result} ｜L5 {snapshot.l5_result}"
        )
        self.snapshot_reason_var.set(reason_text)
        peak = f"第 {snapshot.peak_window} 个" if snapshot.peak_window is not None else "尚无"
        self.snapshot_metrics_var.set(
            f"持续时间 {snapshot.duration_seconds:.2f} 秒  |  "
            f"推理窗口 {snapshot.inference_windows} 个  |  最高分窗口 {peak}"
        )
        self.history_tree.insert(
            "",
            "end",
            values=(
                f"{snapshot.detected_at:%H:%M:%S}",
                snapshot.keyword,
                f"{snapshot.decision_max_score:.4f}",
                f"{snapshot.threshold:.4f}",
                "唤醒" if snapshot.result == "WAKE" else "拒绝",
                "—" if snapshot.result == "WAKE" else reason_text,
            ),
        )
        children = self.history_tree.get_children()
        while len(children) > self.episode_tracker.max_history:
            self.history_tree.delete(children[0])
            children = self.history_tree.get_children()
        self.history_tree.yview_moveto(1.0)
        self._append(
            self.live_log,
            f"检测结束 | {result_text} | 最高判定分数：{snapshot.decision_max_score:.4f} | {reason_text}",
        )

    def _clear_latest_result(self) -> None:
        self.episode_tracker.clear_latest()
        self.snapshot_time_var.set("尚无检测结果")
        self.snapshot_keyword_var.set("—"); self.snapshot_result_var.set("—")
        self.snapshot_raw_var.set("—"); self.snapshot_decision_var.set("—"); self.snapshot_threshold_var.set("—")
        self.snapshot_preprocess_var.set("一级能量 — ｜二级 VAD — ｜三级连续帧 —")
        self.snapshot_detection_var.set("L1 — ｜L2 — ｜L3 — ｜L4 — ｜L5 —")
        self.snapshot_reason_var.set("—")
        self.snapshot_metrics_var.set("持续时间 —  |  推理窗口 —  |  最高分窗口 —")

    def _clear_detection_history(self) -> None:
        self.episode_tracker.clear_history()
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

    def _refresh_snapshot_age(self) -> None:
        if self._closing:
            return
        if self.episode_tracker.latest is not None:
            self.snapshot_time_var.set(self._snapshot_time_text(self.episode_tracker.latest))
        self.root.after(1000, self._refresh_snapshot_age)

    def _set_runtime_state(self, state: str) -> None:
        self.state_var.set(STATE_TEXT[state])
        self.state_label.configure(style="Wake.Status.TLabel" if state == "WAKE" else "Stopped.Status.TLabel" if state == "STOPPED" else "Status.TLabel")

    @staticmethod
    def _translate_rejection_reason(reason: str | None) -> str:
        """Translate backend/runtime reason codes without inferring a reason in the UI."""

        mapping = {
            "RAW_SCORE_BELOW_THRESHOLD": "模型分数低于阈值",
            "RAW_OR_SMOOTHED_SCORE_BELOW_THRESHOLD": "模型分数低于阈值",
            "L1_FAILED": "L1 连续帧确认未通过",
            "L1_CONSECUTIVE_SCORE_PENDING": "L1 连续帧确认未通过",
            "L2_FAILED": "L2 峰值/背景比未通过",
            "L2_BACKGROUND_RATIO_FAILED": "L2 峰值/背景比未通过",
            "COOLDOWN": "正处于冷却时间",
            "L3_COOLDOWN_ACTIVE": "正处于冷却时间",
            "L4_LOST": "L4 多关键词竞争未胜出",
            "L4_ARBITRATION_FAILED": "L4 多关键词竞争未胜出",
            "L5_FAILED": "L5 前后静音/能量变化未通过",
            "L5_TRANSITION_PENDING": "L5 前后静音/能量变化未通过",
            "FINAL_WAKE_EVENT": "无（已成功唤醒）",
            "NO_NEW_SCORE": "等待检测",
            "REJECT_TIMEOUT": "检测超时（5 秒内未唤醒）",
            "NO_VALID_MODEL_SCORE": "模型未产生有效分数",
        }
        return mapping.get(str(reason or ""), "等待检测")

    def start_listening(self, *, show_error: bool = True) -> None:
        from tkinter import messagebox
        if self.capture and self.capture.running: return
        try:
            keyword = self.live_phrase_var.get().strip()
            if not keyword: raise ValueError("请先输入唤醒词")
            model = self._current_model()
            backend = model.create_backend(keyword)
            backend.load(resolve_project_path(PROJECT_ROOT, self.model_path_var.get()))
            raw = self.config["detection"]
            detection = DetectionLogic(DetectionConfig(wake_threshold=float(self.threshold_var.get()), consecutive_wake_frames=int(raw["consecutive_wake_frames"]), peak_background_ratio=float(raw["peak_background_ratio"]), background_alpha=float(raw["background_alpha"]), cooldown_seconds=float(raw["cooldown_seconds"]), arbitration_margin=float(raw["arbitration_margin"]), pre_silence_frames=int(raw["pre_silence_frames"]), post_silence_frames=int(raw["post_silence_frames"])))
            self.engine = StreamingWakeWordEngine(
                backend,
                frame_ms=int(self.config["audio"]["frame_ms"]),
                pre_roll_seconds=float(self.config["audio"]["pre_roll_seconds"]),
                tail_inference_seconds=0.8,
                detection=detection,
            )
            device_text = self.microphone_var.get()
            if device_text.startswith("设备不可用"): raise RuntimeError(device_text)
            device = int(device_text.split(":", 1)[0])
            def enqueue(frame: np.ndarray) -> None:
                try: self.audio_queue.put_nowait(frame)
                except queue.Full:
                    try: self.audio_queue.get_nowait()
                    except queue.Empty: pass
                    self.audio_queue.put_nowait(frame)
            self.capture = MicrophoneCapture(enqueue, frame_ms=30); self.capture.start(device)
            self.episode_tracker.cancel_active()
            self._set_runtime_state("IDLE"); self.final_decision_var.set("未触发"); self.reject_reason_var.set("等待检测")
            runtime_text = (
                f"rolling：{model.window_seconds:.1f} 秒 / {model.hop_seconds:.2f} 秒 | smoothing：{model.smoothing}"
                if model.runtime_mode == "rolling_window"
                else "原生流式推理"
            )
            self._append(self.live_log, f"开始监听 | 模型：{model.display_name} | {runtime_text} | 麦克风：{device_text}")
        except Exception as exc:
            self.stop_listening()
            detail = str(exc)
            friendly = (
                "无法打开麦克风。可能原因：设备正被占用、设备不支持 16 kHz，或系统没有授予麦克风权限。"
                if "audio" in detail.lower() or "portaudio" in detail.lower() or "stream" in detail.lower()
                else f"无法开始监听：{detail}"
            )
            if hasattr(self, "live_log"):
                self._append(self.live_log, f"启动失败 | {detail}")
            if show_error:
                messagebox.showerror("无法开始监听", friendly)
            else:
                raise

    def stop_listening(self) -> None:
        running = bool(self.capture and self.capture.running)
        if self.capture: self.capture.stop()
        self.capture, self.engine = None, None
        self.episode_tracker.cancel_active()
        self._set_runtime_state("STOPPED"); self.score_progress_var.set(0.0); self.score_var.set("—"); self.raw_score_var.set("—")
        model = self._current_model()
        self.smoothing_var.set(model.smoothing if model.runtime_mode == "rolling_window" else "原生流式"); self.inference_state_var.set("未激活")
        self.energy_state_var.set("等待检测"); self.vad_state_var.set("等待检测"); self.speech_frames_var.set("等待检测")
        for var in self.detection_vars: var.set("等待")
        self.detection_vars[3].set("单关键词模式：无需竞争"); self.final_decision_var.set("未触发"); self.reject_reason_var.set("监听已停止")
        if running: self._append(self.live_log, "已停止监听")

    def _process_audio(self, frame: np.ndarray) -> None:
        if self.engine is None: return
        state = self.engine.process_frame(frame)
        backend_state: dict[str, object] = {}
        score_state = getattr(self.engine.backend, "score_state", None)
        if callable(score_state):
            try:
                backend_state = score_state()
            except Exception:
                backend_state = {}
        raw_value = getattr(state, "raw_wake_score", None)
        raw_score = float(backend_state.get("raw_score", 0.0) if raw_value is None else (raw_value or 0.0))
        decision_value = getattr(state, "decision_wake_score", None)
        decision_fallback = backend_state.get("decision_score", raw_score)
        decision_score = float(decision_fallback if decision_value is None else (decision_value or 0.0))
        has_score = decision_score > 0 or raw_score > 0
        self.raw_score_var.set(f"{raw_score:.4f}" if has_score else "—")
        self.score_var.set(f"{decision_score:.4f}" if has_score else "—")
        self.score_progress_var.set(max(0.0, min(1.0, decision_score)))

        smoothing = backend_state.get("smoothing") if isinstance(backend_state.get("smoothing"), dict) else {}
        smoothing_mode = str(smoothing.get("mode", "raw")) if backend_state else "原生流式"
        self.smoothing_var.set(smoothing_mode)
        tail_frames = int(getattr(state, "tail_silence_frames", 0) or 0)
        tail_required = int(getattr(state, "tail_required_frames", 0) or 0)
        if not state.kws_active:
            self.inference_state_var.set("未激活")
        elif tail_frames > 0:
            self.inference_state_var.set(f"尾部识别中：{tail_frames} / {tail_required} 帧")
        elif backend_state:
            window_seconds = float(backend_state.get("window_seconds", 2.0))
            hop_seconds = float(backend_state.get("hop_seconds", 0.20))
            self.inference_state_var.set(f"正在滑窗识别（{window_seconds:.1f} 秒窗口 / {hop_seconds:.2f} 秒步长）")
        else:
            self.inference_state_var.set("正在识别（Model A）")
        self.reject_reason_var.set(self._translate_rejection_reason(getattr(state, "rejection_reason", None)))

        energy_passed = state.energy >= state.adaptive_threshold
        self.energy_state_var.set(f"{'通过' if energy_passed else '未通过'}（{state.energy_dbfs:.1f} dBFS）")
        self.vad_state_var.set("语音" if state.vad else "非语音" if energy_passed else "等待一级通过")
        self.speech_frames_var.set(f"{min(state.speech_frame_count, 3)} / 3" + (" ✓" if state.speech_frame_count >= 3 else ""))
        streak, _, passed = state.l1_status.partition(":")
        if decision_score <= 0:
            self.detection_vars[0].set("等待"); self.detection_vars[1].set("等待")
        else:
            self.detection_vars[0].set(("通过 " if passed == "True" else "监控中 ") + streak)
            l2_passed = str(getattr(state, "l2_status", "False")).lower() == "true"
            self.detection_vars[1].set(f"{'通过' if l2_passed else '拒绝'}（{state.l2_ratio:.2f}）")
        self.detection_vars[2].set(f"冷却中（{state.cooldown:.1f} 秒）" if state.cooldown > 0 else "✓ 可触发")
        self.detection_vars[3].set("单关键词模式：无需竞争")
        self.detection_vars[4].set({"waiting": "等待", "pending_post_silence": "监控中：等待后置静音", "passed": "通过"}.get(state.l5_status, state.l5_status.replace("post_silence_", "监控中：后置静音 ")))
        if state.final_wake_event:
            self._set_runtime_state("WAKE"); self.final_decision_var.set("✓ 已成功唤醒")
            self._append(self.live_log, f"已唤醒 | 关键词：{state.keyword or self.live_phrase_var.get()} | 判定分数：{decision_score:.4f} | 原始分数：{raw_score:.4f}")
        elif state.cooldown > 0:
            self._set_runtime_state("COOLDOWN"); self.final_decision_var.set("未触发")
        elif state.kws_active and tail_frames > 0:
            self._set_runtime_state("TAIL"); self.final_decision_var.set("未触发")
        elif state.kws_active:
            self._set_runtime_state("EVALUATING"); self.final_decision_var.set("未触发")
        elif state.vad:
            self._set_runtime_state("SPEECH"); self.final_decision_var.set("未触发")
        elif energy_passed:
            self._set_runtime_state("SOUND"); self.final_decision_var.set("未触发")
        else:
            self._set_runtime_state("IDLE"); self.final_decision_var.set("未触发")
        snapshot = self.episode_tracker.update(state, keyword=self.live_phrase_var.get())
        if snapshot is not None:
            self._latch_snapshot(snapshot)
            request_final_wake_playback(
                self.playback,
                snapshot,
                resolve_project_path(PROJECT_ROOT, self.config["awake_wav"]),
            )

    def _poll(self) -> None:
        if self._closing: return
        for _ in range(30):
            try: frame = self.audio_queue.get_nowait()
            except queue.Empty: break
            self._process_audio(frame)
        for _ in range(100):
            try: target, line = self.log_queue.get_nowait()
            except queue.Empty: break
            if target == "playback":
                self._append(self.live_log, line)
                continue
            if line.startswith("__START_ERROR__:"):
                detail = line.split(":", 2)[-1]
                if target == "generation":
                    self.gen_status_var.set("生成失败")
                    self._append(self.generation_log, f"数据生成失败 | {detail}")
                else:
                    self.train_status_var.set("启动失败")
                    self._append(self.training_log, f"训练进程启动失败 | {detail}")
            elif line.startswith("__DONE__:"):
                code = int(line.rsplit(":", 1)[-1])
                if target == "generation":
                    if self.gen_status_var.get() != "已停止": self.gen_status_var.set("已完成" if code == 0 else "生成失败")
                    message = "数据生成完成，可打开输出目录。" if code == 0 else f"数据生成失败（退出码 {code}），完整原因见下方日志。"
                    self._append(self.generation_log, message)
                    if code == 0:
                        self._refresh_generation_result()
                else:
                    if code != 0: self.train_status_var.set("训练失败")
                    self._append(self.training_log, "训练进程已正常结束。" if code == 0 else f"训练进程失败（退出码 {code}）。")
                    self._refresh_training_status()
            else:
                if target == "generation":
                    match = re.search(r"completed_(?:voice|record)=(\d+)/(\d+) records=(\d+)", line)
                    if match:
                        completed, total, records = (int(value) for value in match.groups())
                        self.gen_progress_var.set(f"{completed:,} / {total:,} 条")
                        self.gen_positive_var.set(str(records))
                self._append(self.generation_log if target == "generation" else self.training_log, f"{'生成' if target == 'generation' else '训练'}进程 | {line}")
        self.root.after(30, self._poll)

    def _generation_request(self) -> GenerationRequest:
        mode = self.gen_scale_mode.get()
        total = int(self.gen_count.get())
        return GenerationRequest(
            self.gen_phrase.get(), total, self.gen_source.get(), bool(self.gen_noise.get()),
            Path(self.gen_output.get()), scale_mode=mode,
            custom_total=total if mode == "自定义" else None,
            input_folder=Path(self.gen_input.get()) if self.gen_input.get().strip() else None,
        )

    def _refresh_generation_result(self) -> None:
        manifest_path = resolve_project_path(PROJECT_ROOT, self.gen_output.get()) / "DatasetManifest.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = list(raw.get("records", []))
            labels: dict[str, int] = {}
            for row in records:
                label = str(row.get("label", "unknown"))
                labels[label] = labels.get(label, 0) + 1
            duration = sum(float(row.get("duration_seconds") or 0.0) for row in records)
            try:
                display_path = manifest_path.parent.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                display_path = f"…/{manifest_path.parent.name}"
            self.gen_success_detail_var.set(
                "✓ 数据集生成完成  ·  "
                f"正样本 {labels.get('positive', 0):,}  ·  困难负样本 {labels.get('hard_negative', 0):,}  ·  "
                f"普通负样本 {labels.get('negative', 0):,}  ·  环境样本 {labels.get('ambient', 0):,}  ·  "
                f"总量 {len(records):,}  ·  总时长 {duration / 60:.1f} 分钟  ·  {display_path}"
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.gen_success_detail_var.set(f"生成进程已结束，但 manifest 统计读取失败：{exc}")

    def generation_dry_run(self) -> None:
        from tkinter import messagebox
        try:
            command = build_generation_command(PROJECT_ROOT, self._generation_request()); self.gen_status_var.set("配置检查通过")
            self._append(self.generation_log, "生成前检查通过（仅检查配置，未生成数据）"); self._append(self.generation_log, "技术命令 | " + subprocess.list2cmdline(command))
        except Exception as exc: messagebox.showerror("生成前检查失败", str(exc))

    def start_generation(self) -> None:
        from tkinter import messagebox
        try: command = build_generation_command(PROJECT_ROOT, self._generation_request())
        except Exception as exc: messagebox.showerror("无法启动数据生成", str(exc)); return
        self.gen_status_var.set("正在生成"); self._append(self.generation_log, f"开始生成数据 | 唤醒词：{self.gen_phrase.get()} | 输出：{self.gen_output.get()}")
        def run() -> None:
            try:
                code = self.generation_process.start(command, lambda line: self.log_queue.put(("generation", line)), cwd=PROJECT_ROOT)
            except Exception as exc:
                self.log_queue.put(("generation", f"__START_ERROR__:{type(exc).__name__}:{exc}")); return
            self.log_queue.put(("generation", f"__DONE__:{code}"))
        threading.Thread(target=run, daemon=True).start()

    def stop_generation(self) -> None:
        self.generation_process.stop(); self.gen_status_var.set("已停止"); self._append(self.generation_log, "已请求停止本界面启动的生成任务。")

    def _training_request(self) -> TrainingRequest:
        model = self.model_registry.by_display_name(self.train_model.get())
        return TrainingRequest(Path(self.train_dataset.get()), model.id, self.train_phrase.get(), Path(self.train_output.get()))

    def training_dry_run(self) -> None:
        from tkinter import messagebox
        try:
            command = build_training_command(PROJECT_ROOT, self._training_request()); self._append(self.training_log, "训练配置检查通过（仅检查配置，未启动训练）"); self._append(self.training_log, "技术命令 | " + subprocess.list2cmdline(command))
        except Exception as exc: messagebox.showerror("训练配置检查失败", str(exc))

    def start_training(self) -> None:
        from tkinter import messagebox
        try: command = build_training_command(PROJECT_ROOT, self._training_request())
        except Exception as exc: messagebox.showerror("无法启动训练", str(exc)); return
        if not messagebox.askyesno("确认开始训练", "这将启动一个新的长时训练进程。是否继续？"): return
        def run() -> None:
            try:
                code = self.training_process.start(command, lambda line: self.log_queue.put(("training", line)), cwd=PROJECT_ROOT)
            except Exception as exc:
                self.log_queue.put(("training", f"__START_ERROR__:{type(exc).__name__}:{exc}")); return
            self.log_queue.put(("training", f"__DONE__:{code}"))
        self._append(self.training_log, f"开始训练 | 模型：{self.train_model.get()} | 数据集：{self.train_dataset.get()}")
        threading.Thread(target=run, daemon=True).start()

    def stop_training(self) -> None:
        self.training_process.stop(); self._append(self.training_log, "已请求停止本界面启动的训练；外部 PowerShell 训练不受影响。")

    def _find_training_status(self) -> Path | None:
        selected = Path(self.train_output.get()) / "TRAINING_STATUS.json"
        if selected.is_file(): return selected
        candidates = []
        for path in (PROJECT_ROOT / "runs").rglob("TRAINING_STATUS.json"):
            if "test" in {part.lower() for part in path.parts}: continue
            try: candidates.append((path.stat().st_mtime, path))
            except OSError: pass
        return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]

    def _refresh_training_status_periodic(self) -> None:
        if self._closing: return
        self._refresh_training_status(); self.root.after(2000, self._refresh_training_status_periodic)

    def _refresh_training_status(self) -> None:
        path = self._find_training_status()
        if path is None: self.train_status_source_var.set("未找到 TRAINING_STATUS.json；尚无训练状态可显示"); return
        try: data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): self.train_status_source_var.set(f"状态文件正在更新，稍后重试：{path}"); return
        names = {"RUNNING": "正在训练", "COMPLETED": "已完成", "STOPPED": "已停止", "FAILED": "失败", "PREPARING": "准备中"}
        self.train_status_var.set(names.get(str(data.get("status", "")), str(data.get("status") or "未开始")))
        current = int(data.get("current_step") or 0); planned = int(data.get("planned_steps_after_early_stop") or data.get("planned_steps") or 0)
        self.train_step_var.set(f"step {current} / {planned}" if planned else f"step {current}"); self.train_progress["value"] = min(100, current / planned * 100) if planned else 0
        phase = data.get("phase"); self.train_phase_var.set(f"阶段 {phase}（总阶段数未提供）" if phase is not None else "尚无阶段信息")
        loss = data.get("last_loss"); lr = data.get("learning_rate")
        self.train_loss_var.set(f"{float(loss):.4f}" if loss is not None else "尚无数据"); self.train_lr_var.set(f"{float(lr):.3g}" if lr is not None else "尚无数据")
        elapsed = data.get("elapsed_seconds_this_process", data.get("total_elapsed_time_seconds", data.get("elapsed_seconds")))
        if elapsed is None: self.train_elapsed_var.set("尚无数据")
        else:
            hours, remainder = divmod(max(0, int(float(elapsed))), 3600); minutes = remainder // 60
            self.train_elapsed_var.set(f"{hours} 小时 {minutes} 分" if hours else f"{minutes} 分钟")
        if elapsed is not None and current > 0 and planned > current:
            eta_seconds = int(float(elapsed) / current * (planned - current))
            eta_hours, eta_remainder = divmod(max(0, eta_seconds), 3600)
            eta_minutes = eta_remainder // 60
            self.train_eta_var.set(f"约 {eta_hours} 小时 {eta_minutes} 分" if eta_hours else f"约 {eta_minutes} 分钟")
        elif planned and current >= planned:
            self.train_eta_var.set("已完成")
        else:
            self.train_eta_var.set("尚无数据")
        val = data.get("last_validation") if isinstance(data.get("last_validation"), dict) else {}
        for variable, key in ((self.train_recall_var, "recall"), (self.train_fpr_var, "fpr"), (self.train_worst_recall_var, "worst_source_recall")):
            value = val.get(key); variable.set(f"{float(value) * 100:.2f}%" if value is not None else "尚无评估结果")
        checkpoint = data.get("best_checkpoint") or data.get("last_checkpoint") or data.get("last_successful_checkpoint")
        self.train_checkpoint_var.set(Path(str(checkpoint)).name if checkpoint else "尚无 checkpoint")
        threshold = val.get("threshold"); self.train_best_threshold_var.set(f"{float(threshold):.4f}" if threshold is not None else "尚无评估结果")
        self.train_status_source_var.set(f"只读状态来源：{path}（不读取 Test，不控制该训练进程）")

    def inspect_model(self) -> None:
        from tkinter import messagebox
        try:
            info = inspect_tflite_model(resolve_project_path(PROJECT_ROOT, self.model_path_var.get()))
            model = self._current_model()
            deployment = model.deployment
            expected_sha256 = str(deployment.get("sha256", ""))
            if expected_sha256 and info.sha256 != expected_sha256:
                raise RuntimeError("当前模型 SHA256 与冻结部署配置不一致")
            recall = deployment.get("validation_recall")
            target = float(deployment.get("validation_target_recall", 0.98))
            target_met = bool(deployment.get("validation_target_met", False))
            validation_text = (
                "配置未提供"
                if recall is None
                else f"{float(recall) * 100:.1f}%（{'已达到' if target_met else '未达到'} {target * 100:.0f}%）"
            )
            values = {"model": model.display_name, "format": "TensorFlow Lite", "quant": "Full INT8" if info.full_int8 else "非完整 INT8", "size": f"{info.kib:.2f} KiB", "input": f"{info.input_shape} {info.input_dtype.upper()}", "output": f"{info.output_shape} {info.output_dtype.upper()}", "sha": info.sha256, "validation": validation_text}
            for key, value in values.items(): self.model_info_vars[key].set(value)
            self.deploy_int8_var.set("✓ 已生成 INT8 TFLite 模型" if info.full_int8 else "✗ 当前模型并非完整 INT8 输入 / 输出")
            self.deploy_size_var.set(f"✓ 模型大小 {info.kib:.2f} KiB，可供嵌入式部署评估")
        except Exception as exc: messagebox.showerror("模型信息读取失败", str(exc))

    def _copy_sha(self) -> None:
        value = self.model_info_vars["sha"].get()
        if value != "等待检查": self.root.clipboard_clear(); self.root.clipboard_append(value)

    def prepare_export(self) -> None:
        from tkinter import filedialog, messagebox
        output = filedialog.askdirectory(title="选择 ESP32-S3 部署包输出目录")
        if not output: return
        try:
            report = prepare_esp32s3_package(resolve_project_path(PROJECT_ROOT, self.model_path_var.get()), Path(output)); packaged = Path(str(report.get("model", {}).get("packaged_path", "")))
            self.export_path_var.set(str(Path(output).resolve())); self.export_contents_var.set(f"{packaged.name}、model_info.json、README.txt"); self.deploy_hardware_var.set("● 尚未进行真实 ESP32-S3 板端测试")
            messagebox.showinfo("部署包已生成", f"部署包路径：\n{Path(output).resolve()}\n\n注意：尚未进行真实 ESP32-S3 板端测试。")
        except Exception as exc: messagebox.showerror("ESP32-S3 部署包生成失败", str(exc))

    def close(self) -> None:
        self._closing = True; self.stop_listening(); self.playback.close(wait=False); self.generation_process.stop(); self.training_process.stop(); self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="WakeWord Studio 中文教师演示界面")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/demo/teacher_demo.yaml")
    parser.add_argument("--ui-smoke", action="store_true")
    parser.add_argument("--model-a-listen-smoke", action="store_true")
    args = parser.parse_args()
    import tkinter as tk
    root = tk.Tk()
    if args.ui_smoke or args.model_a_listen_smoke: root.withdraw()
    app = WakeWordStudioApp(root, args.config)
    if args.model_a_listen_smoke:
        class _SmokeCapture:
            def __init__(self, frame_callback, *, frame_ms: int = 30):  # noqa: ANN001
                self.frame_callback = frame_callback
                self.frame_ms = frame_ms
                self.running = False

            def start(self, device: int | None = None) -> None:
                del device
                self.running = True

            def stop(self) -> None:
                self.running = False

        original_capture = globals()["MicrophoneCapture"]
        globals()["MicrophoneCapture"] = _SmokeCapture
        app.model_var.set(MODEL_A)
        app._apply_model_defaults()
        app.microphone_var.set("0: Model A smoke input")
        expected = app.model_registry.by_id("model_a")
        if float(app.threshold_var.get()) != expected.threshold:
            raise RuntimeError("Model A UI smoke threshold binding mismatch")
        try:
            app.start_listening(show_error=False)
            if app.engine is None or app.capture is None or not app.capture.running:
                raise RuntimeError("Model A UI smoke failed to start listening")
            root.update_idletasks()
            app.stop_listening()
            if app.engine is not None or app.capture is not None:
                raise RuntimeError("Model A UI smoke failed to stop listening cleanly")
        finally:
            globals()["MicrophoneCapture"] = original_capture
        print(
            "MODEL_A_UI_LISTEN_SMOKE PASS "
            f"model_loaded=true start=true stop=true threshold={app.threshold_var.get()} "
            "capture=synthetic",
            flush=True,
        )
        app.close()
        return
    if args.ui_smoke:
        root.update_idletasks()
        page_sizes = {
            name: (page.winfo_reqwidth(), page.winfo_reqheight())
            for name, page in (
                ("generation", app.generation_tab), ("training", app.training_tab),
                ("live", app.live_tab), ("deployment", app.export_tab),
            )
        }
        oversized = {name: size for name, size in page_sizes.items() if size[0] > 1180 or size[1] > 690}
        if oversized:
            raise RuntimeError(f"UI page request exceeds 1200x780 content area: {oversized}")
        model = app.model_registry.by_id("model_b")
        backend = model.create_backend(app.live_phrase_var.get())
        backend.load(model.model_path)
        info = inspect_tflite_model(model.model_path)
        if info.sha256 != str(model.deployment["sha256"]):
            raise RuntimeError("UI smoke model SHA256 mismatch")
        if float(app.threshold_var.get()) != model.threshold:
            raise RuntimeError("UI smoke threshold binding mismatch")
        print(f"UI_STARTUP_SMOKE PASS tabs=4 page_sizes={page_sizes} default_model={app.model_var.get()} model_loaded=true threshold={app.threshold_var.get()} window=2.0 hop=0.20 smoothing=raw state={app.state_var.get()}", flush=True)
        app.close(); return
    root.mainloop()


if __name__ == "__main__":
    main()
