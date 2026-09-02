"""Record and diagnose real microphone RepCNN scores without touching Test data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase3.scripts.evaluate_repcnn_model_b_frozen import Int8Scorer  # noqa: E402
from phase3.scripts.preflight_repcnn_model_b import (  # noqa: E402
    FeatureLoader,
    build_adapter,
)
from wakeword_studio.backends.repcnn import RepCNNBackend  # noqa: E402
from wakeword_studio.diagnostics.live_repcnn import (  # noqa: E402
    SAMPLE_RATE_HZ,
    amplitude_statistics,
    percentile_summary,
    read_pcm16_wav,
    score_windows,
    spectral_statistics,
    vad_speech_intervals,
    write_pcm16_wav,
    write_score_artifacts,
)
from wakeword_studio.training.streaming_windows import extract_streaming_window  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs/models/repcnn_performance_v1.yaml"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs/qingxiaojia/repcnn_performance_v1/formal/user_run_01/phase3c_model_b_frozen"
    / "final_model/qingxiaojia_repcnn_performance_v1_best1750_full_int8.tflite"
)
DEFAULT_THRESHOLD = 0.84375
DEFAULT_ROOT = PROJECT_ROOT / "phase5/artifacts/live_diagnostics"
DEFAULT_LEVEL_BASELINE = DEFAULT_ROOT / "train_validation_level_baseline.json"


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


class DualPipelineScorer:
    """The deployed Live backend beside the actual formal evaluator objects."""

    def __init__(self, config_path: Path, model_path: Path):
        self.config_path = config_path.resolve()
        self.model_path = model_path.resolve()
        self.config: dict[str, Any] = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        )
        self.adapter = build_adapter(self.config)
        if self.adapter.test_loaded:
            raise RuntimeError("Test access is prohibited in live diagnostics")
        self.feature_loader = FeatureLoader(self.adapter, self.config)
        self.formal_scorer = Int8Scorer(self.model_path)
        self.live_backend = RepCNNBackend(hop_seconds=0.20)
        self.live_backend.load(self.model_path)

    def live_score(self, pcm16: np.ndarray) -> float:
        self.live_backend.reset_stream()
        scores = self.live_backend.stream_scores(np.asarray(pcm16, dtype=np.int16))
        if "你好，青小甲" not in scores:
            raise RuntimeError("Live backend did not score an exact two-second window")
        return float(scores["你好，青小甲"])

    def offline_score(self, pcm16: np.ndarray) -> float:
        clip = np.asarray(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        feature = np.asarray(self.feature_loader.frontend(clip)[0], dtype=np.float32)
        return float(self.formal_scorer.score(feature)["score"])

    def provenance(self) -> dict[str, object]:
        return {
            "formal_pipeline": {
                "frontend_object": "phase3.preflight.FeatureLoader.frontend",
                "inference_object": "phase3.evaluate.Int8Scorer",
            },
            "live_pipeline": "wakeword_studio.backends.repcnn.RepCNNBackend",
            "config_path": str(self.config_path),
            "model_path": str(self.model_path),
            "frontend": self.config["frontend"],
            "test_loaded": self.adapter.test_loaded,
        }


def resolve_device(query: str | None) -> tuple[int | None, str]:
    import sounddevice as sd

    devices = sd.query_devices()
    if query is None or not query.strip():
        default = int(sd.default.device[0])
        return default, str(devices[default]["name"])
    query = query.strip()
    if query.isdigit():
        index = int(query)
        if index < 0 or index >= len(devices) or int(devices[index]["max_input_channels"]) < 1:
            raise ValueError(f"Invalid input device index: {index}")
        return index, str(devices[index]["name"])
    matches = [
        (index, str(row["name"]))
        for index, row in enumerate(devices)
        if int(row["max_input_channels"]) > 0 and query.casefold() in str(row["name"]).casefold()
    ]
    if not matches:
        raise ValueError(f"No input device matches: {query}")
    # Prefer the shortest matching system name to avoid virtual duplicate aliases.
    return min(matches, key=lambda item: (len(item[1]), item[0]))


def list_devices() -> None:
    import sounddevice as sd

    rows = []
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) > 0:
            rows.append(
                {
                    "index": index,
                    "name": str(device["name"]),
                    "max_input_channels": int(device["max_input_channels"]),
                    "default_sample_rate": float(device["default_samplerate"]),
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def record_pcm16(device: int | None, seconds: float) -> np.ndarray:
    import sounddevice as sd

    frames = int(round(seconds * SAMPLE_RATE_HZ))
    print(
        f"即将录制 {seconds:.1f} 秒：请保留约 1 秒前静音，说‘你好，青小甲’，再保留约 1 秒后静音。",
        flush=True,
    )
    for value in (3, 2, 1):
        print(value, flush=True)
        time.sleep(0.5)
    print("录音开始", flush=True)
    captured = sd.rec(
        frames,
        samplerate=SAMPLE_RATE_HZ,
        channels=1,
        dtype="int16",
        device=device,
        blocking=True,
    )
    print("录音结束", flush=True)
    return np.asarray(captured, dtype=np.int16).reshape(-1)


def load_level_baseline(path: Path | None) -> tuple[dict[str, object] | None, float | None]:
    if path is None or not path.is_file():
        return None, None
    baseline = json.loads(path.read_text(encoding="utf-8"))
    target = float(baseline["overall"]["rms_pcm16"]["median"])
    return baseline, target


def level_comparison(
    recording: dict[str, float], baseline: dict[str, object] | None
) -> dict[str, object] | None:
    if baseline is None:
        return None
    reference = baseline["overall"]["rms_pcm16"]  # type: ignore[index]
    rms = recording["rms_pcm16"]
    below = rms < float(reference["p05"])
    above = rms > float(reference["p95"])
    return {
        "train_validation_p05": reference["p05"],
        "train_validation_median": reference["median"],
        "train_validation_p95": reference["p95"],
        "recording_rms_pcm16": rms,
        "outside_train_validation_p05_p95": bool(below or above),
        "diagnosis": "INPUT_LEVEL_DOMAIN_SHIFT" if below or above else "LEVEL_WITHIN_REFERENCE_RANGE",
    }


def root_cause_rule(
    window_report: dict[str, object],
    threshold: float,
    level_report: dict[str, object] | None,
) -> dict[str, object]:
    summary = window_report["summary"]  # type: ignore[index]
    maximum = float(summary["max_offline_score"])
    minimum = min(float(row["offline_score"]) for row in window_report["windows"])  # type: ignore[index]
    delta = float(summary["max_abs_delta"])
    if delta > 1e-7:
        primary = "RUNTIME_PIPELINE_BUG"
    elif maximum >= threshold and maximum - minimum >= 0.30:
        primary = "WINDOW_COVERAGE_OR_TIMING"
    elif maximum < threshold:
        if level_report and level_report["diagnosis"] == "INPUT_LEVEL_DOMAIN_SHIFT":
            primary = "MODEL_GENERALIZATION_FAILURE_OR_INPUT_LEVEL_DOMAIN_SHIFT"
        else:
            primary = "MODEL_GENERALIZATION_FAILURE"
    else:
        primary = "RAW_MODEL_SCORE_HEALTHY_CHECK_DETECTION_SEPARATELY"
    return {
        "primary": primary,
        "rules": {
            "pipeline_bug": "同一窗口 max_abs_delta > 1e-7",
            "window_timing": "双 pipeline 一致，且部分窗口越过冻结阈值、窗口间跨度 >= 0.30",
            "model_generalization": "双 pipeline 一致，所有窗口均低于冻结阈值",
            "input_level_shift": "真人整段 RMS 位于合法 Train/Validation 正样本 p05-p95 之外",
        },
        "not_a_threshold_tuning_result": True,
    }


def analyze_wav(
    wav_path: Path,
    output_dir: Path,
    scorer: DualPipelineScorer,
    *,
    microphone_name: str | None,
    threshold: float,
    baseline_path: Path | None,
    recording_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    pcm16, rate = read_pcm16_wav(wav_path)
    baseline, target_rms = load_level_baseline(baseline_path)
    levels = amplitude_statistics(pcm16)
    windows = score_windows(
        pcm16,
        live_score=scorer.live_score,
        offline_score=scorer.offline_score,
        target_rms_pcm16=target_rms,
    )
    comparison = level_comparison(levels, baseline)
    report = {
        "schema": "wakeword-studio.live-repcnn-diagnostic/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "wav_path": str(wav_path.resolve()),
        "microphone_name": microphone_name,
        "audio": {
            "sample_rate_hz": rate,
            "channels": 1,
            "sample_width_bytes": 2,
            "encoding": "PCM16",
            "number_of_samples": len(pcm16),
            "duration_seconds": len(pcm16) / rate,
            **levels,
            **spectral_statistics(pcm16),
            "vad_speech_intervals": vad_speech_intervals(pcm16),
        },
        "recording_metadata": recording_metadata,
        "frozen_threshold": threshold,
        "pipeline_provenance": scorer.provenance(),
        "window_scores": windows,
        "input_level_comparison": comparison,
        "root_cause_rule_result": root_cause_rule(windows, threshold, comparison),
        "v2_test_loaded": False,
        "v1_external_test_loaded": False,
        "threshold_modified": False,
        "detection_logic_used": False,
    }
    write_score_artifacts(output_dir, report)
    return report


def command_record_analyze(args: argparse.Namespace) -> None:
    device_index, device_name = resolve_device(args.device)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output_dir = (args.output_root / timestamp).resolve()
    pcm16 = record_pcm16(device_index, args.seconds)
    wav_path = output_dir / "recording.wav"
    write_pcm16_wav(wav_path, pcm16)
    metadata = {
        "microphone_index": device_index,
        "microphone_name": device_name,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "duration_seconds": len(pcm16) / SAMPLE_RATE_HZ,
        **amplitude_statistics(pcm16),
        "vad_speech_intervals": vad_speech_intervals(pcm16),
        "score_independent_recording": True,
    }
    json_write(output_dir / "recording_metadata.json", metadata)
    scorer = DualPipelineScorer(args.config, args.model)
    report = analyze_wav(
        wav_path,
        output_dir,
        scorer,
        microphone_name=device_name,
        threshold=args.threshold,
        baseline_path=args.level_baseline,
        recording_metadata=metadata,
    )
    print_summary(report, output_dir)


def command_analyze(args: argparse.Namespace) -> None:
    scorer = DualPipelineScorer(args.config, args.model)
    output_dir = args.output_dir or args.wav.parent / f"{args.wav.stem}_diagnostic"
    report = analyze_wav(
        args.wav,
        output_dir.resolve(),
        scorer,
        microphone_name=args.device_label,
        threshold=args.threshold,
        baseline_path=args.level_baseline,
    )
    print_summary(report, output_dir)


def print_summary(report: dict[str, object], output_dir: Path) -> None:
    window_summary = report["window_scores"]["summary"]  # type: ignore[index]
    print(
        json.dumps(
            {
                "保存目录": str(output_dir.resolve()),
                "RMS": report["audio"]["rms_pcm16"],  # type: ignore[index]
                "Peak": report["audio"]["peak_pcm16"],  # type: ignore[index]
                "窗口数": report["window_scores"]["window_count"],  # type: ignore[index]
                "最大_LIVE_SCORE": window_summary["max_live_score"],
                "最大_OFFLINE_SCORE": window_summary["max_offline_score"],
                "最大_DELTA": window_summary["max_abs_delta"],
                "Top5": [
                    {
                        "start_ms": row["start_ms"],
                        "end_ms": row["end_ms"],
                        "score": row["offline_score"],
                    }
                    for row in window_summary["top5"]
                ],
                "初步根因": report["root_cause_rule_result"]["primary"],  # type: ignore[index]
                "TEST_LOADED": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def command_parity(args: argparse.Namespace) -> None:
    scorer = DualPipelineScorer(args.config, args.model)
    rows = []
    for split in ("train", "validation"):
        samples = scorer.adapter.deterministic_sample(
            split, "positive", args.per_split, purpose="phase5-live-offline-parity"
        )
        for sample in samples:
            float_clip, feature = scorer.feature_loader.audio_and_feature(sample)
            pcm16 = np.clip(np.rint(float_clip * 32768.0), -32768, 32767).astype(np.int16)
            formal = float(scorer.formal_scorer.score(feature)["score"])
            live = scorer.live_score(pcm16)
            independent_offline = scorer.offline_score(pcm16)
            rows.append(
                {
                    "split": split,
                    "record_id": sample.record_id,
                    "source": sample.source,
                    "speaker_id": sample.speaker_id,
                    "window_start_ms": sample.window.window_start_ms,
                    "window_end_ms": sample.window.window_end_ms,
                    "formal_evaluator_score": formal,
                    "live_backend_score": live,
                    "independent_offline_score": independent_offline,
                    "live_formal_delta": live - formal,
                    "offline_formal_delta": independent_offline - formal,
                }
            )
    maximum = max(abs(float(row["live_formal_delta"])) for row in rows)
    report = {
        "schema": "wakeword-studio.live-offline-parity/v1",
        "status": "PASS" if maximum <= 1e-7 else "FAIL",
        "rows": rows,
        "max_abs_live_formal_delta": maximum,
        "pipeline_provenance": scorer.provenance(),
        "allowed_splits": ["train", "validation"],
        "test_loaded": False,
    }
    json_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Live/offline parity failed")


def summarize_group(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "rms_pcm16": percentile_summary([float(row["rms_pcm16"]) for row in rows]),
        "peak_pcm16": percentile_summary([float(row["peak_pcm16"]) for row in rows]),
        "log_energy_dbfs": percentile_summary([float(row["log_energy_dbfs"]) for row in rows]),
    }


def command_levels(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.fasttrack_config.read_text(encoding="utf-8"))
    adapter = build_adapter(config)
    if adapter.test_loaded:
        raise RuntimeError("Test access is prohibited")
    loader = FeatureLoader(adapter, config)
    rows: list[dict[str, object]] = []
    for split in ("train", "validation"):
        samples = adapter.samples(split, "positive")
        for index, sample in enumerate(samples, start=1):
            float_clip, _ = loader.audio_and_feature(sample)
            pcm16 = np.clip(np.rint(float_clip * 32768.0), -32768, 32767).astype(np.int16)
            rows.append(
                {
                    "split": split,
                    "record_id": sample.record_id,
                    "source": sample.source,
                    "speaker_id": sample.speaker_id,
                    **amplitude_statistics(pcm16),
                }
            )
            if index % 250 == 0:
                print(f"LEVELS split={split} completed={index}/{len(samples)}", flush=True)
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
        by_split[str(row["split"])].append(row)
    report = {
        "schema": "wakeword-studio.train-validation-positive-level-baseline/v1",
        "positive_policy": "qingxiaojia_v3_fasttrack eligible positives only",
        "overall": summarize_group(rows),
        "by_split": {key: summarize_group(value) for key, value in sorted(by_split.items())},
        "by_source": {key: summarize_group(value) for key, value in sorted(by_source.items())},
        "record_count": len(rows),
        "test_loaded": False,
    }
    json_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_compare_devices(args: argparse.Namespace) -> None:
    realtek = json.loads(args.realtek_report.read_text(encoding="utf-8"))
    freebuds = json.loads(args.freebuds_report.read_text(encoding="utf-8"))

    def extract(report: dict[str, object]) -> dict[str, object]:
        audio = report["audio"]  # type: ignore[index]
        summary = report["window_scores"]["summary"]  # type: ignore[index]
        return {
            "microphone_name": report.get("microphone_name"),
            "rms_pcm16": audio["rms_pcm16"],
            "peak_pcm16": audio["peak_pcm16"],
            "duration_seconds": audio["duration_seconds"],
            "spectral_centroid_hz": audio["spectral_centroid_hz"],
            "high_band_energy_ratio_4k_8k": audio["high_band_energy_ratio_4k_8k"],
            "max_offline_score": summary["max_offline_score"],
            "max_live_score": summary["max_live_score"],
        }

    left, right = extract(realtek), extract(freebuds)
    comparison = {
        "schema": "wakeword-studio.live-device-comparison/v1",
        "phrase": "你好，青小甲",
        "realtek": left,
        "freebuds_5i": right,
        "absolute_differences": {
            key: abs(float(left[key]) - float(right[key]))
            for key in (
                "rms_pcm16",
                "peak_pcm16",
                "duration_seconds",
                "spectral_centroid_hz",
                "high_band_energy_ratio_4k_8k",
                "max_offline_score",
            )
        },
        "interpretation": (
            "DEVICE_DOMAIN_SHIFT_CANDIDATE"
            if abs(float(left["max_offline_score"]) - float(right["max_offline_score"])) >= 0.20
            else "NO_LARGE_SCORE_SHIFT_IN_THIS_PAIR"
        ),
        "test_loaded": False,
    }
    json_write(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-devices")

    record = sub.add_parser("record-analyze")
    record.add_argument("--device", help="设备 index 或名称子串，例如 Realtek / FreeBuds")
    record.add_argument("--seconds", type=float, default=4.0)
    record.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    record.add_argument("--level-baseline", type=Path, default=DEFAULT_LEVEL_BASELINE)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--wav", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path)
    analyze.add_argument("--device-label")
    analyze.add_argument("--level-baseline", type=Path, default=DEFAULT_LEVEL_BASELINE)

    parity = sub.add_parser("parity")
    parity.add_argument("--per-split", type=int, default=2)
    parity.add_argument("--output", type=Path, default=DEFAULT_ROOT / "live_offline_parity.json")

    levels = sub.add_parser("levels")
    levels.add_argument(
        "--fasttrack-config",
        type=Path,
        default=PROJECT_ROOT / "configs/models/repcnn_performance_v2_fasttrack.yaml",
    )
    levels.add_argument("--output", type=Path, default=DEFAULT_LEVEL_BASELINE)

    compare = sub.add_parser("compare-devices")
    compare.add_argument("--realtek-report", type=Path, required=True)
    compare.add_argument("--freebuds-report", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=DEFAULT_ROOT / "device_comparison.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list-devices":
        list_devices()
    elif args.command == "record-analyze":
        command_record_analyze(args)
    elif args.command == "analyze":
        command_analyze(args)
    elif args.command == "parity":
        command_parity(args)
    elif args.command == "levels":
        command_levels(args)
    elif args.command == "compare-devices":
        command_compare_devices(args)


if __name__ == "__main__":
    main()
