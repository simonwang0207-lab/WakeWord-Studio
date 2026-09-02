"""Score-independent acoustic/timing audit of the five frozen live recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.diagnostics.live_repcnn import read_pcm16_wav  # noqa: E402


RECORDS = (
    "20260830T224435+0800",
    "20260830T224901+0800",
    "20260830T224942+0800",
    "20260830T225023+0800",
    "20260830T225102+0800",
)


def energy_pattern(audio: np.ndarray, bin_ms: int = 200) -> dict[str, object]:
    width = 16_000 * bin_ms // 1000
    values = []
    for start in range(0, len(audio), width):
        chunk = np.asarray(audio[start : start + width], np.float64)
        values.append(float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0)
    weights = np.square(values)
    centers = (np.arange(len(values)) + 0.5) * bin_ms
    center = float(np.sum(centers * weights) / np.sum(weights)) if np.sum(weights) else 0.0
    peak = int(np.argmax(values)) if values else 0
    return {
        "bin_ms": bin_ms,
        "rms_by_bin": values,
        "energy_center_ms": center,
        "peak_energy_bin_start_ms": peak * bin_ms,
    }


def correlation(rows: list[dict[str, object]], field: str) -> float | None:
    x = np.asarray([float(row[field]) for row in rows], np.float64)
    y = np.asarray([float(row["b1_max_score"]) for row in rows], np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=PROJECT_ROOT / "phase5/artifacts/live_diagnostics",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=PROJECT_ROOT / "phase6/artifacts/b2_checkpoint_benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "phase6/artifacts/LIVE_FAILURE_ANALYSIS.json",
    )
    args = parser.parse_args()
    b2500 = json.loads(
        (args.benchmark_root / "ckpt-2500_b1_vs_b2_live_diagnostic.json").read_text(encoding="utf-8")
    )
    b3000 = json.loads(
        (args.benchmark_root / "ckpt-3000_b1_vs_b2_live_diagnostic.json").read_text(encoding="utf-8")
    )
    if b2500["test_loaded"] or b3000["test_loaded"]:
        raise RuntimeError("Benchmark provenance indicates Test access")
    by2500 = {row["record"]: row for row in b2500["records"]}
    by3000 = {row["record"]: row for row in b3000["records"]}
    rows: list[dict[str, object]] = []
    for record in RECORDS:
        diagnostic = json.loads(
            (args.diagnostics_root / record / "diagnostic_report.json").read_text(encoding="utf-8")
        )
        audio, _ = read_pcm16_wav(args.diagnostics_root / record / "recording.wav")
        metadata = diagnostic["audio"]
        intervals = metadata["vad_speech_intervals"]
        first_start = min(float(row["start_ms"]) for row in intervals)
        last_end = max(float(row["end_ms"]) for row in intervals)
        vad_duration = sum(float(row["end_ms"]) - float(row["start_ms"]) for row in intervals)
        windows2500 = by2500[record]["windows"]
        windows3000 = by3000[record]["windows"]
        best_b1 = max(windows3000, key=lambda row: float(row["b1_score"]))
        best_b2500 = max(windows2500, key=lambda row: float(row["b2_score"]))
        best_b3000 = max(windows3000, key=lambda row: float(row["b2_score"]))
        containing = [
            row
            for row in windows3000
            if float(row["start_ms"]) <= first_start
            and float(row["end_ms"]) >= last_end
        ]
        temporal = energy_pattern(audio)
        rows.append(
            {
                "record": record,
                "speech_start_ms": first_start,
                "speech_end_ms": last_end,
                "vad_duration_ms": vad_duration,
                "leading_silence_ms": first_start,
                "trailing_silence_ms": 4000.0 - last_end,
                "rms_pcm16": metadata["rms_pcm16"],
                "peak_pcm16": metadata["peak_pcm16"],
                "spectral_centroid_hz": metadata["spectral_centroid_hz"],
                "high_band_energy_ratio_4k_8k": metadata["high_band_energy_ratio_4k_8k"],
                "full_speech_window_count": len(containing),
                "b1_max_score": by3000[record]["b1_max_score"],
                "b1_best_window_start_ms": best_b1["start_ms"],
                "b1_best_window_contains_full_vad": best_b1 in containing,
                "b2_2500_max_score": by2500[record]["b2_max_score"],
                "b2_2500_best_window_start_ms": best_b2500["start_ms"],
                "b2_3000_max_score": by3000[record]["b2_max_score"],
                "b2_3000_best_window_start_ms": best_b3000["start_ms"],
                "b2_3000_best_window_contains_full_vad": best_b3000 in containing,
                "temporal_energy": temporal,
            }
        )
    correlations = {
        field: correlation(rows, field)
        for field in (
            "vad_duration_ms",
            "leading_silence_ms",
            "rms_pcm16",
            "peak_pcm16",
            "spectral_centroid_hz",
        )
    }
    report = {
        "schema": "wakeword-studio.live-failure-analysis/v1",
        "classification": [
            "LIVE_DIAGNOSTIC_ONLY",
            "NOT_FOR_THRESHOLD_SELECTION",
            "NOT_A_TEST_SET",
        ],
        "records": rows,
        "small_sample_correlations_descriptive_only": correlations,
        "findings": [
            "五条 VAD 时长均小于 2 秒，且每条都有至少一个固定窗口完整覆盖 VAD；不能把低分简单归因于唤醒词超过窗口。",
            "高分与低分录音的 RMS、VAD 时长存在重叠；单一音量或时长不足以解释 B1 分裂。",
            "B1 最佳窗口并不总是完整覆盖 VAD，说明窗口位置/静音上下文会显著改变分数，但不是唯一原因。",
            "B2-3000 在同一固定窗口协议下显著抬高五条最大分，支持 B1 声学泛化不足是主要因素之一。",
            "只有五条、无音素级对齐，无法确定具体是发音、语速、韵律还是麦克风频响导致剩余差异。",
        ],
        "thresholds_modified": False,
        "test_loaded": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
