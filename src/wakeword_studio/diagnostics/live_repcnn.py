"""Deterministic recording/window/level diagnostics for RepCNN live audio."""

from __future__ import annotations

import csv
import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


SAMPLE_RATE_HZ = 16_000
WINDOW_SAMPLES = 32_000
HOP_SAMPLES = 3_200


@dataclass(frozen=True, slots=True)
class FixedWindow:
    index: int
    start_sample: int
    end_sample: int
    start_ms: float
    end_ms: float
    trailing_padding_samples: int


def plan_fixed_windows(
    number_of_samples: int,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: float = 2.0,
    hop_seconds: float = 0.20,
) -> list[FixedWindow]:
    """Plan all windows before inference, including one deterministic tail anchor.

    For audio shorter than one window, the sole window starts at zero and is
    zero-padded. For longer audio, regular hop starts are generated first. If the
    final regular window does not end at the WAV boundary, an exact tail-anchored
    window is appended once. No score can influence this plan.
    """

    if number_of_samples < 0:
        raise ValueError("number_of_samples must be non-negative")
    window = int(round(sample_rate_hz * window_seconds))
    hop = int(round(sample_rate_hz * hop_seconds))
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    if number_of_samples <= window:
        starts = [0]
    else:
        last_start = number_of_samples - window
        starts = list(range(0, last_start + 1, hop))
        if starts[-1] != last_start:
            starts.append(last_start)
    return [
        FixedWindow(
            index=index,
            start_sample=start,
            end_sample=start + window,
            start_ms=1000.0 * start / sample_rate_hz,
            end_ms=1000.0 * (start + window) / sample_rate_hz,
            trailing_padding_samples=max(0, start + window - number_of_samples),
        )
        for index, start in enumerate(starts)
    ]


def extract_fixed_window(pcm16: np.ndarray, window: FixedWindow) -> np.ndarray:
    audio = np.asarray(pcm16, dtype=np.int16).reshape(-1)
    result = np.zeros(window.end_sample - window.start_sample, dtype=np.int16)
    source = audio[window.start_sample : min(window.end_sample, len(audio))]
    result[: len(source)] = source
    return result


def write_pcm16_wav(path: Path, pcm16: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(pcm16, dtype="<i2").reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(audio.tobytes())


def read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        content = handle.readframes(frames)
    if channels != 1 or width != 2 or rate != SAMPLE_RATE_HZ:
        raise ValueError(
            f"Diagnostic WAV must be 16 kHz mono PCM16; got rate={rate}, "
            f"channels={channels}, sample_width={width}"
        )
    return np.frombuffer(content, dtype="<i2").copy(), rate


def amplitude_statistics(pcm16: np.ndarray) -> dict[str, float]:
    samples = np.asarray(pcm16, dtype=np.float64).reshape(-1)
    if not len(samples):
        return {
            "rms_pcm16": 0.0,
            "rms_normalized": 0.0,
            "peak_pcm16": 0.0,
            "peak_normalized": 0.0,
            "mean_square_energy": 0.0,
            "log_energy_dbfs": -180.0,
        }
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    return {
        "rms_pcm16": rms,
        "rms_normalized": rms / 32768.0,
        "peak_pcm16": peak,
        "peak_normalized": peak / 32768.0,
        "mean_square_energy": float(np.mean(samples * samples)),
        "log_energy_dbfs": 20.0 * math.log10(max(rms / 32768.0, 1e-9)),
    }


def spectral_statistics(pcm16: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> dict[str, float]:
    audio = np.asarray(pcm16, dtype=np.float64).reshape(-1) / 32768.0
    if not len(audio):
        return {"spectral_centroid_hz": 0.0, "high_band_energy_ratio_4k_8k": 0.0}
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
    frequencies = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate_hz)
    total = float(np.sum(spectrum))
    if total <= 1e-20:
        return {"spectral_centroid_hz": 0.0, "high_band_energy_ratio_4k_8k": 0.0}
    return {
        "spectral_centroid_hz": float(np.sum(frequencies * spectrum) / total),
        "high_band_energy_ratio_4k_8k": float(
            np.sum(spectrum[(frequencies >= 4000.0) & (frequencies <= 8000.0)]) / total
        ),
    }


def vad_speech_intervals(
    pcm16: np.ndarray,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    frame_ms: int = 30,
    aggressiveness: int = 2,
) -> list[dict[str, float]]:
    import webrtcvad

    audio = np.asarray(pcm16, dtype="<i2").reshape(-1)
    frame_samples = sample_rate_hz * frame_ms // 1000
    vad = webrtcvad.Vad(aggressiveness)
    speech_frames: list[bool] = []
    for start in range(0, len(audio), frame_samples):
        frame = audio[start : start + frame_samples]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)))
        speech_frames.append(bool(vad.is_speech(frame.astype("<i2").tobytes(), sample_rate_hz)))
    intervals: list[dict[str, float]] = []
    start_index: int | None = None
    for index, speech in enumerate([*speech_frames, False]):
        if speech and start_index is None:
            start_index = index
        elif not speech and start_index is not None:
            intervals.append(
                {
                    "start_ms": float(start_index * frame_ms),
                    "end_ms": float(min(index * frame_ms, len(audio) * 1000 / sample_rate_hz)),
                }
            )
            start_index = None
    return intervals


def safe_rms_normalize(
    pcm16: np.ndarray,
    *,
    target_rms_pcm16: float,
    maximum_gain_db: float = 18.0,
) -> tuple[np.ndarray, dict[str, float]]:
    audio = np.asarray(pcm16, dtype=np.int16).reshape(-1)
    current = amplitude_statistics(audio)["rms_pcm16"]
    if current <= 1e-9:
        return audio.copy(), {"gain": 1.0, "gain_db": 0.0, "clipped_fraction": 0.0}
    requested = target_rms_pcm16 / current
    maximum_gain = 10.0 ** (maximum_gain_db / 20.0)
    gain = min(requested, maximum_gain)
    value = audio.astype(np.float64) * gain
    clipped = float(np.mean((value > 32767.0) | (value < -32768.0)))
    normalized = np.clip(np.rint(value), -32768, 32767).astype(np.int16)
    return normalized, {
        "gain": float(gain),
        "gain_db": 20.0 * math.log10(max(gain, 1e-12)),
        "clipped_fraction": clipped,
    }


def score_windows(
    pcm16: np.ndarray,
    *,
    live_score: Callable[[np.ndarray], float],
    offline_score: Callable[[np.ndarray], float],
    target_rms_pcm16: float | None = None,
) -> dict[str, object]:
    windows = plan_fixed_windows(len(pcm16))
    rows: list[dict[str, object]] = []
    for window in windows:
        clip = extract_fixed_window(pcm16, window)
        live = float(live_score(clip))
        offline = float(offline_score(clip))
        row: dict[str, object] = {
            **asdict(window),
            "live_score": live,
            "offline_score": offline,
            "delta": live - offline,
            "abs_delta": abs(live - offline),
            "level": amplitude_statistics(clip),
        }
        if target_rms_pcm16 is not None:
            normalized, gain = safe_rms_normalize(
                clip, target_rms_pcm16=target_rms_pcm16
            )
            row["diagnostic_rms_normalization"] = {
                **gain,
                "target_rms_pcm16": target_rms_pcm16,
                "live_score": float(live_score(normalized)),
                "offline_score": float(offline_score(normalized)),
            }
        rows.append(row)
    ranked = sorted(rows, key=lambda row: float(row["offline_score"]), reverse=True)
    offline_scores = [float(row["offline_score"]) for row in rows]
    live_scores = [float(row["live_score"]) for row in rows]
    return {
        "window_policy": {
            "window_seconds": 2.0,
            "hop_seconds": 0.20,
            "tail_anchor": True,
            "planned_before_inference": True,
        },
        "window_count": len(rows),
        "windows": rows,
        "summary": {
            "max_live_score": max(live_scores),
            "max_offline_score": max(offline_scores),
            "max_abs_delta": max(float(row["abs_delta"]) for row in rows),
            "mean_live_score": float(np.mean(live_scores)),
            "mean_offline_score": float(np.mean(offline_scores)),
            "best_window": ranked[0],
            "top5": ranked[:5],
        },
    }


def write_score_artifacts(output_dir: Path, report: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = report["window_scores"]["windows"]  # type: ignore[index]
    with (output_dir / "window_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "index",
                "start_ms",
                "end_ms",
                "trailing_padding_samples",
                "live_score",
                "offline_score",
                "delta",
                "abs_delta",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }
