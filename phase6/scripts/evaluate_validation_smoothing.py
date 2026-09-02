"""Compare raw/mean/max-mean rolling scores using Validation records only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase3.scripts.evaluate_repcnn_model_b_frozen import Int8Scorer  # noqa: E402
from phase3.scripts.preflight_repcnn_model_b import build_adapter  # noqa: E402
from wakeword_studio.dataset.manifest import sha256_file  # noqa: E402
from wakeword_studio.dataset.repcnn_adapter import REPCNN_LABELS  # noqa: E402
from wakeword_studio.diagnostics.live_repcnn import plan_fixed_windows  # noqa: E402
from wakeword_studio.frontends import load_inference_audio  # noqa: E402
from wakeword_studio.runtime.score_smoothing import RollingScoreSmoother  # noqa: E402
from wakeword_studio.training.repcnn_fasttrack import validation_rank  # noqa: E402
from wakeword_studio.training.repcnn_finalization import operating_points  # noqa: E402


CANDIDATES = (
    ("raw", "raw", 1, 0.0),
    ("mean_2", "mean", 2, 0.0),
    ("mean_3", "mean", 3, 0.0),
    ("max_mean_hybrid_2", "max_mean_hybrid", 2, 0.5),
    ("max_mean_hybrid_3", "max_mean_hybrid", 3, 0.5),
)


def float_window(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    result = np.zeros(end - start, np.float32)
    source = np.asarray(audio, np.float32)[start : min(end, len(audio))]
    result[: len(source)] = source
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only rolling score smoothing comparison")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/models/repcnn_performance_v2_fasttrack.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "phase6/artifacts/VALIDATION_SMOOTHING_REPORT.json",
    )
    parser.add_argument("--limit-per-label", type=int, default=0)
    parser.add_argument("--allow-validation-smoothing-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_validation_smoothing_evaluation:
        raise SystemExit("Validation smoothing evaluation is gated")

    started = time.perf_counter()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    adapter = build_adapter(config)
    if adapter.test_loaded:
        raise RuntimeError("Held-out Test access is prohibited")
    scorer = Int8Scorer(args.model.resolve())
    from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend

    frontend = MicroFrontend(
        sample_rate=16000, window_size_ms=30, window_step_ms=20, num_channels=40
    )
    rows: list[dict[str, object]] = []
    for label in REPCNN_LABELS:
        samples = adapter.samples("validation", label)
        if args.limit_per_label:
            samples = tuple(
                adapter.deterministic_sample(
                    "validation",
                    label,
                    args.limit_per_label,
                    purpose="phase6-validation-smoothing-subset",
                )
            )
        for sample in samples:
            audio = load_inference_audio(sample.audio_path)
            plans = plan_fixed_windows(len(audio), hop_seconds=0.20)
            raw_scores: list[float] = []
            for plan in plans:
                clip = float_window(audio, plan.start_sample, plan.end_sample)
                feature = np.asarray(frontend(clip)[0], np.float32)
                raw_scores.append(float(scorer.score(feature)["score"]))
            method_scores: dict[str, float] = {}
            for name, mode, window, weight in CANDIDATES:
                smoother = RollingScoreSmoother(
                    mode, window_size=window, hybrid_max_weight=weight
                )
                method_scores[name] = max(smoother.update(value) for value in raw_scores)
            rows.append(
                {
                    "record_id": sample.record_id,
                    "label": label,
                    "source": sample.source,
                    "window_count": len(raw_scores),
                    "scores": method_scores,
                }
            )
            if len(rows) % 100 == 0:
                print(f"SMOOTHING_HEARTBEAT validation_records={len(rows)}", flush=True)

    targets = [1 if row["label"] == "positive" else 0 for row in rows]
    labels = [str(row["label"]) for row in rows]
    sources = [str(row["source"]) for row in rows]
    methods: dict[str, object] = {}
    for name, _, window, weight in CANDIDATES:
        points = operating_points(
            [float(row["scores"][name]) for row in rows], targets, labels, sources
        )
        methods[name] = {
            "window_size": window,
            "hybrid_max_weight": weight,
            "operating_points": points,
            "selection_metrics": points["fpr_caps"]["fpr_at_most_10pct"],
        }
    raw_metrics = methods["raw"]["selection_metrics"]
    improving = [
        name
        for name in methods
        if name != "raw"
        and not methods[name]["selection_metrics"]["operating_point_degenerate"]
        and validation_rank(methods[name]["selection_metrics"])
        > validation_rank(raw_metrics)
    ]
    full_validation = args.limit_per_label == 0
    recommendation = (
        max(improving, key=lambda name: validation_rank(methods[name]["selection_metrics"]))
        if improving and full_validation
        else "raw"
    )
    report = {
        "schema": "wakeword-studio.validation-smoothing-comparison/v1",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model.resolve()),
        "dataset_split": "validation_only",
        "window_seconds": 2.0,
        "hop_seconds": 0.20,
        "record_count": len(rows),
        "limited_smoke": not full_validation,
        "sampling": (
            "all_validation_records"
            if full_validation
            else f"deterministic_hash_sample_{args.limit_per_label}_per_label"
        ),
        "methods": methods,
        "recommendation": recommendation,
        "recommendation_rule": "keep raw unless full Validation rank strictly improves",
        "raw_remains_runtime_default_until_full_validation": True,
        "test_loaded": False,
        "live_diagnostic_wavs_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
