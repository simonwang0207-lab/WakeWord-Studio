"""Replay the frozen five-record live diagnostic set on an existing B2 TFLite.

The recordings are development diagnostics only.  This script neither loads
Test nor performs threshold/model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase3.scripts.evaluate_repcnn_model_b_frozen import Int8Scorer  # noqa: E402
from phase6.scripts.benchmark_b2_checkpoint import (  # noqa: E402
    FROZEN_RECORDS,
    aggregate,
    max_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2-model", type=Path, required=True)
    parser.add_argument("--b2-threshold", type=float, required=True)
    parser.add_argument("--old-b2-model", type=Path)
    parser.add_argument("--old-b2-threshold", type=float)
    parser.add_argument(
        "--b1-model",
        type=Path,
        default=PROJECT_ROOT
        / "runs/qingxiaojia/repcnn_performance_v1/formal/user_run_01/phase3c_model_b_frozen"
        / "final_model/qingxiaojia_repcnn_performance_v1_best1750_full_int8.tflite",
    )
    parser.add_argument("--b1-threshold", type=float, default=0.84375)
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=PROJECT_ROOT / "phase5/artifacts/live_diagnostics",
    )
    args = parser.parse_args()

    from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend

    frontend = MicroFrontend(
        sample_rate=16000, window_size_ms=30, window_step_ms=20, num_channels=40
    )
    b1 = Int8Scorer(args.b1_model.resolve())
    b2 = Int8Scorer(args.b2_model.resolve())
    old_b2 = Int8Scorer(args.old_b2_model.resolve()) if args.old_b2_model else None
    if (old_b2 is None) != (args.old_b2_threshold is None):
        raise ValueError("--old-b2-model and --old-b2-threshold must be provided together")
    rows: list[dict[str, object]] = []
    b1_values: list[float] = []
    b2_values: list[float] = []
    old_b2_values: list[float] = []
    for record in FROZEN_RECORDS:
        wav = args.diagnostics_root.resolve() / record / "recording.wav"
        b1_max, b2_max, windows = max_scores(wav, frontend, b1, b2)
        b1_values.append(b1_max)
        b2_values.append(b2_max)
        row = {
            "record": record,
            "window_count": len(windows),
            "b1_max_score": b1_max,
            "new_b2_final_max_score": b2_max,
            "b1_pass": b1_max >= args.b1_threshold,
            "new_b2_final_pass": b2_max >= args.b2_threshold,
        }
        if old_b2 is not None:
            old_b1_max, old_b2_max, old_windows = max_scores(
                wav, frontend, b1, old_b2
            )
            if old_b1_max != b1_max or len(old_windows) != len(windows):
                raise RuntimeError("Fixed-window B1 replay changed between comparisons")
            old_b2_values.append(old_b2_max)
            row.update(
                {
                    "old_b2_final_max_score": old_b2_max,
                    "old_b2_final_pass": old_b2_max >= args.old_b2_threshold,
                }
            )
        rows.append(row)
    report = {
        "classification": [
            "LIVE_DIAGNOSTIC_ONLY",
            "NOT_FOR_THRESHOLD_SELECTION",
            "NOT_A_TEST_SET",
        ],
        "window_seconds": 2.0,
        "hop_seconds": 0.20,
        "b1_threshold": args.b1_threshold,
        "new_b2_final_threshold": args.b2_threshold,
        "old_b2_final_threshold": args.old_b2_threshold,
        "records": rows,
        "b1_aggregate": aggregate(b1_values, args.b1_threshold),
        "new_b2_final_aggregate": aggregate(b2_values, args.b2_threshold),
        "old_b2_final_aggregate": (
            aggregate(old_b2_values, args.old_b2_threshold)
            if old_b2 is not None
            else None
        ),
        "test_loaded": False,
        "threshold_modified": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
