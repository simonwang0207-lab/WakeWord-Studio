"""Validation-only audit for the independently preserved B2 best weights.

This script never writes into ``phase6_finalization`` and never loads Test.
It exists to audit historical best weights that are not retained as TensorFlow
CheckpointManager checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase3.scripts.preflight_repcnn_model_b import build_model  # noqa: E402
from phase4.scripts.run_repcnn_v2_fasttrack_training import (  # noqa: E402
    load_feature_groups,
)
from phase6.scripts.finalize_b2 import (  # noqa: E402
    auc_metrics,
    atomic_json,
    model_scores,
    validation_bundle,
)
from wakeword_studio.dataset.manifest import sha256_file  # noqa: E402
from wakeword_studio.training.repcnn_finalization import operating_points  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only audit of a historical RepCNN best-weights H5"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/models/repcnn_performance_v2_fasttrack.yaml",
    )
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    status = json.loads(args.training_status.resolve().read_text(encoding="utf-8"))
    if status.get("test_loaded") is not False:
        raise RuntimeError("Training status does not prove test_loaded=false")

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    groups = load_feature_groups(config)
    if any(split == "test" for split, _label in groups):
        raise RuntimeError("Historical-best audit unexpectedly loaded Test")

    batches, targets, labels, sources = validation_bundle(groups)
    model = build_model(config)
    weights = args.weights.resolve()
    model.load_weights(weights)
    scores = model_scores(model, batches)
    points = operating_points(scores, targets, labels, sources)
    selection = points["fpr_caps"]["fpr_at_most_10pct"]
    report = {
        "schema": "wakeword-studio.repcnn-b2-historical-best-audit/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": args.step,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "selection_dataset": "validation_only",
        "selection_metrics": selection,
        "operating_points": points,
        **auc_metrics(targets, scores),
        "test_loaded": False,
        "live_diagnostic_wavs_used_for_selection": False,
        "existing_finalization_modified": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"HISTORICAL_BEST_AUDITED step={args.step} "
        f"threshold={selection['threshold']:.8f} "
        f"recall={selection['recall']:.6f} "
        f"worst_source_recall={selection['worst_source_recall']:.6f} "
        f"fpr={selection['fpr']:.6f} test_loaded=false",
        flush=True,
    )
    if args.output is not None:
        atomic_json(args.output.resolve(), report)


if __name__ == "__main__":
    main()
