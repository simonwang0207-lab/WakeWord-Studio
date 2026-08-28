"""Inspect the formal Model A artifact and score held-out manifest records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from microwakeword.inference import Model
from wakeword_studio.frontends import load_inference_audio
from wakeword_studio.dataset.manifest import DatasetManifest


def score_wav(model_path: Path, path: Path) -> float:
    audio = load_inference_audio(path)
    scores = Model(str(model_path), stride=3).predict_clip(audio.astype(np.float32), step_ms=10)
    return float(max(scores, default=0.0))


def describe(scores: list[float]) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def roc_auc(positive: list[float], negative: list[float]) -> float:
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def threshold_sweep(rows: list[dict[str, object]]) -> list[dict[str, float | int]]:
    scores = sorted({float(row["score"]) for row in rows})
    thresholds = [scores[0] - 1e-12]
    thresholds.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    thresholds.append(scores[-1] + 1e-12)
    result = []
    positives = sum(row["label"] == "positive" for row in rows)
    negatives = len(rows) - positives
    for threshold in thresholds:
        tp = sum(row["label"] == "positive" and float(row["score"]) >= threshold for row in rows)
        fp = sum(row["label"] != "positive" and float(row["score"]) >= threshold for row in rows)
        fn = positives - tp
        tn = negatives - fp
        tpr = tp / positives
        fpr = fp / negatives
        result.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tpr": float(tpr),
                "fpr": float(fpr),
                "balanced_accuracy": float((tpr + tn / negatives) / 2),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=Path("outputs/sanity/microwakeword_scores.csv"),
    )
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    args = parser.parse_args()
    manifest = DatasetManifest.load(args.manifest)
    errors = manifest.validate(args.manifest)
    if errors:
        raise SystemExit("Invalid standardized manifest:\n" + "\n".join(errors))
    root = Path(manifest.root)
    rows = []
    selected = manifest.records if args.split == "all" else [r for r in manifest.records if r.split == args.split]
    for index, row in enumerate(selected, start=1):
        rows.append(
            {
                "record_id": row.record_id,
                "label": row.label,
                "split": row.split,
                "audio_path": row.audio_path,
                "score": score_wav(args.model, root / row.audio_path),
            }
        )
        print(f"scored={index}/{len(selected)} label={row.label}", flush=True)
    interpreter = tf.lite.Interpreter(model_path=str(args.model), experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    details_in = interpreter.get_input_details()[0]
    details_out = interpreter.get_output_details()[0]
    by_label = {
        label: [float(row["score"]) for row in rows if row["label"] == label]
        for label in ("positive", "negative", "hard_negative")
    }
    if any(len(scores) < 10 for scores in by_label.values()):
        raise SystemExit(f"Sanity requires at least 10 records per label, got { {k: len(v) for k, v in by_label.items()} }")
    combined_negative = by_label["negative"] + by_label["hard_negative"]
    auc = roc_auc(by_label["positive"], combined_negative)
    sweep = threshold_sweep(rows)
    best_threshold = max(sweep, key=lambda row: (row["balanced_accuracy"], row["tpr"] - row["fpr"]))
    statistics = {label: describe(scores) for label, scores in by_label.items()}
    negative_median = max(statistics["negative"]["median"], statistics["hard_negative"]["median"])
    sanity_pass = bool(
        auc >= 0.75
        and statistics["positive"]["median"] > negative_median
        and statistics["positive"]["mean"] > statistics["negative"]["mean"]
        and statistics["positive"]["mean"] > statistics["hard_negative"]["mean"]
    )
    result = {
        "model": str(args.model.resolve()),
        "bytes": args.model.stat().st_size,
        "kib": args.model.stat().st_size / 1024,
        "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_dtype": np.dtype(details_in["dtype"]).name,
        "output_dtype": np.dtype(details_out["dtype"]).name,
        "unique_operators": sorted({row["op_name"] for row in interpreter._get_ops_details()}),
        "evaluated_split": args.split,
        "records": rows,
        "statistics": statistics,
        "roc_auc": auc,
        "best_threshold": best_threshold,
        "threshold_sweep": sweep,
        "sanity_status": "PASS" if sanity_pass else "FAIL",
        "pass_rule": (
            "ROC AUC >= 0.75 and positive median/mean exceed both negative class distributions"
        ),
    }
    args.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_id", "label", "split", "audio_path", "score"))
        writer.writeheader()
        writer.writerows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "statistics": statistics,
                "roc_auc": auc,
                "best_threshold": best_threshold,
                "sanity_status": result["sanity_status"],
                "scores_csv": str(args.scores_csv.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
