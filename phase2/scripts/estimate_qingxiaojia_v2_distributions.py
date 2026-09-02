"""Read-only full-quota duration/placement simulation for qingxiaojia_v2."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from wakeword_studio.dataset.formal_builder import _stable_seed
from wakeword_studio.dataset.v2_builder import LABELS, SPEECH_FAMILIES, SPLITS, V2DatasetBuilder


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    builder = V2DatasetBuilder(config, args.source_manifest)
    report: dict[str, object] = {}
    processed = 0
    for split in SPLITS:
        durations: list[float] = []
        leading: list[float] = []
        trailing: list[float] = []
        placement: dict[str, int] = {"front": 0, "middle": 0, "back": 0}
        stream_index = 0
        for label in LABELS:
            count = int(config["counts"][split][label])
            for index in range(count):
                record_id = f"{split}-{label}-{index:06d}"
                rng = np.random.default_rng(_stable_seed(f"{config['seed']}:{record_id}"))
                if label == "ambient":
                    durations.append(float(rng.uniform(1.5, 5.0)))
                    processed += 1
                    if processed % 1000 == 0:
                        print(f"DISTRIBUTION HEARTBEAT records={processed}", flush=True)
                    continue
                family = SPEECH_FAMILIES[index % len(SPEECH_FAMILIES)]
                candidates = builder._candidates(split, label, family)
                source = candidates[(index // len(SPEECH_FAMILIES)) % len(candidates)]
                raw = builder._load_source(source)
                placed, start, end, position = builder._place(raw, stream_index, rng)
                stream_index += 1
                durations.append(len(placed) / 16000.0)
                leading.append(start / 16000.0)
                trailing.append((len(placed) - end) / 16000.0)
                placement[position] += 1
                processed += 1
                if processed % 1000 == 0:
                    print(f"DISTRIBUTION HEARTBEAT records={processed}", flush=True)
        report[split] = {
            "duration_seconds": summary(durations),
            "leading_seconds": summary(leading),
            "trailing_seconds": summary(trailing),
            "phrase_placement": placement,
        }
        print(f"DISTRIBUTION SIMULATION split={split} {report[split]}", flush=True)
    means = [report[split]["duration_seconds"]["mean"] for split in SPLITS]
    report["duration_mean_max_gap_seconds"] = round(max(means) - min(means), 6)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"DISTRIBUTION SIMULATION COMPLETE max_mean_gap="
        f"{report['duration_mean_max_gap_seconds']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
