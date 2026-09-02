"""Bounded synthetic-provider smoke for the 12K base/variant pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from build_multikws_12k_dataset import atomic_json, run_generation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-config", type=Path,
        default=PROJECT_ROOT / "configs/multikws/teacher_six_formal_12k.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing smoke: {output}")
    output.mkdir(parents=True)
    config = json.loads(args.formal_config.resolve().read_text(encoding="utf-8"))
    config["dataset"].update({
        "dataset_id": "teacher_six_multikws_v2_group_pipeline_smoke",
        "manifest_profile": "group_pipeline_smoke",
        "experiment_stage": "pipeline_regression_smoke",
        "output_root": str(output),
        "base_counts": {
            "wakeword_per_keyword": {"train": 2, "validation": 2, "test": 2},
            "ordinary_background": {"train": 2, "validation": 2, "test": 2},
            "hard_negative": {"train": 2, "validation": 2, "test": 2},
        },
        "ambient_effective_counts": {"train": 1, "validation": 1, "test": 1},
    })
    config["training"]["effective_train_samples"] = 49
    config_path = output / "SMOKE_CONFIG.json"
    atomic_json(config_path, config)
    calls: list[str] = []

    def fake(job, _text, _voice, _speed):
        calls.append(str(job["base_sample_id"]))
        source_offset = 50 if job["speech_source"] == "voxcpm15" else 0
        frequency = 180 + int(job["index"]) % 30 + source_offset
        timeline = np.arange(8000, dtype=np.float32) / 16000.0
        audio = (0.08 * np.sin(2 * np.pi * frequency * timeline)).astype(np.float32)
        return audio, {"provider": job["speech_source"], "synthetic_pipeline_smoke": True}

    providers = {"kokoro": fake, "voxcpm15": fake}
    interrupted_as_planned = False
    try:
        run_generation(
            config_path, output_root=output, synthesizers=providers, stop_after_base=5
        )
    except KeyboardInterrupt:
        interrupted_as_planned = True
    if not interrupted_as_planned:
        raise RuntimeError("Smoke did not exercise the interruption path")
    info = run_generation(
        config_path, output_root=output, synthesizers=providers, resume=True
    )
    manifest = json.loads((output / "DatasetManifest.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, object]]] = {}
    for record in manifest["records"]:
        groups.setdefault(str(record["base_sample_id"]), []).append(record)
    train_speech_groups = [
        siblings for siblings in groups.values()
        if siblings[0]["split"] == "train" and siblings[0]["speech_source"] != "procedural_ambient"
    ]
    variants_distinct = all(len({row["sha256"] for row in siblings}) == 3
                            for siblings in train_speech_groups)
    report = {
        "schema": "wakeword-studio.multikws-12k-pipeline-smoke/v1",
        "status": "PASS",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "planned_base_speech": len(manifest["base_records"]),
        "actual_synthesis_calls": len(calls),
        "unique_synthesis_calls": len(set(calls)),
        "effective_samples": len(manifest["records"]),
        "split_counts": manifest["split_counts"],
        "base_source_counts": manifest["base_source_counts"],
        "BASE_GROUP_SPLIT_LEAKAGE": manifest["base_group_split_leakage"],
        "TRAIN_VARIANTS_DISTINCT": variants_distinct,
        "RESUME_REUSED_COMPLETED_BASE": len(calls) == len(set(calls)) == len(manifest["base_records"]),
        "TEST_READ": info["TEST_READ"],
    }
    if not all((variants_distinct, report["RESUME_REUSED_COMPLETED_BASE"],
                report["BASE_GROUP_SPLIT_LEAKAGE"] == 0, report["TEST_READ"] is False)):
        raise RuntimeError(f"12K pipeline smoke failed: {report}")
    atomic_json(output / "PIPELINE_SMOKE_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
