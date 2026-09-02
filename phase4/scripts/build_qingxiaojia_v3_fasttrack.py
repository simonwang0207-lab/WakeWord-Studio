"""Build a metadata-only Train/Validation view for Model B v2 fast-track."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.dataset.manifest import DatasetManifest, sha256_file  # noqa: E402
from wakeword_studio.training.repcnn_fasttrack import (  # noqa: E402
    REQUIRED_SOURCES,
    positive_is_eligible,
)
from wakeword_studio.training.streaming_windows import plan_streaming_window  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def grouped(rows: list[object], field: str) -> dict[str, int]:
    if field == "source":
        values = (row.speaker.source for row in rows)  # type: ignore[attr-defined]
    else:
        values = (row.speaker.speaker_id for row in rows)  # type: ignore[attr-defined]
    return dict(sorted(Counter(values).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets/projects/qingxiaojia_v2/DatasetManifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/projects/qingxiaojia_v3_fasttrack",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--maximum-phrase-ms", type=float, default=1950.0)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    source = DatasetManifest.load(source_path)
    source_root = Path(source.root).resolve()
    kept = []
    report: dict[str, object] = {
        "schema": "wakeword-studio.repcnn-fasttrack-data-view/v1",
        "created_at": utc_now(),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "maximum_positive_phrase_ms": args.maximum_phrase_ms,
        "allowed_splits": ["train", "validation"],
        "test_loaded": False,
        "wav_policy": "read_qingxiaojia_v2_paths_in_place",
        "splits": {},
    }
    for split in ("train", "validation"):
        positives = [row for row in source.records if row.split == split and row.label == "positive"]
        eligible = []
        excluded = []
        for row in positives:
            window = plan_streaming_window(
                record_id=row.record_id,
                label=row.label,
                duration_seconds=float(row.duration_seconds or 0.0),
                phrase_start_ms=row.acoustic.phrase_start_ms,
                phrase_end_ms=row.acoustic.phrase_end_ms,
                phrase_placement=row.acoustic.phrase_placement,
                window_ms=2000.0,
                seed=args.seed,
            )
            if positive_is_eligible(
                phrase_start_ms=row.acoustic.phrase_start_ms,
                phrase_end_ms=row.acoustic.phrase_end_ms,
                full_phrase_contained=window.full_phrase_contained,
                maximum_phrase_ms=args.maximum_phrase_ms,
            ):
                eligible.append(row)
            else:
                excluded.append(row)
        for required_source in REQUIRED_SOURCES:
            if not any(row.speaker.source == required_source for row in eligible):
                raise RuntimeError(f"{split} has zero eligible positives for {required_source}")
        kept.extend(eligible)
        kept.extend(
            row
            for row in source.records
            if row.split == split and row.label != "positive"
        )
        report["splits"][split] = {  # type: ignore[index]
            "eligible_positive": len(eligible),
            "excluded_positive": len(excluded),
            "eligible_by_source": grouped(eligible, "source"),
            "excluded_by_source": grouped(excluded, "source"),
            "eligible_by_speaker": grouped(eligible, "speaker"),
            "excluded_by_speaker": grouped(excluded, "speaker"),
        }

    view = DatasetManifest(
        wake_word=source.wake_word,
        records=kept,
        source_kind=source.source_kind,
        root=str(source_root),
        generator={
            "name": "qingxiaojia_v3_fasttrack_metadata_view",
            "source_manifest": str(source_path),
            "source_manifest_sha256": report["source_manifest_sha256"],
            "positive_eligibility": (
                f"phrase_duration_ms <= {args.maximum_phrase_ms:g} and complete phrase "
                "contained in the 2000 ms RepCNN input"
            ),
            "invalid_positive_policy": "exclude_without_truncation_or_relabeling",
            "test_loaded": False,
        },
        coverage_policy=source.coverage_policy,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = view.save(output_dir / "DatasetManifest.json")
    manifest_errors = view.validate(manifest_path, check_files=False)
    if manifest_errors:
        raise RuntimeError("Fast-track manifest validation failed: " + "; ".join(manifest_errors[:10]))
    report["output_manifest"] = str(manifest_path)
    report["output_manifest_sha256"] = sha256_file(manifest_path)
    report["output_records"] = len(kept)
    report_path = output_dir / "FASTTRACK_DATA_VIEW.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
