"""Snapshot a read-only safety guard for the separately running B2 process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ctypes
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/qingxiaojia/repcnn_performance_v2_fasttrack/formal/user_run_01",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "phase6/artifacts/B2_RUNNING_GUARD.json",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "TRAINING_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    pid = int(status["pid"])
    alive = process_alive(pid)
    safe = (
        status.get("status") == "RUNNING"
        and status.get("test_loaded") is False
        and alive
    )
    report = {
        "schema": "wakeword-studio.b2-running-guard/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "status_path": str(status_path),
        "status_sha256_at_snapshot": sha256(status_path),
        "pid": pid,
        "process_alive": alive,
        "training_status": status.get("status"),
        "current_step": status.get("current_step"),
        "planned_steps": status.get("planned_steps"),
        "last_update": status.get("last_update"),
        "test_loaded": status.get("test_loaded"),
        "protection_policy": {
            "kill_or_restart": False,
            "modify_phase4_trainer_or_config": False,
            "modify_run_checkpoints": False,
            "checkpoint_access": "read_only",
        },
        "guard_passed": safe,
    }
    if not safe:
        raise RuntimeError(f"B2 running guard failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
