"""Launch WakeWord Studio in a standalone local desktop window."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from wakeword_studio.webapp import serve


PROJECT_ROOT = Path(__file__).resolve().parent


def find_edge() -> Path:
    candidates = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    command = shutil.which("msedge")
    if command:
        candidates.insert(0, Path(command))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 Microsoft Edge WebView 运行环境，请先安装或修复 Microsoft Edge")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def wait_for_server(port: int, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("本地 WakeWord Studio 服务未能按时启动")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="只检查桌面运行环境，不打开窗口")
    args = parser.parse_args()
    edge = find_edge()
    if args.check:
        print(f"WAKEWORD_STUDIO_DESKTOP_READY edge={edge}")
        return
    port = args.port or free_port()
    worker = threading.Thread(
        target=serve,
        args=(PROJECT_ROOT, PROJECT_ROOT / "configs/demo/teacher_demo.yaml"),
        kwargs={"port": port, "open_browser": False},
        daemon=True,
    )
    worker.start()
    wait_for_server(port)
    profile = Path(tempfile.mkdtemp(prefix="wakeword-studio-desktop-"))
    try:
        subprocess.run(
            [
                str(edge), f"--app=http://127.0.0.1:{port}",
                f"--user-data-dir={profile}", "--start-maximized",
                "--no-first-run", "--disable-background-mode",
            ],
            check=False,
        )
    finally:
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
