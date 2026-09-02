"""Serialized wake-confirmation playback with observable lifecycle events."""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class PlaybackRequest:
    episode_id: str
    path: Path


def play_audio_blocking(path: Path) -> None:
    """Play one file to completion; callers serialize concurrent requests."""

    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path.resolve()), winsound.SND_FILENAME)
        return

    import sounddevice as sd
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    sd.play(audio, rate, blocking=True)


class WakePlaybackQueue:
    """FIFO playback: every accepted final WAKE request is played exactly once."""

    def __init__(
        self,
        log: Callable[[str], None],
        *,
        player: Callable[[Path], None] = play_audio_blocking,
    ) -> None:
        self._log = log
        self._player = player
        self._queue: queue.Queue[PlaybackRequest | None] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self.request_count = 0
        self.started_count = 0
        self.playback_count = 0
        self.skipped_count = 0
        self._worker = threading.Thread(
            target=self._run,
            name="wake-playback",
            daemon=True,
        )
        self._worker.start()

    def request(self, path: Path, *, episode_id: str) -> bool:
        resolved = Path(path).resolve()
        with self._lock:
            if self._closed:
                self.skipped_count += 1
                self._log(f"PLAYBACK_SKIPPED episode={episode_id} reason=queue_closed")
                return False
            if not resolved.is_file():
                self.skipped_count += 1
                self._log(f"PLAYBACK_SKIPPED episode={episode_id} reason=file_not_found")
                return False
            self.request_count += 1
        self._log(f"PLAYBACK_REQUESTED episode={episode_id} path={resolved}")
        self._queue.put(PlaybackRequest(episode_id=episode_id, path=resolved))
        return True

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            try:
                if request is None:
                    return
                with self._lock:
                    self.started_count += 1
                self._log(f"PLAYBACK_STARTED episode={request.episode_id}")
                try:
                    self._player(request.path)
                except Exception as exc:  # hardware/backend failures remain visible
                    with self._lock:
                        self.skipped_count += 1
                    self._log(
                        "PLAYBACK_SKIPPED "
                        f"episode={request.episode_id} reason=playback_error:{type(exc).__name__}"
                    )
                else:
                    with self._lock:
                        self.playback_count += 1
                    self._log(f"PLAYBACK_FINISHED episode={request.episode_id}")
            finally:
                self._queue.task_done()

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            with self._lock:
                completed = self.playback_count + self.skipped_count
                requested = self.request_count
            if self._queue.unfinished_tasks == 0 and completed >= requested:
                return True
            time.sleep(0.005)
        return False

    def close(self, *, wait: bool = False, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        if wait:
            self._worker.join(timeout=float(timeout))

