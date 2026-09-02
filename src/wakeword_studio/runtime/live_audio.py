"""Thin, optional sounddevice adapter for 16 kHz mono PCM16 capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..audio import TARGET_CHANNELS, TARGET_SAMPLE_RATE_HZ


@dataclass(frozen=True, slots=True)
class MicrophoneDevice:
    index: int
    name: str
    channels: int
    default_sample_rate: float

    @property
    def display_name(self) -> str:
        return f"{self.index}: {self.name}"


def _sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Microphone support requires sounddevice in the selected Python environment"
        ) from exc
    return sd


def list_input_devices() -> list[MicrophoneDevice]:
    sd = _sounddevice()
    devices = []
    for index, row in enumerate(sd.query_devices()):
        channels = int(row["max_input_channels"])
        if channels > 0:
            devices.append(
                MicrophoneDevice(
                    index=index,
                    name=str(row["name"]),
                    channels=channels,
                    default_sample_rate=float(row["default_samplerate"]),
                )
            )
    return devices


class MicrophoneCapture:
    def __init__(self, frame_callback: Callable[[np.ndarray], None], *, frame_ms: int = 30):
        self.frame_callback = frame_callback
        self.frame_ms = int(frame_ms)
        self.frame_samples = TARGET_SAMPLE_RATE_HZ * self.frame_ms // 1000
        self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self, device: int | None = None) -> None:
        if self.running:
            raise RuntimeError("Microphone capture is already running")
        sd = _sounddevice()

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            del time_info
            if status:
                # PortAudio status is non-fatal; the next valid frame remains usable.
                pass
            if frames != self.frame_samples:
                return
            self.frame_callback(np.frombuffer(bytes(indata), dtype="<i2").copy())

        self._stream = sd.RawInputStream(
            samplerate=TARGET_SAMPLE_RATE_HZ,
            blocksize=self.frame_samples,
            channels=TARGET_CHANNELS,
            dtype="int16",
            device=device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
