"""PC microphone Phase 1A demo with complete per-frame diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

from wakeword_studio.audio import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE_HZ,
    TARGET_SAMPLE_WIDTH_BYTES,
)
from wakeword_studio.backends.microwakeword import MicroWakeWordBackend
from wakeword_studio.runtime.detection_logic import DetectionConfig, DetectionLogic
from wakeword_studio.runtime.engine import StreamingWakeWordEngine


def play_awake(path: Path) -> None:
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path.resolve()), winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        import sounddevice as sd
        import soundfile as sf

        audio, rate = sf.read(path, dtype="float32")
        sd.play(audio, rate)


def iter_wav(path: Path, frame_samples: int):
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != TARGET_CHANNELS
            or handle.getsampwidth() != TARGET_SAMPLE_WIDTH_BYTES
            or handle.getframerate() != TARGET_SAMPLE_RATE_HZ
        ):
            raise ValueError("Offline WAV must be mono, PCM16, 16 kHz")
        while data := handle.readframes(frame_samples):
            frame = np.frombuffer(data, dtype="<i2")
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)))
            yield frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--awake-wav", type=Path, default=Path("assets/i_am_awake.wav"))
    parser.add_argument("--input-wav", type=Path, help="Deterministic offline test instead of microphone")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--wake-threshold", type=float, default=0.55)
    parser.add_argument("--wake-frames", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1.35)
    parser.add_argument("--post-silence-frames", type=int, default=2)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--no-playback", action="store_true", help="Keep offline CI/tests silent")
    parser.add_argument("--quiet", action="store_true", help="Write JSONL without per-frame stdout")
    args = parser.parse_args()
    backend = MicroWakeWordBackend()
    backend.load(args.model)
    detection = DetectionLogic(
        DetectionConfig(
            wake_threshold=args.wake_threshold,
            consecutive_wake_frames=args.wake_frames,
            peak_background_ratio=args.ratio,
            post_silence_frames=args.post_silence_frames,
        )
    )
    engine = StreamingWakeWordEngine(
        backend, frame_ms=30, pre_roll_seconds=1.5, detection=detection
    )
    frame_samples = TARGET_SAMPLE_RATE_HZ * 30 // 1000
    log_handle = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.jsonl.open("w", encoding="utf-8")

    def consume(frame: np.ndarray) -> None:
        state = engine.process_frame(frame)
        line = json.dumps(state.to_dict(), ensure_ascii=False)
        if not args.quiet:
            print(line, flush=True)
        if log_handle:
            log_handle.write(line + "\n")
            log_handle.flush()
        if state.final_wake_event and not args.no_playback:
            play_awake(args.awake_wav)

    if args.input_wav:
        for frame in iter_wav(args.input_wav, frame_samples):
            consume(frame)
        if log_handle:
            log_handle.close()
        return
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Microphone mode requires: python -m pip install sounddevice") from exc
    print("Listening for ‘你好，青小甲’... Ctrl+C to stop.")
    deadline = time.monotonic() + args.seconds
    with sd.RawInputStream(
        samplerate=TARGET_SAMPLE_RATE_HZ,
        blocksize=frame_samples,
        channels=TARGET_CHANNELS,
        dtype="int16",
    ) as stream:
        while time.monotonic() < deadline:
            data, overflowed = stream.read(frame_samples)
            if overflowed:
                print('{"warning":"microphone overflow"}', flush=True)
            consume(np.frombuffer(data, dtype="<i2"))
    if log_handle:
        log_handle.close()


if __name__ == "__main__":
    main()
