# WakeWord Studio — Phase 0 Plan

## Scope and decision target

This phase is a technical-selection POC only. It will not build the production GUI, complete ESP32 firmware, or a large training platform.

Candidates to be tested with real code and artifacts:

1. microWakeWord / MixedNet
2. LiveKit Embedded Wakeword / RepCNN
3. Google Research `kws_streaming` / DS-TC-ResNet

The decision will select two models for offline Chinese custom wake words, automated synthetic-data generation and training, INT8 TFLite/TFLite Micro deployment on ESP32-S3, with at least one final model realistically targeting 50–100 KiB.

## Machine reconnaissance (2026-08-28)

- Working directory: `F:\ZJU_intership\task\4`
- Project directory: `F:\ZJU_intership\task\4\WakeWord-Studio`
- Git status at start: current working directory was not a Git repository
- OS: Microsoft Windows build 26200, x64
- CPU: 13th Gen Intel Core i7-13700H; 20 logical processors
- RAM: 15.80 GiB total; 2.34 GiB available at inspection time (85% load)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- GPU VRAM: 8188 MiB reported by `nvidia-smi`
- NVIDIA driver: 591.59
- CUDA compiler (`nvcc`): not found in PATH
- Python on PATH: 3.12.7 at `D:\anaconda12.7\python.exe`
- Python launcher (`py`): not found
- Conda: 24.11.0
- uv: 0.11.24
- pip: 24.2 (base Python 3.12)
- Git: 2.52.0.windows.1
- CMake: not found in PATH
- TensorFlow in base environment: not installed
- PyTorch in base environment: installed
- Disk free: C: 10.59 GiB; D: 88.90 GiB; E: 49.97 GiB; F: 305.23 GiB

The base Conda/Python environment will not be modified. Candidate-specific dependencies will be isolated under `.envs/`.

Verified isolated environments so far:

- `.envs/microwakeword`: CPython 3.10.20, TensorFlow 2.21.0, CPU device only; microWakeWord imports successfully
- `.envs/kokoro`: CPython 3.10.20, Kokoro 0.9.4, PyTorch 2.13.0; Chinese TTS inference ran on CPU
- `.envs/livekit`: CPython 3.11.5, TensorFlow 2.21.0 CPU, LiveKit Embedded Wakeword editable install at commit `726403d432d11d3a4327f0a367d43db78f4e3d78`

Windows CIM and `systeminfo` were denied by the managed sandbox; CPU/RAM were subsequently obtained through read-only registry and Win32 memory APIs. Native-Windows TensorFlow 2.21 reported that GPU support is unavailable, even though `nvidia-smi` detects the RTX 4060.

## Execution plan and evidence

For each candidate, preserve actual commands, logs, errors, exported models, byte sizes, SHA256, inference outputs, and TFLite operator lists.

### A. microWakeWord / MixedNet

- Pin upstream commit and record license/runtime requirements.
- Establish a legal Chinese TTS path for “你好，青小甲”.
- Generate only a small Phase 0 dataset.
- Stop after the first successful Chinese batch for user listening approval.
- Run minimal training, checkpoint/export, full INT8 conversion, positive/negative PC inference, operator inspection, and size analysis.

### B. LiveKit Embedded Wakeword / RepCNN

- Pin upstream commit and inspect maintenance/license/dependencies.
- Exercise the official generate/augment/train/export/eval workflow where practical.
- Run minimal training, INT8 TFLite export, inference, operator inspection, and distinguish architecture claims from real ESP32-S3 firmware evidence.

### C. Google Research `kws_streaming` / DS-TC-ResNet

- Pin upstream commit and inspect license/dependencies.
- Try a minimal custom wake/unknown/silence data path.
- Test non-streaming training to streaming TFLite conversion and INT8 quantization as far as the isolated Windows environment reasonably permits.
- Record extra engineering required relative to A/B.

## Stop conditions

- No single training run should be expected to exceed one hour; reduce samples/epochs/channels first.
- Do not retry the same dependency or network failure more than three reasonable variants.
- Before any download expected to exceed about 2 GiB, report its purpose, size, and destination to the user.
- After the first successful Chinese TTS batch, update `phase0/NEED_USER_ACTION.md`, provide 5–10 WAV paths, and wait for the user to confirm pronunciation/quality.

## Completion checklist

- [x] microWakeWord actual POC
- [x] RepCNN actual POC
- [x] DS-TC-ResNet compatibility investigation and best-effort POC
- [x] At least two actual TFLite exports
- [x] Actual model byte sizes and SHA256
- [x] At least two successful PC inference paths
- [x] TFLite operators inspected
- [x] ESP32-S3 compatibility graded by evidence
- [x] Chinese TTS actually generated and user-audited
- [x] `PHASE0_REPORT.md` completed
- [x] Final recommended pair justified by POC evidence

## Added final-system constraints (teacher addendum, 2026-08-28)

The Phase 0 comparison must also evaluate, for every candidate:

- raw unthresholded score availability and score cadence;
- single multi-class, multi-output, or parallel-binary multi-keyword deployment;
- calibration-aware arbitration (never treat `softmax(binary probabilities)` as automatically probabilistic);
- ordinary local dataset-folder import through a future common `DatasetAdapter`;
- speaker/age/prosody diversity without claiming pitch-shifted adult speech is real child speech;
- real background mixing at configured SNR;
- compatibility with Energy/adaptive threshold → WebRTC VAD → 3 speech-frame pre-gate;
- use of approximately 1–2 seconds of pre-roll where a window/state must be warmed;
- the five post-model layers: consecutive wake score, peak/background ratio, cooldown, multi-keyword arbitration, and before/after silence;
- separate configuration for `vad_consecutive_frames = 3` and `wake_consecutive_frames = N`.

The future abstraction names are `WakeWordBackend` / `WakeWordEngine`. The wake response text is “我醒来了” and the reserved asset path is `assets/i_am_awake.wav`; neither playback nor the full runtime is implemented in Phase 0.

Additional completion conditions:

- [x] At least two candidates verified to expose usable raw scores
- [x] All candidates have a multi-wake-word route and resource-cost analysis
- [x] Dataset-folder import difficulty assessed for all candidates
- [x] Age-diversity generation limits assessed without unsupported claims
- [x] Actual configured-SNR noise mixing demonstrated
- [x] VAD-gated activation and pre-roll compatibility assessed
- [x] Five-layer DetectionLogic compatibility scored
- [x] Final report contains all added comparison fields
