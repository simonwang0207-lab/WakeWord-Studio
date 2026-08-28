# LiveKit Embedded Wakeword / RepCNN Phase 0 compatibility evidence

Assessment date: 2026-08-28

## Measured result

- Upstream commit: `726403d432d11d3a4327f0a367d43db78f4e3d78` (2026-03-20)
- License: Apache-2.0
- Official CLI stages run: `augment`, `train`, `export --quantize`, `eval`
- Training graph parameters: 2,241; fused inference classifier: 1,473
- Float model: 12,288 bytes; SHA-256 `c38b4052129c555304d8082f13bb37bd41f78970b9326bbdc11fbc4fa0238bcf`
- Official `--quantize` model: 12,304 bytes, but **float32 input/output and float operators**; it is not a full-integer model.
- Full-integer calibration add-on: 11,856 bytes (11.578 KiB), INT8 input/output; SHA-256 `bf8718a8cb53d835e1756961325c7aea91d897f179c202c08ffe0671b0cfd234`
- Estimated export arithmetic: 5.639 M MACs per 99-frame classifier invocation.

The actual full-INT8 operators are `RESHAPE`, `CONV_2D`,
`DEPTHWISE_CONV_2D`, `ADD`, `MEAN`, `FULLY_CONNECTED`, and `LOGISTIC`.
All seven are registered in ESPHome's current 20-op `micro_wake_word` resolver.

## ESP32-S3 evidence grade

Overall: **UNCERTAIN**.

- Operator compatibility is **VERIFIED at source level** against ESPHome dev.
- General TFLM/ESP-NN support for ESP32-S3 is available from Espressif.
- The LiveKit README itself says microWakeWord/ESPHome compatibility is a TODO and
  “has not been verified yet.” No physical-board firmware example for this artifact
  was run in Phase 0.
- Its model input is a complete `(1, 99, 40)` stateless window. ESPHome's existing
  streaming wrapper can technically view dimension 1 as a stride, but that would
  naturally invoke only after collecting 99 feature frames, not provide a score on
  every 20 ms hop. A dedicated rolling-window runtime is needed for dense score cadence.
- At 5.639 M MACs per invocation, invoking three binary models on every 20 ms hop is
  not realistic without a measured, lower cadence and on-board latency benchmark.

Primary sources:

- https://github.com/livekit/embedded-wakeword
- https://raw.githubusercontent.com/esphome/esphome/dev/esphome/components/micro_wake_word/streaming_model.cpp
- https://github.com/espressif/esp-tflite-micro

## Added system-constraint assessment

- Raw score available: **YES**. The TFLite output is a sigmoid probability before any wrapper threshold.
- Streaming score available: **YES on PC / PARTIAL for embedded**. The upstream PC `StreamingWakeWordModel` returned 76 rolling-window scores in a 3.5-second test. Embedded cadence is not verified.
- Threshold bypass: **YES**; direct interpreter output was measured.
- Multi-keyword support: **EASY as parallel binary models**. The upstream PC API loads a list of classifiers, computes the frontend once, and returns a dictionary of independent scores. There is no native multi-class/multi-head training path.
- Arbitration: convert calibrated binary probabilities to logits and calibrate per model before temperature softmax, or use winner-plus-margin. Softmax over the three raw probabilities is not mathematically justified.
- Three-model cost: approximately 3x classifier flash/arena/invocations; this tiny artifact is only about 35 KiB total model files, but compute is approximately 16.9 M MACs per arbitration instant.
- Local dataset import: **EASY–MODERATE**. Normal 16 kHz WAVs only need to be copied into four official split folders, then `augment` creates the expected `.npy` caches. This adapter was actually exercised with 100 Kokoro WAVs.
- Age diversity: **PARTIAL**. The reused Kokoro Chinese model has many speakers, but no verified child/young/middle-aged/elderly labels. Real age coverage still needs labelled public or recorded speech.
- Noise support: **GOOD**. The official augmentation command ran. A direct background mix test measured 4.999, 9.999, and 14.999 dB for configured 5/10/15 dB SNR targets.
- Gated activation: **YES WITH PRE-ROLL**. Because the classifier is stateless over ~2 seconds, keep a 1–2 second raw-audio ring buffer; after three VAD speech frames, feed the history and continue rolling windows.
- DetectionLogic compatibility: **4/5 on PC, 3/5 pending embedded cadence**. L1/L2 use continuous scores; L3 and L5 are wrapper logic; L4 needs per-model calibration. The VAD three-frame gate and wake-score N-frame confirmation remain separate settings.

## Chinese generation limitation

The pinned embedded repository's generator imports PyTorch and expects
`data/piper/en-us-libritts-high.pt`, uses CMUDict/ARPAbet, and is English-specific.
Its official `generate` command is therefore not a valid Chinese generator. Phase 0
used the already human-approved Apache-2.0 Kokoro Chinese data instead. The first CLI
attempt also exposed a Windows UTF-8 bug in `load_config`, locally fixed by opening
YAML with `encoding="utf-8"`.
