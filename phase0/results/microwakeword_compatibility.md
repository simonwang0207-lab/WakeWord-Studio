# microWakeWord / MixedNet ESP32-S3 compatibility evidence

Assessment date: 2026-08-28

## Result

- Model-format evidence: **PASS**. The exported model is a TFLite streaming model with `int8` input and `uint8` output, matching ESPHome's current `micro_wake_word` loader checks.
- Operator-set evidence: **PASS (source-level)**. Every operator found in the exported INT8 model is registered by ESPHome's current 20-op streaming resolver.
- Physical ESP32-S3 execution: **NOT TESTED**. No board was connected during Phase 0, so the overall deployment rating is **PROBABLE**, not confirmed.

## Measured artifact

- INT8 model: `phase0/artifacts/models/microwakeword_smoke/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
- Size: 52,944 bytes (51.703 KiB)
- SHA-256: `b8cb6ccbccafe81baa72495156145301c9472ea1e20f22e9bb8cec94ca6de76e`
- Input: signed INT8, streaming feature tensor with 40 microfrontend channels
- Output: unsigned UINT8 scalar probability

## Operator comparison

Exported unique operators:

`ASSIGN_VARIABLE`, `CALL_ONCE`, `CONCATENATION`, `CONV_2D`,
`DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `LOGISTIC`, `QUANTIZE`,
`READ_VARIABLE`, `RESHAPE`, `SPLIT_V`, `STRIDED_SLICE`, `VAR_HANDLE`.

ESPHome dev branch's current `StreamingModel::register_streaming_ops_()` registers all
13 of those operators. It also registers `MUL`, `ADD`, `MEAN`, `AVERAGE_POOL_2D`,
`MAX_POOL_2D`, `PAD`, and `PACK`, which this artifact does not need.

Primary source:
https://raw.githubusercontent.com/esphome/esphome/dev/esphome/components/micro_wake_word/streaming_model.cpp

ESPHome also verifies a 3-D INT8 input whose last dimension is the 40-channel
preprocessor feature size, and a 1x1 UINT8 output. Those checks match this export.

Espressif maintains an ESP32/ESP32-S3 port of TensorFlow Lite Micro with ESP-NN
optimizations, but that general platform support is not proof that this exact model's
tensor arena allocation and invocation succeed on a physical board.

Primary source:
https://github.com/espressif/esp-tflite-micro

## Remaining board risks

1. Tensor-arena usage was not measured on-device. ESPHome now probes the configured
   manifest arena size, then tries 1.5x and 2x, but the manifest still needs a sensible
   initial value.
2. Operator versions and ESP-NN kernel behavior can vary with the ESPHome/TFLM pin.
3. Latency, PSRAM/internal-RAM placement, microphone frontend continuity, and false
   activations under real acoustic conditions remain unverified.
4. The 60-step smoke model does not separate the two held-out examples meaningfully;
   compatibility success must not be confused with wake-word quality.
