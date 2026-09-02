# Phase 3A — Model B RepCNN implementation and preflight

Model A remained frozen. Every Phase 3A data path was restricted to the frozen
`qingxiaojia_v2` Train and Validation records; held-out Test audio was not loaded.

## Native implementation audit

1. **Architecture:** LiveKit Embedded Wakeword RepCNN xxlarge: initial 3×3 Conv2D,
   11 RepDS blocks with dilation cycle `[1,2,4]`, global average pooling, and one
   sigmoid output. Training-time DW3×3/DW1×1/identity BN branches are fused into
   one DW3×3 plus PW1×1 pair per block for deployment.
2. **Input feature:** TFLM microfrontend PCAN/log filterbank features, scaled with
   `uint16 × 0.0390625` to float32.
3. **Input tensor:** `(1,99,40)`.
4. **Window:** 30 ms frontend window, 20 ms hop, 2.0 s classifier context.
5. **Streaming:** stateless rolling 99-frame buffer; after warm-up the host may
   invoke once per new 20 ms feature hop.
6. **Native loss:** clip-level focal binary cross-entropy (`gamma=2`) with optional
   mixup, label smoothing, SpecAugment, and scheduled negative weighting.
7. **Output:** one raw sigmoid probability; no threshold is embedded.
8. **Parameters:** 64,257 training; 53,505 after RepCNN/BN fusion.
9. **Quantization:** representative-data TFLite conversion restricted to built-in
   INT8 ops, with INT8 input and output.
10. **Compatibility:** valid TFLite and TFLM-compatible operator set; physical-board
    compatibility and cadence remain unverified.
11. **Ops:** `RESHAPE`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `MEAN`,
    `FULLY_CONNECTED`, `LOGISTIC`.
12. **Why ~110 KiB:** the fused graph has 53,505 scalar parameters; INT8 weights,
    per-channel quantization tables, tensors and FlatBuffer/operator metadata produce
    the measured 112,816-byte artifact. Size is small relative to compute: the current
    converter reports about 210.102 M MACs per 99-frame invocation, superseding the
    old Phase 0 compute estimate.

RepCNN is naturally clip-level because global average pooling removes the time axis
before the single sigmoid. Model B therefore keeps native clip-level focal loss;
`negative`, `hard_negative`, and `ambient` are explicit target-0 providers. The
microWakeWord v3 frame/hard-max objective was not copied.

## Preflight results

- Dataset adapter: PASS; reads WAVs in place and retains speaker/source/text/SNR,
  phrase interval, split, target, and deterministic window metadata.
- Source/label audit: PASS. Kokoro and VoxCPM1.5 are each 50% of positive,
  ordinary-negative, and hard-negative Train records. Procedural ambient is recorded
  as a semantic ambient-only exception. Batch is 16 positive / 4 ordinary negative /
  8 hard-negative / 4 ambient.
- Target audit: PASS; 10 positive, 10 ordinary negative, 10 hard-negative, 5 ambient;
  errors = 0.
- Tiny overfit: PASS after 150 steps and one no-gradient BN calibration pass. Score
  means changed from 0.5 to positive 0.98118, ordinary negative 0.03165,
  hard-negative 0.02957, ambient 0.02418.
- Full INT8: PASS; 112,816 bytes (110.172 KiB); input int8 scale 0.1014093161,
  zero-point -128; output int8 scale 0.00390625, zero-point -128. Dequantization is
  `real_score = scale * (raw - zero_point)`.
- Benchmark: PASS; 180 CPU steps; mean 1.42456 s/step, P95 1.43128 s/step;
  first/last 20-step focal-loss means 0.16649 → 0.01648; Validation output remained
  nonconstant; checkpoint and strict resume passed with zero weight/prediction error.
- Peak process RAM: 3.430 GiB. Native Windows TensorFlow did not expose a GPU, so
  TensorFlow VRAM use was 0; the NVIDIA system snapshot was 1,911/8,188 MiB used by
  the machine overall.

Formal training is configured for 7,200 total steps (6,000 primary + 600 refinement
+ 600 fine-tuning), with 500 stabilization steps before deterministic mixup and
SpecAugment, full Validation-only checkpoint selection, early stopping, and exact
all-variable/optimizer resume. The runner is gated by `--allow-formal-training` and
was not launched during Phase 3A.
