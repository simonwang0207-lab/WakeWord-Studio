# Phase 2F — qingxiaojia_v2 training preflight

Status: **READY — awaiting `START V2 FORMAL TRAINING`**. No formal training was
started during this phase.

## Frozen inputs

- Dataset: `datasets/projects/qingxiaojia_v2`
- Manifest SHA-256: `50e3857e9941d910b640039dd70e73c39e331cc368816c378849ca9774f1973c`
- Samples: 15,200 total; Train 12,000; Validation 1,600; held-out Test 1,600.
- Speaker and source-group leakage remain zero. The exact split table is stored
  in `phase2/results/qingxiaojia_v2_freeze.json`.
- The feature store contains only Train and Validation (13,600 records). Neither
  v2 Test nor the immutable v1 external Test was loaded.

## Frozen training plan

The v2 config is `configs/models/microwakeword_tiny_v2.yaml`. It retains the v1
microWakeWord / MixedNet Tiny network exactly: 19,697 parameters and the same
3-second frontend shape. Planned training is 15,000 steps with batch size 64,
equivalent to 960,000 replacement-sampled examples or 80 nominal passes over
the 12,000-row Train split. This is enough repeated coverage for the deliberately
diverse v2 data without increasing steps merely because v2 is larger. Early
stopping remains enabled after a 4,000-step warm-up.

Validation and checkpoint intervals are both 500 steps. Validation is the only
split permitted for checkpoint selection, early stopping, and threshold
selection. v2 Test is evaluated only after model and threshold freeze. The v1
external Test is evaluated once after the final model freeze and cannot tune the
threshold.

## Streaming window alignment

The preflight checks all 13,600 Train/Validation rows and writes 30 human-readable
audit rows: 10 positive, 10 ordinary negative, and 10 hard-negative. Results:

- 3,380 positive rows contain the complete annotated phrase interval.
- 20 slow Validation positives have a phrase span longer than the unchanged
  3-second model context (maximum 3.827 seconds). They use the causal terminal
  decision window ending at `phrase_end_ms`, so the effective final 3 seconds are
  retained without changing model architecture.
- All 10,200 negative-class rows have no target phrase interval.
- Alignment failures: zero.

The audit is at
`runs/qingxiaojia/microwakeword_tiny_v2/preflight/streaming_window_alignment_audit.csv`.

A rejected 4-second preflight attempt is intentionally preserved as guard
evidence. It stopped at step zero when the parameter-count check detected 20,225
parameters. The accepted 3-second benchmark restored the exact 19,697-parameter
architecture.

## 150-step benchmark and resume probe

Benchmark directory:
`runs/qingxiaojia/microwakeword_tiny_v2/20260829T150007Z_preflight_benchmark_150`

- Selected device: TensorFlow CPU. Native-Windows TensorFlow 2.21 does not expose
  the RTX 4060 as a TensorFlow GPU; hardware free VRAM was 6,020 MiB but was not
  used. Peak process working set was 740.426 MiB.
- Mean measured step time: 0.091021 s; P95: 0.100582 s.
- Mean Validation overhead: 0.526036 s.
- Mean checkpoint overhead: 0.130721 s; best-weight save: 0.063293 s.
- Loss first-20 mean: 0.682069; last-20 mean: 0.320698.
- Every loss and gradient norm was finite; optimizer iterations reached 150.
- Validation produced 1,597 distinct rounded scores with standard deviation
  0.012976, so outputs were not constant.
- Training paused after checkpoint 75, then strictly restored the model,
  optimizer, and global step and completed steps 76–150.

At these measured rates, 15,000 steps plus 30 Validation runs, 30 checkpoints,
and expected best-weight saves are estimated at 1,386.9 seconds, about 23 minutes
7 seconds. Actual completion time depends on machine load and early stopping.

Because the accepted architecture and streaming export path are unchanged, the
best current size estimate is the v1 measured 52,840 bytes / 51.602 KiB, within
the required 50–100 KiB interval. This remains an estimate until the frozen v2
checkpoint is exported.

Formal launch remains gated by `phase2/scripts/start_microwakeword_v2_formal.ps1`.
The launcher refuses to run without `-Approved`, starts a hidden independent
local process, and creates `TRAINING_STATUS.json`, `training.log`, checkpoints,
best/last weights, launcher stdout/stderr, heartbeat updates, and a persistent
`RESUME_COMMAND.txt` in the timestamped formal run directory.

