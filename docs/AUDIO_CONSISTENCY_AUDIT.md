# PHASE 1A-STABILIZATION: Audio/Data Consistency Audit

Date: 2026-08-28  
Canonical audio contract: **16,000 Hz, mono, PCM 16-bit WAV**; internal waveform: float32.

## Executive conclusion

The old `DatasetManifest.json` value of 24,000 Hz was not a metadata-only bug. The
referenced Phase 0 Kokoro WAV files are genuinely 24 kHz mono PCM16. However, the
completed 300-step microWakeWord model was trained from mmap features created only
after those WAVs were explicitly converted to 16 kHz. Its training frontend therefore
matches its 16 kHz inference frontend, and the refined model remains technically
reusable for sanity inference.

All formal WakeWord Studio data entry paths now standardize to 16 kHz mono PCM16 before
writing a v2 manifest. The manifest stores both `original_sample_rate_hz` and the actual
standardized `sample_rate_hz`.

## Real WAV header audit

The following records were selected from the old manifest (three per label). Values
come from the RIFF/WAV headers, not filenames or generator metadata.

| Label | Relative source path | Actual Hz | Channels | Width | Samples | Duration (s) | Manifest Hz |
|---|---|---:|---:|---:|---:|---:|---:|
| positive | `positive/positive_000_zf_001_s0.90.wav` | 24000 | 1 | 2 bytes | 60600 | 2.525 | 24000 |
| positive | `positive/positive_001_zf_002_s0.95.wav` | 24000 | 1 | 2 bytes | 60600 | 2.525 | 24000 |
| positive | `positive/positive_002_zf_003_s1.00.wav` | 24000 | 1 | 2 bytes | 58200 | 2.425 | 24000 |
| negative | `negative/negative_011_zf_021_s0.95.wav` | 24000 | 1 | 2 bytes | 60000 | 2.500 | 24000 |
| negative | `negative/negative_012_zf_022_s1.00.wav` | 24000 | 1 | 2 bytes | 85800 | 3.575 | 24000 |
| negative | `negative/negative_013_zf_023_s1.05.wav` | 24000 | 1 | 2 bytes | 57600 | 2.400 | 24000 |
| hard-negative | `negative/negative_000_zf_001_s0.90.wav` | 24000 | 1 | 2 bytes | 52800 | 2.200 | 24000 |
| hard-negative | `negative/negative_001_zf_002_s0.95.wav` | 24000 | 1 | 2 bytes | 52800 | 2.200 | 24000 |
| hard-negative | `negative/negative_002_zf_003_s1.00.wav` | 24000 | 1 | 2 bytes | 45000 | 1.875 | 24000 |

Source root:
`phase0/artifacts/datasets/microwakeword_tts_smoke`.

## 24 kHz provenance and complete data chain

### A. TTS original output

Kokoro produces a 24 kHz waveform. The historical generator records that contract in
`generate_kokoro_smoke_dataset.py` (`SAMPLE_RATE`) and writes the returned array as a
24 kHz PCM16 WAV in `synthesize()`. The observed headers independently confirm it.

### B. Saved generated WAV

The Phase 0 generated WAV is 24 kHz mono PCM16. No resampling happened before that file
was saved.

### C. Augmentation rate

For the 300-step training path, `prepare_microwakeword_features.py:load_16k()` loads and
resamples first. `feature_generator()` then calls `augment()` on that 16 kHz float32
waveform and sends the result to `generate_features_for_clip()`. The augmentation is
stored as mmap features rather than as another WAV, so it is operating at the correct
16 kHz sample timeline.

The current formal generator uses `generate_dataset.py:synthesize()` to resample Kokoro
24 kHz output through the canonical helper before `environment_augment()` and before
writing a 16 kHz PCM16 WAV.

### D. Why the old manifest said 24000

The old `DatasetAdapter.from_generator_manifest()` pointed directly at the Phase 0
source WAV and copied `sample_rate_hz` from the generator manifest. Since the file was
actually 24 kHz, this metadata was truthful but did not represent a standardized formal
dataset. The adapter now writes a separate standardized WAV and derives manifest values
from its verified output header.

### E. Training frontend input for the 300-step model

The 300-step config points to
`phase0/artifacts/features/microwakeword_smoke/{positive,negative}`, not directly to WAV
files. Those mmap features were built at 19:32:41-43 by
`prepare_microwakeword_features.py:load_16k()` followed by
`generate_features_for_clip()`. The 300-step weight was created later at 22:22:11.

The historical implementation performed:

1. `soundfile.read()` of the actual 24 kHz WAV;
2. mono conversion if needed;
3. `scipy.signal.resample_poly(..., 16000 / source_rate)`;
4. float32 clipping;
5. microfrontend feature extraction.

Verdict: option **2 — 24 kHz WAV was automatically resampled to 16 kHz before feature
extraction**. The 300-step model did not treat 24 kHz samples as if they were 16 kHz.

### F. Inference frontend input

`frontends.py:load_inference_audio()` uses the same canonical loader as
`load_training_audio()`. `sanity_microwakeword.py:score_wav()` and
`MicroWakeWordBackend.evaluate()` use that inference entry point. The upstream
microWakeWord frontend consumes 160 samples per 10 ms feature step and its TensorFlow
frontend declares `sample_rate=16000`.

### G. Automatic resampling

Yes. Resampling now exists in two intentional safety layers:

- `audio.py:standardize_wav()` at dataset ingestion, producing persisted 16 kHz WAVs;
- the shared training/inference loader, protecting direct callers that supply a
  non-standard WAV.

Both use the single `TARGET_SAMPLE_RATE_HZ` constant and polyphase resampling.

## Stabilization changes

- Added the canonical audio contract and conversion utilities in
  `src/wakeword_studio/audio.py`.
- Added paired training/inference entry points in `src/wakeword_studio/frontends.py`.
- Updated `DatasetAdapter` to write a separate standardized dataset and refuse to
  overwrite an existing standardized WAV.
- Added `original_sample_rate_hz`; v2 `sample_rate_hz` is taken from the actual output
  header.
- Manifest validation now checks actual rate, metadata rate, channels, and sample width.
- Kept split assignment stratified per label.
- Updated formal generator, sanity inference, runtime defaults, microphone input, and
  acknowledgement generation to use the shared audio contract.
- Declared SciPy as a project dependency; silent fallback resampling is not allowed.

## Tests

The test suite covers:

- 24 kHz WAV to 16 kHz;
- 48 kHz stereo WAV to 16 kHz mono;
- output PCM16;
- manifest/header equality;
- identical training and inference loading;
- per-label train/validation/test distribution;
- existing runtime gates, detection logic, and pre-roll behavior.

Final result: `9 passed`.

## Canonical small sanity dataset

Final dataset:
`phase1/artifacts/datasets/qingxiaojia_generated_sanity_polyphase_16k`.

- 10 positive, 10 negative, 10 hard-negative;
- new synthesis speeds and new deterministic environment augmentation seed;
- all 30 headers verified as 16 kHz, mono, 2-byte PCM;
- all records preserve `original_sample_rate_hz: 24000` and record
  `sample_rate_hz: 16000`;
- per label: 8 train, 1 validation, 1 test;
- manifest validation errors: none.

Two diagnostic datasets were retained but excluded from the final score:

- `qingxiaojia_sanity_16k` overlaps the old training source and was used only to verify
  adapter conversion;
- `qingxiaojia_generated_sanity_16k` used a non-canonical lightweight resampler during
  environment isolation checking.

## Final raw-score sanity

Final model:
`phase1/artifacts/models/microwakeword_qingxiaojia_300_run/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`.

| Label | Count | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|---:|
| positive | 10 | 0.749412 | 0.735294 | 0.658824 | 0.870588 | 0.075503 |
| negative | 10 | 0.693333 | 0.686275 | 0.658824 | 0.737255 | 0.024540 |
| hard-negative | 10 | 0.705882 | 0.694118 | 0.662745 | 0.800000 | 0.046895 |

- ROC AUC (positive vs combined negative/hard-negative): **0.6875**.
- Best threshold in the diagnostic sweep: `0.723529`.
- At that threshold: TPR `0.60`, FPR `0.15`, balanced accuracy `0.725`.
- Sanity result: **FAIL**. Positive central tendency is higher, but the distributions
  overlap heavily; one favorable threshold is not evidence of robust separation.

Per-sample scores are in `outputs/sanity/microwakeword_scores.csv`; full statistics and
the threshold sweep are in `outputs/sanity/microwakeword_sanity.json`.

## Failure analysis and training decision

The frontend/sample-rate mismatch hypothesis is rejected. Remaining likely causes are:

- only 50 positive and 50 negative source WAVs;
- a single TTS engine and limited acoustic diversity;
- hard-negatives were folded into one broad negative feature set;
- the 300-step refinement is intentionally short;
- the 51.7 KiB MixedNet artifact has limited capacity;
- validation samples reuse TTS voice identities even when speed and environment differ.

Class weighting was balanced, so simple class imbalance is not the primary suspect.
No further training was started. A future confirmed training phase should use more
independent speakers/sources, a dedicated hard-negative curriculum, held-out speaker
groups, larger validation sets, and a planned step/capacity sweep.

## Safe Git initialization proposal

Do not initialize until reviewed. When approved:

1. Create `.gitignore` before `git init`.
2. Commit source, tests, docs, small text configs, firmware source, and reproducibility
   manifests.
3. Ignore `.envs/`, `.python/`, `.uv-cache/`, `__pycache__/`, `.pytest_cache/`, model
   checkpoints, SavedModels, TFLite/ONNX/H5 files, mmap features, generated WAV datasets,
   TensorBoard events, caches, and large logs.
4. Keep small JSON/CSV result summaries only when they are intentional reproducibility
   records; store large artifacts outside Git or under an artifact release system.
5. Run `git status --short` and inspect every candidate before the first `git add`.

ESP-IDF remains PENDING/BLOCKED and was not touched in this phase.
