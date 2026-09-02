# qingxiaojia_v2 dataset plan

Status: **BUILT / QA PASSED / AWAITING HUMAN LISTENING**

This plan repairs the Phase 2C generalization gap without overwriting
`qingxiaojia_v1`, its Test split, the Phase 2B baseline, or Phase 2C outputs.
The v2 `DatasetManifest` and audio build now exist; training is not authorized.

## Why v2 exists

Phase 2C showed that the main failure was not streaming conversion. Float
streaming Test Recall was 29.5% and INT8 streaming Test Recall was 27.5%, while
v1 Validation used only Kokoro `zm_041` and Test introduced Kokoro `zm_053`,
`zm_056`, and MeloTTS `ZH`. The v2 design therefore prioritizes source/speaker
coverage and duration/silence matching rather than a larger model.

## Planned scale

The first v2 round contains 15,200 records. The existing 900-record v1 Test is
an additional external benchmark and is not counted in v2.

| Split | Positive | Ordinary negative | Hard negative | Ambient | Total |
|---|---:|---:|---:|---:|---:|
| Train | 3,000 | 4,800 | 2,700 | 1,500 | 12,000 |
| Validation | 400 | 640 | 360 | 200 | 1,600 |
| Held-out Test | 400 | 640 | 360 | 200 | 1,600 |
| Total | 3,800 | 6,080 | 3,420 | 1,900 | 15,200 |

## Speaker and source allocation

The preferred allocation below is conditional on a model-level license gate.
Speaker IDs are allocation slots and must be frozen only after the candidate
model's speaker list and output have been audited.

| Split | Source family | Speaker policy | Planned records |
|---|---|---|---:|
| Train | Kokoro | 6 existing v1-train voices; no `zm_041`, `zm_053`, or `zm_056` | 6,600 |
| Train | permissive cloning candidate using AISHELL-3 references | speaker slots frozen only after POC | 5,400 |
| Validation | Kokoro | 3 voices disjoint from Train and all v1 external-Test voices | 800 |
| Validation | permissive cloning candidate using AISHELL-3 references | speakers disjoint from Train/Test | 800 |
| Held-out Test | permissive cloning candidate using AISHELL-3 references | speakers unseen by Train/Validation | 1,600 |

Hard gate: AISHELL-3 is Apache-2.0 and supports selective reference-speaker
access with official age/gender/accent metadata. VoxCPM1.5 is the current
Apache-2.0 cloning-model candidate, but its install/download and listening POC
are not yet approved or complete. No cloned output is approved for formal v2.

Piper `chaowen` passed the pronunciation/audio-quality listening gate on
2026-08-29. Its card says its own dataset is CC0, but it was fine-tuned from
Xiao Ya, whose BZNSYP dataset is marked non-commercial. Its final gate is
`ALLOW_FOR_RESEARCH_AUDIT_ONLY` and
`FORMAL_DATASET_USE_BLOCKED_PENDING_LICENSE_REVIEW`. The current listening
files remain outside every formal manifest even though listening passed.

MeloTTS `ZH` remains outside v2 Train/Validation so the existing v1 external
Test retains an unseen TTS family. The v1 Test is immutable and is always
reported both overall and separately for Kokoro and MeloTTS.

## Leakage rules

- `source_group_id = source_family:speaker_id` belongs to exactly one split.
- A `source_utterance_id`, its speed variants, and all augmentations belong to
  exactly one split.
- Text templates may recur across splits, but not the same synthesized or real
  source utterance.
- Split assignment happens before augmentation and is checked in CI.
- A source-family/speaker allocation file is frozen before bulk generation.

## Duration, silence, and phrase placement targets

All three v2 splits use the same target distributions. Validation is not made
artificially clean or narrow.

| Dimension | Target distribution |
|---|---|
| Total duration | 1.5–2.2 s: 25%; 2.2–3.0 s: 35%; 3.0–4.0 s: 30%; 4.0–5.0 s: 10% |
| Leading silence | 0–0.15 s: 30%; 0.15–0.50 s: 30%; 0.50–1.00 s: 25%; 1.00–1.50 s: 15% |
| Trailing silence | 0–0.20 s: 25%; 0.20–0.60 s: 30%; 0.60–1.10 s: 30%; 1.10–1.50 s: 15% |
| Wake-phrase placement | front: 30%; middle: 40%; back: 30% |

The builder must store `phrase_start_ms`, `phrase_end_ms`, measured leading and
trailing silence, and a duration-bin ID. Reports must show count, mean, standard
deviation, median, p05/p25/p75/p95, and histogram bins for Train, Validation,
and Test. A split fails QA when any target bin differs by more than 5 percentage
points or a duration mean differs by more than 0.15 s from another split.

## Noise and SNR targets

- Clean: 20%.
- Room/office/fan: 40% combined.
- Street/transit: 15%.
- TV speech/music: 15%.
- Device/reverb/impulse-response conditions: 10%.
- For noisy records: SNR 20 dB or above 15%, 10–20 dB 25%, 5–10 dB 25%,
  and 0–5 dB 15% of all records; the remaining 20% is clean.
- Noise clips and room impulse responses are group-split before mixing so the
  exact background recording cannot leak across splits.

## Hard negatives

The manually revised inventory is retained. `你好，小甲` and `你好，青甲` each
target 7.5% of hard negatives, for 15% combined. Adjacent phonetic confusions
target 35%; broader greeting/name/command negatives target 50%. No single text,
speaker, or source utterance may contribute more than 10% of a hard-negative
split. Sampling begins broad and raises Tier-1 near-misses only in the final
third of training; this preserves the known false-accept cases without creating
a two-phrase classifier.

## Minimal streaming-objective repair

The current baseline loads a fixed `spectrogram_length` window using
`truncate_start`, gives the whole clip one binary label, and evaluates a single
clip output. Deployment instead processes streaming packets and uses score
aggregation/consecutive-frame logic. The first v2 experiment keeps MixedNet
Tiny unchanged and changes only sample construction and evaluation:

1. Randomly place the phrase and retain its exact interval.
2. Draw fixed-length training windows from the whole sequence.
3. Label a window positive only when the complete phrase is inside the valid
   receptive interval. Boundary/partial-phrase windows enter an ignore band in
   round one, avoiding contradictory labels.
4. Include background-only, ordinary-negative, hard-negative, and partial
   streaming windows in every batch.
5. Select checkpoints and thresholds using full-clip streaming inference with
   state reset per clip and the deployed max/consecutive-three aggregation.
6. Report both window metrics and utterance-level streaming metrics.

Initial batch target: 35% aligned positive windows, 25% partial/background
windows from positive clips, 25% hard-negative windows, and 15% ordinary
negative/ambient windows. The main risk is incorrect phrase timestamps; any
record without exact synthetic placement or reviewed alignment cannot supply a
positive window label.

## Age and real-voice route

See `docs/AGE_DIVERSITY_PLAN.md`. TTS speaker age remains unknown. Pitch/rate
transforms remain acoustic proxies and never count as child or elderly coverage.
Common Voice Mandarin is a candidate for real ordinary-negative/background
speech with self-reported demographic metadata, but no bulk download is
authorized. Real positive wake phrases require consented, purpose-recorded
speakers because arbitrary corpus text cannot be relabeled as the wake phrase.

## Resource estimate

- Canonical PCM16 audio: approximately 1.5 GB for 15,200 records.
- Fixed float features: approximately 0.6–0.8 GB.
- Source utterances, manifests, reports, and resumable intermediates: 0.6–1.2 GB.
- Expected incremental formal artifacts: 2.7–3.5 GB, excluding reusable model
  caches and isolated TTS environments.
- CPU generation/augmentation/feature extraction: approximately 3–6 hours,
  with resumable manifests and a heartbeat at least every 50 records.

These estimates must be recalculated from a 100-record pilot before bulk build.

## Completed build result — 2026-08-29

The 100-record pilot and 15,200-record formal build passed. The actual
speaker-disjoint allocation is:

| Split | Kokoro | VoxCPM1.5 using AISHELL-3 references |
|---|---|---|
| Train | zf_001, zf_003, zf_006, zm_009, zm_013, zm_020 | SSB0197, SSB0273, SSB0632, SSB0710 |
| Validation | zf_017, zm_031 | SSB0393, SSB0434 |
| Test | zf_021, zm_041 | SSB0737 |

The immutable v1 Test retains Kokoro `zm_053`, `zm_056`, and MeloTTS `ZH` as
external unseen speakers/source. Piper `chaowen` is absent from v2.

Actual totals are 12,000 Train, 1,600 Validation, and 1,600 Test. Kokoro and
VoxCPM1.5 each supply 6,650 records; 1,900 records are procedural ambient.
Total audio is 12.579 hours and 1,449,807,774 WAV bytes. Full QA found zero
speaker, source-group, or source-utterance leakage; zero duplicate hashes; zero
partial files; and zero audio/manifest violations. Train/Validation/Test mean
durations are 2.979/3.023/2.938 seconds, with a maximum mean gap of 0.086
seconds. Phrase placement is exactly 30% front, 40% middle, and 30% back among
speech records in every split.

AISHELL-3 A/B/C/D remains verified demographic metadata only. Every record has
`perceived_age_verified=false`; age was not used as a hard balancing target.
The canonical manifest is `datasets/projects/qingxiaojia_v2/DatasetManifest.json`
and the audit is `datasets/projects/qingxiaojia_v2/reports/quality_audit.json`.

## Gates before formal build or training

1. **Passed 2026-08-29:** human accepted the seven clean originals plus the
   replacement `positive_04_careful.wav`; the rejected pause artifact is kept
   under `rejected/`.
2. A model-level license is documented for every formal source family.
3. The source/speaker allocation and external-Test hashes are frozen.
4. A 100-record pilot passes leakage, duration/silence, audio-contract, and
   streaming-window-label QA.
5. Only then build 15,200 records. Training still requires a separate explicit
   `START V2 FORMAL TRAINING` authorization.
