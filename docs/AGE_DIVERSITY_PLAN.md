# Age diversity plan

Status: **planned; real-age coverage is not yet satisfied**

## Metadata contract

Every record carries the following fields:

- `speaker_id`: stable pseudonymous speaker identifier.
- `source_type`: `tts`, `public_corpus`, or `consented_recording`.
- `source_family`: engine/model/corpus/recording-session family.
- `age_group`: `child`, `young`, `middle_aged`, `elderly`, or null.
- `age_verified`: true only when project-controlled documentation verifies the
  band; false for self-report, TTS, inference, or missing age.
- `age_source`: `verified`, `reported`, or `unknown`.
- `gender_if_available`: optional self-reported/source-provided value; null is
  valid and no value is inferred acoustically.
- `acoustic_age_proxy`: optional transform description, stored outside speaker
  demographics and never counted as age coverage.

## Coverage target

The first real-voice pilot targets at least 8 speakers in each age group (32
speakers total), with at least 4 speakers per group in Train and at least 2
group-disjoint speakers per group in Validation. Test speakers are entirely
held out. Speaker-level imbalance must not exceed 2:1 across age groups.

| Group | Primary route | Metadata quality | Current status |
|---|---|---|---|
| Child | purpose-recorded wake phrase with guardian consent and supervised session | verified only when consent/session records support the band | needs manual collection |
| Young | purpose recording plus Mandarin Common Voice negatives | verified for project recording; reported for Common Voice | not collected |
| Middle-aged | purpose recording plus Mandarin Common Voice negatives | verified/reported | not collected |
| Elderly | targeted purpose recording plus Mandarin Common Voice negatives | verified/reported | not collected |

Common Voice demographic fields are optional and self-reported, so they map to
`age_source=reported`, `age_verified=false`. Its clips can supply ordinary
negative/background speech only unless the transcript genuinely contains the
wake phrase and passes manual review. Access is now through Mozilla Data
Collective and may require account/terms acceptance or a large download; that
step requires user authorization and is not part of the current run.

## Privacy and split rules

- Store only pseudonymous speaker IDs and coarse age bands in the manifest.
- Keep consent/verification documents outside the audio repository.
- Split by speaker before augmentation; no real speaker crosses Train, Val, and
  Test.
- Do not infer age or gender from voice.
- Do not relabel pitch-shifted adult TTS as child or elderly.
- Report metrics by age group only when each group has enough independent
  speakers to avoid identifying or overinterpreting individuals.

If verified child or elderly recordings are unavailable, v2 reports the gap as
`NOT COVERED`; it does not fabricate coverage from acoustic proxies.

