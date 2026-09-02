# Phase 2D — Age Extreme Supplement POC

Audit date: 2026-08-29

## Scope and interpretation

This POC seeks coverage of materially different age-related acoustic traits;
it does not train or evaluate an age classifier. AISHELL-3 A/B/C/D values are
retained only as official demographic metadata. They are not mapped to audible
child, young-adult, middle-aged, or elderly voice classes.

The existing VoxCPM1.5 POC remains accepted for pronunciation, speaker
diversity, and gender diversity. Official age metadata is available, but
perceptual age diversity was not demonstrated. VoxCPM1.5 therefore remains a
candidate source for `qingxiaojia_v2`, subject to the later formal-dataset gate.

## Younger-end selective-source audit

The local official AISHELL-3 `spk-info.txt` contains 218 speakers but only two
group-A (`<14`) entries: SSB0393 and SSB1126. SSB0393 was already selected and
reviewed in the original seven-speaker POC. SSB1126 has no WAV in the official
AISHELL Hugging Face mirror at pinned revision
`f20d5db4a31fe779ef07bb1af4ea92da5c786622`. OpenSLR provides the full archive,
not a per-speaker download. Consequently, zero new group-A references were
obtained and the requested five-to-eight new A speakers do not exist in this
dataset.

AISHELL-3 is distributed under Apache-2.0. Sources:

- https://openslr.org/93/
- https://huggingface.co/datasets/AISHELL/AISHELL-3

## Older-end selective-source audit

Mozilla Data Collective lists Common Voice Scripted Speech 26.0 — Chinese
(China), released 2026-06-17, under CC0-1.0. Its metadata contains:

- `sixties`: 6 clips from 2 speakers;
- `seventies`: 5 clips from 1 speaker.

The dataset therefore contains the desired demographic candidates. It cannot
be used for this bounded POC through the current official delivery path: the
page declares `hasSampleDownload=false`, and the API/SDK grants a presigned URL
for the single 22,965,003,653-byte (21.39 GB) archive after authentication and
terms acceptance. There is no official per-speaker, per-age, or per-clip
download endpoint. No archive or audio was downloaded.

CC0-1.0 covers the dataset. Mozilla Data Collective terms additionally prohibit
speaker re-identification and re-hosting/re-sharing the dataset; any future raw
reference must remain local and preserve provenance without attempting to
identify the speaker. Sources:

- https://mozilladatacollective.com/datasets/cmqim47x700tunq074za20dq1
- https://mozilladatacollective.com/api-reference/docs

## Gate result

No new raw reference is ready for listening and no VoxCPM clone was generated.
The POC is paused at selective acquisition, as required. If a legal selective
source later becomes available, its manifest must retain
`reference_age_group`, `reference_age_group_source`, `source_family`, and
`speaker_id`. Human review may additionally set `perceived_age_reviewed` and
`perceived_age_category`; these fields must never be represented as verified
age metadata.
