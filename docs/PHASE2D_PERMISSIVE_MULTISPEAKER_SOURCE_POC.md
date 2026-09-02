# Phase 2D permissive multi-speaker source POC

Status: **VOXCPM1.5 POC GENERATED; HUMAN LISTENING REQUIRED**

This phase is diagnostic and POC-only. It does not authorize bulk generation,
dataset admission, or model training.

## Piper Chaowen license gate

Final decision:

- `ALLOW_FOR_RESEARCH_AUDIT_ONLY`
- `FORMAL_DATASET_USE_BLOCKED_PENDING_LICENSE_REVIEW`

The Piper voices repository is MIT-licensed and Chaowen's direct dataset is
marked CC0, but Chaowen is documented as fine-tuned from Xiao Ya. Xiao Ya's
model card identifies BZNSYP and marks the dataset non-commercial. The existing
listening samples remain available for research audit but are excluded from the
formal `qingxiaojia_v2` manifest.

Official records:

- https://huggingface.co/rhasspy/piper-voices/commit/10eb5c756ae21b759c8344d54aef86f9399ae92d
- https://huggingface.co/rhasspy/piper-voices/commit/1f265c6bbb3612a8581fc384c91b23fe0b5a0297
- https://huggingface.co/rhasspy/piper-voices/tree/main

## AISHELL-3 source audit

AISHELL-3 is suitable as a permissive reference-speaker source. OpenSLR lists
Apache-2.0, about 85 hours, 218 native Mandarin speakers, and explicit gender,
age-group, and native-accent metadata. The official age codes are preserved
verbatim:

| Code | Official range |
|---|---|
| A | `<14` |
| B | `14-25` |
| C | `26-40` |
| D | `>41` |

No project-inferred child, adult, or elderly label is added. Every reference
record uses `age_group_source = verified_dataset_metadata`; this verifies the
dataset annotation, not an independent age check by this project.

The OpenSLR full archive is approximately 18/19 GB, but it is not required for
the POC. The official AISHELL Hugging Face organization mirror exposes
`spk-info.txt`, transcripts, and WAVs by speaker/file. This run selectively
downloaded one reference WAV for each of seven speakers at pinned revision
`f20d5db4a31fe779ef07bb1af4ea92da5c786622`:

| Age code | Speakers | Gender coverage |
|---|---|---|
| A | SSB0393 | female |
| B | SSB0273, SSB0632 | male, female |
| C | SSB0710, SSB0197 | male, female |
| D | SSB0434, SSB0737 | male, female |

The seven copied reference WAVs total 2,236,858 bytes. Their hashes, official
speaker metadata, transcripts, paths, audio properties, and source revision are
recorded in
`phase2/artifacts/permissive_multispeaker_poc/aishell3_references/reference_manifest.json`.

Official records:

- https://openslr.org/93/
- https://www.openslr.org/resources/93/
- https://huggingface.co/datasets/AISHELL/AISHELL-3
- https://huggingface.co/datasets/AISHELL/AISHELL-3/blob/main/spk-info.txt

## VoxCPM1.5 feasibility and download gate

VoxCPM1.5 is an Apache-2.0, Chinese/English, 0.8B-parameter voice-cloning model.
The official repository requires Python `>=3.10,<3.13`, PyTorch `>=2.5`, and
CUDA `>=12`. Its comparison table reports about 6 GB VRAM. The local RTX 4060
Laptop GPU has 8,188 MiB total VRAM; 5,982 MiB was free during this audit, so
the POC is plausible only marginally and should run with other GPU applications
closed. Windows remains a compatibility risk, especially around FFmpeg,
TorchCodec, and CUDA/PyTorch combinations.

Official model files total about 1.95 GB (`model.safetensors` about 1.6 GB and
`audiovae.pth` about 346 MB). A clean CUDA/PyTorch environment and the remaining
audio dependencies make the actual network transfer exceed 2 GB. Conservative
requirements before starting are:

- model target:
  `phase2/artifacts/permissive_multispeaker_poc/voxcpm15/model`;
- isolated environment target: `F:\ZJU_intership\task\4\.vcp15`;
- expected network transfer: roughly 4-6 GB including model and dependencies;
- recommended free disk: 10-12 GB to cover environment and download caches;
- expected setup/download time: 15-45 minutes, network dependent.

Official records:

- https://huggingface.co/openbmb/VoxCPM1.5
- https://huggingface.co/openbmb/VoxCPM1.5/tree/main
- https://github.com/OpenBMB/VoxCPM
- https://github.com/OpenBMB/VoxCPM/blob/main/pyproject.toml

The user approved this download on 2026-08-29. Preflight passed with 289.1 GiB
free on drive F:, no residual Python process, and about 6.0 GiB free GPU memory.
The installation nevertheless stopped before environment creation:

1. the first shallow clone of the official GitHub repository ended with
   `Recv failure: Connection was reset`;
2. the second shallow clone ended with `Could not connect to server`;
3. the fallback PyPI/Hugging Face availability query produced no output and was
   terminated instead of being left running.

No independent environment, model directory, generated audio, or residual
Python/Git process remains. Empty partial clone directories were removed; the
AISHELL-3 references were not changed. This satisfies the repeated-failure
stop rule. The POC may resume only after GitHub and package-index connectivity
is restored or the official source/model files are supplied locally.

Consequently, no cloned wake-word POC audio exists yet and there is no new
listening gate. Formal v2 admission remains blocked until installation succeeds,
generation produces no more than 12 clips, speaker diversity is audible,
pronunciation passes human review, and output licensing/provenance is recorded.

### Manual model and wheel recovery update

The user later supplied the official model and source ZIP locally. All six
required model files are readable; `model.safetensors` has a valid safetensors
JSON header and `audiovae.pth` has a valid PyTorch ZIP header. The model totals
1,953,226,505 bytes. Browser-downloaded PyTorch 2.10.0+cu128 and torchaudio
2.10.0+cu128 wheels were also installed offline in the independent Python
3.10.20 environment.

POC execution remains stopped because the browser ZIP has no `.git` metadata
and three local editable-install attempts failed inside `setuptools-scm` after
it searched upward into the parent WakeWord-Studio repository and hit Git's
dubious-ownership check. No dependency transaction was committed, model loading
was not attempted, no audio was generated, and no process remains. The next
safe option is an explicitly approved PYTHONPATH-based install that avoids
building the ZIP as a package.

### PYTHONPATH recovery and bounded POC result

The user approved the PYTHONPATH recovery. The official runtime dependencies
were installed in the independent `.vcp15` environment without packaging the
browser ZIP, modifying global Git configuration, or touching another project
environment. PyTorch 2.10.0+cu128 reports CUDA 12.8 and successfully recognizes
the RTX 4060 Laptop GPU.

TorchCodec's Windows DLL could not load. VoxCPM1.5 only requires basic PCM WAV
input for this POC, so the isolated POC process replaced `torchaudio.load` with
a libsndfile-backed reader. The adapter passed shape, sample-rate, finite-value,
and peak checks; the repository and system FFmpeg were not modified.

The local model loaded in 5.458 seconds. Ten clips were generated from seven
AISHELL-3 references: seven positives and three hard negatives, covering
official age groups A/B/C/D and both genders. Total generation time was 17.541
seconds; maximum CUDA allocated memory was 2,733.6 MiB and maximum reserved
memory was 3,396 MiB. All outputs are 44.1 kHz PCM16, finite, non-empty, and
unclipped. There are no partial files or residual Python processes.

The listening manifest is
`phase2/artifacts/permissive_multispeaker_poc/voxcpm15/listening_manifest.json`.
Formal dataset eligibility remains false until the user explicitly accepts the
pronunciation, speaker diversity, age-group perceptual diversity, and audio
quality of these ten files.

### Listening decision and age-metadata interpretation

Human review accepted VoxCPM1.5 pronunciation, speaker diversity, and gender
diversity. VoxCPM1.5 remains an approved candidate source for the future
`qingxiaojia_v2` multi-speaker dataset. The existing ten-file POC is not itself
admitted to a formal dataset by this decision.

The AISHELL-3 A/B/C/D values remain verified demographic metadata only. They
must not be presented as perceptual acoustic classes such as child, young,
middle-aged, or elderly. Human review did not demonstrate perceptual age
diversity in the seven-speaker POC. Future age coverage is therefore evaluated
as a separate human-reviewed acoustic-extreme supplement, not as an age
classification task.
