# NEED USER ACTION — Phase 0.5 / 1A

## 1. Install/activate ESP-IDF (blocks required ESP32-S3 build)

Current checks: `idf.py`, CMake, Ninja, `C:\Espressif`, `%USERPROFILE%\esp`, and
`%USERPROFILE%\.espressif` are all absent.

Fastest official Windows path:

1. Run `winget install Espressif.EIM` (administrator/network approval may appear).
2. Open **Espressif Installation Manager**.
3. Choose **Easy Installation** and install the latest stable ESP-IDF (v6.0.1 is current;
   ESP-IDF v5.5.x is also acceptable).
4. Open the installed **IDF PowerShell/Terminal** and run `where idf.py`.
5. Send the displayed path here (or simply reply `ESP-IDF installed`).

Do not flash hardware yet. The immediate acceptance target is only `set-target` + `build`.

Official guide:
https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/windows-setup-update.html

## 2. Later: microphone + listening confirmation

Once the automated tests are complete, a 30-second human test is needed:

- allow microphone access;
- say “你好，青小甲”;
- listen once to `assets/i_am_awake.wav` and confirm it says “我醒来了”.

The Python microphone package `sounddevice` is not currently installed. Installation will
be requested only when the rest of the demo is ready.

## 3. Configure the local Git identity (completed)

Completed on 2026-08-29. The repository-local identity is configured and baseline commit
`59bf30d` (`chore: establish wakeword studio baseline`) exists.

## 4. Listen to the second independent Chinese TTS audit (completed)

Completed on 2026-08-29. The initial no-terminal-punctuation set was rejected because
several final `jia` syllables lost most of the `a` vowel. The v2 terminal-boundary set
fixed the truncation and passed listening. Formal generation will use a terminal period.

```powershell
explorer phase1\artifacts\tts_second_source_relisten_v2\listen\positive
explorer phase1\artifacts\tts_second_source_relisten_v2\listen\hard_negative
```

MeloTTS is accepted as one independent TTS family and one additional voice. The four
punctuation candidates are not treated as four speakers, and no age metadata is inferred.

## 5. Review the formal dataset sample — COMPLETED

Completed on 2026-08-29. The user confirmed `revised hard-negative listening passed`.
The revised 12-phrase hard-negative curriculum is accepted for the Phase 2A dataset.

The formal `qingxiaojia_v1` dataset is built and passes automated audio/manifest/leakage
validation. The first listening pass found three exact-pinyin collisions and repeated
`jia1` variants in the hard-negative labels. Those labels have been replaced; before
starting the first long microWakeWord baseline, re-listen to the compact 26-file folder:

```powershell
explorer datasets\projects\qingxiaojia_v1\listening_review
```

Files `01` through `12` are one clean example of every revised hard-negative curriculum phrase.
The remaining files sample MeloTTS/Kokoro positives plus augmented ordinary negatives and
ambient audio. `metadata.jsonl` records the expected text, source, noise, and tier.

Please confirm:

- all 12 hard negatives say the recorded text and remain distinguishable from the target;
- positive samples still contain the complete wake phrase after augmentation;
- noise mixing has no clipping, severe distortion, or inaudible speech;
- ambient samples contain no accidental wake phrase.

Gate result: **PASSED**. A later baseline-training phase may use this dataset, subject to
its own explicit configuration, runtime estimate, and monitored-command notice.

## 6. Start the Phase 2B formal baseline — NEED USER ACTION

The 150-step timing benchmark is complete. Formal training remains gated and has not
started. Before authorizing it, configure Windows so the display may turn off but the
computer does **not** enter sleep or hibernation while plugged in.

When ready, reply exactly:

`START FORMAL TRAINING`

That approval will launch one independent background process, verify its PID, first
heartbeat, status file, log, checkpoint directory, and resume command, then return control
to the user while training continues independently.

## 7. Phase 2D Piper chaowen replacement listening gate — COMPLETED

Seven original files passed listening. Listen only to the replacement:

`phase2/artifacts/tts_third_source_audit/piper_chaowen/listen/positive/positive_04_careful.wav`

The rejected `positive_04_pause.wav` is preserved under `rejected/` for audit.
The user confirmed `piper chaowen replacement listening passed` on 2026-08-29.
Pronunciation and audio quality are accepted. The separate model-license gate
still blocks formal source admission, bulk generation, and v2 training.

## 8. Phase 2D VoxCPM1.5 POC network gate — BLOCKED / NEED USER ACTION

AISHELL-3 selective reference preparation is complete: seven speakers, one WAV
per speaker, 2,236,858 bytes total, with official A/B/C/D age groups plus gender
and accent metadata. The full 18/19 GB archive was not downloaded.

VoxCPM1.5 itself is about 1.95 GB, but its clean CUDA/PyTorch environment and
audio dependencies make the expected transfer about 4-6 GB. Reserve 10-12 GB
free disk. Proposed targets:

- model: `phase2/artifacts/permissive_multispeaker_poc/voxcpm15/model`
- environment: `F:\ZJU_intership\task\4\.vcp15`

Expected installation/download time is 15-45 minutes, network dependent. The
RTX 4060 8 GB is marginally compatible with the official approximately 6 GB
VRAM estimate; other GPU applications should be closed during the POC.

The user approved the small POC download on 2026-08-29. Preflight passed, but
the official GitHub shallow clone failed twice: first by connection reset and
then by inability to connect to port 443. A subsequent package-index query also
stalled without output and was actively terminated. The independent environment
and model directory were never created, and there are no residual processes or
partial clone files.

Please restore access to both GitHub and the Python package index, or provide a
local copy of the official VoxCPM source/model. A quick manual connectivity test
is:

```powershell
git ls-remote https://github.com/OpenBMB/VoxCPM.git HEAD
D:\anaconda12.7\python.exe -m pip index versions voxcpm
```

When both commands return normally, reply:

`VOXCPM NETWORK READY`

### Manual CUDA PyTorch wheel gate — NEED USER ACTION

The user supplied the VoxCPM1.5 model and repository locally. Model integrity
passed. The isolated Python 3.10.20 environment was created successfully at
`F:\ZJU_intership\task\4\.vcp15`, but the automatic CUDA PyTorch download was
stopped at the user's request. It was downloading the approximately 2.7 GiB
Windows CUDA 12.8 PyTorch wheel; nothing was installed and no process remains.

Download these two exact CPython 3.10 / Windows x64 wheels in a browser:

- `torch-2.10.0+cu128-cp310-cp310-win_amd64.whl`
- `torchaudio-2.10.0+cu128-cp310-cp310-win_amd64.whl`

Official indexes:

- https://download.pytorch.org/whl/cu128/torch/
- https://download.pytorch.org/whl/cu128/torchaudio/

Place both files in:

`F:\ZJU_intership\task\4\WakeWord-Studio\phase2\artifacts\permissive_multispeaker_poc\voxcpm15\wheels`

Then reply:

`PYTORCH CUDA WHEELS READY`

### Local VoxCPM source build gate — BLOCKED / NEED USER DECISION

The two browser-downloaded wheels passed ZIP/header and SHA-256 checks and were
installed offline into `.vcp15`:

- `torch==2.10.0+cu128`
- `torchaudio==2.10.0+cu128`

The local VoxCPM repository installation then failed three times before any
dependency transaction was committed. The browser-extracted repository has no
own `.git` metadata, so `setuptools-scm` searches upward into WakeWord-Studio
and Git rejects that parent repository as dubious ownership. A temporary
setuptools-scm version and a process-local safe-directory setting did not clear
the build backend failure. The repeated-failure gate is now active; do not keep
retrying automatically.

Only torch and torchaudio are installed. `torch` import is not yet possible.
`uv pip check` reports these missing direct torch dependencies:

- `filelock`
- `typing-extensions>=4.10.0`
- `sympy>=1.13.3`
- `networkx>=2.5.1`
- `jinja2`
- `fsspec>=0.8.5`

The local repository additionally declares `torchcodec`,
`transformers>=4.36.2`, `einops`, `gradio>=6,<7`, `inflect`, `addict`,
`wetext`, `modelscope>=1.22.0`, `datasets>=3,<4`, `huggingface-hub`,
`pydantic`, `tqdm`, `simplejson`, `sortedcontainers`, `soundfile`, `librosa`,
`matplotlib`, `funasr`, `spaces`, `argbind`, and `safetensors`.

A safe recovery option is to avoid packaging the ZIP entirely: install the
declared dependencies into `.vcp15`, then run with the local
`repo\src` directory on `PYTHONPATH`. This does not modify the repository or
global Git configuration. It requires a new explicit approval because the
automatic installation retry limit has been reached. Reply:

`APPROVE VOXCPM PYTHONPATH RECOVERY`

### VoxCPM1.5 10-file listening gate — NEED USER ACTION

PYTHONPATH recovery succeeded. Ten local-only VoxCPM1.5 files were generated
from seven AISHELL-3 reference speakers. Automated checks passed: 10 WAVs,
44.1 kHz PCM16, no NaN, no clipping, no partial file, and no residual process.

Listen to:

`phase2/artifacts/permissive_multispeaker_poc/voxcpm15/listen`

Please verify:

- all seven positive files completely say `你好，青小甲`;
- files `08` and `10` say `你好，小甲`, and file `09` says `你好，青甲`;
- the seven speakers sound materially different rather than like one voice;
- age groups A/B/C/D have at least some audible diversity, without treating
  perceived age as independently verified metadata;
- there is no clipping, truncated final syllable, boundary noise, or other
  obvious quality defect.

Do not admit these files to `qingxiaojia_v2` until the listening result is
explicitly recorded.

### Phase 2D age-extreme reference gate — BLOCKED BY SELECTIVE ACCESS

The existing VoxCPM1.5 listening gate is resolved for pronunciation, speaker
diversity, and gender diversity. Perceptual age diversity was not demonstrated,
but this does not invalidate VoxCPM1.5 as a formal multi-speaker candidate.

No new raw reference needs listening yet. AISHELL-3 contains only two official
group-A speakers in total: the already reviewed SSB0393 and SSB1126. The pinned
official selective mirror contains no SSB1126 WAV, while OpenSLR exposes the
full approximately 18/19 GB archive rather than a per-speaker download.

Common Voice Scripted Speech 26.0 Chinese (China) has suitable age metadata,
including two `sixties` speakers and one `seventies` speaker. However, its
official Mozilla Data Collective page has `hasSampleDownload=false`, and the
authenticated API/SDK downloads the single 21.39 GB archive rather than clips
selected by speaker or age. The project must not download that archive under
the current POC boundary.

Resume only if a permitted, selectively acquired raw reference is supplied or
an official selective endpoint becomes available. The user must listen to raw
references before any new VoxCPM cloning. For each selected reference, the
later bounded clone step may generate exactly one `你好，青小甲` file, with at
most six files total.

### qingxiaojia_v2 formal dataset listening gate — NEED USER ACTION

The Phase 2E build is complete and automated QA passed. The dataset contains
15,200 canonical WAVs with zero speaker/source-utterance leakage, duplicate
hash, partial file, manifest mismatch, or v1 external-benchmark mutation.

Listen to the 15 WAV files in:

`datasets/projects/qingxiaojia_v2/listening_review`

Use `metadata.json` in the same directory to see each split, label, source,
speaker, text, phrase placement, padding, noise, and SNR. Confirm:

- all six positives completely say `你好，青小甲`;
- ordinary negatives do not contain the wake phrase;
- the four selected hard negatives say `你好，小甲` or `你好，青甲` and remain
  distinguishable from the positive;
- front/middle/back placement and leading/trailing padding do not truncate the
  phrase;
- noisy and ambient files have no clipping, boundary artifact, or corrupt WAV.

Training remains explicitly disabled. After listening, record the result; a
separate authorization is still required before feature preparation or formal
training.

### qingxiaojia_v2 listening gate — RESOLVED; formal training approval pending

The user recorded `QINGXIAOJIA_V2 LISTENING PASSED`. Phase 2F preflight is now
complete: the dataset is frozen, streaming-window alignment passed, and the
150-step benchmark plus strict checkpoint resume probe passed. Formal training
has not started. The only remaining action is an explicit reply:

`START V2 FORMAL TRAINING`
