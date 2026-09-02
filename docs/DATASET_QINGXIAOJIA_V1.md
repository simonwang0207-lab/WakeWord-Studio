# Qingxiaojia v1 Dataset Quality Report

- Manifest: `F:\ZJU_intership\task\4\WakeWord-Studio\datasets\projects\qingxiaojia_v1\DatasetManifest.json`
- Total samples: **9,000**
- Total duration: **6.547 hours**
- Dataset WAV bytes: **719.6 MiB**
- Canonical contract: **16,000 Hz / mono / PCM16**
- Manifest/WAV validation errors: **0**
- Duration outliers (<0.3 s or >5.0 s): **4**

Age caveat: no TTS voice supplies verified or reported age metadata. Acoustic
pitch/speed proxies are counted separately and must not be described as real child
or senior voices.

Source-holdout caveat: MeloTTS has only one Chinese speaker and is kept entirely
in test to prevent speaker leakage. This provides unseen-family evaluation, but
the current training speech families are only: `kokoro`.

## Labels

| Value | Count |
|---|---:|
| ambient | 1000 |
| hard_negative | 2000 |
| negative | 4000 |
| positive | 2000 |

## Splits

| Value | Count |
|---|---:|
| test | 900 |
| train | 7200 |
| validation | 900 |

## Split × label

| Split | Positive | Negative | Hard negative | Ambient |
|---|---:|---:|---:|---:|
| train | 1600 | 3200 | 1600 | 800 |
| validation | 200 | 400 | 200 | 100 |
| test | 200 | 400 | 200 | 100 |

## Source distribution

| Value | Count |
|---|---:|
| kokoro | 7815 |
| melotts | 185 |
| procedural_ambient_proxy | 1000 |

## Source × split distribution

| Value | Count |
|---|---:|
| test:kokoro | 615 |
| test:melotts | 185 |
| test:procedural_ambient_proxy | 100 |
| train:kokoro | 6400 |
| train:procedural_ambient_proxy | 800 |
| validation:kokoro | 800 |
| validation:procedural_ambient_proxy | 100 |

## Speaker distribution

| Value | Count |
|---|---:|
| kokoro:zf_001 | 720 |
| kokoro:zf_003 | 720 |
| kokoro:zf_006 | 720 |
| kokoro:zf_017 | 720 |
| kokoro:zf_021 | 720 |
| kokoro:zm_009 | 715 |
| kokoro:zm_013 | 715 |
| kokoro:zm_020 | 707 |
| kokoro:zm_031 | 663 |
| kokoro:zm_041 | 800 |
| kokoro:zm_053 | 334 |
| kokoro:zm_056 | 281 |
| melotts:ZH | 185 |
| procedural_ambient_proxy:none | 1000 |

## Age metadata distribution

| Value | Count |
|---|---:|
| unknown:unknown | 9000 |

## Acoustic age proxy distribution

| Value | Count |
|---|---:|
| higher_pitch_and_faster | 800 |
| lower_pitch_and_slower | 800 |
| none | 7400 |

## Noise distribution

| Value | Count |
|---|---:|
| clean | 1000 |
| fan | 1142 |
| keyboard | 1142 |
| music | 1142 |
| office | 1145 |
| room | 1145 |
| street | 1142 |
| tv_speech | 1142 |

## SNR distribution (dB)

| Value | Count |
|---|---:|
| 0 | 1736 |
| 10 | 1750 |
| 20 | 1778 |
| 5 | 1736 |
| none | 2000 |

## Hard-negative tier distribution

| Value | Count |
|---|---:|
| 1 | 1006 |
| 2 | 664 |
| 3 | 330 |

## Duration

- Minimum: 1.008 s
- Mean: 2.619 s
- Maximum: 5.200 s

## Deterministic listening list

The following paths were sampled with seed `20260829`:

- `train/positive/train-positive-001235.wav` — split=train, label=positive, source=kokoro, speaker=zf_021, text='你好，青小甲'
- `train/positive/train-positive-000844.wav` — split=train, label=positive, source=kokoro, speaker=zm_013, text='你好，青小甲'
- `train/positive/train-positive-000698.wav` — split=train, label=positive, source=kokoro, speaker=zf_021, text='你好，青小甲'
- `test/positive/test-positive-000123.wav` — split=test, label=positive, source=kokoro, speaker=zm_053, text='你好，青小甲'
- `train/positive/train-positive-001345.wav` — split=train, label=positive, source=kokoro, speaker=zm_031, text='你好，青小甲'
- `train/positive/train-positive-001337.wav` — split=train, label=positive, source=kokoro, speaker=zm_013, text='你好，青小甲'
- `test/negative/test-negative-000324.wav` — split=test, label=negative, source=melotts, speaker=ZH, text='明天早上提醒我开会'
- `train/negative/train-negative-000286.wav` — split=train, label=negative, source=kokoro, speaker=zm_020, text='帮我查一下今天的日程'
- `train/negative/train-negative-000171.wav` — split=train, label=negative, source=kokoro, speaker=zf_021, text='周末我们一起去公园吧'
- `test/negative/test-negative-000321.wav` — split=test, label=negative, source=melotts, speaker=ZH, text='请打开客厅的灯'
- `train/negative/train-negative-000230.wav` — split=train, label=negative, source=kokoro, speaker=zm_009, text='小甲，你好'
- `train/negative/train-negative-000591.wav` — split=train, label=negative, source=kokoro, speaker=zm_009, text='清早起来空气很好'
- `validation/hard_negative/validation-hard_negative-000047.wav` — split=validation, label=hard_negative, source=kokoro, speaker=zm_041, text='你好，小瑞'
- `train/hard_negative/train-hard_negative-001522.wav` — split=train, label=hard_negative, source=kokoro, speaker=zf_001, text='你好，小安'
- `train/hard_negative/train-hard_negative-000056.wav` — split=train, label=hard_negative, source=kokoro, speaker=zf_021, text='你好，青甲'
- `train/hard_negative/train-hard_negative-000069.wav` — split=train, label=hard_negative, source=kokoro, speaker=zm_009, text='青小甲'
- `test/hard_negative/test-hard_negative-000006.wav` — split=test, label=hard_negative, source=kokoro, speaker=zm_053, text='你好吗，青小甲'
- `test/hard_negative/test-hard_negative-000175.wav` — split=test, label=hard_negative, source=kokoro, speaker=zm_053, text='你好，小甲'
- `train/ambient/train-ambient-000064.wav` — split=train, label=ambient, source=procedural_ambient_proxy, speaker=none, text=None
- `train/ambient/train-ambient-000611.wav` — split=train, label=ambient, source=procedural_ambient_proxy, speaker=none, text=None
- `train/ambient/train-ambient-000167.wav` — split=train, label=ambient, source=procedural_ambient_proxy, speaker=none, text=None
- `train/ambient/train-ambient-000720.wav` — split=train, label=ambient, source=procedural_ambient_proxy, speaker=none, text=None
- `train/ambient/train-ambient-000449.wav` — split=train, label=ambient, source=procedural_ambient_proxy, speaker=none, text=None
- `train/ambient/train-ambient-000141.wav` — split=train, label=ambient, source=procedural_ambient_proxy, speaker=none, text=None

## Validation details

- No missing files, audio-header violations, or source/group split leakage detected.
