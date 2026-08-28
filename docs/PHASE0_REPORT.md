# WakeWord Studio — Phase 0 技术选型报告

状态：**COMPLETE（技术选型 POC，不代表最终精度或实机验收）**  
日期：2026-08-28

## 1. Executive Summary

**推荐 Model 1：microWakeWord / MixedNet。** 真实完成训练、streaming 全 INT8 导出、正负 WAV 推理和算子检查；模型 51.703 KiB，命中 50–100 KiB 目标。13 个实际算子全部存在于 ESPHome 当前 `micro_wake_word` resolver。未接实体板，所以 ESP32-S3 结论严格为 **PROBABLE**，不是 CONFIRMED。

**推荐 Model 2：LiveKit Embedded Wakeword / RepCNN。** 官方 `augment → train → export → eval` 已实际跑通，并补出了真正 INT8 输入/输出模型。PC 端提供未阈值 sigmoid score、滚动 score 和多 binary model API，本地数据适配最简单，噪声可精确配置 SNR。其 README 明确说 ESPHome 兼容尚未验证，99 帧整窗模型的 embedded score cadence 也未实测，因此结论是 **UNCERTAIN**。

**Backup：Google `kws_streaming` / DS-TC-ResNet。** 它原生支持 multi-class logits，最适合严格的多关键词 softmax；真实本地目录 loader、两步训练、float/INT8 导出和推理均完成。但依赖仍钉死 2020 年 Python 3.6 `tf_nightly`，源码引用公开 TF 不存在的内部 Keras API，streaming 转换两次各 10 分钟超时。端到端 MFCC 图含浮点 FFT/LOG 岛；classifier-only INT8 图算子干净但需要外部 MFCC adapter，因此不选作当前主线。

Phase 0 没有追求 Recall 98%。所有准确率和概率只证明 pipeline，不代表最终模型质量。

## 2. Candidate comparison

### 2.1 核心 POC

| 项目 | microWakeWord | RepCNN | DS-TC-ResNet |
|---|---|---|---|
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| 上游快照 | `95f16d5951eb97eb8a4047b1042ca6e15b854dda` | `726403d432d11d3a4327f0a367d43db78f4e3d78` | 人工解压的 2026 快照；无 `.git`，commit 不可验证 |
| 环境 | Py3.10 / TF2.21 CPU | Py3.11 / TF2.21 CPU | Py3.10 / TF2.13.1 CPU |
| 中文 | 外接 Kokoro，已试听 | 固定 commit 自带生成器为英文；复用 Kokoro | 无 TTS；训练 loader 对标签语言无关 |
| Auto dataset | 外部 TTS/脚本 | CLI 完整，但内置 TTS 为英文 | 无 TTS；本地目录原生 |
| Streaming | 真正 stateful | PC rolling-window；classifier stateless | 架构支持 state；本机转换超时 |
| INT8 | 全 INT8 streaming | 全 INT8 classifier；官方 `--quantize` 自身仍 float I/O | classifier-only 全 INT8；端到端有浮点岛 |
| 参数 | 19,697 | 2,241 training / 1,473 fused | 2,390 |
| 实测主要文件 | 51.703 KiB | 11.578 KiB | 31.180 KiB end-to-end；14.523 KiB classifier-only |
| 计算量 | 未从日志得到 | 5.639 M MAC/整窗 | 1.312 M end-to-end；0.203 M classifier-only |
| TFLM ops | resolver 全覆盖 | resolver 全覆盖 | classifier 覆盖；end-to-end 不覆盖 |
| ESP32-S3 | **PROBABLE** | **UNCERTAIN** | **FAILED AS EXPORTED / POSSIBLE WITH ADAPTER** |
| 集成成本 | 中 | 低–中 | 高 |

### 2.2 老师新增约束

| 项目 | microWakeWord | RepCNN | DS-TC-ResNet |
|---|---|---|---|
| Raw probability/logit | **YES**，sigmoid probability | **YES**，sigmoid probability | **YES**，multi-class logits |
| Streaming score | **YES** | **YES on PC / PARTIAL embedded** | **PARTIAL**：设计支持，转换超时 |
| Threshold 可绕过 | YES | YES | YES |
| Multi-keyword | **EASY**：parallel binary | **EASY**：parallel binary | **NATIVE**：single multi-class |
| 三关键词成本 | 本 POC 文件约 155.1 KiB + 各自 state/arena；共享 frontend | 文件约 34.7 KiB，但约 16.9 M MAC/arbitration；共享 frontend | 单模型多输出，head 增量很小 |
| Multi-class | DIFFICULT | POSSIBLE_WITH_ADAPTER | NATIVE |
| Dataset folder import | MODERATE | EASY–MODERATE，已实测 | EASY，已实测 |
| Age diversity | PARTIAL | PARTIAL | PARTIAL |
| Noise augmentation | GOOD | **GOOD；5/10/15 dB 已实测** | PARTIAL：原生 volume；精确 SNR 用统一预混 |
| Gated activation | YES_WITH_PRE_ROLL | YES_WITH_PRE_ROLL | YES_WITH_PRE_ROLL，先修 streaming export |
| DetectionLogic | **5/5** | **4/5 PC；3/5 embedded** | **3/5** |

### 2.3 0–5 评分

| 维度 | microWakeWord | RepCNN | DS-TC-ResNet |
|---|---:|---:|---:|
| License / custom wake | 5 / 5 | 5 / 5 | 5 / 5 |
| Maintenance / docs | 3 / 2 | 4 / 4 | 2 / 2 |
| Chinese / auto data | 3 / 3 | 3 / 4 | 2 / 1 |
| Training automation | 3 | 5 | 2 |
| Streaming / raw score | 5 / 5 | 3 / 5 | 2 / 5 |
| INT8 / TFLite / TFLM | 5 / 5 / 5 | 4 / 5 / 4 | 3 / 4 / 2 |
| ESP evidence / ops | 4 / 5 | 2 / 5 | 1 / 2 |
| 50–100 KiB potential | 5 | 4 | 4 |
| PC inference | 5 | 5 | 5 |
| Dataset import | 3 | 4 | 5 |
| Multi-keyword route | 4 | 4 | 5 |
| Age / noise | 3 / 4 | 3 / 5 | 3 / 3 |
| DetectionLogic | 5 | 4 | 3 |
| Dependency / integration simplicity | 3 / 4 | 3 / 4 | 1 / 1 |
| **Total / 110** | **91** | **89** | **63** |

评分只表示本次最终软件约束下的实测工程适配度，不是论文精度排名。

## 3. Actual commands

```powershell
# A
.envs\microwakeword\Scripts\python.exe -m microwakeword.model_train_eval --training_config phase0\artifacts\configs\microwakeword_smoke.yaml --train 1 --restore_checkpoint 1 --test_tflite_streaming 1 --test_tflite_streaming_quantized 1 --use_weights best_weights mixednet --pointwise_filters "48,48,48,48" --repeat_in_block "1,1,1,1" --mixconv_kernel_sizes "[5],[7,11],[9,15],[23]" --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3

# B
.envs\livekit\Scripts\livekit-wakeword.exe augment phase0\artifacts\configs\livekit_repcnn_smoke.yaml
.envs\livekit\Scripts\livekit-wakeword.exe train phase0\artifacts\configs\livekit_repcnn_smoke.yaml
.envs\livekit\Scripts\livekit-wakeword.exe export phase0\artifacts\configs\livekit_repcnn_smoke.yaml --quantize
.envs\livekit\Scripts\livekit-wakeword.exe eval phase0\artifacts\configs\livekit_repcnn_smoke.yaml

# C
.envs\kws_streaming\Scripts\python.exe phase0\scripts\stage_kws_streaming_dataset.py ...
.envs\kws_streaming\Scripts\python.exe phase0\scripts\train_kws_streaming_smoke.py --data-dir phase0\artifacts\datasets\kws_streaming_smoke --train-dir phase0\artifacts\models\kws_streaming_smoke
.envs\kws_streaming\Scripts\python.exe phase0\scripts\export_inspect_kws_streaming.py ...
.envs\kws_streaming\Scripts\python.exe phase0\scripts\export_kws_classifier_only.py ...
```

完整日志在 `phase0/logs/`。

必要兼容修复：A 兼容 `pymicro-features 2.0.2`、Keras3 ndarray 和 streaming calibration packet；B 的 Windows 中文 YAML 强制 UTF-8，并补代表性 full-INT8 exporter；C 为缺失的 `_keras_internal` 建最小 shim，上游转换超时后改用同一训练图的公开 converter，部署 batch 改为 1，并隔离导出 classifier-only 图。base Conda 未修改。

## 4. Actual artifacts

| 候选 | Artifact | Bytes | KiB | SHA-256 |
|---|---|---:|---:|---|
| A INT8 streaming | `.../microwakeword_smoke/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite` | 52,944 | 51.703 | `b8cb6ccbccafe81baa72495156145301c9472ea1e20f22e9bb8cec94ca6de76e` |
| B full INT8 | `.../livekit_repcnn_smoke/livekit_repcnn_smoke_full_int8.tflite` | 11,856 | 11.578 | `bf8718a8cb53d835e1756961325c7aea91d897f179c202c08ffe0671b0cfd234` |
| C end-to-end float | `.../kws_streaming_smoke/tflite/ds_tc_resnet_nonstream_float.tflite` | 66,224 | 64.672 | `af180301b82163f29dadc75028201d643d7d5e797475e85ef28332d8656eb229` |
| C INT8-I/O hybrid | `.../ds_tc_resnet_nonstream_full_int8.tflite` | 31,928 | 31.180 | `58647ed5403c2c93ccfa846fd23ea4c8092d2db0ec223a1ba69be8bc3fff1d7c` |
| C classifier full INT8 | `.../ds_tc_resnet_classifier_full_int8.tflite` | 14,872 | 14.523 | `95742ab7c5e86b8cffa11508624b23e0819766d1df00d7eef57df573593b129e` |

C 的 end-to-end “full_int8” 文件实际含浮点 island，不能称纯 INT8；classifier-only 才是干净整型网络，但不含 MFCC。

## 5. Actual inference results

| 模型 | Positive | Negative |
|---|---|---|
| A INT8 streaming | mean 0.49211，last 0.49804 | mean 0.49276，last 0.50196 |
| B float | 0.345386 | 0.231325 |
| B full INT8 | 0.347656 | 0.230469 |
| C float logits `[qingxiaojia,other]` | `[-0.03855,0.64149]` | `[-0.01503,0.37005]` |
| C classifier INT8 logits | `[-0.03528,0.61933]` | `[-0.01176,0.36062]` |

A 非常量但未分离；B 正样本分数较高；C 两步训练未学好。B 的 PC streaming wrapper 在 3.5 秒输入上实际返回 76 个 score，范围 0.21753–0.31954。这些均是 sanity check，不是 accuracy 证据。

## 6. ESP32-S3 compatibility

### A — PROBABLE

INT8 input、UINT8 1×1 output 与 ESPHome loader 匹配。13 个 unique op：`ASSIGN_VARIABLE, CALL_ONCE, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, QUANTIZE, READ_VARIABLE, RESHAPE, SPLIT_V, STRIDED_SLICE, VAR_HANDLE` 全在当前 resolver。仍缺实体板 arena、Invoke、延迟和麦克风验证。

### B — UNCERTAIN

full INT8 仅需 `ADD, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, MEAN, RESHAPE`，resolver 全覆盖。但上游把 ESPHome compatibility 写为未验证 TODO。99 帧整窗模型还需专用 rolling-window runtime 和实机 cadence/latency 测试。

### C — FAILED AS EXPORTED / POSSIBLE WITH ADAPTER

端到端图额外需要 `GATHER, RFFT2D, COMPLEX_ABS, MAXIMUM, LOG`，量化图含 `DEQUANTIZE/QUANTIZE` 浮点岛；当前 resolver 不覆盖。classifier-only 七个 op 全覆盖，但输入 `(1,98,20)` MFCC，不是 ESPHome 的 40-bin microfrontend。需增加外部 MFCC，或用 `preprocess=micro` 重新训练。internal-state streaming 转换两次各 10 分钟超时，没有可验证 artifact。

主证据：

- https://raw.githubusercontent.com/esphome/esphome/dev/esphome/components/micro_wake_word/streaming_model.cpp
- https://github.com/espressif/esp-tflite-micro

## 7. 50–100 KiB analysis

**Tiny Model 选择 A。** microWakeWord 的真实 full INT8 streaming 文件为 51.703 KiB，不是把参数量冒充 KB。B 本次只有 11.578 KiB，可用 filters/blocks 自然扩大，但无需为了凑大小扩大。C 的 64.672 KiB 文件虽然在区间，却是 float、非流式且前端不兼容，不能满足 tiny ESP32 INT8 目标。

## 8. Recommended final pair and runtime constraints

1. `WakeWordBackend` A：microWakeWord/MixedNet，默认 ESP32-S3 tiny、binary-per-keyword。
2. `WakeWordBackend` B：RepCNN，训练体验更好；正式开发前必须做 S3 rolling-window benchmark。
3. Backup：DS-TC-ResNet，仅在 native multi-class logits 的收益大于依赖/前端/streaming 适配成本时启用。

### Multi-keyword / L4

A/B 使用三个 binary model、共享 frontend。不要机械执行 `softmax([pA,pB,pC])`；应先对每模型做 calibration，至少用 `logit(p)` + temperature scaling，再做 softmax，或 winner + minimum margin。C 原生输出 logits，可在 wake classes（含 unknown/silence）上直接 softmax/argmax。

### Gate 与五层 DetectionLogic

- `vad_consecutive_frames = 3`：Energy/adaptive threshold 后的 WebRTC VAD pre-gate。
- `wake_consecutive_frames = N`：模型 score 的 post-process L1。

二者绝不混用。runtime 始终保留约 1–2 秒 raw/feature ring buffer；VAD gate 通过后 reset backend 并回放 pre-roll，避免丢失唤醒词开头。L2 维护 score 背景基线和 peak ratio；L3 wrapper cooldown；L4 使用校准后的多词 score；L5 使用 energy/silence history。本阶段只做兼容性分析，未实现完整逻辑。

抽象命名统一为 `WakeWordBackend` / `WakeWordEngine`，不是 `WakeNetEngine`。唤醒响应为“我醒来了”，未来资产名 `assets/i_am_awake.wav`；Phase 0 未实现播放。

## 9. Risks

- Smoke 数据太小，不能外推 Recall 98%。
- Kokoro 有多中文 speaker，但无可验证的 child/young/middle-aged/elderly 标签；pitch shift 不是儿童声。正式数据需加入有年龄标签的合法公开语音、真人录制和独立真人测试集。
- RepCNN 精确 SNR 已实测；正式 DatasetAdapter 仍需覆盖 room/office/keyboard/fan/street/TV-speech/music/white/pink 和多 SNR。
- A 训练文档引用缺失脚本、easy generator 英文硬编码；B 官方 quantize 不是真 full INT8，embedded 声明未实证。
- C 的 tf_nightly pin、TFA EOL、私有 API、转换超时、MFCC 前端浮点岛使维护风险最高。
- 三模型并行 arena、PSRAM/internal RAM、latency、功耗均未在实体板测量。

## 10. Human intervention record

| 时间 | 原因 | 用户操作 | 解决 |
|---|---|---|---|
| 2026-08-28 | 首批中文 TTS 强制试听 | 试听 8 个 Kokoro WAV，回复“全部正常” | Yes |
| 2026-08-28 | Google clone 三次无法连接 GitHub 443 | 人工放置 `kws_streaming` 目录 | Yes；缺 `.git` commit metadata |

## 11. Added acceptance checklist

- [x] 至少 A/B 提供 raw score
- [x] 三候选 multi-wake-word 和资源成本已分析
- [x] L4 calibration/softmax 数学问题已说明
- [x] 数据目录导入已评估，B/C 实际适配
- [x] 年龄多样性证据缺口已记录
- [x] 5/10/15 dB 噪声混合实测
- [x] VAD gate、pre-roll、state warm-up 已分析
- [x] 五层 DetectionLogic 已分析
- [x] 两种 consecutive-frame 设置已分离
- [x] 最新响应文本和抽象命名已记录

机器侦察见 `docs/PHASE0_PLAN.md`；原始 JSON、算子和兼容性证据见 `phase0/results/`。
