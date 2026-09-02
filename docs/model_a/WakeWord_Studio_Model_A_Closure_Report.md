# WakeWord-Studio — Model A（microWakeWord / MixedNet Tiny）阶段收尾与可追溯技术报告

**报告用途**：Model A 阶段收尾；供后续整体项目报告、答辩材料、A/B 模型对比、系统集成，以及未来重新改进 Model A 时复用。  
**报告原则**：只记录已有项目事实、已确认日志数据和明确标注的分析判断；不为了“98%”目标伪造、筛选或美化数据。  
**当前结论**：Model A 已完成“约 50–100 KB 的可部署小模型”目标，但在当前严格测试协议下无法达到合理的 98% Recall 工作点；v3 的 sequence objective 显著降低误唤醒，但以 Recall 下降为代价。Model A 建议冻结为 **A_LIMITED / Tiny baseline**，后续由 Model B 承担高精度方向。

---

## 0. 证据等级与“哪些数字是怎么来的”

为避免后续整理时把“已验证事实”“历史记录”“推断”混在一起，本报告使用三类证据等级：

### E1 — 当前最终日志直接验证
来自 2026-08-30 的 Phase 2I：
- `validation-freeze`
- `heldout-tests`
- full-INT8 export 日志
- frozen hashes / threshold / Test 结果 / source breakdown / error analysis

这些数据是本报告中可信度最高的最终数据。

### E2 — 项目阶段记录中已确认，但本报告未重新打开原始 artifact 复算
包括：
- v1 数据集规模与早期评估
- Phase 2C float / streaming / INT8 诊断
- qingxiaojia_v2 构建统计与 speaker split
- v2 训练与 Phase 2G 结果
- v3 objective 设计、tiny-overfit、benchmark、训练过程与中途 JSON 序列化故障
- TTS / speaker source 选择过程

这些数据在此前阶段已经通过日志/命令输出确认，但本次收尾报告没有对每一项重新跑脚本复算，因此必须保留“阶段记录”属性。

### E3 — 工程分析 / 后续建议
例如：
- “Tiny 容量可能限制跨声源泛化”
- “Model B 应承担高精度目标”
- “未来应增加真人数据”
这些是基于 E1/E2 的工程判断，不应写成实验事实。

---

# 1. 项目背景与 Model A 的角色

## 1.1 总体任务背景

项目目标是开发本地唤醒词训练与运行软件，面向 ESP32-S3 场景，核心需求包括：

1. 至少接入两种开源唤醒词 / KWS 模型；
2. 能训练自定义中文唤醒词；
3. 其中至少一个最终模型约 **50–100 KB**；
4. 高识别率目标约 **98%**；
5. 自动生成多样语音数据；
6. 支持用户选择数据集目录和模型后自动训练；
7. 本地实时麦克风唤醒；
8. 唤醒后播放“我醒来了”；
9. 系统前置 Energy + WebRTC VAD + 3 个连续 speech frame 的 gate；
10. 后置 DetectionLogic 包括 L1–L5；
11. 未来对接 ESP32-S3，当前没有实体板，因此只能做到模型/固件准备与编译级验证，不能宣称已完成真实硬件运行。

## 1.2 Model A 选型

**Model A**：
- 开源路线：microWakeWord
- 核心架构：MixedNet Tiny / streaming
- 定位：Tiny / low-resource / 50–100 KB 模型
- 目标：优先满足小体积和嵌入式可部署性，然后尽可能提高识别效果

Phase 0 已验证该路线可导出约 52 KB full-INT8 streaming TFLite，因此锁定为 Model A。

## 1.3 Model A 最终定位

截至 v3 最终评估：

- 模型文件：**52,840 bytes**
- **51.6015625 KiB**
- full INT8
- deployment parameter count：**19,697**
- 达成小模型体积目标
- 但无法在合理 FPR 下达到 98% Recall

因此建议最终状态：

> **Model A = A_LIMITED / Tiny baseline / frozen**

不要把它描述成“高精度模型”，也不要伪造“98%”。

---

# 2. 音频与部署基础约束

## 2.1 统一音频格式

Canonical audio format：

- sample rate：**16 kHz**
- mono
- PCM16

早期 Kokoro 生成音频为 24 kHz，但训练 loader 会正确 resample 到 16 kHz，因此没有形成 sample-rate bug。

## 2.2 streaming 逻辑

Model A 最终采用 streaming inference。

v3 deployment score：

```text
sequence_score = max_t(min(p[t], p[t+1], p[t+2]))
```

含义：

> 只有存在连续 3 个 decision frame 都保持高分，才形成较高 sequence score。

v3 中 decision frame：
- frontend step：10 ms
- model stride：3
- decision frame：30 ms
- 3 个连续 decision frame：90 ms

这与后续 DetectionLogic L1 “连续若干帧确认”思想对齐，但要注意：

- 前置的“连续 3 个 WebRTC VAD speech frame”是 **pre-gate**
- Model 输出后的连续分数确认是 **post-model confirmation**
- 两者不是同一个逻辑

---

# 3. qingxiaojia_v1：第一版正式数据与暴露的问题

## 3.1 v1 数据集

路径：

```text
datasets/projects/qingxiaojia_v1
```

阶段记录统计：

- 总样本：**9,000**
- 总时长：约 **6.547 h**
- positive：2,000
- negative：4,000
- hard-negative：2,000
- ambient：1,000
- train：7,200
- validation：900
- test：900
- 16 kHz / mono / PCM16

## 3.2 hard negative 的标签修订

早期发现一个非常重要的数据问题：

某些候选 hard negative 如：

- “倾小甲”
- “清小甲”
- “轻小甲”

与“青小甲”在实际语音中可能高度同音/近乎无法区分。

如果强行把这些样本标为 negative，会造成标签语义矛盾：

> 同样或几乎同样的声音，一部分要求模型输出 1，另一部分要求输出 0。

因此后续删除这些同音性过强、标签不可靠的 hard negative，保留 **12 个拼音层面可区分**的 hard-negative phrase。

这是一次必要的数据标注修复。

## 3.3 v1 的 age 问题

v1 中并没有真实年龄标签。

当时只能使用 acoustic proxy，不能宣称具有真实“儿童 / 青年 / 中年 / 老年”覆盖。

后续所有材料必须避免写成：

> “v1 已覆盖真实多年龄人群”

这是不真实的。

---

# 4. Phase 2C：v1 模型诊断与根因拆解

Phase 2C 的目的不是重新调参数，而是回答：

> 性能差到底是模型没学会、streaming 转换坏了、INT8 量化坏了，还是数据泛化差？

## 4.1 三种评估形态

### 4.1.1 Float non-streaming
阶段记录：

- Test Recall：**9.5%**
- ROC AUC：**0.7095**

用途：
- 检查模型在非 streaming 条件下是否有基本判别能力

### 4.1.2 Float streaming
阶段记录：

- Test Recall：**29.5%**
- ROC AUC：**0.7366**

用途：
- 检查真实 streaming 结构是否明显损坏模型能力

### 4.1.3 INT8 streaming
阶段记录：

- Test Recall：**27.5%**
- ROC AUC：**0.7013**

用途：
- 模拟最终 ESP32 类部署形式

## 4.2 float streaming 与 INT8 的相关性

阶段记录：

- Pearson：**0.9894**
- Spearman：**0.7291**
- MAE：**0.0222**

结论：

> INT8 相对 float streaming 的变化是存在的，但不足以解释总体性能崩溃。

因此“量化把模型彻底搞坏”不是主因。

## 4.3 TFLite 输出反量化公式修复

TFLite 输出：

```text
dtype = uint8
scale = 0.00390625 = 1 / 256
zero_point = 0
```

正确反量化：

```text
real_score = scale * (raw - zero_point)
```

也就是：

```text
raw / 256
```

上游旧代码曾使用：

```text
raw / 255
```

两者最大误差约：

```text
1/255 - 1/256
```

对 score 只造成很小偏差，不能解释 20–50 个百分点的性能问题。

最终 runtime 规则：

> 永远读取 TFLite quantization metadata；禁止硬编码 `/255`。

## 4.4 streaming state leakage 问题

Phase 2C 发现：

> 某些 TFLite interpreter 的内部 streaming state 在不同 WAV 之间可能残留。

如果 `reset_all_variables()` 不可靠，会导致：
- 上一条样本影响下一条样本
- 指标不可复现
- false positive / false negative 被污染

最终修复原则：

> **每条独立 WAV 使用 fresh interpreter**，或必须使用经过回归验证的 reset。

v3 最终评估明确记录：

```text
stream_state_reset = fresh TFLite interpreter per WAV
```

## 4.5 v1 真正的主要问题

Phase 2C 后的判断：

- streaming conversion 不是主要问题
- INT8 不是主要问题
- 量化公式小误差不是主要问题
- 真正问题更偏向：
  - 数据分布
  - speaker/source 泛化
  - 训练 objective 与实际实时触发条件不一致

---

# 5. 数据源扩展与许可审计

## 5.1 Piper / chaowen POC

Piper chaowen 的发音和 speaker quality POC 结果不错。

但其 license chain 存在不清晰 / 可能受 Xiao Ya / BZNSYP 非商业约束影响，因此项目状态定为：

```text
ALLOW_FOR_RESEARCH_AUDIT_ONLY
```

没有把它作为正式主数据源。

这是正确做法：
- 技术可用 ≠ 许可可用于正式项目交付

## 5.2 AISHELL-3

AISHELL-3：
- Apache-2.0
- 官方 speaker metadata 包括 age / gender / accent
- 总 speaker 数约 218

项目只选择性下载了少量 reference WAV（约 7 条，约 2.2 MB），没有下载完整约 19 GB 数据集。

年龄分组元数据曾按：
- A `<14`
- B `14–25`
- C `26–40`
- D `>41`

但人工试听发现：

> “reference_age_group” 并不等于听感上真的能明确听出儿童/青年/中年/老年差异。

因此正式结论应写：

```text
reference_age_group = known
perceived_age_verified = false
```

不能把 metadata age 直接包装成“感知年龄多样性已验证”。

## 5.3 Common Voice

Common Voice zh-CN metadata 中有 sixties / seventies 等高年龄 speaker，但选择性下载不方便，完整包约 21 GB 量级，因此项目没有为此继续大规模下载。

年龄极端样本不再作为阻塞项目的必要条件。

## 5.4 VoxCPM1.5

本地完成 VoxCPM1.5 环境搭建：

- 独立环境：
  ```text
  F:\ZJU_intership\task\4\.vcp15
  ```
- Python：3.10.20
- torch：2.10.0+cu128
- CUDA：可见 RTX 4060 Laptop GPU 8 GB
- model load：约 5.458 s
- peak allocated：约 2.73 GB
- peak reserved：约 3.396 GB

POC：
- 10 samples
- 7 positive
- 3 hard negative
- 7 speakers
- 有 gender 与官方 age group 元数据

人工试听结论：
- speaker 差异明显
- 发音完整
- pronunciation / speaker / gender diversity POC 通过
- perceptual age diversity 未验证

因此 VoxCPM1.5 被纳入 v2 的正式多 speaker source。

---

# 6. qingxiaojia_v2：数据重构

## 6.1 数据集总体

路径：

```text
datasets/projects/qingxiaojia_v2
```

阶段记录：

- 总样本：**15,200**
- 总时长：**12.579 h**
- WAV：约 **1.35 GiB**
- 项目总量：约 **1.41 GiB**
- Train：12,000
- Validation：1,600
- Test：1,600

Manifest SHA-256：

```text
50e3857e9941d910b640039dd70e73c39e331cc368816c378849ca9774f1973c
```

v3 最终 evaluation 仍验证使用该 hash。

## 6.2 speaker-disjoint split

### Train
Kokoro：
- zf_001
- zf_003
- zf_006
- zm_009
- zm_013
- zm_020

VoxCPM：
- SSB0197
- SSB0273
- SSB0632
- SSB0710

### Validation
Kokoro：
- zf_017
- zm_031

VoxCPM：
- SSB0393
- SSB0434

### Test
Kokoro：
- zf_021
- zm_041

VoxCPM：
- SSB0737

目的：

> 避免同一个 speaker 同时出现在 Train / Validation / Test，降低 speaker memorization。

## 6.3 source composition

阶段记录：

- VoxCPM：6,650（43.75%）
- Kokoro：6,650（43.75%）
- ambient：剩余部分
- 非 ambient speech 中两类 source 各 50%
- MeloTTS：**0 条进入 v2**
- `zm_053`、`zm_056`、MeloTTS `ZH` 保持为 v1 external unseen family

## 6.4 duration 对齐

阶段记录：

- Train：2.979 ± 0.847 s
- Validation：3.023 ± 0.843 s
- Test：2.938 ± 0.840 s
- min ≈ 1.5 s
- max ≈ 5.0 s

这样减少 v1 中 Train/Val/Test duration/silence pattern 差异过大的问题。

## 6.5 wake phrase placement

位置比例：

- front：30%
- middle：40%
- back：30%

同时对 leading/trailing silence 做多样化。

目的：

> 避免模型只学“唤醒词总是在固定位置”。

## 6.6 SNR / noise

非 ambient speech：
- clean
- 20 dB
- 10 dB
- 5 dB
- 0 dB

每档约 2,660。

这意味着 v2 Test 本身包含很困难的 0 dB 场景，因此不能把其总体 Recall 直接等价成“普通室内近场使用成功率”。

## 6.7 hard negatives

v2：
- hard negative：3,420
- 12 个修订后的 phrase
- 每个约 284–286 条
- “你好，小甲”：284
- “你好，青甲”：284

数据泄漏检查：
- 记录为 0 leakage

所有 WAV：
- 16 kHz
- mono
- PCM16

## 6.8 v2 QA

正式状态从：

```text
BUILT_QA_PASSED_AWAITING_HUMAN_LISTENING
```

进入人工试听。

用户试听 15 个样本后通过。

---

# 7. v2 Model A：clip-level objective 版本

## 7.1 preflight

v2 formal training 前：

- planned steps：15,000
- batch：64
- TensorFlow 2.21 / Windows CPU
- RTX 4060 在 native Windows TF>=2.11 下没有用于 TensorFlow training
- mean：约 0.091 s/step
- estimated training time：约 23 min

streaming alignment audit：
- Train/Val 共 13,600 条
- 全部通过
- positive 使用 phrase_start / phrase_end
- 3,380 positive 完整包含 phrase
- 20 个较长 Validation positives 使用 terminal 3 s causal window
- negatives 不含 phrase

resume 测试通过。

预计 INT8 size：
- 52,840 bytes

## 7.2 正式训练结果

v2：
- planned：15,000 steps
- early stopped：7,500
- best checkpoint：3,500
- best training Validation F1：**0.491499**

## 7.3 v2 frozen Phase 2G 结果

最终 INT8：
- 52,840 bytes
- 51.602 KiB
- deployment params：19,697
- input：int8
- input scale：0.10196079
- input zero point：-128
- output：uint8
- output scale：1/256
- output zero point：0
- fresh interpreter per WAV

### Validation best-F1 operating point

阶段记录：

- threshold：**0.94921875**
- Recall：**65%**
- Precision：**56.64%**
- F1：**60.54%**
- FPR：**16.58%**

Recall ≥90 / 95 / 98 时：
- threshold 约 0.25390625
- FPR：100%
- Precision：25%

因此：

> **NO REASONABLE 98% OPERATING POINT**

### v2 held-out Test

阶段记录：

- Recall：**75.75%**
- Precision：**55.49%**
- F1：**64.06%**
- FPR：**20.25%**
- FRR：**24.25%**
- ROC AUC：**0.838822**
- PR AUC：**0.578366**
- TP 303
- FP 243
- TN 957
- FN 97

### v1 external Test，用 v2 model

- Recall：**61.50%**
- Precision：**45.39%**
- F1：**52.23%**
- FPR：**21.14%**
- ROC AUC：**0.726546**

### v2 Test source breakdown

阶段记录：

- zf_021 Recall：80%
- zm_041 Recall：100%
- VoxCPM SSB0737 Recall：61.5%

external：
- zm_053：87.14%
- zm_056：92.31%
- MeloTTS ZH：3.08%

### v2 hard negative

- hard-negative FPR：**43.06%**
- ordinary-negative FPR：约 **13.75%**
- ambient：表现较好

score distribution：

- positive mean：0.8885；median：0.9961
- hard negative mean：0.6648；median：0.8125
- ordinary negative mean：0.4478；median：0.2539
- ambient mean：0.2727；median：0.2539

关键诊断：

> positive 与 hard-negative score overlap 严重，特别是 near-phonetic phrase；ambient 不是主要矛盾。

---

# 8. 为什么从 v2 改为 v3 sequence objective

v2 的 clip-level BCE 主要优化：

> 一个 clip 最终是不是 positive。

但真实部署逻辑不是“整个 clip 有没有任意一个高分峰值”，而是：

> 实时流中要出现稳定、连续的唤醒证据。

因此 v2 存在 objective mismatch：

- 训练目标：clip-level
- 部署目标：streaming temporal confirmation

这导致：
- 单个 accidental peak 就可能造成 false accept
- hard-negative near-phonetic 很容易产生高峰

v3 的目标是：

> 让 positive 在 phrase 结束后出现稳定连续高分，同时 hard negative 整段保持低分。

---

# 9. v3 sequence objective 设计

配置：

```text
configs/models/microwakeword_tiny_v3_sequence.yaml
```

## 9.1 loss

```text
L = L_frame + 0.5 * L_hardmax
```

### L_frame
weighted frame BCE：

- positive end-region frame weight：32
- other frames：1

### L_hardmax
对 hard-negative：

```text
mean BCE(0, max_t(p_t))
```

作用：
- 直接惩罚 hard negative 中最高的 accidental wake peak

## 9.2 target

positive：
- phrase complete 后的第 1–3 个 decision frames = 1
- 其它 frames = 0

decision frame：
- 30 ms
- 3 frames = 90 ms

hard negative：
- 全部 102 streaming frames = 0
- 再加 max-score penalty

## 9.3 target audit

抽样：
- 10 positive：每个恰好 3 个 positive target frame
- 10 hard negative：全部 0
- 5 ordinary negative：全部 0

audit：
- **0 errors**

## 9.4 tiny overfit

24 samples：
- 250 steps + 100 no-grad BN calibration

阶段记录：

positive end：
- 0.4999 → 0.7515

prephrase max：
- 0.5007 → 0.0486

hardneg max：
- 0.5015 → 0.0121

loss：
- 0.35934 → 0.00032

target 3 frames avg：
- 0.5003 → 0.8097

min：
- 0.7443

“你好，小甲”：
- 0.5011 → 0.00353

“你好，青甲”：
- 0.5015 → 0.01659

结论：

> objective 至少在 tiny subset 上能正确学习 desired behavior。

## 9.5 benchmark

CPU：
- mean：0.127784 s/step
- p95：0.148534 s/step
- validation overhead：1.329 s
- peak RAM：约 1 GB

150-step benchmark：
- loss：0.8935 → 0.4245
- strict resume at 75：PASS
- ROC AUC：0.5725
- output 非恒定

计划：
- 15,000 steps
- validation / checkpoint interval：500
- early stopping
- estimated：约 32m38s

---

# 10. v3 正式训练过程与工程故障

run：

```text
runs\qingxiaojia\microwakeword_tiny_v3_sequence\formal\20260829T162135Z
```

## 10.1 第一次中断

v3 正式训练在 step 500 后 worker 消失。

状态文件仍显示 RUNNING。

根因：

```text
false_accepts_by_group
```

包含 NumPy `int64`，JSON serialization 失败。

这不是：
- 模型崩坏
- loss 崩坏
- checkpoint 损坏

而是：
- monitoring/status JSON 的类型序列化 bug

## 10.2 修复原则

新增 generic recursive JSON normalizer：

- `np.integer -> int`
- `np.floating -> float`
- `np.bool_ -> bool`
- `np.ndarray -> list`
- nested dict/list recursive normalization

并增加 regression tests。

## 10.3 recovery

后续 worker：
- 从 `ckpt-500` 恢复
- strict restore 成功
- 继续 formal training

中途确认：
- `ckpt-2500` 曾存在
- 由于 `CheckpointManager(max_to_keep=5)`，后续 ckpt 会淘汰早期 trainer checkpoint
- 但 best step 对应的：
  ```text
  best_weights.weights.h5
  ```
  保留

## 10.4 训练结束

最终日志：

```text
VALIDATION step=7500
threshold=0.00076047
recall=0.590000
precision=0.696165
f1=0.638701
fpr=0.085833
roc_auc=0.762973
best=False
stale=8
```

然后：

```text
EARLY_STOPPING
step=7500
best_step=2500
best_f1=0.6537966537966537
```

最终：

```text
TRAINING_COMPLETED
step=7500
early_stopped=True
best_step=2500
best_f1=0.6537966537966537
```

注意：

训练日志中的 `threshold=0.00076047` 不是最终 INT8 deployment threshold。  
最终 frozen evaluation 使用正确 dequantized probability score 后重新在 Validation 上选择 threshold。

---

# 11. v3 冻结导出

最终使用：

```text
best_weights.weights.h5
```

对应：

- best step：2,500
- checkpoint SHA-256：
  ```text
  fd48f7179e4a7ae7bf7f208eb47945ef8e21c6ea719469665001c539c9bf98e5
  ```

明确：

```text
ckpt_7500_used = false
```

## 11.1 模型

最终 TFLite：

```text
phase2i_v3_frozen_final\final_model\
tflite_stream_state_internal_quant\
stream_state_internal_quant.tflite
```

最终：
- bytes：**52,840**
- KiB：**51.6015625**
- matches expected nominal bytes：true
- SHA-256：
  ```text
  994f08b799364f02f6fc06273cccd4a155722af25f1b61a88f4e5b2621a2d41c
  ```
- deployment parameter count：**19,697**
- full_int8：true

operators：

- ASSIGN_VARIABLE
- CALL_ONCE
- CONCATENATION
- CONV_2D
- DEPTHWISE_CONV_2D
- FULLY_CONNECTED
- LOGISTIC
- QUANTIZE
- READ_VARIABLE
- RESHAPE
- SPLIT_V
- STRIDED_SLICE
- VAR_HANDLE

## 11.2 quantization

input：
- dtype：int8
- shape：[1, 3, 40]
- scale：0.10196078568696976
- zero point：-128

output：
- dtype：uint8
- shape：[1, 1]
- scale：0.00390625
- zero point：0

公式：

```text
real_score = scale * (raw - zero_point)
```

明确：

```text
raw_div_255_used = false
```

## 11.3 unresolved parameter-count discrepancy

导出过程中 Keras model summary 曾显示：

- Total params：23,697
- Trainable params：19,313
- Non-trainable params：4,384

最终 deployment metadata：

- parameter_count：19,697

该差异目前尚未在本阶段完全解释。

可能与：
- streaming state
- BN / non-trainable variable
- deployment graph 的统计口径

有关，但这是 **E3 推测**，不能作为已证明结论。

后续正式总报告如果要写参数量，建议：
- 明确注明“deployment parameter count = 19,697”
- 如果展示 Keras summary，同时保留训练图的 23,697 total
- 不要把两者混成同一个口径

---

# 12. v3 Validation threshold freeze

selection split：

```text
v2_validation_only
```

selection rule：

```text
best sequence F1;
tie -> recall, precision, FPR, threshold
```

Test 在 threshold freeze 前未访问：

```text
v2_test_audio_accessed = false
v1_external_test_audio_accessed = false
```

## 12.1 frozen threshold

```text
0.3671875
```

Validation：
- count：1,600
- Recall：**52.00%**
- Precision：**72.2222%**
- F1：**60.4651%**
- FRR：**48.00%**
- FPR：**6.6667%**
- TP：208
- FP：80
- TN：1,120
- FN：192
- ROC AUC：**0.78461354**
- PR AUC：**0.60558539**

## 12.2 Recall target sweep

### Recall ≥90%
threshold：
```text
0.1640625
```

- Recall：92%
- Precision：29.2528%
- F1：44.3908%
- FPR：**74.1667%**

结论：
```text
NO REASONABLE 90% OPERATING POINT
```

### Recall ≥95%
threshold：
```text
0.12890625
```

实际上 Recall：
- 100%

但：
- Precision：25%
- FPR：100%
- TN：0

### Recall ≥98%
同样 threshold：
```text
0.12890625
```

- Recall：100%
- FPR：100%

结论：

> **NO REASONABLE 98% OPERATING POINT**

合理性策略：
```text
validation_fpr <= 0.01
```

---

# 13. v3 final v2 held-out Test

threshold：
```text
0.3671875
```

Test 使用后没有改变 threshold：

```text
threshold_changed_after_test = false
```

## 13.1 overall

count：
- 1,600

metrics：
- Recall：**59.00%**
- Precision：**74.44795%**
- F1：**65.82985%**
- FRR：**41.00%**
- FPR：**6.75%**
- TP：236
- FP：81
- TN：1,119
- FN：164
- ROC AUC：**0.82870208**
- PR AUC：**0.69874107**

## 13.2 category breakdown

### positive
- count：400
- accepted：236
- rejected：164
- Recall：59%
- FRR：41%

score：
- mean：0.62325195
- median：0.75
- std：0.36089020
- min：0.12890625
- max：0.99609375

### ordinary negative
- count：640
- false accepts：32
- FPR：**5.00%**

score：
- mean：0.21190186
- median：0.1640625

### hard negative
- count：360
- false accepts：49
- FPR：**13.6111%**

score：
- mean：0.26827257
- median：0.1640625

### ambient
- count：200
- false accepts：0
- FPR：**0%**

score：
- mean：0.144765625
- median：0.12890625

## 13.3 source breakdown

### Kokoro
count：
- 700

- Recall：**94.5%**
- Precision：78.4232%
- F1：85.7143%
- FRR：5.5%
- FPR：10.4%
- TP：189
- FP：52
- TN：448
- FN：11
- ROC AUC：**0.962125**
- PR AUC：**0.905385**

### VoxCPM1.5
count：
- 700

- Recall：**23.5%**
- Precision：61.8421%
- F1：34.0580%
- FRR：76.5%
- FPR：5.8%
- TP：47
- FP：29
- TN：471
- FN：153
- ROC AUC：**0.615235**
- PR AUC：**0.439859**

这是 v3 最大的问题之一：

> 同一个模型在 Kokoro 与 VoxCPM source 上 Recall 差异极大。

不能用 Kokoro 94.5% 替代 overall 59% 作为“最终识别率”。

## 13.4 特别 hard-negative

### “你好，小甲”
- count：30
- false accepts：4
- FPR：**13.3333%**

### “你好，青甲”
- count：30
- false accepts：0
- FPR：**0%**

## 13.5 hard-negative false accepts by text

v2 Test：

- “你好吗，青小甲”：7
- “你好，小安”：1
- “你好，小甲”：4
- “你好，星小甲”：6
- “你好，请小甲”：9
- “你好，金小甲”：8
- “你好，青小佳”：3
- “你好，青小杰”：10
- “青小甲”：1

说明：

> v3 虽然大幅压低 hard-negative FPR，但 near-phonetic confusion 并未消失。

---

# 14. v3 final v1 external Test

该 Test 不参与 threshold selection。

Manifest SHA-256：

```text
70b089652a7f8eb407c9d23ccc0efe7e33ce241fad2309f87f35702dc4752391
```

threshold：

```text
0.3671875
```

## 14.1 overall

count：
- 900

- Recall：**45.00%**
- Precision：**64.2857%**
- F1：**52.9412%**
- FRR：**55.00%**
- FPR：**7.14286%**
- TP：90
- FP：50
- TN：650
- FN：110
- ROC AUC：**0.76654643**
- PR AUC：**0.55720447**

## 14.2 categories

positive：
- 200
- Recall 45%

negative：
- 400
- false accepts 23
- FPR 5.75%

hard negative：
- 200
- false accepts 27
- FPR **13.5%**

ambient：
- 100
- false accepts 0
- FPR 0%

## 14.3 unseen source breakdown

### Kokoro zm_053
- count：334
- Recall：**41.4286%**
- Precision：44.6154%
- F1：42.9630%
- FPR：13.6364%
- ROC AUC：0.732386
- PR AUC：0.361909

### Kokoro zm_056
- count：281
- Recall：**92.3077%**
- Precision：81.0811%
- F1：86.3309%
- FPR：6.48148%
- ROC AUC：0.974751
- PR AUC：0.903178

### MeloTTS ZH
- count：185
- Recall：**1.53846%**
- Precision：100%（仅 1 TP / 0 FP；不能理解成整体性能很好）
- F1：3.0303%
- FPR：0%
- ROC AUC：0.518333
- PR AUC：0.383242

MeloTTS 结果说明：

> 模型几乎不接受该 source 的 positive，因此 precision=100% 并没有实际价值；核心问题是 Recall≈1.5%。

---

# 15. v2 vs v3：最终可引用比较

最终 Phase 2I comparison block：

| Metric | v2 | v3 | 变化 |
|---|---:|---:|---:|
| Validation F1 | 0.605355 | 0.604651 | 基本持平 |
| v2 Test Recall | 75.75% | 59.00% | -16.75 pp |
| v2 Test FPR | 20.25% | 6.75% | -13.50 pp |
| Hard-negative FPR | 43.0556% | 13.6111% | -29.4445 pp |
| Ordinary-negative FPR | 13.75% | 5.00% | -8.75 pp |
| Ambient FPR | 0% | 0% | 不变 |
| v1 external Recall | 61.50% | 45.00% | -16.50 pp |
| v1 external FPR | 21.1429% | 7.1429% | -14.00 pp |
| v2 Test ROC AUC | 0.838822 | 0.828702 | -0.01012 |
| INT8 bytes | 52,840 | 52,840 | 不变 |

### 结论

v3 的变化不是“模型整体变强”，而是：

> **更保守、更少误唤醒，但更容易漏唤醒。**

最明显的收益：
- hard-negative FPR：43.06% → 13.61%
- overall FPR：20.25% → 6.75%

最明显的代价：
- Recall：75.75% → 59%
- external Recall：61.5% → 45%

---

# 16. 问题—改进—结果链路

## 16.1 问题：v1 speaker/source 泛化差
**改进**：
- speaker-disjoint split
- VoxCPM + Kokoro 多 source
- duration / silence / placement distribution 对齐
- 增加更困难噪声

**结果**：
- v2 相比 v1 在部分 unseen source recall 上改善
- 但 source dependency 仍很严重

## 16.2 问题：同音 hard-negative 标签矛盾
**改进**：
- 删除“倾/清/轻小甲”等无法可靠作为 negative 的近乎同音项
- 保留 12 个更有语音可分性的 hard-negative phrase

**结果**：
- 标签语义更合理
- 但 near-phonetic false accept 仍是难点

## 16.3 问题：INT8 output 处理可能错误
**改进**：
- 使用 TFLite quantization metadata
- `scale*(raw-zero_point)`
- 禁止 `/255`

**结果**：
- 量化解释被规范化
- 排除了该小误差作为主要性能瓶颈

## 16.4 问题：TFLite state leakage
**改进**：
- 每 WAV fresh interpreter

**结果**：
- frozen evaluation 明确 state isolation
- 提高结果可信度与复现性

## 16.5 问题：clip-level BCE 与实时触发目标不一致
**改进**：
- frame-level target
- phrase end 后 3-frame positive region
- hard-negative max penalty
- deployment score = 3 consecutive frame minimum 的时间最大值

**结果**：
- hard-negative FPR 大幅下降
- 但 Recall 同时明显下降

## 16.6 问题：JSON status serialization 导致 formal worker 中断
**改进**：
- recursive NumPy JSON normalization
- regression tests
- strict checkpoint restore

**结果**：
- formal v3 training 成功从 ckpt-500 恢复并完成

## 16.7 问题：export 目录 PermissionError
**现象**：
- TensorFlow SavedModel 导出残留 `variables_temp/*.tempstate...`
- 普通目录 ACL 写入测试其实通过

**处理**：
- 保留失败目录备份
- 从干净输出目录重新 export

**结果**：
- 成功生成 52,840-byte TFLite

这说明错误更可能与半完成 export 产物/目录状态有关，而不是整体 F 盘 ACL 权限不足。

---

# 17. 98% 目标的真实状态

必须明确：

### 当前 Model A v3
在 frozen Validation 上：

- 90%+ Recall：FPR 74.17%
- 95/98% Recall：实际上要降低 threshold 到全负样本也通过，FPR 100%

因此：

> **Model A 当前没有合理的 98% operating point。**

不要写：
- “Model A 识别率 98%”
- “Accuracy 98%”
- “Kokoro 94.5% 接近 98% 所以算 98%”
- “Melo precision 100% 所以达标”

这些都不符合真实 overall wake recall。

正确项目叙事：

- Model A 完成 50–100 KB Tiny 模型
- Model B / 完整系统承担高精度目标
- 标准场景与压力测试要分开报告

---

# 18. Model A 当前可以客观宣称的成果

可以真实宣称：

1. 已完成一个 **51.602 KiB** 的 full-INT8 streaming wake-word model；
2. deployment model 有约 **19.7k 参数**（按最终 exporter metadata）；
3. 支持 16 kHz mono PCM16；
4. 支持 streaming state；
5. 正确处理 INT8 quantization metadata；
6. 每条 WAV state isolation；
7. qingxiaojia_v2 有 15,200 样本、12.579 h、speaker-disjoint Train/Val/Test；
8. 数据包含 positive / ordinary negative / hard negative / ambient；
9. 包含 clean / 20 / 10 / 5 / 0 dB；
10. v3 sequence objective 将：
    - hard-negative FPR 43.06% → 13.61%
    - overall FPR 20.25% → 6.75%
11. 代价是 Recall：
    - 75.75% → 59%
12. Model A 未达到合理 98% Recall；
13. source 泛化严重不均：
    - Kokoro v2 Test Recall 94.5%
    - VoxCPM 23.5%
    - external MeloTTS ZH 1.54%
14. Model A 更适合作为 tiny / low-resource baseline，而非最终 high-accuracy mode。

---

# 19. 当前不能宣称的内容

禁止写：

- “已在 ESP32-S3 真机达到 XX ms latency”
- “已在真机长期运行无误唤醒”
- “模型达到 98%”
- “所有年龄段真人覆盖”
- “VoxCPM/AISHELL speaker 的听感年龄已经验证”
- “MeloTTS 泛化良好”
- “RepCNN 一定优于 Model A”
- “参数量就是 23,697”或“就是 19,697”而不说明统计口径差异
- “所有测试都是普通室内场景”
- “Test 用于调 threshold”

---

# 20. Frozen evaluation integrity

最终 v3 evaluation 明确：

```text
sequence_score_formula =
max_t(min(p[t], p[t+1], p[t+2]))
```

```text
clip_level_max_score_used = false
```

```text
output_formula =
scale * (raw - zero_point)
```

```text
raw_div_255_used = false
```

```text
fresh_interpreter_per_wav = true
```

Test order：

1. v2 held-out Test
2. v1 external Test

```text
threshold_changed_after_test = false
```

这套 protocol 后续 Model B 应尽量保持相同思想，以便公平比较。

---

# 21. 最终工件 / 哈希 / 路径应保存

## 21.1 dataset

v2：

```text
datasets/projects/qingxiaojia_v2
```

Manifest SHA-256：

```text
50e3857e9941d910b640039dd70e73c39e331cc368816c378849ca9774f1973c
```

## 21.2 v3 run

```text
runs\qingxiaojia\microwakeword_tiny_v3_sequence\formal\20260829T162135Z
```

## 21.3 best weights

```text
best_weights.weights.h5
```

step：
```text
2500
```

SHA-256：

```text
fd48f7179e4a7ae7bf7f208eb47945ef8e21c6ea719469665001c539c9bf98e5
```

## 21.4 final TFLite

```text
phase2i_v3_frozen_final\final_model\
tflite_stream_state_internal_quant\
stream_state_internal_quant.tflite
```

SHA-256：

```text
994f08b799364f02f6fc06273cccd4a155722af25f1b61a88f4e5b2621a2d41c
```

## 21.5 config hash

```text
e749872c7d1ec59714a932959078bdf453b74797bf3ef95780c7dc868a7782ed
```

## 21.6 evaluation dependency hashes

Phase 2I 记录：

```text
phase2i_script
13d7f3d452813150c519e6589963c86705a78a6c8e159b4c7f0707484338e97e
```

```text
phase2g_metric_helpers
554950e55331711adc03b74b26e4d5b06af855ff7f23695521cc2a2f5bfb5fe1
```

```text
sequence_model_builder
6493d2b40ab050943214cef53be86e4a6113e67b16b7ab01ff98cb76c7f4c59a
```

```text
runtime_config
e9fa02b88b9d1843e6a8c6297df5ba86aed4f7469ef8e497ad4e2128e71321f8
```

```text
sequence_score
bebb815e48e2c7c679024dcaaac0d9daee28b12ba41d18901c5a0fb349b3882e
```

```text
json_normalization
3e8bdbc6a8117e61a423151e31a04b319a13348435149a7d7957b6b8703a0997
```

## 21.7 v1 external manifest

```text
70b089652a7f8eb407c9d23ccc0efe7e33ce241fad2309f87f35702dc4752391
```

---

# 22. 建议额外归档的文件

Model A freeze 时建议完整保存：

```text
configs/models/microwakeword_tiny_v3_sequence.yaml
```

```text
best_weights.weights.h5
```

```text
TRAINING_STATUS.json
```

```text
training.log
```

```text
phase2i_v3_frozen_final/
```

其中至少：
- `checkpoint_freeze.json`
- final TFLite
- threshold freeze JSON
- final evaluation JSON
- export metadata
- error analysis

以及：
- qingxiaojia_v2 manifest
- v1 external manifest
- training / evaluation scripts
- JSON normalizer
- sequence score helper
- relevant tests
- pip/environment snapshot（如已有）
- Git commit hash（如已有）

如果当前还没有统一 archive manifest，后续应补一个只读归档清单，而不是重新训练。

---

# 23. 如果未来重新改进 Model A，优先级建议

此节是 E3 工程建议，不是当前实验事实。

## P0：先不要 v4
当前 A 已经经历：
- v1 数据问题
- v2 数据修复
- v3 objective 修复

继续立即 v4 容易陷入低收益循环。

## P1：真人数据
当前 source 主要是 TTS。

最值得增加的是：
- 多个真人 speaker
- 每人重复真实唤醒词
- 正常近场
- 不同语速
- 不同音量
- 少量常见环境噪声

目的是减少 TTS domain shortcut。

## P2：source-balanced training audit
重点确认：
- positive 和 negative 在每种 source 中都充分出现
- batch 不会让 source 与 label 高度相关

尤其避免：
```text
Kokoro ≈ positive
VoxCPM ≈ difficult/negative
```
这类 shortcut。

## P3：更平滑的 temporal objective
当前 v3 3-frame rule 过于保守。

可考虑：
- positive target window 稍宽
- soft temporal label
- consecutive-frame requirement 与 score threshold 联合
- hard-negative penalty 权重重新平衡

但所有 tuning 只能使用 Train / Validation。

## P4：capacity
52 KB Tiny 可能确实难以同时做到：
- high recall
- low hard-negative FPR
- cross-source generalization

因此更现实的是把高精度任务交给 Model B。

---

# 24. 对后续 Model B 的直接经验继承

必须继承：

1. 16 kHz canonical audio；
2. Train / Val / Test 严格分离；
3. Test 不调 threshold；
4. speaker-disjoint；
5. hard negatives 必须存在；
6. source balance audit；
7. correct INT8 metadata；
8. fresh state / proven reset；
9. tiny overfit；
10. benchmark；
11. strict resume；
12. frozen threshold；
13. source breakdown；
14. hard-negative breakdown；
15. external unseen-source test；
16. 不用单个最好 source 的结果代替 overall。

Model B 不必照抄：
- microWakeWord 的网络结构
- v3 的具体 3-frame loss
- 15,000 steps

应尊重 RepCNN 原生训练方式。

---

# 25. 一句话阶段结论

> Model A microWakeWord / MixedNet Tiny 已成功完成约 52 KB full-INT8 streaming 模型的工程闭环。通过 v1→v2 的数据重构和 v2→v3 的 deployment-aligned sequence objective，误唤醒尤其是 hard-negative FPR 得到显著改善，但 Recall 与跨 TTS/source 泛化仍不足，当前不存在合理的 98% Recall operating point。因此 Model A 应冻结为低资源 Tiny baseline，后续由更高容量 Model B 和完整 WakeWordEngine 承担高精度目标。

---

# 26. 数据来源与指标计算说明

## Recall / TPR

```text
Recall = TP / (TP + FN)
```

表示：
> 真正说了唤醒词的样本中，有多少成功触发。

## FRR

```text
FRR = FN / (TP + FN) = 1 - Recall
```

表示：
> 真正的唤醒词被漏掉的比例。

## Precision

```text
Precision = TP / (TP + FP)
```

表示：
> 所有触发中，有多少是真的唤醒词。

## FPR

```text
FPR = FP / (FP + TN)
```

表示：
> 负样本被错误唤醒的比例。

## F1

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

## ROC AUC

基于连续 score，而不是单个 frozen threshold：
- 衡量 positive / negative 排序区分能力
- 不等价于某个具体 operating point 的实际 Recall/FPR

## PR AUC

在 positive class 较重要时观察 precision-recall tradeoff。

## sequence score

v3：

```text
max_t(min(p[t], p[t+1], p[t+2]))
```

它不是简单：
```text
max_t(p[t])
```

后者在 v3 final evaluation 中明确：

```text
clip_level_max_score_used = false
```

---

# 27. 仍需在最终总报告中显式保留的未解决问题

1. training Keras total params 23,697 与 deployment parameter_count 19,697 的统计口径差异；
2. 为什么 VoxCPM Test Recall 只有 23.5%，而 Kokoro 94.5%；
3. 为什么 MeloTTS ZH external Recall 只有 1.54%；
4. v3 sequence objective 如何在降低 FPR 的同时追回 Recall；
5. standard real-world acceptance set 尚未完成；
6. ESP32-S3 真机尚未验证；
7. 当前没有证据证明 Model A 达到 98%；
8. 年龄 metadata 与感知年龄不是同一概念；
9. TTS benchmark 不等于真人麦克风使用性能；
10. Model A 的结果应作为模型级 benchmark，不应被包装成完整 WakeWordEngine 最终系统指标。

---

# 28. Model A 冻结建议

冻结标记建议：

```text
MODEL_A_STATUS = FROZEN
MODEL_A_QUALITY = A_LIMITED
MODEL_A_ROLE = TINY_BASELINE
MODEL_A_98PCT = NOT_ACHIEVED
```

除非后续整体项目完成后还有明确时间和新数据来源，否则不再启动 v4。

后续主线：

```text
Model B RepCNN
→ A/B comparison
→ unified WakeWordBackend
→ Energy
→ WebRTC VAD
→ 3-speech-frame pre-gate
→ DetectionLogic L1–L5
→ GUI
→ live microphone acceptance
→ ESP32-S3 export / compile preparation
```

---

**报告结束。**
