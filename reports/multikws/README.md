# Teacher Six Multi-KWS：Validation 收口与冻结 Test 报告

> Validation 部分与模型选择由正式 run artifact 自动生成；Test 部分只读取已经存在的 immutable `TEST_REPORT.json` 汇总，Phase 10 未重新推理、未重新校准、未覆盖 Test artifact。
> `selection_source=validation_only`，`THRESHOLD_CHANGED_AFTER_TEST=false`，`98PCT=NOT_ACHIEVED`。

## 1. 实验目标

一个 7-class softmax 模型同时识别六个提示词；第 0 类是 `background`，不是六个独立 binary model。

| class | keyword_id | 显示文本 |
|---:|---|---|
| 0 | background | 背景/拒识 |
| 1 | qingxiaojia | 你好，青小甲 |
| 2 | doudou | 你好，豆豆 |
| 3 | diandian | 你好，点点 |
| 4 | xiaorui | 你好，小瑞 |
| 5 | duoduo | 你好，多多 |
| 6 | jizhiwa | 你好，吉智娃 |

## 2. Dataset 与公平协议

- Dataset ID：`teacher_six_multikws_v2_formal_12k`；dataset SHA256：`27c9d0ed7273bd81262009bd45e2431d8b8183796c1d9bee8a7e6ae66970d77c`。
- Train / Validation / Test：9000 / 1500 / 1500。
- 来源计数：Kokoro 5400、VoxCPM1.5 5400、procedural ambient 1200。
- 两模型使用同一 dataset、split、deterministic epoch sampler、seed 与 Validation calibration/ranking protocol；PTQ representative split 均为 `train`。
- Test 未用于特征提取、训练、校准或模型选择；冻结 operating point 后已执行一次正式 held-out Test，Phase 10 仅汇总既有报告。
- 数据是 multi-source / multi-speaker；`AGE_VERIFIED=false`，multi-speaker 不等于 multi-age。

## 3. 模型训练与部署概况

`parameter_count` 严格采用 `TRAINING_REPORT.json` 的可训练参数统计口径。

| 模型 | Architecture | Steps / Epochs | Early stop | Params | Estimated MACs | TFLite bytes / KiB | SHA256 | Full INT8 | Hardware verified |
|---|---|---:|---|---:|---:|---:|---|---|---|
| bcresnet | `{"architecture":"broadcasted_residual_network","channels":40,"depth":8,"width_multiplier":1.0,"subbands":4,"temporal_dilations":[1,2,4],"dropout":0.1,"activation":"relu"}` | 6486 / 23.0 | true | 19287 | 6589720 | 108080 / 105.55 | `1176f3752b0a7a7056efa8dad5a917f1177d50e3ebeef434d1a87af387a2070a` | true | false |
| convmixer | `{"architecture":"acoustic_convmixer","hidden_dim":48,"depth":6,"kernel_size":[7,5],"patch_size":[3,2],"stride":[2,2],"dropout":0.1,"activation":"relu"}` | 4512 / 16.0 | true | 25783 | 24192336 | 60408 / 58.99 | `acc517399e72a41f3161d700702fb71db4826face2be7184f90d91375034d476` | true | false |

## 4. BC-ResNet vs ConvMixer 总体 Validation 对比

| 模型 | 阶段 | Macro Recall | Macro Precision | Macro F1 | Micro Accuracy | Worst Keyword Recall | Background FAR | Background Rejection Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bcresnet | Float | 94.44% | 91.25% | 92.71% | 91.73% | 87.33% | 12.33% | 87.67% |
| bcresnet | Full INT8 | 90.00% | 89.49% | 89.37% | 88.20% | 78.67% | 14.50% | 85.50% |
| convmixer | Float | 92.44% | 92.83% | 92.55% | 92.73% | 88.00% | 6.83% | 93.17% |
| convmixer | Full INT8 | 92.89% | 88.77% | 90.56% | 90.53% | 86.67% | 13.00% | 87.00% |

## 5. 六关键词逐项对比

### Float

| 关键词 | BC Recall | BC Precision | BC F1 | Conv Recall | Conv Precision | Conv F1 |
|---|---:|---:|---:|---:|---:|---:|
| qingxiaojia / 你好，青小甲 | 94.00% | 92.76% | 93.38% | 88.00% | 92.96% | 90.41% |
| doudou / 你好，豆豆 | 97.33% | 88.48% | 92.70% | 94.00% | 92.16% | 93.07% |
| diandian / 你好，点点 | 94.67% | 90.45% | 92.51% | 88.67% | 95.00% | 91.72% |
| xiaorui / 你好，小瑞 | 96.67% | 95.39% | 96.03% | 97.33% | 86.39% | 91.54% |
| duoduo / 你好，多多 | 87.33% | 95.62% | 91.29% | 93.33% | 95.24% | 94.28% |
| jizhiwa / 你好，吉智娃 | 96.67% | 84.80% | 90.34% | 93.33% | 95.24% | 94.28% |

### Full INT8（最终部署候选）

| 关键词 | BC Recall | BC Precision | BC F1 | Conv Recall | Conv Precision | Conv F1 |
|---|---:|---:|---:|---:|---:|---:|
| qingxiaojia / 你好，青小甲 | 89.33% | 88.16% | 88.74% | 86.67% | 93.53% | 89.97% |
| doudou / 你好，豆豆 | 78.67% | 90.08% | 83.99% | 93.33% | 87.50% | 90.32% |
| diandian / 你好，点点 | 88.00% | 93.62% | 90.72% | 96.67% | 85.80% | 90.91% |
| xiaorui / 你好，小瑞 | 98.67% | 94.87% | 96.73% | 97.33% | 78.07% | 86.65% |
| duoduo / 你好，多多 | 87.33% | 95.62% | 91.29% | 90.67% | 94.44% | 92.52% |
| jizhiwa / 你好，吉智娃 | 98.00% | 74.62% | 84.73% | 92.67% | 93.29% | 92.98% |

## 6. Float → INT8 量化稳定性

下表为 `INT8 - Float`；正值表示数值升高，负值表示下降。

| 模型 | Macro Recall | Macro F1 | Worst Keyword Recall | Background FAR |
|---|---:|---:|---:|---:|
| bcresnet | -4.44 pp | -3.34 pp | -8.67 pp | +2.17 pp |
| convmixer | +0.44 pp | -1.99 pp | -1.33 pp | +6.17 pp |

BC-ResNet 存在明显 PTQ sensitivity，尤其 `doudou`；ConvMixer 的 Recall 量化稳定性明显更好，但其 Background FAR 在量化后仍明显升高。INT8 的个别指标高于 Float 只表示该冻结 operating point 和有限 Validation 样本上的观测差异，不表示量化使模型本质变强。

## 7. Confusion / Error Analysis

`关键词 → background` 是 false reject/rejection；`关键词 → 另一个关键词` 才是 keyword-to-keyword confusion。最终输出为 background 可能来自 top-1 本来就是 background，也可能来自非 background top-1 未通过 threshold/margin。

| 模型 | 阶段 | true → predicted | count |
|---|---|---|---:|
| bcresnet | float | duoduo → background | 13 |
| bcresnet | float | diandian → background | 8 |
| bcresnet | float | qingxiaojia → background | 7 |
| bcresnet | float | duoduo → doudou | 6 |
| bcresnet | float | xiaorui → background | 5 |
| bcresnet | float | doudou → background | 4 |
| bcresnet | float | jizhiwa → background | 3 |
| bcresnet | float | qingxiaojia → jizhiwa | 2 |
| bcresnet | int8 | doudou → background | 27 |
| bcresnet | int8 | diandian → background | 16 |
| bcresnet | int8 | duoduo → background | 13 |
| bcresnet | int8 | qingxiaojia → background | 12 |
| bcresnet | int8 | doudou → qingxiaojia | 5 |
| bcresnet | int8 | duoduo → doudou | 4 |
| bcresnet | int8 | qingxiaojia → jizhiwa | 4 |
| bcresnet | int8 | jizhiwa → background | 3 |
| convmixer | float | diandian → background | 15 |
| convmixer | float | qingxiaojia → background | 11 |
| convmixer | float | jizhiwa → background | 8 |
| convmixer | float | duoduo → doudou | 5 |
| convmixer | float | doudou → background | 3 |
| convmixer | float | doudou → duoduo | 3 |
| convmixer | float | duoduo → background | 3 |
| convmixer | float | qingxiaojia → xiaorui | 3 |
| convmixer | int8 | jizhiwa → background | 9 |
| convmixer | int8 | qingxiaojia → background | 9 |
| convmixer | int8 | duoduo → doudou | 6 |
| convmixer | int8 | diandian → background | 5 |
| convmixer | int8 | qingxiaojia → doudou | 4 |
| convmixer | int8 | qingxiaojia → xiaorui | 4 |
| convmixer | int8 | doudou → duoduo | 3 |
| convmixer | int8 | duoduo → background | 3 |

关键词 → background（最终输出第 0 类）的拒识统计：

| 模型 | 阶段 | qingxiaojia | doudou | diandian | xiaorui | duoduo | jizhiwa |
|---|---|---:|---:|---:|---:|---:|---:|
| bcresnet | float | 7 | 4 | 8 | 5 | 13 | 3 |
| bcresnet | int8 | 12 | 27 | 16 | 2 | 13 | 3 |
| convmixer | float | 11 | 3 | 15 | 3 | 3 | 8 |
| convmixer | int8 | 9 | 2 | 5 | 2 | 3 | 9 |

近音关键词 doudou / diandian / duoduo 的定向混淆：

| 模型 | 阶段 | doudou→diandian | doudou→duoduo | diandian→doudou | diandian→duoduo | duoduo→doudou | duoduo→diandian |
|---|---|---:|---:|---:|---:|---:|---:|
| bcresnet | float | 0 | 0 | 0 | 0 | 6 | 0 |
| bcresnet | int8 | 0 | 0 | 0 | 0 | 4 | 0 |
| convmixer | float | 0 | 3 | 1 | 0 | 5 | 1 |
| convmixer | int8 | 2 | 3 | 0 | 0 | 6 | 2 |

- bcresnet float：background 最常被误识别为 `jizhiwa`（24 条）。
- bcresnet int8：background 最常被误识别为 `jizhiwa`（44 条）。
- convmixer float：background 最常被误识别为 `xiaorui`（16 条）。
- convmixer int8：background 最常被误识别为 `xiaorui`（32 条）。

## 8. Source Generalization

| 模型 | 阶段 | 关键词 | Kokoro Recall | VoxCPM1.5 Recall |
|---|---|---|---:|---:|
| bcresnet | Float | qingxiaojia | 100.00% | 88.00% |
| bcresnet | Float | doudou | 100.00% | 94.67% |
| bcresnet | Float | diandian | 100.00% | 89.33% |
| bcresnet | Float | xiaorui | 100.00% | 93.33% |
| bcresnet | Float | duoduo | 98.67% | 76.00% |
| bcresnet | Float | jizhiwa | 100.00% | 93.33% |
| bcresnet | Full INT8 | qingxiaojia | 94.67% | 84.00% |
| bcresnet | Full INT8 | doudou | 80.00% | 77.33% |
| bcresnet | Full INT8 | diandian | 97.33% | 78.67% |
| bcresnet | Full INT8 | xiaorui | 100.00% | 97.33% |
| bcresnet | Full INT8 | duoduo | 100.00% | 74.67% |
| bcresnet | Full INT8 | jizhiwa | 100.00% | 96.00% |
| convmixer | Float | qingxiaojia | 97.33% | 78.67% |
| convmixer | Float | doudou | 97.33% | 90.67% |
| convmixer | Float | diandian | 89.33% | 88.00% |
| convmixer | Float | xiaorui | 100.00% | 94.67% |
| convmixer | Float | duoduo | 100.00% | 86.67% |
| convmixer | Float | jizhiwa | 100.00% | 86.67% |
| convmixer | Full INT8 | qingxiaojia | 100.00% | 73.33% |
| convmixer | Full INT8 | doudou | 97.33% | 89.33% |
| convmixer | Full INT8 | diandian | 100.00% | 93.33% |
| convmixer | Full INT8 | xiaorui | 100.00% | 94.67% |
| convmixer | Full INT8 | duoduo | 100.00% | 81.33% |
| convmixer | Full INT8 | jizhiwa | 98.67% | 86.67% |

Kokoro 上接近满分不能外推为真实系统接近 98%；VoxCPM1.5 明显更差，表明跨 source / reference speaker 泛化仍是主要瓶颈。procedural ambient 不是 speech source，因此没有 per-keyword Recall。

## 9. 部署成本

BC-ResNet 的 TRAINING_REPORT estimated MACs 为 6,589,720，ConvMixer 为 24,192,336（约 3.67×）。BC-ResNet 计算量更小；ConvMixer TFLite 更小且 INT8 Validation 更稳，但计算量明显更高。实际 ESP32-S3 latency 尚未验证。若 exporter 日志另有 MAC 数字，它与 TRAINING_REPORT 的静态 `estimated_macs` 属于不同统计口径，不应混用。

## 10. 指标与 operating point 说明

- **Worst Keyword Recall**：六个关键词 Recall 的最小值。
- **Background FAR**：真实 background 在冻结决策逻辑后被输出为任一关键词的比例。
- **Background Rejection Rate**：真实 background 最终输出第 0 类的比例，与同一集合上的 FAR 互补。
- **Per-source Per-keyword Recall**：在指定 speech source 且真实标签为该关键词的子集上，最终正确输出该关键词的比例。
- **Float→INT8 degradation**：同一 Validation 与冻结 operating point 下 INT8 指标减 Float 指标；`pp` 是百分点。
- **Threshold / margin threshold**：运行时代码先稳定降序取 top-1/top-2。只有 top-1 不是 background、top-1 score ≥ threshold 且 top-1−top-2 ≥ margin threshold 时才接受关键词；否则输出第 0 类。
- **MACs**：模型一次前向的静态乘加次数估算，不等同于真实硬件 latency。
- **Full INT8/PTQ**：用 Train representative samples 做训练后量化，TFLite 输入、输出及受支持算子均为 INT8；未做量化感知训练。

## 11. 当前阶段结论

- BC-ResNet：`ROLE=COMPUTE_LIGHT_BASELINE`。
- ConvMixer：`ROLE=PRIMARY_CANDIDATE`；Final INT8 overall / worst-keyword 更均衡、PTQ stability 更好且文件更小，但 MACs 明显更高。
- `98PCT=NOT_ACHIEVED`；不能声称六关键词达到 98%。

## 12. 当前限制

- Validation 是合成 benchmark；real microphone acceptance 未完成。
- 冻结 held-out Test 已完成；不得再用它调整 threshold、margin 或超参数。
- ESP32-S3 hardware runtime 未验证。
- `AGE_VERIFIED=false`。
- source / speaker 泛化仍不足。

## 13. 冻结 Test：Validation vs Test

以下数值来自 `reports/multikws/test/{bcresnet,convmixer}/TEST_REPORT.json`，没有重新运行 Test。

| 模型 | Split | Macro Recall | Macro Precision | Macro F1 | Micro Accuracy | Worst Recall | Background FAR |
|---|---|---:|---:|---:|---:|---:|---:|
| BC-ResNet | Validation INT8 | 90.00% | 89.49% | 89.37% | 88.20% | 78.67% | 14.50% |
| BC-ResNet | Test INT8 | 89.89% | 86.59% | 87.94% | 88.13% | 74.00% | 14.50% |
| ConvMixer | Validation INT8 | 92.89% | 88.77% | 90.56% | 90.53% | 86.67% | 13.00% |
| ConvMixer | Test INT8 | 94.22% | 87.41% | 90.47% | 89.73% | 88.00% | 17.00% |

ConvMixer 保持 `PRIMARY_CANDIDATE`：Test Macro Recall、Worst Recall 与文件大小优于 BC-ResNet；BC-ResNet 仍是低计算量 `COMPUTE_LIGHT_BASELINE`。ConvMixer Test Background FAR 升到 17%，且没有真实连续麦克风 False Wakes/hour 结果，因此不能把离线 Recall 外推成产品验收通过。

## 14. Test 六关键词 Recall

| 关键词 | BC-ResNet | ConvMixer |
|---|---:|---:|
| 你好，青小甲 | 74.00% | 94.00% |
| 你好，豆豆 | 94.00% | 97.33% |
| 你好，点点 | 91.33% | 96.00% |
| 你好，小瑞 | 97.33% | 98.67% |
| 你好，多多 | 86.67% | 91.33% |
| 你好，吉智娃 | 96.00% | 88.00% |

- BC-ResNet 最大问题是 `qingxiaojia → jizhiwa`（27 条）。
- ConvMixer 的主要错误包括 `jizhiwa → background`（12 条）与 `duoduo → doudou`（8 条）。
- 两模型仍表现出 source gap。BC-ResNet 的 VoxCPM `duoduo` Recall 为 73.33%；ConvMixer 的 VoxCPM `jizhiwa` Recall 为 76.00%。Kokoro 上的高分不能代表真人麦克风。

最终结论保持：`98PCT=NOT_ACHIEVED`，`REAL_MIC_ACCEPTANCE=PENDING`，`ESP32S3_RUNTIME_VERIFIED=false`。
