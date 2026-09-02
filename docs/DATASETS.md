# 数据集复现与 GitHub 发布策略

## 1. 为什么 GitHub 不包含训练音频

本仓库的 GitHub 快照面向源码、实验协议、配置、报告和小型部署模型。完整数据目录包含数 GiB WAV、特征缓存、partial records 与本机 TTS cache，不适合普通 Git，也可能涉及语音来源许可与隐私边界。因此 `.gitignore` 排除 `datasets/`、音频格式、NPZ/NPY 和 phase artifacts；本地文件不删除。

## 2. Teacher-Six 12K 正式数据集

| 字段 | 值 |
|---|---|
| Dataset ID | `teacher_six_multikws_v2_formal_12k` |
| Total | 12,000 |
| Train / Validation / Test | 9,000 / 1,500 / 1,500 |
| Wake classes | 六词各 1,200 |
| Background | ordinary 2,400 + hard negative 1,200 + ambient 1,200 |
| Speech sources | Kokoro 5,400 + VoxCPM1.5 5,400 |
| Procedural ambient | 1,200 |
| Audio | 16 kHz、mono、PCM16 |
| Speaker/reference disjoint | Kokoro=true；VoxCPM=true |
| Base-group split leakage | 0 |
| Age verified | false |

Hashes：

```text
dataset_sha256=27c9d0ed7273bd81262009bd45e2431d8b8183796c1d9bee8a7e6ae66970d77c
manifest_sha256=8c4f8008c6344efb575a491c19256686d8321896285b0519c90b8ce766695116
manifest_file_sha256=ee31f0e94f16d58864b9db2125c1c4c8f99e5ca982e5e440796371cfd5afde46
```

Git 中保留：

- `configs/multikws/teacher_six_formal_12k.json`
- `configs/multikws/teacher_six_keywords.json`
- `artifacts/datasets/teacher_six_multikws_v2_formal_12k/DATASET_INFO.json`
- `artifacts/datasets/teacher_six_multikws_v2_formal_12k/MANIFEST_METADATA.json`
- `reports/multikws/README.md`

Git 中不保留：

- `datasets/projects/teacher_six_multikws_v2_formal_12k/**/*.wav`
- 27.8 MB 的 Train/Validation feature NPZ；
- 21.4 MB、含 12,000 条逐样本记录和原开发机绝对路径的完整 manifest；
- base TTS cache、partial JSONL 和本地 generation status。

## 3. 数据分布

| 类别 | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 每个 wakeword（共六个） | 900 | 150 | 150 | 1,200 |
| Ordinary background speech | 1,800 | 300 | 300 | 2,400 |
| Hard-negative speech | 900 | 150 | 150 | 1,200 |
| Procedural ambient | 900 | 150 | 150 | 1,200 |
| **全部样本** | **9,000** | **1,500** | **1,500** | **12,000** |

每个 keyword × split 中 Kokoro 与 VoxCPM 1.5 平衡：Train 各 450，Validation/Test 各 75。Ordinary 与 hard-negative speech 也在两个 source 间平衡。Ambient 不算 speech source。

## 4. 增强配置

正式 Train 配置启用 speed 0.90/0.97/1.04/1.10、gain -4 至 +3 dB、leading silence 40–220 ms、trailing silence 40–250 ms、reverb/far-field 概率 0.4、SNR 5/10/15/20 dB，以及 office、fan/AC、keyboard、TV/speech、babble、street、car、classroom、cafe、device/mic 噪声。完整 effective config 在 `teacher_six_formal_12k.json`。

## 5. 重建流程

数据生成依赖本地 Kokoro 与 VoxCPM1.5/AISHELL-3 reference 环境；这些模型、reference audio 和 cache 不随 Git 分发。准备好合法来源与兼容环境后：

1. 检查 `configs/multikws/teacher_six_formal_12k.json` 中相对输出路径和 provider 配置；
2. 运行 `phase9/scripts/build_multikws_12k_dataset.py --help` 核对当前命令参数；
3. 先执行项目的 Phase 9 smoke/preflight；
4. 生成正式数据并保留 `DatasetManifest.json`、`DATASET_INFO.json` 和状态文件；
5. 用 `phase9/scripts/audit_multikws_dataset.py` 做 split/source/leakage 审计；
6. 用 `phase9/scripts/extract_multikws_features.py` 只提取 Train/Validation 特征；
7. 在重建完成后比较 dataset、manifest logical hash 与记录分布。

不要把本报告中的原开发机路径当作必需路径，也不要在没有合法 source/reference 的情况下伪造 speaker 或年龄 metadata。

## 6. Test 口径

原 Teacher-Six Test 在训练与选择阶段未读取，模型与 operating point 冻结后执行过一次。结果现已知；如果未来依据这些结果修改数据或模型，原 Test 只能作为已知回归集，必须建立新的 untouched holdout 才能做下一次无偏最终评估。
