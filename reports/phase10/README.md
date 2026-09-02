# WakeWord Studio Phase 10

## 1. 系统架构

默认产品形态保持 `Browser Web Dashboard + Python backend`。浏览器将本地麦克风转换为 16 kHz mono PCM16，通过原有 HTTP 音频入口送入 `WakeWordEngine`；引擎仍执行 adaptive energy gate、WebRTC VAD、连续 3 帧语音门、2 秒 pre-roll、TFLite backend 与五层 DetectionLogic。Phase 10 通过既有 `WakeWordBackend` 增加动态 N-class Multi-KWS adapter，没有建立旁路 Demo。

## 2. 模型列表

- Binary：Model A microWakeWord/MixedNet、Model B RepCNN、历史 BC-ResNet binary、历史 ConvMixer binary。
- Multi-KWS：Teacher-Six BC-ResNet Full INT8（`COMPUTE_LIGHT_BASELINE`）、Teacher-Six ConvMixer Full INT8（`PRIMARY_CANDIDATE`）。
- Imported：用户导入且接口兼容的 TFLite；只有同时注册 trainer、architecture config 和 vocabulary metadata 才能训练。

Teacher-Six 的 threshold、margin、class names、SHA256、dataset/model version 与 Validation/Test 摘要来自冻结 artifact，不在 Web/JS/runtime 中散落硬编码。

## 3. Binary vs Multi-KWS

Binary backend 输出一个唤醒分数。Multi-KWS backend 输出 `[1,N]`，class 0 永远是 background，其余 class 来自 vocabulary。运行时返回 Top1、Top2、margin、background score、具体关键词和 rejection reason；类别数不是固定 7。

## 4. Teacher-Six final Test

| 模型 | Test Macro Recall | Test Macro F1 | Worst Recall | Background FAR | TFLite |
|---|---:|---:|---:|---:|---:|
| BC-ResNet | 89.89% | 87.94% | 74.00% | 14.50% | 108080 bytes |
| ConvMixer | 94.22% | 90.47% | 88.00% | 17.00% | 60408 bytes |

完整的逐词、混淆与 source 分析见 `reports/multikws/README.md`。`98PCT = NOT_ACHIEVED`。Phase 10 没有重跑 Test，也没有根据 Test 调 threshold/margin。

## 5. 当前默认模型

默认 active model 是 Teacher-Six ConvMixer Full INT8。它是 Validation-only 选择后冻结的 primary candidate，Test 的综合 Recall 和 worst-keyword Recall 也优于 BC-ResNet；BC-ResNet 保留为计算量较低的 baseline。模型切换必须显式点击“设为当前模型”，训练成功或失败都不会修改 active model；“回滚上一模型”恢复上一次激活记录。

## 6. 实时 pipeline 与判定

`Raw mic → Energy gate → WebRTC VAD → 3 speech frames → pre-roll → 99×40 frontend → Full INT8 TFLite → Top1/Top2 → DetectionLogic L1–L5 → accepted keyword → FIFO playback`

常见拒绝原因包括 `BACKGROUND_TOP1`、`LOW_TOP1_SCORE`、`LOW_MARGIN`、`TEMPORAL_EVIDENCE_INSUFFICIENT`、`BACKGROUND_RATIO_FAILED`、`COOLDOWN` 和 `POST_SILENCE_PENDING`。唤醒后 cooldown 防止重复播放“我醒来了”。

## 7. 新增提示词工作流

“＋ 新增提示词”第一次只调用 preflight：生成稳定 ASCII keyword ID、追加 class ID、不可变新 vocabulary、旧六词 replay 计划、background/ambient/ordinary/hard-negative 计划以及独立 run/dataset 名。它不会启动 TTS 或训练。

第二次显式点击“确认创建任务（不启动）”只在独立 run directory 写入新词生成配置、扩展词表、训练配置和 `USER_ACTION_COMMANDS.json`。正式执行时先只生成新词/新 hard-negative，再用 hardlink（不可用时 copy）组合旧 12K replay；合并器不解码 frozen Test WAV，源 dataset 不被覆盖。

正式流程是：

`扩展词表 → 新词数据 → 旧类 replay → 新 dataset version → ConvMixer/BC-ResNet 训练 → Validation calibration → freeze threshold/margin → Full INT8 → Validation INT8 → READY_CANDIDATE → 用户激活`

`ADD_KEYWORD_REQUIRES_RETRAIN = true`。Softmax 分类头增加 class 后不能零训练热插拔。

## 8. 防止遗忘

新 dataset 必须包含旧六类正样本 replay、background ordinary speech、confusion-aware hard negatives 与 ambient；旧 `teacher_six_multikws_v2_formal_12k` 保持 immutable，新数据使用例如 `teacher_six_plus_xiaozhi_v1`。旧 class ID 不重排，新 class 只追加到末尾。

## 9. 数据生成与 WAV 导入

自动数据沿用 Kokoro、VoxCPM1.5、augmentation、ordinary negatives、ambient 与 manifest pipeline。用户 WAV 被复制并标准化为 16 kHz mono PCM16，不覆盖源文件；坏 WAV 单独报告并跳过，不丢弃其它有效录音。音频永久保存默认关闭。

## 10. Job 与 Candidate

Multi-KWS job 状态为 `PENDING / DATA_PREPARING / READY_TO_TRAIN / TRAINING / VALIDATING / QUANTIZING / EVALUATING / READY_CANDIDATE / FAILED / CANCELLED`。cancel 保留 checkpoint，resume 使用同一独立 run directory。Candidate 只有具备训练报告、Float/INT8 Validation confusion、模型 metadata、Full INT8 和 SHA256 才能注册；注册也不会自动激活。

## 11. 真人麦克风验收

“六词快速验收”建议每词 10 次，记录 attempts/correct/wrong_keyword/rejected 与 hit rate，产物标为 `REAL_MIC_ACCEPTANCE`，绝不标为科研 Test。

`REAL_MIC_ACCEPTANCE = PENDING`

## 12. Background FAR vs False Wakes/hour

Background FAR 是离线 background 样本被错误接受的比例。False Wakes/hour 是连续真实环境音频里按观察时长归一化的误唤醒频率。二者数据单位、采样方式和产品含义不同，不能互换。首次本地验收建议连续测至少 5 分钟；稳定性评估建议累计 30–60 分钟。

## 13. ESP32-S3

保留 TFLite Micro inspect/header/package 路径，但当前没有实板测试。ConvMixer 估算 MACs 约 24.19M，BC-ResNet 约 6.59M；真实 latency 和峰值内存仍待实板验证。

`ESP32S3_RUNTIME_VERIFIED = false`

## 14. 当前限制

- 六词 98% 目标未达成。
- 真人麦克风验收尚未完成。
- ConvMixer Test Background FAR 为 17%，source gap 仍存在。
- 年龄标签未验证；multi-speaker 不等于 multi-age。
- 当前没有 ESP32-S3 实板 latency/内存证据。
- 新关键词正式数据生成与训练必须由用户显式启动，本轮未执行。
