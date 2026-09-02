# Model B augmentation audit（Train/Validation only）

结论：B2 数据本身已经包含多类噪声和部分合成房间响应；不能把 reverb/noise 视为完全缺失，也不应在最后阶段重新生成大型数据集。

| 项目 | 当前 B2 | B2.1 低风险预案 |
|---|---|---|
| 环境噪声 | 已存在 procedural room/office/fan/street/music/device/TV speech 等 | 保留，不重复生成 |
| Reverb / room response | manifest 元数据存在 synthetic room/office/device response | 保留，不额外堆叠 RIR |
| Temporal variation | 训练集已有 phrase placement，但 B2 feature batch 没有额外小幅时移 | Train-only 零填充 ±3 feature frames，概率 0.5 |
| SpecAugment | B2 config 明确关闭 | Train-only，最多 3 个时间帧和 2 个频率 bin，概率 0.5 |
| Microphone EQ / bandwidth | 没有明确的设备频响增强 | 只提供受限 frequency-tilt hook；默认 0 dB、关闭，必须先做 Validation ablation |
| 真人 5 条录音 | 仅诊断 | 禁止训练、阈值或 augmentation 选择 |

安全边界：所有新增增强只作用于 Train feature batch；Validation 不增强；Test 不可达；当前 B2 config/run 不修改。B2.1 是否启动必须由用户在 B2 完成并审阅 Validation 后决定。
