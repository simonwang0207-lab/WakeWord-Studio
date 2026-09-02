# 阶段 6 性能结论（P6/P7）

## 已排除

- Live backend 与正式离线 frontend/inference 对同一窗口分数一致，已排除 runtime frontend 分叉。
- 五条真人录音的 VAD 时长为 1.20～1.74 秒，每条都至少存在一个 2 秒窗口完整覆盖语音，已排除“唤醒词整体必然超过模型窗口”。
- 高低分录音的 RMS、Peak、VAD 时长存在重叠，已排除单一输入音量或单一语音时长解释全部失败。
- 五条录音只作开发诊断，未用于 threshold、checkpoint、smoothing 或 augmentation 选择。

## 最可能问题

1. B1 对真人发音/韵律/设备声学分布的泛化不足。ckpt-3000 在完全相同窗口协议下把五条最大分均值从 0.625 提高到 0.925，说明训练方向能显著改变真人响应。
2. 窗口位置与静音上下文是次要但真实的影响因素。部分最佳窗口没有完全覆盖 VAD 尾部，分数也随 0.20 秒位移明显变化；因此采用 0.20 秒 rolling hop 和 0.8 秒 tail 有工程价值。
3. 只有五条、没有音素级对齐，不能进一步断言具体是声母、韵母、语速、韵律还是麦克风频响造成差异。

## 当前已有增强

- 多类 procedural room/office/fan/street/music/device/TV speech 噪声。
- manifest 中已有部分 synthetic room/office/device response。
- 训练样本已有 front/middle/back phrase placement。
- mixup、focal loss、source/speaker/hard-negative 分层采样。

## B2.1 新增但尚未启动

- Train-only mild SpecAugment：最多 3 个时间 frame、2 个频率 bin，概率 0.5。
- Train-only 零填充 temporal shift：±3 feature frames，概率 0.5，不循环回绕。
- 独立 750-step fine-tune；从最终 B2 最佳权重继续；只用 Validation 选择。
- microphone frequency-tilt hook 已提供，但默认 0 dB/关闭，必须先做 Validation ablation。

## 现在不值得做

- 不值得根据五条真人录音降低 threshold、选择 checkpoint 或启用 smoothing。
- 不值得重新生成大型噪声/RIR 数据集；现有数据已有对应覆盖，且时间成本高。
- 不值得立即启用 microphone EQ 或永久 RMS normalization；缺少负样本 tradeoff 证据。
- 不值得在 B2 正式训练完成前启动 B2.1。
- 不值得修改 GUI DetectionLogic 来掩盖 raw score 问题。
