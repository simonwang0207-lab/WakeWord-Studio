# Frontend specification

`FRONTEND_PARITY_REQUIRED=true`

这两个 ONNX **不接收 WAV/PCM**。输入是与训练完全一致的 `[1,99,40]` float32 TFLite Micro microfrontend filterbank；板端若不复现该前端，模型结果没有可比性。

## 从 PCM 到模型输入

1. 音频必须为 16,000 Hz、mono。文件生成规范为 PCM16；特征提取读取为 float32 `[-1,1]`。
2. 每条输入固定为 32,000 samples（2.000 s）。更长时中心裁剪；更短时左右近似等量补零，奇数个缺失 sample 的额外 1 个零放在右侧。不做 VAD、静音裁剪或响度归一化。
3. float 音频通过 `clip(audio * 32768, -32768, 32767).astype(int16)` 转为有符号 PCM16。
4. 正式 cache 使用 `pymicro-features 2.0.2` 的 TFLite Micro microfrontend：30 ms frame（480 samples）、内部 10 ms step（160 samples）、40 个 filterbank channels、125–7500 Hz。
5. TFLM frontend 顺序执行切窗、512-point FFT（257 个非冗余频点）、filterbank、noise reduction、PCAN auto gain、log scale。关键固定参数：`smoothing_bits=10`、`even_smoothing=0.025`、`odd_smoothing=0.06`、`min_signal_remaining=0.05`、`pcan_strength=0.95`、`pcan_offset=80`、`gain_bits=21`、`enable_log=true`、`scale_shift=6`。
6. C frontend 的 uint16 输出乘 `0.0390625` 得到 float32。没有额外 mean/std normalization、CMVN、MFCC/DCT 或模型侧 input scaling。
7. 每个 2 s clip 新建/reset frontend state；先产生 10 ms-hop 特征，再取 `frames[::2]`，得到 20 ms hop 的 99 帧。
8. 张量布局是 `[batch, time, filterbank_channel]`：batch=1，99 按时间从早到晚，40 个通道从低频到高频。最终输入必须 contiguous float32，shape `[1,99,40]`。

正式 Validation feature cache：`datasets/projects/teacher_six_multikws_v2_formal_12k/train_validation_features.npz`  
SHA256：`b8f48798358c91bb126252f2f42165baf5761f31cce4822cc651f8879831d8f6`  
metadata 明确 `TEST_READ=false`。本次还用真实 Validation WAV 重算第一个样本，得到 `backend=pymicro-features`、shape `[99,40]`、与 cache `max_abs_diff=0.0`，从而确认上述实际路径。

## 重要边界

- ONNX 仅包含分类网络，不包含上述音频 frontend。
- 40 维是 filterbank feature，不是普通浮点 mel-spectrogram，也不是 40 维 MFCC。
- 若芯片 SDK 自带“Mel/MFCC”，不能只按尺寸相同就替代；必须用 test vectors 验证数值一致性。
- test vectors 已经是 frontend 输出，可先绕过麦克风链路验证 ONNX/芯片编译器，再验证真实 PCM frontend。
