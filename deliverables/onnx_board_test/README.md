# Teacher-Six Multi-KWS ONNX 板端测试包

这是两个 **7-class Multi-KWS 模型**，不是六个独立模型。它们同时支持 background、你好青小甲、你好豆豆、你好点点、你好小瑞、你好多多、你好吉智娃。

## 文件

- `models/BCResNet_TeacherSix_MultiKWS_FP32.onnx`：BC-ResNet FP32，计算量较低。
- `models/ConvMixer_TeacherSix_MultiKWS_FP32.onnx`：ConvMixer FP32，PC/offline 正式结果更好、模型文件较小，但静态 MAC 约为 BC-ResNet 的 3.67 倍。
- `labels.txt`：输出 class order。
- `model_info.json`：来源 checkpoint、shape、opset、hash、正式结果摘要。
- `frontend_spec.md`：板端必须严格复现的音频前端。
- `verification/test_vectors/`：固定 Validation features 与两个模型的期望输出。

两个模型输入均为 `input_features`、float32、`[1,99,40]`；输出均为 `class_scores`、float32、`[1,7]`。`class_scores[0]` 是 background，`class_scores[1:7]` 依次对应 `labels.txt`。输出是 softmax class scores。

**模型本身不直接接收 WAV/PCM。** 板端必须先实现与 `frontend_spec.md` 完全一致的 TFLite Micro microfrontend。`FRONTEND_PARITY_REQUIRED=true`。

## 正式结果摘要（读取既有冻结报告，未重跑 Test）

- BC-ResNet：Test Macro Recall 89.89%，Worst Keyword Recall 74.00%，Background FAR 14.50%；静态估算 6,589,720 MAC。优点是算力较低；现有正式 INT8 相对 Float 的量化退化更明显。
- ConvMixer：Test Macro Recall 94.22%，Worst Keyword Recall 88.00%，Background FAR 17.00%；静态估算 24,192,336 MAC。整体效果更好、文件小，但计算量约为 BC-ResNet 的 3–4 倍，且 Background FAR 相对更高。

这些 ONNX 从正式 Float best checkpoint 直接恢复并导出，不是从 INT8 TFLite 反向转换。threshold/margin 没有改变；ONNX 输出本身也没有内置运行时 threshold/margin 判定。

## 建议板端测试

1. 芯片工具链能否转换，及 operator support；
2. Flash、SRAM/PSRAM、tensor/workspace 占用；
3. 单次 inference latency、real-time factor、连续运行稳定性；
4. 先用 test vectors 比较 7 个输出，再接真实麦克风 frontend；
5. 分别测六个词、相似词/硬负例、背景语音和环境噪声；
6. 比较 BC-ResNet 的低算力优势与 ConvMixer 的 PC/offline 效果优势，再决定部署模型。

ONNX checker 与 ONNX Runtime PASS 只证明桌面 ONNX artifact 合法且与 Float Keras 数值一致，不代表芯片已经运行成功：`CHIP_RUNTIME_VERIFIED=false`、`ESP32S3_RUNTIME_VERIFIED=false`。
