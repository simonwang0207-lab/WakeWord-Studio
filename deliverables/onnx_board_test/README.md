# Teacher-Six Multi-KWS ONNX 板端测试包

这是两个 **7-class Multi-KWS 模型**。它们同时支持 background、你好青小甲、你好豆豆、你好点点、你好小瑞、你好多多、你好吉智娃。

## 文件

- `models/BCResNet_TeacherSix_MultiKWS_FP32.onnx`：BC-ResNet FP32，计算量较低。
- `models/ConvMixer_TeacherSix_MultiKWS_FP32.onnx`：ConvMixer FP32，PC/offline 正式结果更好、模型文件较小，但静态 MAC 约为 BC-ResNet 的 3.67 倍。
- `labels.txt`：输出 class order。
- `model_info.json`：来源 checkpoint、shape、opset、hash、正式结果摘要。
- `frontend_spec.md`：板端必须严格复现的音频前端。
- `verification/test_vectors/`：固定 Validation features 与两个模型的期望输出。

两个模型输入均为 `input_features`、float32、`[1,99,40]`；输出均为 `class_scores`、float32、`[1,7]`。`class_scores[0]` 是 background，`class_scores[1:7]` 依次对应 `labels.txt`。输出是 softmax class scores。

**模型本身不直接接收 WAV/PCM。** 板端必须先实现与 `frontend_spec.md` 完全一致的 TFLite Micro microfrontend。`FRONTEND_PARITY_REQUIRED=true`。

## 测试结果

- BC-ResNet：Test Macro Recall 89.89%，Worst Keyword Recall 74.00%，Background FAR 14.50%；静态估算 6,589,720 MAC。优点是算力较低；现有正式 INT8 相对 Float 的量化退化更明显。
- ConvMixer：Test Macro Recall 94.22%，Worst Keyword Recall 88.00%，Background FAR 17.00%；静态估算 24,192,336 MAC。整体效果更好、文件小，但计算量约为 BC-ResNet 的 3–4 倍，且 Background FAR 相对更高。


