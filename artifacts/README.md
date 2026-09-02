# Release Artifacts

本目录保存适合普通 Git 分发的小型冻结 artifact。以下模型均由 WakeWord-Studio 使用项目数据训练得到，不是直接下载的中文成品唤醒模型。
## 发布模型

| Model ID | 文件 | 任务 | 角色 | 大小 | SHA256 |
|---|---|---|---|---:|---|
| `model_a`（兼容 ID） | `models/binary/microwakeword_mixednet_full_int8.tflite` | microWakeWord Tiny / Binary | HISTORICAL | 52,840 B | `994f08b799364f02f6fc06273cccd4a155722af25f1b61a88f4e5b2621a2d41c` |
| `model_b`（兼容 ID） | `models/binary/repcnn_full_int8.tflite` | RepCNN / Binary | HISTORICAL | 112,816 B | `6acfecf7cc8651c1fba52809eee1d89abbcffa0a48bd46662b2e58ac23ce064f` |
| `bcresnet_binary_formal` | `models/binary/bcresnet_binary_full_int8.tflite` | Binary | HISTORICAL | 108,784 B | `474ad90681a75acfd51fa41df1c69d43aa27ce1e2bf6f97054fa1529f370cc87` |
| `convmixer_binary_formal` | `models/binary/convmixer_binary_full_int8.tflite` | Binary | HISTORICAL | 59,984 B | `236893035d0806aef6b085079f5ac706403bfb2889f74881d6dda70b23cd1580` |
| `teacher_six_bcresnet` | `models/teacher_six/teacher_six_bcresnet_full_int8.tflite` | Multi-KWS | COMPUTE_LIGHT_BASELINE | 108,080 B | `1176f3752b0a7a7056efa8dad5a917f1177d50e3ebeef434d1a87af387a2070a` |
| `teacher_six_convmixer` | `models/teacher_six/teacher_six_convmixer_full_int8.tflite` | Multi-KWS | PRIMARY_CANDIDATE / current active | 60,408 B | `acc517399e72a41f3161d700702fb71db4826face2be7184f90d91375034d476` |

每个 `.tflite` 旁边的 `.metadata.json` 记录 model ID、架构、任务、输入输出、类别数、字节数、SHA256、冻结 threshold/margin、Validation/Test 摘要、来源 run 和硬件验证状态。汇总索引为 `metadata/MODEL_MANIFEST.json`。

## 数据集元数据

`datasets/teacher_six_multikws_v2_formal_12k/` 只包含精简的正式数据集信息与 hash，不包含 12,000 个 WAV，也不包含带原开发机绝对路径的完整逐样本 manifest。重建方式见 [`../docs/DATASETS.md`](../docs/DATASETS.md)。

## 发布边界

- Full INT8 模型：包含；
- 原始训练 WAV：不包含；
- feature NPZ/NPY：不包含；
- Keras weights/checkpoint：不包含；
- TTS 下载缓存与虚拟环境：不包含；
- ESP32-S3 实板验证：尚未完成，所有 metadata 均为 `hardware_runtime_verified=false`。
