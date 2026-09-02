# Third-party notices and publication boundaries

本文件用于记录 WakeWord-Studio 当前公开快照涉及的主要第三方工具、模型和数据来源。它不是法律意见，也不会替代各上游项目自己的许可证文本。发布者和使用者仍需以实际下载版本、commit 和许可证文件为准。

## 直接运行依赖

| 组件 | 用途 | 当前使用方式 | 上游许可/状态 |
|---|---|---|---|
| LiteRT (`ai-edge-litert`) | 已发布 TFLite 模型的本地推理 | Python runtime dependency，不在本仓库复制其源码或 wheel | Apache-2.0；以安装版本自带许可证为准 |
| pymicro-features | 运行时 16 kHz/40-channel 声学前端 | Python runtime dependency，不在本仓库复制其源码或 wheel | Apache-2.0；以安装版本自带许可证为准 |
| TensorFlow | 模型训练、量化和导出；普通实时运行不需要 | 可选的 Python training dependency，不在本仓库复制其源码或 wheel | Apache-2.0；以安装版本自带许可证为准 |
| LiveKit Embedded Wakeword | Binary 模型训练与历史兼容能力 | `pyproject.toml` 的 `training` extra 固定到上游 commit；不 vendor 上游仓库 | 上游仓库声明 Apache-2.0 |
| NumPy / SciPy / SoundFile / PyYAML / pypinyin / webrtcvad-wheels | 数值、音频、配置、中文处理和语音活动检测 | Python dependency；`webrtcvad-wheels` 提供 `webrtcvad` 导入接口 | 分别遵循各包安装版本的许可证 |
| tf2onnx / ONNX / ONNX Runtime | Phase 11 FP32 ONNX 导出与验证 | optional `export` dependencies | 分别遵循各包安装版本的许可证 |

## 数据生成与 reference 来源

这些来源用于本地数据生成或语音 reference。其完整模型、下载缓存和原始音频均不随 GitHub 仓库发布。

| 来源 | 本项目用途 | 仓库是否分发原始内容 | 发布前复核 |
|---|---|---:|---|
| Kokoro / Kokoro-82M | 合成 wake word、ordinary speech 和 hard-negative speech | 否 | 上游模型/常用 wrapper 声明 Apache-2.0；使用者仍需核对实际下载版本 |
| VoxCPM1.5 | 第二 speech source，结合 reference conditioning 合成语音 | 否 | OpenBMB VoxCPM 仓库提供 Apache-2.0 LICENSE；reference 音频还涉及说话人授权边界 |
| AISHELL-3 / OpenSLR 93 | VoxCPM reference speaker 音频 | 否 | OpenSLR 93 页面标明 Apache-2.0；本仓库不重新分发 19 GB corpus 或 reference WAV |
| 本地用户 WAV | 可选的自定义数据导入 | 否 | 上传/训练者必须拥有采集、使用与再处理授权 |

## 发布模型

`artifacts/models/` 和 `deliverables/onnx_board_test/models/` 中的模型由本项目训练得到，不是把 Kokoro、VoxCPM 或 AISHELL-3 的原始权重复制进仓库。除 artifact metadata 或文件中另有声明外，项目权利人选择按根目录 Apache-2.0 许可证发布这些模型。

这一项目许可证不替代第三方权利核查：

- 使用者仍需遵守实际使用的 TTS、reference corpus 和依赖版本的许可证；
- 语音克隆/reference conditioning 必须具备合法的说话人授权；
- 不应把 GitHub Public 状态本身当作第三方数据授权；
- 不应上传原始/增强语音、reference WAV、模型下载缓存或第三方大模型权重。

## 本地排除项

以下内容通过 `.gitignore` 排除且保留在开发机：

- `.envs/`、`.venv/` 与包/模型缓存；
- `datasets/`、`runs/`、checkpoint、feature arrays 和日志；
- `g2pW/` 本地模型缓存；
- TTS/ASR 上游 clone、weights、wheels 和 Hugging Face cache；
- `.env`、credential、私钥、runtime feedback 与麦克风测试会话。

## 上游链接

- LiveKit Embedded Wakeword: <https://github.com/livekit/livekit-embedded-wakeword>
- Kokoro-82M: <https://huggingface.co/hexgrad/Kokoro-82M>
- VoxCPM: <https://github.com/OpenBMB/VoxCPM>
- AISHELL-3 / OpenSLR 93: <https://www.openslr.org/93/>
