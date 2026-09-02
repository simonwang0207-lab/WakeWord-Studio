# Phase 10 前 Existing Feature Inventory

`PRESERVE = true`

本清单记录 Phase 10 修改前已经存在的能力。Phase 10 只允许兼容升级；下列模型、入口、页面、API 与 artifact 不得删除或覆盖。

## A. 模型与 Backend

- `WakeWordBackend` 抽象：`train / evaluate / export / load / stream_scores / reset_stream`。
- `MicroWakeWordBackend`：microWakeWord / MixedNet，原生流式 `[1,3,40] → [1,1]` Full INT8/UINT8。
- `RepCNNBackend`：2 秒 rolling window、0.20 秒 hop、MicroFrontend `[1,99,40] → [1,1]` Full INT8。
- 已注册产品模型：Model A — microWakeWord Tiny；Model B — RepCNN。
- 已有但未进入产品 registry 的正式 artifact：historical BC-ResNet binary、historical ConvMixer binary、Teacher-Six BC-ResNet Multi-KWS、Teacher-Six ConvMixer Multi-KWS。
- Imported TFLite：用户模型复制到 `models/imported/<model_id>/model.tflite`，校验接口并记录 SHA256；只有存在 trainer/config 时才可训练。

## B. 训练与评估入口

- Binary：microWakeWord v1/v2/v3、RepCNN performance/fast-track/robust、Phase 8 fair BC-ResNet/ConvMixer。
- Multi-KWS：12K dataset builder、Train/Validation feature extractor、BC-ResNet/ConvMixer formal trainer、Validation report、冻结 held-out Test evaluator。
- 训练支持 checkpoint、latest/best、resume、Validation-only calibration、Full INT8 PTQ。
- Teacher-Six formal Test 已完成，原始 Validation/Test JSON 与 TFLite 是 immutable artifact。

## C. Browser Web Dashboard 页面

- 数据集生成。
- 模型训练。
- 实时唤醒。
- 模型与部署。
- 默认入口是 localhost Browser Dashboard；legacy UI 与可选 desktop launcher 文件仍存在，但默认不使用桌面 WebView。

## D. HTTP/API/WebSocket

- GET：`/api/bootstrap`、`/api/live/state`、`/api/job/<kind>`。
- POST：`/api/live/audio`、`/api/live/start`、`/api/live/stop`。
- POST：`/api/generation/preflight`、`/api/generation/start`、`/api/generation/stop`。
- POST：`/api/training/preflight`、`/api/training/start`、`/api/training/stop`。
- POST：`/api/model/inspect`、`/api/model/deploy`、`/api/model/import`。
- 静态路由：`/`、`/app.css`、`/overrides.css`、`/app.js`。
- Phase 10 前没有 WebSocket；实时浏览器 PCM16 通过 `/api/live/audio` POST 输入，状态通过轮询获取。

## E. Registry / Config

- `configs/demo/teacher_demo.yaml` 是产品模型、provider、检测配置的来源。
- `ModelRegistry` 支持按 id/display/config key 查询与 backend factory。
- `configs/demo/user_models.json` 用于持久化 imported models（文件可在首次导入时创建）。

## F. 实时麦克风链路

Browser `getUserMedia` → 16 kHz PCM16 → `/api/live/audio` → `StreamingWakeWordEngine` → adaptive energy gate → WebRTC VAD → 连续 3 帧 speech gate → 2 秒 pre-roll → backend/frontend/TFLite → DetectionLogic → episode tracker → FIFO 播放“我醒来了”。

## G. DetectionLogic

- L1：连续 wake score / temporal streak。
- L2：peak/background EMA ratio。
- L3：cooldown。
- L4：多 score arbitration margin。
- L5：pre-silence / post-silence transition。
- 已输出逐层状态与 rejection reason，但 Phase 10 前尚无 Multi-KWS background/top1/top2 语义。

## H. TFLite inference

- Model A：`pymicro_features` 10 ms frontend，3 帧切片，INT8 input / UINT8 output。
- RepCNN/Binary rolling models：LiveKit `MicroFrontend`，99×40，逐窗口 INT8 interpreter。
- Offline Multi-KWS evaluator：99×40 Full INT8，逐样本 inference；Phase 10 前尚无实时 Multi-KWS backend。

## I. Imported-model

- UI 可上传 `.tflite`、选择 binary backend、指定冻结 threshold。
- 后端验证大小、接口和 Full INT8 状态，生成稳定 model id 与 SHA256。
- 已明确“推理兼容”不等于“可训练架构”。

## J. 新增提示词

- 底层 `MultiKWSVocabulary/add_keyword` 已保证 background=0、追加 class id、旧 class 不重排，并声明 `ADD_KEYWORD_REQUIRES_RETRAIN=true`。
- Phase 10 前 UI/API 尚无新增词向导、dataset replay plan、candidate/activation job。

## K. Dataset pipeline

- Kokoro、多 speaker、本地 WAV 导入、VoxCPM1.5、ordinary speech、hard negatives、ambient、augmentation、manifest、resume。
- Multi-KWS 12K pipeline 支持 deterministic sample id、speaker/reference split、Train/Validation/Test 冻结。
- 本地目录导入会标准化到 16 kHz mono PCM16 且不覆盖源文件。

## L. 导出与部署

- 模型检查、SHA256、TFLite interface/quantization/operator audit。
- ESP32-S3 package/header 生成与预审；无实板，因此 hardware runtime 未验证。
- `PRESERVE = true`：Phase 10 后必须继续保留上述页面、API、binary backend、训练入口、import 与部署能力。
