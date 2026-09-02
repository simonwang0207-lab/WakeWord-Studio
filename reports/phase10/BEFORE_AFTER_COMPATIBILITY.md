# Phase 10 Before → After Compatibility Matrix

`OLD_FEATURES_REMOVED = false`

| 能力 | Phase 10 前 | Phase 10 后 | 兼容结论 |
|---|---|---|---|
| 默认 UI | Browser Dashboard + Python backend | 保持，仍为 `run_studio.py` 默认入口 | 保留 |
| 数据集页面 | 四类数据、provider、预检/生成 | 原页面与 API 均保留 | 保留 |
| Binary 训练 | Model A / Model B 注册 trainer | 原 launcher、按钮和 API 保留 | 保留 |
| 实时链路 | Energy → WebRTC VAD → 3 帧 → pre-roll → backend → L1–L5 | 原链路不旁路；新增 N-class backend 与可解释 L4 | 兼容升级 |
| Binary backend | microWakeWord、RepCNN | 原类与 factory 分支保留 | 保留 |
| Multi-KWS | 离线 trainer/evaluator | 新增动态 N-class 实时 backend | 新增 |
| 模型 registry | Model A/B + imported | 再纳入历史 binary 和 Teacher-Six；字段扩展有默认值 | 兼容升级 |
| 模型切换 | 实时页临时选择 | 增加显式 activate、持久化 active 和 rollback | 新增 |
| 模型导入 | Binary TFLite 推理/部署 | 保留；明确 inference-compatible ≠ training-compatible | 保留 |
| 新增提示词 | 底层 append-class contract | UI/API preflight、replay 计划、hard negative、job contract | 新增 |
| WAV 导入 | 标准化且不覆盖源文件 | 单个坏 WAV 被记录并跳过，不使全部有效文件失败 | 兼容升级 |
| Runtime log | binary score 与 L1–L5 | 增加 Top1/Top2/margin/background/model metadata/reason | 兼容升级 |
| 真人验收 | 无 | `REAL_MIC_ACCEPTANCE` schema/API/UI 入口 | 新增 |
| 背景误唤醒 | 无连续时长指标 | False Wakes/hour session；与 Background FAR 区分 | 新增 |
| 部署 | inspect、SHA、ESP32 package | 原入口保留，registry 卡片信息更完整 | 保留 |
| Formal Test | 已完成的 immutable artifact | 未重跑、未调参、未覆盖；只读汇总到 README | 保留 |

四个旧页面、旧静态路由和旧 HTTP API 均继续存在。Phase 10 新增 API 不更改旧 payload 的必填字段。
