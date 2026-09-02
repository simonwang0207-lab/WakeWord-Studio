# UI 线程接入说明（仅接口，不修改 GUI）

## 1. Model B hop

Model B 注册信息中的 `hop_seconds` 应从 `0.30` 改为 `0.20`。部署窗口保持 `2.0` 秒，不允许改成其他长度。

## 2. Rolling backend 初始化

```python
backend = RepCNNBackend(
    keyword="你好，青小甲",
    window_seconds=2.0,
    hop_seconds=0.20,
    smoothing_mode="raw",
)
backend.load(model_path)
engine = StreamingWakeWordEngine(
    backend,
    pre_roll_seconds=2.0,
    tail_inference_seconds=0.8,
)
```

逐帧继续调用：

```python
runtime_log = engine.process_frame(pcm16_frame)
```

backend 在累积满 2 秒后，每 0.20 秒产生一次新分数。VAD 结束后 engine 至少保持 0.8 秒 tail inference；30 ms frame 下为 27 帧。

## 3. `score_state()` 字段

```python
state = backend.score_state()
```

- `raw_score`：本次 TFLite 直接反量化分数。
- `decision_score`：送入 DetectionLogic 的分数；默认 raw 模式下与 `raw_score` 相同。
- `window_seconds`：固定为 `2.0`。
- `hop_seconds`：应为 `0.20`。
- `smoothing.mode`：`raw`、`mean` 或 `max_mean_hybrid`；Validation 完整比较前必须保持 `raw`。
- `smoothing.window_size`：平滑历史长度。
- `smoothing.hybrid_max_weight`：hybrid 的 max 权重。
- `smoothing.history`：当前 causal 分数历史。

## 4. UI 显示 decision score

UI 主分数应读取：

```python
runtime_log.decision_wake_score
```

诊断区可同时显示 `runtime_log.raw_wake_score`。当前 raw 默认下两者相等；未来只有 Validation 明确批准 smoothing 后才可能不同。

## 5. UI 显示拒绝原因

读取：

```python
runtime_log.rejection_reason
```

可能值包括：`NO_NEW_SCORE`、`RAW_OR_SMOOTHED_SCORE_BELOW_THRESHOLD`、`L1_CONSECUTIVE_SCORE_PENDING`、`L2_BACKGROUND_RATIO_FAILED`、`L3_COOLDOWN_ACTIVE`、`L4_ARBITRATION_FAILED`、`L5_TRANSITION_PENDING`、`FINAL_WAKE_EVENT`。

## 6. UI 显示 tail 状态

读取：

```python
runtime_log.tail_silence_frames
runtime_log.tail_required_frames
runtime_log.kws_active
```

推荐显示为“尾部推理：`tail_silence_frames / tail_required_frames`”。`kws_active=true` 表示 rolling backend 仍在采集/推理；不要在 VAD 刚结束时立即 reset backend。

## 接入边界

UI 不应自行改变阈值、L1/L2/L5 或 smoothing 策略。阈值来自最终 freeze 文件；smoothing 在完整 Validation 有真实 Recall/FPR tradeoff 改善前保持 `raw`。
