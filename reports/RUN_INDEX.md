# 正式 Run 索引

本索引保留关键实验 lineage，而不把完整 `runs/`、checkpoint、feature cache 或 TensorBoard 日志提交到 Git。路径均为仓库根目录下的**原开发机本地参考路径**；GitHub clone 使用 `artifacts/models/` 中的发布副本。

| Run / Model | 任务 | 主要结果 | 发布 artifact | 本地 run 归档 |
|---|---|---|---|---|
| Model A microWakeWord v3 | Binary | Test Recall 59.00%，FPR 6.75%；Tiny streaming baseline | `artifacts/models/binary/microwakeword_mixednet_full_int8.tflite` | 本地保留，Git 排除 |
| Model B RepCNN fasttrack | Binary | Validation Recall 81.01%，Worst-source 68.75%，FPR 10% | `artifacts/models/binary/repcnn_full_int8.tflite` | 本地保留，Git 排除 |
| BC-ResNet Binary `user_run_01` | Binary | INT8 Validation Recall 54.75%，Worst-source 38.19%，FPR 10% | `artifacts/models/binary/bcresnet_binary_full_int8.tflite` | 本地保留，Git 排除 |
| ConvMixer Binary `user_run_01` | Binary | INT8 Validation Recall 86.71%，Worst-source 76.39%，FPR 9.92% | `artifacts/models/binary/convmixer_binary_full_int8.tflite` | 本地保留，Git 排除 |
| Teacher-Six BC `v2_12k_user_run_02` | Multi-KWS | Test Macro Recall 89.89%，Worst 74.00%，Background FAR 14.50% | `artifacts/models/teacher_six/teacher_six_bcresnet_full_int8.tflite` | 本地保留，Git 排除 |
| Teacher-Six Conv `v2_12k_user_run_01` | Multi-KWS | Test Macro Recall 94.22%，Worst 88.00%，Background FAR 17.00% | `artifacts/models/teacher_six/teacher_six_convmixer_full_int8.tflite` | 本地保留，Git 排除 |

## 本地来源路径

```text
runs/qingxiaojia/microwakeword_tiny_v3_sequence/formal/20260829T162135Z/phase2i_v3_frozen_final
runs/qingxiaojia/repcnn_performance_v2_fasttrack/formal/user_run_01/phase6_finalization_v2
runs/qingxiaojia/bcresnet_binary/formal/user_run_01
runs/qingxiaojia/convmixer_binary/formal/user_run_01
runs/multikws/teacher_six/bcresnet/formal/v2_12k_user_run_02
runs/multikws/teacher_six/convmixer/formal/v2_12k_user_run_01
```

## 报告入口

- Teacher-Six：`reports/multikws/README.md`
- Validation model selection：`reports/multikws/MODEL_SELECTION_VALIDATION.json`
- Teacher-Six frozen Test：`reports/multikws/test/`
- 项目完整阶段总结：`reports/PROJECT_INTERIM_REPORT.md`
- Model A 历史收口：`docs/model_a/WakeWord_Studio_Model_A_Closure_Report.md`
- Model B 历史收口：`docs/model_b/WakeWord_Studio_Model_B_Interim_Closure_Report.md`

完整 run 不进入 Git 的原因是其中包含大型 checkpoint、weights、feature arrays、日志和机器相关状态。发布模型的字节数与 SHA256 已与原 run 逐一核对。
