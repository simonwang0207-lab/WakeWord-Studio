# WakeWord-Studio — Model B (LiveKit Embedded Wakeword / RepCNN) 阶段性冻结报告

> 状态：**训练已冻结，Validation 已完成并冻结阈值，held-out Test 尚未运行**
>
> 原则：与 Model A closure report 一致，保存关键工程事实、训练过程、冻结产物、Validation 结果、风险与后续动作；不伪造未运行数据，不把单一 source 的 98% Recall 写成 overall 98%，不隐去 false positive。

## 0. 证据等级
- **E1 当前原始证据**：本轮正式训练日志、`TRAINING_STATUS.json`、Phase 3C frozen export/Validation 输出、阈值冻结 JSON。
- **E2 已确认阶段记录**：此前完成并确认过的 Phase 3A preflight、tiny overfit、INT8 smoke、benchmark。
- **E3 工程分析**：现象解释、风险和后续建议，不等同于测量结果。

## 1. 模型定位
- 项目：`F:\ZJU_intership\task\4\WakeWord-Studio`
- 唤醒词：**你好，青小甲**
- Model B：**LiveKit Embedded Wakeword / RepCNN**
- 角色：Performance model
- 原生推理：2.0 s clip-level sigmoid classifier
- 输入：`[1,99,40]`
- 约 20 ms hop
- 不使用 Model A v3 的 3-frame sequence score；连续确认应属于 DetectionLogic。

## 2. qingxiaojia_v2 数据集
- 总样本：15,200
- Train：12,000
- Validation：1,600
- Test：1,600
- 总时长约 12.579 h
- 16 kHz / mono / PCM16
- Manifest SHA256：
`50e3857e9941d910b640039dd70e73c39e331cc368816c378849ca9774f1973c`

### Speaker split
Train:
- Kokoro：zf_001,zf_003,zf_006,zm_009,zm_013,zm_020
- VoxCPM1.5：SSB0197,SSB0273,SSB0632,SSB0710

Validation:
- Kokoro：zf_017,zm_031
- VoxCPM1.5：SSB0393,SSB0434

Test:
- Kokoro：zf_021,zm_041
- VoxCPM1.5：SSB0737

类别：positive / ordinary negative / hard-negative / ambient。Test 不参与 threshold tuning。

## 3. Phase 3A 预检（E2）
- RepCNN xxlarge
- 64 filters
- 11 RepDS blocks
- training graph 多分支 RepDSBlock
- export 时 RepCNN/BN 融合到 DW/PW Conv
- objective：Focal BCE，gamma=2
- 500-step 稳定期后启用 deterministic mixup / SpecAugment
- Target audit：10 positive + 10 ordinary + 10 hard-negative + 5 ambient，0 errors，PASS
- Tiny overfit：
  - loss 0.239936 -> 0.000024
  - positive 0.5000 -> 0.9812
  - ordinary negative 0.5000 -> 0.0316
  - hard-negative 0.5000 -> 0.0296
  - ambient 约 0.0242
- 180-step benchmark PASS
- strict resume：weight error=0，prediction error=0
- CPU / TensorFlow 2.21 native Windows
- mean sec/step=1.42456，p95=1.43128
- RAM peak 约 3.430 GiB
- MAC estimate：约 **210.102M MACs / 99-frame invocation**
- ESP32-S3 latency 尚未实测

## 4. 正式训练
Run:
`F:\ZJU_intership\task\4\WakeWord-Studio\runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01`

TRAINING_STATUS:
- status=COMPLETED
- final_step=4250
- planned_steps=7200
- early_stopped=true
- stale_evaluations=10
- test_loaded=false
- best_validation_f1=0.7813953488372092
- last_loss=0.05440231412649155
- mean_recent_loss=0.08486745685338974
- learning_rate=0.0005
- negative_weight=3.3608834560355607
- elapsed_seconds_this_process=9213.281254899994

### Training Validation timeline
| Step | F1 | Recall | FPR | Best |
|---:|---:|---:|---:|:---:|
|250|0.600219|0.6850|0.199167|YES|
|500|0.623482|0.7700|0.233333|YES|
|750|0.750000|0.7275|0.070833|YES|
|1000|0.734450|0.7675|0.107500|NO|
|1250|0.743707|0.8125|0.124167|NO|
|1500|0.746898|0.7525|0.087500|NO|
|**1750**|**0.781395**|**0.8400**|**0.103333**|**YES — FINAL BEST**|
|2000|0.444733|0.4275|0.165000|NO|
|2250|0.703911|0.7875|0.150000|NO|
|2500|0.731308|0.7825|0.119167|NO|
|2750|0.595745|0.7700|0.271667|NO|
|3000|0.421496|1.0000|0.915000|NO|
|3250|0.738208|0.7825|0.112500|NO|
|3500|0.648770|0.7250|0.170000|NO|
|3750|0.704463|0.7300|0.114167|NO|
|4000|0.720096|0.7525|0.112500|NO|
|4250|0.650343|0.8300|0.240833|NO|

step 3000 虽 Recall=100%，FPR=91.5%，不可视为达到可用 100%。

## 5. 冻结权重
Best weights：
`F:\ZJU_intership\task\4\WakeWord-Studio\runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\best_weights.weights.h5`

- best step=1750
- SHA256=`5b026e5ccc7aec1c0bc758953c75c4c3a963d6edcccb52e5387be632c452ab92`
- config SHA256=`b8cb56a9c25d78cdba17affc068b9bd356be24713e41fdd233374615a4ce32b2`
- `last_weights.weights.h5` / ckpt-4250 不作为最终模型
- ckpt-1750 被 retention 淘汰不影响单独保存的 best weights

## 6. Frozen full-INT8 export
模型：
`F:\ZJU_intership\task\4\WakeWord-Studio\runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\phase3c_model_b_frozen\final_model\qingxiaojia_repcnn_performance_v1_best1750_full_int8.tflite`

- SHA256=`02e45f8a9047179eb9c5c089d402ddea23907b91a08cc0ddb462cdaef8e813d8`
- size=112,816 bytes / 110.172 KiB
- training params=64,257
- deployment params=53,505
- input int8 `[1,99,40]`, scale=0.10140931606292725, zero_point=-128
- output int8 `[1,1]`, scale=0.00390625, zero_point=-128
- 正确反量化：`real_score = scale * (raw - zero_point)`
- STATELESS clip inference

## 7. Frozen Validation threshold
Selection split：v2 Validation only，count=1600。
Test loaded：NO。

### Best-F1 operating point
- threshold=**0.84375**
- Recall=**82.25%**
- Precision=**75.1142%**
- F1=**78.5203%**
- FRR=**17.75%**
- FPR=**9.0833%**
- TP=329 FP=109 TN=1091 FN=71
- ROC AUC=**0.9294583333**
- PR AUC=**0.8192574791**

### Recall >= 90%
- threshold=0.6328125
- Recall=90.25%
- Precision=59.6694%
- F1=71.8408%
- FPR=20.3333%
- TP=361 FP=244 TN=956 FN=39
- verdict：**NO REASONABLE 90% OPERATING POINT**

### Recall >= 95%
- threshold=0.3828125
- Recall=95.00%
- Precision=47.0880%
- F1=62.9660%
- FPR=35.5833%
- TP=380 FP=427 TN=773 FN=20
- verdict：**NO REASONABLE 95% OPERATING POINT**

### Recall >= 98%
- threshold=0.2421875
- Recall=98.00%
- Precision=40.7484%
- F1=57.5624%
- FPR=**47.50%**
- TP=392 FP=570 TN=630 FN=8
- verdict：**NO REASONABLE 98% OPERATING POINT**

不能把“通过降阈值做到 98% Recall”表述为“系统达到可用 98%”。

## 8. Validation 类别分解（threshold 0.84375）
- Positive：400，accepted 329，Recall 82.25%，FRR 17.75%
- Ordinary negative：640，accepted 33，FPR **5.15625%**
- Hard-negative：360，accepted 75，FPR **20.8333%**
- Ambient：200，accepted 1，FPR **0.5%**

主要错误源：near-phonetic hard-negative。

## 9. Source breakdown
### Kokoro
- count=700
- Recall=**98.00%**
- Precision=71.7949%
- F1=82.8753%
- FPR=15.4%
- TP=196 FP=77 TN=423 FN=4
- ROC AUC=0.960295
- PR AUC=0.88273277

### VoxCPM1.5
- count=700
- Recall=**66.50%**
- Precision=81.0976%
- F1=73.0769%
- FPR=6.2%
- TP=133 FP=31 TN=469 FN=67
- ROC AUC=0.906925
- PR AUC=0.79852969

### Procedural ambient
- count=200
- FPR=0.5%

存在显著 source generalization gap。Kokoro 单 source 的 98% Recall 不能代表 overall 98%。

## 10. Speaker breakdown
| Speaker | Count | Recall | Precision | F1 | FPR | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
|SSB0393|356|0.73|0.869048|0.793478|0.042969|0.959063|0.902427|
|SSB0434|344|0.60|0.75|0.666667|0.081967|0.841701|0.684370|
|zf_017|356|0.99|0.755725|0.857143|0.125000|0.984805|0.965381|
|zm_031|344|0.97|0.683099|0.801653|0.184426|0.935348|0.819640|

## 11. Special hard-negative
原始 JSON 中文 text 在 PowerShell 中出现 mojibake，但两个预定义重点 group 的数值仍有效：
- “你好，小甲”：30 条，11 false accepts，FPR **36.6667%**
- “你好，青甲”：30 条，7 false accepts，FPR **23.3333%**

需修报告层 UTF-8 显示/序列化，不能改变 frozen metrics 或 threshold。

## 12. Model A vs B 当前已知结果
### Model A v3
- INT8=52,840 bytes / 51.602 KiB
- frozen Validation：Recall 52%，Precision 72.2222%，F1 60.4651%，FPR 6.6667%
- v2 Test：Recall 59.00%，Precision 74.44795%，F1 65.82985%，FPR 6.75%，hard-negative FPR 13.6111%，ordinary FPR 5.00%，ambient 0%，ROC AUC 0.82870208，PR AUC 0.69874107
- v1 external：Recall 45.00%，FPR 7.14286%，ROC AUC 0.76654643

### Model B 当前 frozen Validation
- INT8=112,816 bytes / 110.172 KiB
- Recall 82.25%
- Precision 75.114%
- F1 78.520%
- FPR 9.083%
- ROC AUC 0.929458
- PR AUC 0.819257

在 Model B held-out Test 未运行前，不宣布 B 最终 Test 优于 A。

## 13. 当前问题
1. hard-negative FPR 20.83%，近音混淆仍强。
2. Kokoro / VoxCPM1.5 source gap 明显。
3. 98% Recall 需要 47.5% FPR，不可部署。
4. 110 KiB 文件不代表计算轻；约 210M MACs / 2s invocation。
5. report text 存在 mojibake，需修 UTF-8 输出层。
6. ESP32-S3 真机 latency、RAM、flash、实时 cadence 未验证。

## 14. 当前冻结状态
- `MODEL_B_TRAINING_STATUS = FROZEN`
- `MODEL_B_VALIDATION_STATUS = FROZEN`
- `MODEL_B_TEST_STATUS = PENDING`
- `MODEL_B_98PCT = NOT_ACHIEVED_ON_VALIDATION`
- frozen threshold：**0.84375**
- 禁止根据 Test 回头调 threshold

## 15. 后续严格顺序
1. 只修 evaluator/report 的 UTF-8 中文显示问题，不改模型/指标/阈值。
2. unit test + Validation-only smoke，确认 Test unopened。
3. 保持 threshold=0.84375。
4. 人工运行 v2 held-out Test。
5. 同一 threshold 人工运行 v1 external Test。
6. 生成 A vs B 同口径最终对比。
7. 再决定是否优化 Model B，或进入完整 WakeWordEngine / VAD / DetectionLogic / 真实麦克风验收。

## 16. 尚未完成，禁止伪造
- Model B v2 held-out Test：PENDING
- Model B v1 external Test：PENDING
- 真人麦克风 Recall / FAR / false triggers per hour：PENDING
- 多人真实语音 98% acceptance：PENDING
- ESP32-S3 真机 latency / RAM / flash：PENDING
- WakeWordEngine 端到端系统指标：PENDING

## 17. 原始产物索引
- `runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\training.log`
- `runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\TRAINING_STATUS.json`
- `runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\best_weights.weights.h5`
- `runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\phase3c_model_b_frozen\threshold_freeze.json`
- `runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\phase3c_model_b_frozen\v2_validation_report.json`
- `runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\phase3c_model_b_frozen\v2_validation_threshold_sweep.csv`
- `runs\qingxiaojia\repcnn_performance_v1\formal\user_run_01\phase3c_model_b_frozen\final_model\qingxiaojia_repcnn_performance_v1_best1750_full_int8.tflite`

## 18. 阶段结论
Model B RepCNN 已在 step 1750 冻结最佳权重。最终 full-INT8 模型 112,816 bytes。Frozen Validation 的 best-F1 工作点为 threshold 0.84375，Recall 82.25%、Precision 75.11%、F1 78.52%、FPR 9.08%、ROC AUC 0.92946、PR AUC 0.81926。Model B 的判别能力明显强于 Tiny Model A，但仍不存在合理 98% operating point；98% Recall 需要约 47.5% FPR。主要问题是近音 hard-negative 与 VoxCPM1.5 泛化。held-out Test、真人麦克风和 ESP32-S3 运行尚未验证。
