# microWakeWord Tiny v1 Formal Baseline — Final Acceptance

## Training

- Status: **COMPLETED** by early stopping
- Steps: **11,500 / 15,000 planned**
- Wall-clock elapsed: **339.287 s** (5 min 39.287 s)
- Best checkpoint: `best_weights.weights.h5`, step **7,500**
- Best training validation at threshold 0.5: Recall **0.9900**, Precision **0.857143**, F1 **0.918794**
- No training exception, failed checkpoint, NaN, incomplete artifact, or residual Python process was found in the final audit.

## Frozen full-INT8 streaming validation

The deployment threshold was selected using Validation only. The held-out Test split was not used for threshold selection.

- Rule: maximize Validation F1 subject to Recall >= 0.98; ties by Precision, FPR, then higher threshold
- Threshold: **1.0**
- Recall: **0.9900**
- Precision: **0.6875**
- F1: **0.811475**
- FPR: **0.128571**
- Confusion matrix: TP **198**, FP **90**, TN **610**, FN **2**

The threshold is already at the maximum quantized score, yet 90 Validation negatives also saturate at 1.0. This is a serious false-positive/calibration warning.

## Held-out Test

- Samples: **900** (200 positive, 400 ordinary negative, 200 hard-negative, 100 ambient)
- Recall / TPR: **0.1900**
- Precision: **0.703704**
- F1: **0.299213**
- FRR: **0.8100**
- FPR: **0.022857**
- ROC AUC: **0.766693**
- PR AUC: **0.491891**
- Confusion matrix: TP **38**, FP **16**, TN **684**, FN **162**

| Category | Count | Accepted | Rejected | Result |
|---|---:|---:|---:|---|
| Positive | 200 | 38 | 162 | Recall 19.0%, FRR 81.0% |
| Ordinary negative | 400 | 2 | 398 | 2 false accepts, FPR 0.5% |
| Hard-negative | 200 | 14 | 186 | 14 false accepts, FPR 7.0% |
| Ambient | 100 | 0 | 100 | 0 false accepts, FPR 0% |

## Test source comparison

| Source | Samples (positive / negative) | Recall | Precision | F1 | FRR | FPR | ROC AUC | PR AUC | Confusion (TP/FP/TN/FN) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kokoro held-out | 615 (135 / 480) | 21.481% | 67.442% | 32.584% | 78.519% | 2.917% | 0.779799 | 0.511195 | 29/14/466/106 |
| MeloTTS test-only | 185 (65 / 120) | 13.846% | 81.818% | 23.684% | 86.154% | 1.667% | 0.839936 | 0.693901 | 9/2/118/56 |

The remaining 100 Test records are source `procedural` ambient clips and produced no false accepts.

## Error analysis

- False negatives: **162** — Kokoro 106, MeloTTS 56. They occur in clean audio (22) and across all tested SNRs: 0 dB 33, 5 dB 37, 10 dB 34, 20 dB 36. The failure is therefore not confined to one noise level.
- False positives: **16** — 14 hard-negatives and 2 ordinary negatives.
- Hard-negative false accepts: `你好，小甲` **8**, `你好，青甲` **6**; all are tier-2 near-phrase confusions.
- Ordinary-negative false accepts: `现在几点了` **2**.
- No ambient clip was falsely accepted.

## Final full-INT8 streaming model

- Path: `runs/qingxiaojia/microwakeword_tiny_v1/formal/20260828T190330Z/final_model/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
- Size: **52,840 bytes / 51.602 KiB**
- Parameters: **19,697**
- Input / output dtype: **int8 / uint8**
- SHA-256: `69c2eb075975033c30eae6a12c6377a071c2698c12f378a8fb9a1277a631f831`
- 50–100 KiB requirement: **PASS**

## Conclusion and recommendation

The best in-training Validation Recall reached 99%, but the frozen full-INT8 held-out Test Recall is only 19%; the model therefore **does not meet the 98% held-out Recall target**. Moreover, forcing >=98% Recall on deployment Validation produces a severe 12.857% Validation FPR even at the maximum score threshold.

The dominant bottleneck is cross-split/source generalization plus score saturation/calibration after streaming INT8 export, not model size. Before any additional long training, the recommended next step is a bounded diagnostic: compare float vs full-INT8 scores on the same Validation/Test clips, audit source/speaker leakage and acoustic-distribution differences between splits, and inspect the saturated score=1.0 cases. Only after that evidence should the data recipe, loss/sampling, or architecture be changed.

## Key artifacts

- Metrics: `runs/qingxiaojia/microwakeword_tiny_v1/formal/20260828T190330Z/final_evaluation/metrics.json`
- Threshold report: `runs/qingxiaojia/microwakeword_tiny_v1/formal/20260828T190330Z/final_evaluation/threshold_report.json`
- Error analysis: `runs/qingxiaojia/microwakeword_tiny_v1/formal/20260828T190330Z/final_evaluation/error_analysis/`
- Per-sample scores: `runs/qingxiaojia/microwakeword_tiny_v1/formal/20260828T190330Z/final_evaluation/scores.csv`
- Training log: `runs/qingxiaojia/microwakeword_tiny_v1/formal/20260828T190330Z/training.log`
