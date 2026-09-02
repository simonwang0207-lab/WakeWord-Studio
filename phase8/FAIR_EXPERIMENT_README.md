# 模型公平对照实验运行说明

## 固定实验契约

- 任务：`binary_kws`，关键词为“你好，青小甲”。
- 数据：冻结 `qingxiaojia_v2` WAV/特征缓存，以及冻结 RepCNN B2 使用的精确 eligibility/sample view。
- 仅允许 `train`、`validation`；加载器不枚举 Test 文件名，所有报告必须为 `test_loaded=false`。
- Frontend：与 RepCNN B2 相同的 TFLM microfrontend，输入 `[1, 99, 40]`。
- 采样、focal objective、学习率/负样本权重日程、Validation 间隔、early stopping 均与 B2 对齐。
- threshold 仅由 Validation 选择：先约束 overall FPR ≤ 0.10，再按 `(worst-source Recall, overall Recall, -source gap, F1, precision, -FPR)` 排序。
- Smoke 结果带 `formal_result=false`，比较报告生成器拒绝把它写入正式结果表。

## 运行环境

在 Windows PowerShell 中进入 WSL：

```powershell
wsl.exe -d Ubuntu
```

进入 WSL 后：

```bash
cd /mnt/f/ZJU_intership/task/4/WakeWord-Studio
source ~/venvs/wakeword-gpu/bin/activate
python phase8/scripts/gpu_probe.py
```

成功条件是 `gpu_count=1`、`gpu_op_executed=true`，且 op device 包含 `GPU:0`。另开终端可监控：

```bash
watch -n 1 nvidia-smi
```

## USER ACTION REQUIRED：BC-ResNet 正式训练

预计 RTX 4060 Laptop GPU 总耗时约 15–45 分钟；图编译后的普通 train step 预计约 0.02–0.10 秒，另有每 500 step 的全 Validation 开销。实际时间受 `/mnt/f` mmap 随机 I/O、功耗模式和散热影响；训练读取的是约 225 MiB 的冻结特征缓存，不复制 WAV。

```bash
cd /mnt/f/ZJU_intership/task/4/WakeWord-Studio
source ~/venvs/wakeword-gpu/bin/activate
mkdir -p runs/qingxiaojia/bcresnet_binary/formal/user_run_01
set -o pipefail
python phase8/scripts/run_fair_binary_kws.py \
  --config configs/models/bcresnet_binary_fair.json \
  --run-dir runs/qingxiaojia/bcresnet_binary/formal/user_run_01 \
  --allow-formal-training \
  2>&1 | tee runs/qingxiaojia/bcresnet_binary/formal/user_run_01/training.log
```

## USER ACTION REQUIRED：ConvMixer 正式训练

预计总耗时约 15–45 分钟；普通 train step 与 BC-ResNet 同量级，ConvMixer 的计算量略高但 INT8 图更小。其余影响因素相同。

```bash
cd /mnt/f/ZJU_intership/task/4/WakeWord-Studio
source ~/venvs/wakeword-gpu/bin/activate
mkdir -p runs/qingxiaojia/convmixer_binary/formal/user_run_01
set -o pipefail
python phase8/scripts/run_fair_binary_kws.py \
  --config configs/models/convmixer_binary_fair.json \
  --run-dir runs/qingxiaojia/convmixer_binary/formal/user_run_01 \
  --allow-formal-training \
  2>&1 | tee runs/qingxiaojia/convmixer_binary/formal/user_run_01/training.log
```

## 中断与恢复

按 `Ctrl+C` 后，trainer 会先保存 checkpoint，并把状态写为 `INTERRUPTED_RESUMABLE`。恢复时不要新建目录，使用原目录：

```bash
python phase8/scripts/run_fair_binary_kws.py \
  --config configs/models/bcresnet_binary_fair.json \
  --run-dir runs/qingxiaojia/bcresnet_binary/formal/user_run_01 \
  --allow-formal-training --resume
```

ConvMixer 恢复方式相同，只需替换 config 和 run-dir。不要给 smoke 使用 `--resume`。

## 输出与成功判据

每个正式目录包含：

- `TRAINING_STATUS.json`：状态、设备、step、loss、Validation、checkpoint；
- `training.log`：stdout/stderr 日志；
- `checkpoints/`：可恢复 checkpoint；
- `best_single.weights.h5`：Validation 排序最佳权重；
- `export/*_formal_full_int8.tflite`：最终 Full INT8；
- `threshold_freeze.json`：Final INT8 Validation threshold；
- `FORMAL_RESULT.json`：统一指标与 per-source/operating points。

成功必须同时满足：`status=COMPLETED`、`formal_result=true`、`test_loaded=false`、GPU op 为真、INT8 输入输出形状分别为 `[1,99,40]` 和 `[1,1]`。若进程退出后缺任一文件，不算完成。

正式训练都结束后生成统一报告：

```bash
python3 phase8/scripts/build_fair_comparison.py \
  --bcresnet-result runs/qingxiaojia/bcresnet_binary/formal/user_run_01/FORMAL_RESULT.json \
  --convmixer-result runs/qingxiaojia/convmixer_binary/formal/user_run_01/FORMAL_RESULT.json \
  --output-dir runs/qingxiaojia/fair_model_comparison/formal
```

需要回传：两个 `TRAINING_STATUS.json`、两个 `FORMAL_RESULT.json`、两个 `threshold_freeze.json`、两个 `training.log`，以及生成的 `FAIR_COMPARISON.json/.md`。在这之前不能宣称哪个结构更好。

