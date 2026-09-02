# GitHub 发布前审计

审计日期：2026-09-02  
项目快照：WakeWord-Studio Phase 11A  
审计边界：repository packaging、README/许可边界收口、可运行性验证与 staging dry-run；未训练模型、未重跑 Formal Test、未删除本地数据或 run、未提交或推送。

## Repository snapshot

| 项目 | 事实 |
|---|---|
| Git repository | 是 |
| 当前分支 | `master`（推送前将规范化为 `main`） |
| 最近 commit | `59bf30d chore: establish wakeword studio baseline` |
| 历史 commit | 已存在，共 1 个 |
| Remote | `origin = https://github.com/simonwang0207-lab/WakeWord-Studio.git` |
| Staged files | 0 |
| 发布候选文件 | 312 |
| 发布候选未压缩总量 | 约 3.99 MiB（4,187,015 bytes） |
| GitHub repository | `simonwang0207-lab/WakeWord-Studio` 已创建 |
| GitHub CLI | 已安装且 keyring 登录有效 |
| 根 LICENSE | Apache-2.0 |
| Visibility | `PUBLIC` |

工作树包含既有 Phase 1–11A 源码、配置、测试与报告的大量 tracked modifications/untracked files。本次没有覆盖、回退或删除这些用户工作；正式发布只纳入经过 `.gitignore` 和 staging dry-run 审计的候选。

## 本地总体大小

本地目录很大，但绝大部分是环境、缓存、数据集与 run，不属于发布候选。最大顶层目录如下（扫描时近似值）：

| 排名 | 目录 | 大小 |
|---:|---|---:|
| 1 | `.envs/` | 8,464 MiB |
| 2 | `phase0/` | 6,531.83 MiB |
| 3 | `phase2/` | 5,396.46 MiB |
| 4 | `datasets/` | 3.614 GiB |
| 5 | `.cache/` | 3,099.92 MiB |
| 6 | `runs/` | 2.10 GiB |
| 7 | `phase1/` | 874.80 MiB |
| 8 | `g2pW/` | 152.12 MiB |
| 9 | `phase7/` | 22.60 MiB |
| 10 | `phase9/` | 10.68 MiB |
| 11 | `outputs/` | 3.50 MiB |
| 12 | `phase5/` | 1.24 MiB |
| 13 | `src/` | 1.14 MiB |
| 14 | `phase6/` | 0.95 MiB |
| 15 | `reports/` | 0.88 MiB |
| 16 | `phase3/` | 0.23 MiB |
| 17 | `tests/` | 0.15 MiB |
| 18 | `docs/` | 0.11 MiB |
| 19 | `firmware/` | 0.11 MiB |
| 20 | `phase4/` | 0.07 MiB |

本地共发现 48 个大于 50 MiB 的文件，其中 28 个大于 100 MiB；分布为 `.envs` 16、`.cache` 14、`phase0` 9、`phase2` 5、`phase1` 2、`g2pW` 1、`runs` 1。它们全部不在发布候选中。

最大 20 个本地文件如下。缓存中的相同二进制可能出现多个副本；路径以仓库根目录为基准，大小为扫描近似值。

| 排名 | 大小 | 文件/类别 |
|---:|---:|---|
| 1 | 2,734.54 MiB | `phase2/artifacts/.../voxcpm15/wheels/torch-*.whl` |
| 2 | 1,529.23 MiB | `phase2/artifacts/.../voxcpm15/model/model.safetensors` |
| 3 | 1,007.66 MiB | `.envs/livekit/.../_pywrap_tensorflow_common.dll` |
| 4 | 1,007.66 MiB | `phase0/artifacts/uv-cache/.../_pywrap_tensorflow_common.dll` |
| 5 | 1,007.66 MiB | `.envs/microwakeword/.../_pywrap_tensorflow_common.dll` |
| 6 | 1,007.66 MiB | `phase0/artifacts/uv-cache/.../_pywrap_tensorflow_common.dll`（第二缓存副本） |
| 7 | 641.13 MiB | `phase1/artifacts/.../transformers/...` 模型文件 |
| 8 | 641.06 MiB | `phase0/artifacts/uv-cache/.../_pywrap_tensorflow_internal.pyd` |
| 9 | 641.06 MiB | `.envs/kws_streaming/.../_pywrap_tensorflow_internal.pyd` |
| 10 | 329.99 MiB | `phase2/artifacts/.../voxcpm15/model/audiovae/...` |
| 11 | 312.09 MiB | `phase0/artifacts/tts/hf-cache/.../Kokoro-82M...` |
| 12 | 290.95 MiB | `.envs/melotts/.../torch_cpu.dll` |
| 13 | 290.95 MiB | `phase2/artifacts/piper_runtime/venv/.../torch_cpu.dll` |
| 14 | 290.95 MiB | `.envs/kokoro/.../torch_cpu.dll` |
| 15 | 290.95 MiB | `phase0/artifacts/uv-cache/.../torch_cpu.dll` |
| 16 | 290.95 MiB | `.cache/uv/.../torch_cpu.dll` |
| 17 | 198.15 MiB | `phase1/artifacts/.../MeloTTS` checkpoint |
| 18 | 178.99 MiB | `.cache/uv/.../unidic-lite/.../sys.dic` |
| 19 | 178.99 MiB | `.cache/uv/.../unidic-lite/.../build/.../sys.dic` |
| 20 | 178.99 MiB | `.envs/melotts/.../unidic_lite/dicdir/sys.dic` |

发布候选中最大文件仅约 0.40 MiB；候选中大于 50 MiB和大于 100 MiB的文件均为 0。Phase 11 的两个 ONNX 分别为 127,395 B 和 119,627 B，正式 ZIP 为 235,220 B，均无需 Git LFS。

## GitHub 包含与排除

包含：根 README、第三方许可边界、`pyproject.toml`、启动器、`src/`、Phase 代码/脚本、`configs/`、`tests/`、Web UI、ESP32-S3 skeleton、小型正式 TFLite、Teacher-Six FP32 ONNX 与固定 test vectors、精简 dataset metadata、阶段报告、正式实验 JSON/Markdown、Registry 与 runtime config。

排除但保留在本机：

- `.envs/`、`.venv/`、Python/IDE cache；
- `g2pW/` 本地第三方模型缓存；
- `datasets/` 中 WAV 与完整逐样本 manifest；
- `runs/`、checkpoint、weights、feature NPY/NPZ、TensorBoard 日志；
- `outputs/`、下载/TTS/Hugging Face/uv cache、`phase*/artifacts/`；
- WAV（唯一例外为固定的小型提示音 `assets/i_am_awake.wav`）；
- 本地 `.env`、key、certificate、credential/secret JSON 与 PID/lock/temp 文件。

这些规则只通过 `.gitignore` 实现，没有删除任何本地文件。

## 最终发布模型

| Model ID | 发布路径 | 字节数 | SHA256 | 状态 |
|---|---|---:|---|---|
| `model_a` | `artifacts/models/binary/microwakeword_mixednet_full_int8.tflite` | 52,840 | `994f08b799364f02f6fc06273cccd4a155722af25f1b61a88f4e5b2621a2d41c` | HISTORICAL |
| `model_b` | `artifacts/models/binary/repcnn_full_int8.tflite` | 112,816 | `6acfecf7cc8651c1fba52809eee1d89abbcffa0a48bd46662b2e58ac23ce064f` | HISTORICAL |
| `bcresnet_binary_formal` | `artifacts/models/binary/bcresnet_binary_full_int8.tflite` | 108,784 | `474ad90681a75acfd51fa41df1c69d43aa27ce1e2bf6f97054fa1529f370cc87` | HISTORICAL |
| `convmixer_binary_formal` | `artifacts/models/binary/convmixer_binary_full_int8.tflite` | 59,984 | `236893035d0806aef6b085079f5ac706403bfb2889f74881d6dda70b23cd1580` | HISTORICAL |
| `teacher_six_bcresnet` | `artifacts/models/teacher_six/teacher_six_bcresnet_full_int8.tflite` | 108,080 | `1176f3752b0a7a7056efa8dad5a917f1177d50e3ebeef434d1a87af387a2070a` | COMPUTE_LIGHT_BASELINE |
| `teacher_six_convmixer` | `artifacts/models/teacher_six/teacher_six_convmixer_full_int8.tflite` | 60,408 | `acc517399e72a41f3161d700702fb71db4826face2be7184f90d91375034d476` | PRIMARY_CANDIDATE / snapshot active |

以上文件由原 run artifact 复制而来，没有移动或改写源 artifact；字节数和 SHA256 已核对。每个模型旁均有 metadata，汇总清单为 `artifacts/metadata/MODEL_MANIFEST.json`。六个发布模型合计不足 0.5 MiB，因此 `GIT_LFS_REQUIRED=false`。

## Phase 11 ONNX 交付

| 模型 | 发布路径 | 字节数 | SHA256 | 本机验证 |
|---|---|---:|---|---|
| BC-ResNet Teacher-Six FP32 | `deliverables/onnx_board_test/models/BCResNet_TeacherSix_MultiKWS_FP32.onnx` | 127,395 | `f32f764ce6bbe90a43f272e794d6869c2861aeb573d02c6bccc9bee369382375` | ONNX checker / ORT / Float equivalence PASS |
| ConvMixer Teacher-Six FP32 | `deliverables/onnx_board_test/models/ConvMixer_TeacherSix_MultiKWS_FP32.onnx` | 119,627 | `c56b158b12cd654ff88d751992b945b3372df0c918f78eef29137bcb9aa18e70` | ONNX checker / ORT / Float equivalence PASS |
| 板端交付 ZIP | `deliverables/WakeWord_Models_ONNX_Board_Test.zip` | 235,220 | `e76a8463ec6c501c46da9d6069644c465a8f14cf972a636783628894658018c1` | ZIP scope / CRC / internal checksums PASS |

ONNX Runtime PASS 只证明 PC 端模型可加载和数值等价，不代表芯片实板已经运行；`CHIP_RUNTIME_VERIFIED=false`。

## Dataset 与 run policy

Teacher-Six 12K WAV 不上传。仓库保留生成/vocabulary config、12K 分布、source/split/augmentation 信息以及 dataset、logical manifest、manifest file SHA256；详见 `docs/DATASETS.md` 和 `artifacts/datasets/teacher_six_multikws_v2_formal_12k/`。

完整 `runs/` 不上传。六个关键正式 run 的 lineage、主要结果与发布 artifact 映射保存在 `reports/RUN_INDEX.md`。原 run、checkpoint、Test 报告与本地 artifact 均未删除；发布报告中的已知 Test 数字没有被重算或修改。

## Secret、隐私与 nested repository

- 发布候选的 private-key header、`sk-`、GitHub/Hugging Face token prefix，以及常见 key/token/secret/password 赋值模式扫描结果为 0。
- 未发现候选 `.env`、`.key`、`.pem`；单个历史 commit 中也未发现高风险 token prefix。`SECRET_RISK_FOUND=false`。
- 文档与历史实验 config/result 中仍有原开发机绝对路径。这些路径是 provenance，不是 Web UI 的必要路径。标准 clone/install/launch 和六个运行模型已使用相对路径。
- `src/wakeword_studio/phase10.py` 中生成的 WSL 正式训练命令仍含原开发机 `/mnt/f/...`；它不影响 Web UI 或现有模型启动，但在其他机器重训前必须改为当地路径。
- `WakeWord Studio.vbs` 优先使用 clone 内标准 `.venv`，并兼容原开发机 `.envs/livekit`。
- 本地 nested Git 位于 `.cache/uv/...` 与 `phase0/artifacts/upstream/{livekit-embedded-wakeword,microWakeWord}/.git`。它们均被忽略，没有删除，也不会作为嵌套 repository 提交。
- `THIRD_PARTY_NOTICES.md` 已记录 LiveKit、Kokoro、VoxCPM、AISHELL-3 和发布模型的许可复核边界；项目根许可证已由仓库所有者确认为 Apache-2.0。

## 依赖与启动验证

- `pyproject.toml` 核心依赖保留既有版本边界；新增 `runtime` extra，固定 `tensorflow==2.21.0`，并将 `livekit-wakeword` 固定到上游 commit `726403d432d11d3a4327f0a367d43db78f4e3d78`。没有升级 NumPy、Torch 等现有核心栈。
- Editable package metadata dry-run：PASS（使用 `--no-build-isolation --no-deps`，没有修改项目依赖版本）。
- 当前正式 livekit 环境实际加载 `teacher_six_convmixer`：PASS；Registry 6 个 artifact 的大小/SHA256 校验全部通过。
- 完整测试：`187 passed, 6 warnings`。pytest cache provider 在受限 Windows workspace 写 cache 时会挂起；使用 `-p no:cacheprovider` 后断言和进程退出均正常。
- 真实 Web 启动 smoke：`run_studio.py --no-browser --port 8876` 启动成功，`/` 与 `/api/bootstrap` 均返回 HTTP 200；随后测试服务已停止。
- `pip check`：PASS；`wakeword_studio.webapp` import：PASS；Controller/Registry bootstrap：PASS。

## Staging dry-run

`git add --dry-run --all -- .` 已通过；未实际暂存，`git diff --cached --name-only` 为空。确认发布候选包括 README、第三方说明、项目报告、`src/`、Phase、configs、tests、`artifacts/models/` 与 ONNX deliverables；`.envs/`、datasets、runs、phase artifacts、cache、`g2pW/` 和大模型/checkpoint 被排除。

最终复核得到 312 个候选文件、约 3.99 MiB，dry-run 检查 252 个新增/修改项，staged files 仍为 0。预计上传规模不足 4 MiB，不需要 Git LFS。

## Remaining risks

1. Apache-2.0 项目许可证不会替使用者取得第三方数据、语音 reference、商标、人格权或其他权利；相关来源和边界见 `THIRD_PARTY_NOTICES.md`。
2. 历史训练 config 含原开发机路径，复现者需要按本机重写；当前实时 Web UI 不依赖这些路径。
3. 干净安装需要联网访问 Python package index 与 pinned GitHub commit，并要求 Git 和 Python 3.11；本轮没有创建第二套联网干净环境。
4. 正式 ESP32-S3 实板验收仍未完成；模型 metadata 明确记录 `hardware_runtime_verified=false`。

结论：代码、文档、Apache-2.0 许可边界、可运行性、模型 artifact、GitHub 认证、Public 远端和 staging dry-run 已收口，可进入正式 commit/push。`READY_FOR_PUBLIC_GITHUB_PUSH=true`。
