# WakeWord-Studio 使用指南

本文说明 Web UI 中“数据集”“模型训练”“实时唤醒”和“模型部署”四个页面的实际用法。所有操作默认在本机完成。

## 1. 启动项目

建议把项目安装到独立的 Conda 环境，不要安装到 Anaconda `base`：

```powershell
git clone https://github.com/simonwang0207-lab/WakeWord-Studio.git
cd WakeWord-Studio
conda create -n wakeword-studio-runtime python=3.11 -y
conda activate wakeword-studio-runtime
python -m pip install --upgrade pip
python -m pip install -e ".[runtime,demo]"
python .\run_studio.py
```

浏览器会打开 <http://127.0.0.1:8765>。PowerShell 窗口是本地服务进程，使用期间不要关闭；停止服务时在该窗口按 `Ctrl+C`。

## 2. 输出路径怎样计算

页面中的相对路径都以项目根目录为基准。例如项目位于：

```text
E:\test\WakeWord-Studio
```

则三个默认输出位置为：

| 功能 | 页面默认值 | 实际位置 |
|---|---|---|
| 数据生成或导入 | `outputs/teacher_generated` | `E:\test\WakeWord-Studio\outputs\teacher_generated` |
| 模型训练 run | `runs/teacher_ui/manual_run` | `E:\test\WakeWord-Studio\runs\teacher_ui\manual_run` |
| 部署包 | `outputs/esp32_deployment` | `E:\test\WakeWord-Studio\outputs\esp32_deployment` |

也可以在页面中直接填写绝对路径，例如 `E:\WakeWordData\qingxiaojia`。输出目录不要和导入的原始语音目录设成同一个目录。

`outputs/`、`runs/` 和 `datasets/` 默认受 `.gitignore` 保护，不会作为普通源码提交到 GitHub。需要长期保留时，请自行备份这些目录。

## 3. 实时唤醒

1. 打开“实时唤醒”。
2. 在“当前模型”中选择模型。切换模型会显式激活所选模型。
3. 选择麦克风；第一次使用时允许浏览器访问麦克风。
4. 点击“开始监听”。
5. 页面会显示声音能量、VAD、Top1、Top2、分数差、阈值和拒绝原因。
6. 测试结束后点击“停止监听”。

默认的 Teacher-Six ConvMixer 同时识别六个提示词。第 0 类 `background` 只表示“不是六个目标提示词”，不会进一步识别环境声音的具体种类。

实时音频在本机处理。默认不会保存麦克风录音；只有使用反馈或验收功能并明确启用保存时，才可能产生本地记录。

## 4. 创建训练数据

### 4.1 从本地 WAV 导入

全新克隆后可直接使用“本地语音文件夹”。推荐先按 label 和 split 整理 WAV：

```text
my_audio/
├─ train/
│  ├─ positive/
│  ├─ hard_negative/
│  ├─ negative/
│  └─ ambient/
├─ validation/
│  ├─ positive/
│  ├─ hard_negative/
│  ├─ negative/
│  └─ ambient/
└─ test/
   ├─ positive/
   ├─ hard_negative/
   ├─ negative/
   └─ ambient/
```

各标签含义：

- `positive`：完整目标唤醒词；
- `hard_negative`：发音相近、缺字或倒序短语；
- `negative`：普通非目标语音；
- `ambient`：环境声、设备噪声或静音。

目录名也接受 `hard-negative` 和 `background`；其中 `background` 会被映射为 `ambient`。如果路径中没有 `train`、`validation` 或 `test`，程序会按每个标签的稳定顺序分配约 80%/10%/10%。正式实验建议事先显式划分 split。

如需记录说话人或年龄，可在输入目录放置 `metadata.csv` 或 `metadata.jsonl`。年龄必须来自真实 metadata；不能根据声音主观推断。`child`、`young`、`middle`、`senior` 目录可以记录报告年龄分组，但仍建议同时写清 label 和 split。

页面操作：

1. 填写目标唤醒词。
2. “语音来源”选择“本地语音文件夹”。
3. 在“真人语音目录”填写原始 WAV 文件夹。
4. 在“输出目录”填写新目录，例如 `outputs/my_wakeword_dataset`。
5. 根据需要启用数据增强。启用后会为有效录音生成确定性的噪声/混响/SNR 版本，不会覆盖原始 WAV。
6. 先点击“生成前检查”，通过后再点击“开始生成”。

导入过程会统一输出为 16 kHz、单声道、PCM16 WAV，并生成：

```text
outputs/my_wakeword_dataset/
├─ DatasetManifest.json
├─ positive/
├─ hard_negative/
├─ negative/
└─ ambient/
```

`DatasetManifest.json` 记录每条音频的相对路径、label、split、来源、说话人 metadata、增强信息、时长和 SHA256。

### 4.2 使用 Kokoro 生成

Kokoro 不属于普通运行依赖。只有本机已经准备好 Kokoro 包、模型缓存和 `configs/demo/teacher_demo.yaml` 所配置的 Python 环境时，Kokoro 才会出现在可用语音来源中。

GitHub 仓库不附带 Kokoro/VoxCPM 权重、缓存或第三方语音 reference。若页面只显示“本地语音文件夹”，这是正常现象，不代表 Web UI 安装失败。


### 配置已有的 Kokoro 环境

Kokoro 环境和模型缓存不必放在 WakeWord-Studio 项目目录中。项目只需要知道 Kokoro 环境所使用的 Python 解释器路径。

首先激活已有的 Kokoro 环境，并查看 Python 的真实位置：

```powershell
conda activate 你的Kokoro环境名
python -c "import sys; print(sys.executable)"
```

例如，Conda 环境可能输出：

```text
D:\anaconda12.7\envs\kokoro\python.exe
```

普通 Python 虚拟环境可能位于：

```text
F:\TTS\kokoro_env\Scripts\python.exe
```

然后打开：

```text
configs/demo/teacher_demo.yaml
```

找到：

```yaml
providers:
  kokoro:
    display_name: Kokoro
    kind: tts
    dependency: kokoro
    python: .envs/kokoro/Scripts/python.exe
```

将 `python` 改成实际解释器的绝对路径。Windows 下建议使用正斜杠：

```yaml
providers:
  kokoro:
    display_name: Kokoro
    kind: tts
    dependency: kokoro
    python: D:/anaconda12.7/envs/kokoro/python.exe
```

也可以使用带单引号的反斜杠路径：

```yaml
    python: 'F:\TTS\kokoro_env\Scripts\python.exe'
```

修改后验证该解释器能够导入 Kokoro 和 PyTorch：

```powershell
& 'D:\anaconda12.7\envs\kokoro\python.exe' -c "import kokoro, torch; print('Kokoro 环境正常'); print('CUDA=', torch.cuda.is_available())"
```

如果输出类似下面内容，说明 Python 环境可用：

```text
Kokoro 环境正常
CUDA= True
```

`CUDA=False` 不代表 Kokoro 无法运行，只表示当前环境将使用 CPU，数据生成速度通常会更慢。

### 配置已有的模型缓存

Kokoro 模型缓存也不必放在项目目录中。默认通常使用当前 Windows 用户的 Hugging Face 缓存目录，例如：

```text
C:\Users\你的用户名\.cache\huggingface
```

如果模型已经位于默认缓存目录，一般不需要额外配置。

如果缓存位于其他位置，可以在启动 WakeWord-Studio 前设置 `HF_HOME`：

```powershell
conda activate wakeword-studio-runtime
$env:HF_HOME = 'F:\ModelCache\huggingface'
python .\run_studio.py
```

这个设置只对当前 PowerShell 窗口及其启动的子进程生效，不会移动或复制模型文件。

配置完成并重新启动 Web UI 后，打开“数据集”页面。“语音来源”中应当出现 `Kokoro`。建议先点击“生成前检查”，确认通过后再开始生成。

### 本地配置注意事项

`configs/demo/teacher_demo.yaml` 是 Git 跟踪文件。个人电脑上的绝对路径通常不适合提交到公共仓库。

提交代码前检查：

```powershell
git status --short
```

如果出现：

```text
M configs/demo/teacher_demo.yaml
```

表示该文件包含本地修改。除非准备把它改成适用于所有用户的通用配置，否则不要执行：

```powershell
git add configs/demo/teacher_demo.yaml
```

其他用户需要根据自己的 Conda、虚拟环境和模型缓存位置配置对应路径。
## 5. 模型训练

### 5.1 当前支持范围

模型训练页不是任意数据集的一键训练器。当前公开版本中：

- Teacher-Six BC-ResNet 和 ConvMixer 是冻结的推理/部署模型，没有连接网页 trainer；
- 可训练项使用历史 binary trainer，并要求配置中指定的数据集、特征和专用训练 Python 环境；
- GitHub 不包含完整训练 WAV、特征缓存、checkpoint 或本机 `.envs`；
- 因此，全新克隆不能仅靠普通 runtime 环境复现正式训练；
- 新生成的数据不会自动成为某个历史模型的合法训练输入。

页面会把没有 trainer 的模型标记为“仅推理/部署”。如果预检提示缺少 `DatasetManifest.json`、训练 Python、特征缓存或数据集不匹配，应补齐对应训练资产，而不是绕过检查。

### 5.2 已准备训练环境时

1. 打开“模型训练”。
2. 填写训练配置所要求的数据集目录；目录内必须有 `DatasetManifest.json`。
3. 选择标记为“可训练”的模型。
4. 为本次实验指定一个新的输出目录，例如 `runs/teacher_ui/experiment_01`。
5. 点击“检查配置”。
6. 确认数据、训练解释器和冻结配置均存在后，再点击“开始训练”。

训练 run 通常写入：

```text
runs/teacher_ui/experiment_01/
├─ TRAINING_STATUS.json
├─ checkpoints/
├─ preserved_best_checkpoint/     # 由具体 trainer 决定
└─ 其它权重、日志或评估文件        # 由具体 trainer 决定
```

训练输出目录保存的是完整 run，而不一定直接包含可部署 TFLite。训练完成也不会自动覆盖、注册或激活页面当前使用的模型。量化、导出、验证和注册必须使用该架构对应的正式流程完成。

TensorFlow 训练依赖可通过以下 extra 安装，但这不会自动补齐数据、特征或 GPU 配置：

```powershell
python -m pip install -e ".[runtime,demo,training]"
```

## 6. 生成部署包

1. 打开“模型部署”。
2. 点击要打包的已发布模型卡片。
3. 点击“验证完整性”，检查文件大小、Full INT8 接口和 SHA256。
4. 在输出框填写目录，例如 `outputs/esp32_deployment`。
5. 点击“生成部署包”。

输出目录包含：

```text
outputs/esp32_deployment/
├─ <所选模型文件>.tflite
├─ model_info.json
└─ README.txt
```

`model_info.json` 记录模型输入输出、量化接口、大小、SHA256 和生成状态。该按钮复制的是模型注册表中明确选择的冻结 artifact，不会扫描训练目录，也不会自行挑选 checkpoint。

部署包只是 TFLite 模型及接口 metadata，不包含完整 ESP32-S3 固件，不会自动烧录设备，也不代表已经通过实板延迟、内存和真实麦克风验收。ONNX 板端测试材料是仓库中的独立交付物，位于 `deliverables/onnx_board_test/`。

## 7. 导入自己的 TFLite

“模型训练”页的“导入已有模型”只用于注册已存在的 TFLite：

1. 填写显示名称、backend 和冻结阈值；
2. 选择 `.tflite` 文件；
3. 提交后，模型会复制到 `models/imported/<model_id>/model.tflite`；
4. 导入成功不等于该模型可训练；
5. 需要在“模型部署”中显式设为当前模型，实时页才会使用它。

导入时会检查文件大小、模型接口和量化类型。不兼容的输入输出形状会被拒绝。

## 8. 常见问题

### 点击监听提示缺少 `pkg_resources`

如果项目是通过 `git clone` 下载的，先更新源码：

```powershell
git pull
```

如果是从 GitHub 下载的 ZIP，ZIP 不会自动更新，请重新下载并解压到新目录。然后在对应的 Conda 环境中重新安装依赖并检查 VAD 包：

```powershell
python -m pip uninstall -y webrtcvad
python -m pip install -e ".[runtime,demo]"
python -c "from importlib.metadata import version; print(version('webrtcvad-wheels'))"
```

预期版本为 `2.0.14`。不要继续使用修复前下载的旧 ZIP。

### 找不到输出文件

先确认启动服务时 PowerShell 所在的项目根目录。相对路径以项目根目录计算；也可以改用绝对路径排除歧义。

### 生成按钮提示 provider 不可用

说明对应 TTS 依赖或专用 Python 环境尚未配置。全新克隆默认仍可选择本地语音文件夹导入 WAV。

### 训练预检提示数据集或 Python 不存在

这表示当前所选模型的历史 trainer 依赖没有随公开仓库提供。普通实时运行环境只负责推理，不等同于正式训练环境。
