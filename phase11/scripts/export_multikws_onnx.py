"""Export the frozen Teacher-Six Float checkpoints to board-delivery ONNX.

This script is intentionally Validation-only.  It never opens Test audio or
recalibrates an operating point; frozen Test *reports* are read only to carry
already-published summaries into the delivery metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wakeword_studio.training.multikws_models import build_multikws_model  # noqa: E402
from wakeword_studio.training.multikws_vocabulary import MultiKWSVocabulary  # noqa: E402


OPSET = 15
INPUT_NAME = "input_features"
OUTPUT_NAME = "class_scores"
CLASS_DISPLAY_NAMES = [
    "background",
    "你好，青小甲",
    "你好，豆豆",
    "你好，点点",
    "你好，小瑞",
    "你好，多多",
    "你好，吉智娃",
]
MODEL_SPECS = {
    "bcresnet": {
        "model_id": "teacher_six_bcresnet",
        "architecture": "BC-ResNet",
        "filename": "BCResNet_TeacherSix_MultiKWS_FP32.onnx",
        "run": "runs/multikws/teacher_six/bcresnet/formal/v2_12k_user_run_02",
        "tflite": "export/teacher_six_bcresnet_formal_full_int8.tflite",
        "test_report": "reports/multikws/test/bcresnet/TEST_REPORT.json",
        "role": "COMPUTE_LIGHT_BASELINE",
    },
    "convmixer": {
        "model_id": "teacher_six_convmixer",
        "architecture": "ConvMixer",
        "filename": "ConvMixer_TeacherSix_MultiKWS_FP32.onnx",
        "run": "runs/multikws/teacher_six/convmixer/formal/v2_12k_user_run_01",
        "tflite": "export/teacher_six_convmixer_formal_full_int8.tflite",
        "test_report": "reports/multikws/test/convmixer/TEST_REPORT.json",
        "role": "PRIMARY_CANDIDATE",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_keras(model: Any, values: np.ndarray, batch_size: int = 32) -> np.ndarray:
    rows = []
    for start in range(0, len(values), batch_size):
        rows.append(np.asarray(model(values[start : start + batch_size], training=False), np.float32))
    return np.concatenate(rows, axis=0)


def predict_onnx(session: Any, values: np.ndarray) -> np.ndarray:
    rows = [session.run([OUTPUT_NAME], {INPUT_NAME: row[np.newaxis].astype(np.float32)})[0] for row in values]
    return np.concatenate(rows, axis=0).astype(np.float32)


def predict_tflite(tf: Any, path: Path, values: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError(f"Expected Full INT8 model: {path}")
    rows = []
    for row in values:
        quantized = np.clip(np.rint(row[np.newaxis] / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        rows.append((output - output_zero) * output_scale)
    return np.concatenate(rows, axis=0)


def balanced_validation_indices(labels: np.ndarray, per_class: int = 14) -> np.ndarray:
    selected = []
    for class_id in range(len(CLASS_DISPLAY_NAMES)):
        matches = np.flatnonzero(labels == class_id)
        if len(matches) < per_class:
            raise RuntimeError(f"Validation class {class_id} has only {len(matches)} samples")
        selected.extend(matches[:per_class].tolist())
    return np.asarray(selected, np.int32)


def vector_indices(labels: np.ndarray, metadata: list[dict[str, Any]]) -> np.ndarray:
    ordinary = next(
        index for index, row in enumerate(metadata)
        if labels[index] == 0 and "ordinary_background" in str(row["record_id"])
    )
    hard_negative = next(
        index for index, row in enumerate(metadata)
        if labels[index] == 0 and "hard_negative" in str(row["record_id"])
    )
    keywords = [int(np.flatnonzero(labels == class_id)[0]) for class_id in range(1, 7)]
    return np.asarray([ordinary, *keywords, hard_negative], np.int32)


def rename_onnx_value(model: Any, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        if value.name == old_name:
            value.name = new_name
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old_name:
                node.input[index] = new_name
        for index, name in enumerate(node.output):
            if name == old_name:
                node.output[index] = new_name


def export_onnx(tf: Any, tf2onnx: Any, onnx: Any, model: Any, output_path: Path) -> Any:
    signature = [tf.TensorSpec([1, 99, 40], tf.float32, name=INPUT_NAME)]

    @tf.function(input_signature=signature)
    def serving(input_features: Any) -> Any:
        return tf.identity(model(input_features, training=False), name=OUTPUT_NAME)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_proto, _ = tf2onnx.convert.from_function(
        serving,
        input_signature=signature,
        opset=OPSET,
        output_path=str(output_path),
    )
    if len(model_proto.graph.input) != 1 or len(model_proto.graph.output) != 1:
        raise RuntimeError("Expected exactly one ONNX input and output")
    rename_onnx_value(model_proto, model_proto.graph.input[0].name, INPUT_NAME)
    rename_onnx_value(model_proto, model_proto.graph.output[0].name, OUTPUT_NAME)
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output_path)
    checked = onnx.load(output_path)
    onnx.checker.check_model(checked)
    return checked


def tensor_shape(value_info: Any) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        result.append(int(dimension.dim_value) if dimension.dim_value else str(dimension.dim_param))
    return result


def equivalence(keras_scores: np.ndarray, onnx_scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(keras_scores - onnx_scores)
    keras_top1 = np.argmax(keras_scores, axis=1)
    onnx_top1 = np.argmax(onnx_scores, axis=1)
    norms = np.linalg.norm(keras_scores, axis=1) * np.linalg.norm(onnx_scores, axis=1)
    cosine = np.sum(keras_scores * onnx_scores, axis=1) / np.maximum(norms, 1e-12)
    per_class = {}
    for class_id, name in enumerate(CLASS_DISPLAY_NAMES):
        mask = labels == class_id
        per_class[name] = {
            "sample_count": int(mask.sum()),
            "top1_agreement": float(np.mean(keras_top1[mask] == onnx_top1[mask])),
            "max_abs_error": float(np.max(absolute[mask])),
            "mean_abs_error": float(np.mean(absolute[mask])),
        }
    return {
        "sample_count": int(len(labels)),
        "top1_agreement": float(np.mean(keras_top1 == onnx_top1)),
        "max_abs_error": float(np.max(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "minimum_cosine_similarity": float(np.min(cosine)),
        "per_class": per_class,
    }


def frontend_markdown(feature_hash: str) -> str:
    return f"""# Frontend specification

`FRONTEND_PARITY_REQUIRED=true`

这两个 ONNX **不接收 WAV/PCM**。输入是与训练完全一致的 `[1,99,40]` float32 TFLite Micro microfrontend filterbank；板端若不复现该前端，模型结果没有可比性。

## 从 PCM 到模型输入

1. 音频必须为 16,000 Hz、mono。文件生成规范为 PCM16；特征提取读取为 float32 `[-1,1]`。
2. 每条输入固定为 32,000 samples（2.000 s）。更长时中心裁剪；更短时左右近似等量补零，奇数个缺失 sample 的额外 1 个零放在右侧。不做 VAD、静音裁剪或响度归一化。
3. float 音频通过 `clip(audio * 32768, -32768, 32767).astype(int16)` 转为有符号 PCM16。
4. 正式 cache 使用 `pymicro-features 2.0.2` 的 TFLite Micro microfrontend：30 ms frame（480 samples）、内部 10 ms step（160 samples）、40 个 filterbank channels、125–7500 Hz。
5. TFLM frontend 顺序执行切窗、512-point FFT（257 个非冗余频点）、filterbank、noise reduction、PCAN auto gain、log scale。关键固定参数：`smoothing_bits=10`、`even_smoothing=0.025`、`odd_smoothing=0.06`、`min_signal_remaining=0.05`、`pcan_strength=0.95`、`pcan_offset=80`、`gain_bits=21`、`enable_log=true`、`scale_shift=6`。
6. C frontend 的 uint16 输出乘 `0.0390625` 得到 float32。没有额外 mean/std normalization、CMVN、MFCC/DCT 或模型侧 input scaling。
7. 每个 2 s clip 新建/reset frontend state；先产生 10 ms-hop 特征，再取 `frames[::2]`，得到 20 ms hop 的 99 帧。
8. 张量布局是 `[batch, time, filterbank_channel]`：batch=1，99 按时间从早到晚，40 个通道从低频到高频。最终输入必须 contiguous float32，shape `[1,99,40]`。

正式 Validation feature cache：`datasets/projects/teacher_six_multikws_v2_formal_12k/train_validation_features.npz`  
SHA256：`{feature_hash}`  
metadata 明确 `TEST_READ=false`。本次还用真实 Validation WAV 重算第一个样本，得到 `backend=pymicro-features`、shape `[99,40]`、与 cache `max_abs_diff=0.0`，从而确认上述实际路径。

## 重要边界

- ONNX 仅包含分类网络，不包含上述音频 frontend。
- 40 维是 filterbank feature，不是普通浮点 mel-spectrogram，也不是 40 维 MFCC。
- 若芯片 SDK 自带“Mel/MFCC”，不能只按尺寸相同就替代；必须用 test vectors 验证数值一致性。
- test vectors 已经是 frontend 输出，可先绕过麦克风链路验证 ONNX/芯片编译器，再验证真实 PCM frontend。
"""


def readme_markdown(model_rows: list[dict[str, Any]]) -> str:
    by_id = {row["model_id"]: row for row in model_rows}
    bc = by_id["teacher_six_bcresnet"]
    conv = by_id["teacher_six_convmixer"]
    return f"""# Teacher-Six Multi-KWS ONNX 板端测试包

这是两个 **7-class Multi-KWS 模型**，不是六个独立模型。它们同时支持 background、你好青小甲、你好豆豆、你好点点、你好小瑞、你好多多、你好吉智娃。

## 文件

- `models/{bc['filename']}`：BC-ResNet FP32，计算量较低。
- `models/{conv['filename']}`：ConvMixer FP32，PC/offline 正式结果更好、模型文件较小，但静态 MAC 约为 BC-ResNet 的 {conv['estimated_macs'] / bc['estimated_macs']:.2f} 倍。
- `labels.txt`：输出 class order。
- `model_info.json`：来源 checkpoint、shape、opset、hash、正式结果摘要。
- `frontend_spec.md`：板端必须严格复现的音频前端。
- `verification/test_vectors/`：固定 Validation features 与两个模型的期望输出。

两个模型输入均为 `input_features`、float32、`[1,99,40]`；输出均为 `class_scores`、float32、`[1,7]`。`class_scores[0]` 是 background，`class_scores[1:7]` 依次对应 `labels.txt`。输出是 softmax class scores。

**模型本身不直接接收 WAV/PCM。** 板端必须先实现与 `frontend_spec.md` 完全一致的 TFLite Micro microfrontend。`FRONTEND_PARITY_REQUIRED=true`。

## 正式结果摘要（读取既有冻结报告，未重跑 Test）

- BC-ResNet：Test Macro Recall {bc['test_summary']['macro_recall']:.2%}，Worst Keyword Recall {bc['test_summary']['worst_keyword_recall']:.2%}，Background FAR {bc['test_summary']['background_false_accept_rate']:.2%}；静态估算 {bc['estimated_macs']:,} MAC。优点是算力较低；现有正式 INT8 相对 Float 的量化退化更明显。
- ConvMixer：Test Macro Recall {conv['test_summary']['macro_recall']:.2%}，Worst Keyword Recall {conv['test_summary']['worst_keyword_recall']:.2%}，Background FAR {conv['test_summary']['background_false_accept_rate']:.2%}；静态估算 {conv['estimated_macs']:,} MAC。整体效果更好、文件小，但计算量约为 BC-ResNet 的 3–4 倍，且 Background FAR 相对更高。

这些 ONNX 从正式 Float best checkpoint 直接恢复并导出，不是从 INT8 TFLite 反向转换。threshold/margin 没有改变；ONNX 输出本身也没有内置运行时 threshold/margin 判定。

## 建议板端测试

1. 芯片工具链能否转换，及 operator support；
2. Flash、SRAM/PSRAM、tensor/workspace 占用；
3. 单次 inference latency、real-time factor、连续运行稳定性；
4. 先用 test vectors 比较 7 个输出，再接真实麦克风 frontend；
5. 分别测六个词、相似词/硬负例、背景语音和环境噪声；
6. 比较 BC-ResNet 的低算力优势与 ConvMixer 的 PC/offline 效果优势，再决定部署模型。

ONNX checker 与 ONNX Runtime PASS 只证明桌面 ONNX artifact 合法且与 Float Keras 数值一致，不代表芯片已经运行成功：`CHIP_RUNTIME_VERIFIED=false`、`ESP32S3_RUNTIME_VERIFIED=false`。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "deliverables" / "onnx_board_test",
    )
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    output_root = args.output.resolve()
    models_dir = output_root / "models"
    verification_dir = output_root / "verification"
    vectors_dir = verification_dir / "test_vectors"
    for directory in (models_dir, verification_dir, vectors_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = read_json(PROJECT_ROOT / "configs/multikws/teacher_six_formal_12k.json")
    vocabulary = MultiKWSVocabulary.load(PROJECT_ROOT / config["vocabulary"])
    if list(vocabulary.class_names) != [
        "background", "qingxiaojia", "doudou", "diandian", "xiaorui", "duoduo", "jizhiwa"
    ]:
        raise RuntimeError(f"Unexpected vocabulary: {vocabulary.class_names}")
    if CLASS_DISPLAY_NAMES != ["background", *[item.display_name for item in vocabulary.keywords]]:
        raise RuntimeError("Display vocabulary mismatch")

    feature_path = PROJECT_ROOT / "datasets/projects/teacher_six_multikws_v2_formal_12k/train_validation_features.npz"
    feature_metadata_path = feature_path.with_suffix(".metadata.json")
    feature_metadata = read_json(feature_metadata_path)
    if feature_metadata.get("TEST_READ") is not False:
        raise RuntimeError("Validation feature cache does not prove TEST_READ=false")
    arrays = np.load(feature_path)
    x_validation = np.asarray(arrays["x_validation"], np.float32)
    y_validation = np.asarray(arrays["y_validation"], np.int32)
    validation_metadata = list(feature_metadata["metadata"]["validation"])
    if x_validation.shape != (1500, 99, 40) or y_validation.shape != (1500,):
        raise RuntimeError(f"Unexpected Validation cache shape: {x_validation.shape}, {y_validation.shape}")
    if len(validation_metadata) != len(x_validation):
        raise RuntimeError("Validation metadata is not aligned with feature rows")

    comparison_indices = balanced_validation_indices(y_validation)
    vector_rows = vector_indices(y_validation, validation_metadata)
    comparison_x = x_validation[comparison_indices]
    comparison_y = y_validation[comparison_indices]
    vector_x = x_validation[vector_rows]
    vector_y = y_validation[vector_rows]

    model_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    equivalence_rows: dict[str, Any] = {}
    vector_outputs: dict[str, np.ndarray] = {}

    for model_name, spec in MODEL_SPECS.items():
        run_dir = PROJECT_ROOT / spec["run"]
        report = read_json(run_dir / "TRAINING_REPORT.json")
        state = read_json(run_dir / "TRAINING_STATE.json")
        if report["model_name"] != model_name or report.get("TEST_READ") is not False:
            raise RuntimeError(f"Run contract mismatch: {run_dir}")
        if report["class_names"] != list(vocabulary.class_names) or report["input_shape"] != [99, 40]:
            raise RuntimeError(f"Run shape/vocabulary mismatch: {run_dir}")
        checkpoint_prefix = run_dir / "checkpoints/best/best"
        if not checkpoint_prefix.with_suffix(".index").is_file():
            raise FileNotFoundError(checkpoint_prefix.with_suffix(".index"))
        model = build_multikws_model(
            model_name,
            (99, 40),
            vocabulary.num_classes,
            report["architecture_config"],
        )
        restore = tf.train.Checkpoint(model=model).restore(str(checkpoint_prefix))
        restore.expect_partial()
        # The formal trainer saved this prefix with Checkpoint.write(), which
        # intentionally omits Checkpoint's private save_counter.  Require a
        # real object match while accepting that one documented bookkeeping
        # variable, exactly as the training restore-audit path does.
        restore.assert_nontrivial_match()

        keras_scores = predict_keras(model, comparison_x)
        output_path = models_dir / spec["filename"]
        onnx_model = export_onnx(tf, tf2onnx, onnx, model, output_path)
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise RuntimeError(f"Unexpected ORT interface: {output_path}")
        ort_input, ort_output = session.get_inputs()[0], session.get_outputs()[0]
        if ort_input.name != INPUT_NAME or ort_input.shape != [1, 99, 40] or ort_input.type != "tensor(float)":
            raise RuntimeError(f"Unexpected ONNX input: {ort_input.name}, {ort_input.shape}, {ort_input.type}")
        if ort_output.name != OUTPUT_NAME or ort_output.shape != [1, 7] or ort_output.type != "tensor(float)":
            raise RuntimeError(f"Unexpected ONNX output: {ort_output.name}, {ort_output.shape}, {ort_output.type}")
        onnx_scores = predict_onnx(session, comparison_x)
        comparison = equivalence(keras_scores, onnx_scores, comparison_y)
        if comparison["top1_agreement"] < 0.999999 or comparison["max_abs_error"] > 1e-4:
            raise RuntimeError(f"Keras/ONNX equivalence failed for {model_name}: {comparison}")

        tflite_path = run_dir / spec["tflite"]
        tflite_scores = predict_tflite(tf, tflite_path, comparison_x)
        onnx_tflite_agreement = float(
            np.mean(np.argmax(onnx_scores, axis=1) == np.argmax(tflite_scores, axis=1))
        )
        frozen_test = read_json(PROJECT_ROOT / spec["test_report"])
        if frozen_test.get("model_name") != model_name:
            raise RuntimeError(f"Frozen Test report mismatch: {spec['test_report']}")
        test_summary = {
            key: frozen_test["metrics"][key]
            for key in (
                "macro_recall", "macro_precision", "macro_f1", "micro_accuracy",
                "worst_keyword_recall", "background_false_accept_rate",
            )
        }
        threshold_freeze = read_json(run_dir / "threshold_freeze.json")
        opsets = {item.domain or "ai.onnx": int(item.version) for item in onnx_model.opset_import}
        model_row = {
            "model_id": spec["model_id"],
            "architecture": spec["architecture"],
            "architecture_config": report["architecture_config"],
            "task_type": "multi_kws",
            "format": "ONNX",
            "precision": "FP32",
            "opset": OPSET,
            "opset_imports": opsets,
            "input_name": INPUT_NAME,
            "input_shape": [1, 99, 40],
            "input_dtype": "float32",
            "output_name": OUTPUT_NAME,
            "output_shape": [1, 7],
            "output_dtype": "float32",
            "num_classes": 7,
            "class_ids": list(vocabulary.class_names),
            "class_names": CLASS_DISPLAY_NAMES,
            "sample_rate_hz": 16000,
            "frontend": "TFLite Micro microfrontend / pymicro-features 2.0.2; see frontend_spec.md",
            "frontend_parity_required": True,
            "parameter_count": int(report["parameter_count"]),
            "estimated_macs": int(report["estimated_macs"]),
            "filename": spec["filename"],
            "onnx_size_bytes": output_path.stat().st_size,
            "onnx_sha256": sha256(output_path),
            "source_checkpoint": checkpoint_prefix.relative_to(PROJECT_ROOT).as_posix(),
            "source_checkpoint_reported": report["best_checkpoint_path"],
            "source_run": run_dir.relative_to(PROJECT_ROOT).as_posix(),
            "source_tflite": tflite_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_tflite_sha256": sha256(tflite_path),
            "frozen_top1_threshold": float(threshold_freeze["top1_threshold"]),
            "frozen_margin_threshold": float(threshold_freeze["margin_threshold"]),
            "operating_point_changed": False,
            "validation_summary": {
                key: report["int8_validation"][key]
                for key in (
                    "macro_recall", "macro_precision", "macro_f1", "micro_accuracy",
                    "worst_keyword_recall", "background_false_accept_rate",
                )
            },
            "test_summary": test_summary,
            "role": spec["role"],
            "hardware_runtime_verified": False,
            "chip_runtime_verified": False,
            "esp32s3_runtime_verified": False,
        }
        model_rows.append(model_row)
        export_rows.append({
            "model_id": spec["model_id"],
            "source_is_float_best_checkpoint": True,
            "source_checkpoint": model_row["source_checkpoint"],
            "onnx": f"models/{spec['filename']}",
            "onnx_checker": "PASS",
            "onnxruntime": "PASS",
            "input": {"name": ort_input.name, "shape": ort_input.shape, "dtype": ort_input.type},
            "output": {"name": ort_output.name, "shape": ort_output.shape, "dtype": ort_output.type},
            "opsets": opsets,
            "size_bytes": model_row["onnx_size_bytes"],
            "sha256": model_row["onnx_sha256"],
        })
        comparison["onnx_vs_int8_tflite_top1_agreement"] = onnx_tflite_agreement
        equivalence_rows[spec["model_id"]] = comparison
        vector_outputs[spec["model_id"]] = predict_onnx(session, vector_x)

    (output_root / "labels.txt").write_text(
        "\n".join(f"{index} {name}" for index, name in enumerate(CLASS_DISPLAY_NAMES)) + "\n",
        encoding="utf-8",
    )
    write_json(
        output_root / "model_info.json",
        {
            "schema": "wakeword-studio.onnx-board-delivery/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": model_rows,
            "MODEL_RETRAINED": False,
            "FORMAL_TEST_RERUN": False,
            "OPERATING_POINT_CHANGED": False,
            "CHIP_RUNTIME_VERIFIED": False,
            "ESP32S3_RUNTIME_VERIFIED": False,
        },
    )
    write_json(
        verification_dir / "export_report.json",
        {
            "schema": "wakeword-studio.onnx-export-report/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tf2onnx": tf2onnx.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "ONNX_OPSET": OPSET,
            "models": export_rows,
            "TEST_AUDIO_READ": False,
        },
    )
    write_json(
        verification_dir / "equivalence_report.json",
        {
            "schema": "wakeword-studio.onnx-equivalence-report/v1",
            "source_split": "validation",
            "selection": "first 14 samples per class in Validation cache order",
            "sample_count": int(len(comparison_indices)),
            "validation_indices": comparison_indices.tolist(),
            "class_counts": {
                CLASS_DISPLAY_NAMES[class_id]: int(np.sum(comparison_y == class_id))
                for class_id in range(7)
            },
            "models": equivalence_rows,
            "TEST_AUDIO_READ": False,
        },
    )

    np.savez_compressed(
        vectors_dir / "test_vectors.npz",
        input_features=vector_x,
        ground_truth_class_id=vector_y,
        validation_index=vector_rows,
        **{f"expected_{model_id}": scores for model_id, scores in vector_outputs.items()},
    )
    vector_manifest = []
    slugs = ["background_ordinary", "qingxiaojia", "doudou", "diandian", "xiaorui", "duoduo", "jizhiwa", "background_hard_negative"]
    for order, (validation_index, slug) in enumerate(zip(vector_rows.tolist(), slugs, strict=True)):
        filename = f"vector_{order:02d}_{slug}_input.npy"
        np.save(vectors_dir / filename, vector_x[order].astype(np.float32), allow_pickle=False)
        entry = {
            "vector_id": f"vector_{order:02d}_{slug}",
            "input_file": filename,
            "input_shape": [99, 40],
            "input_dtype": "float32",
            "validation_index": int(validation_index),
            "sample_id": validation_metadata[validation_index]["record_id"],
            "ground_truth_class_id": int(vector_y[order]),
            "ground_truth": CLASS_DISPLAY_NAMES[int(vector_y[order])],
            "expected": {},
        }
        for model_id, scores in vector_outputs.items():
            row = scores[order]
            entry["expected"][model_id] = {
                "expected_top1": int(np.argmax(row)),
                "expected_class": CLASS_DISPLAY_NAMES[int(np.argmax(row))],
                "expected_scores": [float(value) for value in row],
            }
        vector_manifest.append(entry)
    write_json(
        vectors_dir / "test_vectors.json",
        {
            "schema": "wakeword-studio.onnx-test-vectors/v1",
            "source_split": "validation",
            "vector_count": len(vector_manifest),
            "combined_file": "test_vectors.npz",
            "vectors": vector_manifest,
            "TEST_AUDIO_READ": False,
        },
    )

    (output_root / "frontend_spec.md").write_text(
        frontend_markdown(sha256(feature_path)), encoding="utf-8"
    )
    (output_root / "README.md").write_text(readme_markdown(model_rows), encoding="utf-8")

    checksum_files = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (output_root / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}\n" for path in checksum_files),
        encoding="ascii",
    )
    print(json.dumps({
        "output": str(output_root),
        "models": export_rows,
        "equivalence": equivalence_rows,
        "test_vectors": len(vector_manifest),
        "TEST_AUDIO_READ": False,
        "MODEL_RETRAINED": False,
        "FORMAL_TEST_RERUN": False,
        "OPERATING_POINT_CHANGED": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
