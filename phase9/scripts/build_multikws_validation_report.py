"""Build the frozen Phase 9 Validation report from existing artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEYWORDS = ("qingxiaojia", "doudou", "diandian", "xiaorui", "duoduo", "jizhiwa")
DISPLAY = {
    "qingxiaojia": "你好，青小甲", "doudou": "你好，豆豆", "diandian": "你好，点点",
    "xiaorui": "你好，小瑞", "duoduo": "你好，多多", "jizhiwa": "你好，吉智娃",
}
ROLES = {"bcresnet": "COMPUTE_LIGHT_BASELINE", "convmixer": "PRIMARY_CANDIDATE"}
SUMMARY_KEYS = (
    "macro_recall", "macro_precision", "macro_f1", "micro_accuracy",
    "worst_keyword_recall", "background_false_accept_rate", "background_rejection_rate",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _pp(value: float) -> str:
    return f"{100.0 * float(value):+.2f} pp"


def _load_run(run_dir: Path, expected_model: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    report = _read(run_dir / "TRAINING_REPORT.json")
    float_metrics = _read(run_dir / "confusion_float_validation.json")
    int8_metrics = _read(run_dir / "confusion_int8_validation.json")
    freeze = _read(run_dir / "threshold_freeze.json")
    if report.get("model_name") != expected_model:
        raise ValueError(f"Expected {expected_model}, got {report.get('model_name')}")
    if report.get("TEST_READ") is not False or freeze.get("TEST_READ") is not False:
        raise ValueError(f"{expected_model} artifacts do not preserve TEST_READ=false")
    if float_metrics.get("test_loaded") is not False or int8_metrics.get("test_loaded") is not False:
        raise ValueError(f"{expected_model} Validation artifact reports Test access")
    if report.get("PTQ_REPRESENTATIVE_SPLIT") != "train":
        raise ValueError(f"{expected_model} PTQ representative split is not train")
    if freeze.get("source") != "validation_only":
        raise ValueError(f"{expected_model} operating point is not Validation-only")
    for key, frozen_key in (("threshold", "top1_threshold"), ("margin_threshold", "margin_threshold")):
        if float(int8_metrics[key]) != float(freeze[frozen_key]):
            raise ValueError(f"{expected_model} frozen {key} disagrees with INT8 artifact")
    exports = list((run_dir / "export").glob("*.tflite"))
    if len(exports) != 1:
        raise ValueError(f"{expected_model} must have exactly one exported TFLite")
    tflite = exports[0].resolve()
    actual_sha = _sha256(tflite)
    if actual_sha != str(report["int8_export"]["sha256"]):
        raise ValueError(f"{expected_model} TFLite SHA256 mismatch")
    return {
        "run_dir": run_dir, "report": report, "float": float_metrics,
        "int8": int8_metrics, "freeze": freeze, "tflite": tflite,
        "tflite_sha256": actual_sha,
    }


def _summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in SUMMARY_KEYS}


def _degradation(run: dict[str, Any]) -> dict[str, float]:
    float_metrics, int8_metrics = run["float"], run["int8"]
    return {
        "macro_recall_pp": float(int8_metrics["macro_recall"]) - float(float_metrics["macro_recall"]),
        "macro_f1_pp": float(int8_metrics["macro_f1"]) - float(float_metrics["macro_f1"]),
        "worst_keyword_recall_pp": float(int8_metrics["worst_keyword_recall"]) - float(float_metrics["worst_keyword_recall"]),
        "background_far_pp": float(int8_metrics["background_false_accept_rate"]) - float(float_metrics["background_false_accept_rate"]),
    }


def _background_top_false_accept(metrics: dict[str, Any], class_names: list[str]) -> tuple[str, int]:
    row = metrics["confusion_matrix"][0]
    candidates = [(class_names[index], int(row[index])) for index in range(1, len(class_names))]
    return max(candidates, key=lambda item: (item[1], item[0]))


def _confusion_count(metrics: dict[str, Any], class_names: list[str], truth: str, guess: str) -> int:
    return int(metrics["confusion_matrix"][class_names.index(truth)][class_names.index(guess)])


def _overall_table(runs: dict[str, dict[str, Any]]) -> list[str]:
    labels = {
        "macro_recall": "Macro Recall", "macro_precision": "Macro Precision",
        "macro_f1": "Macro F1", "micro_accuracy": "Micro Accuracy",
        "worst_keyword_recall": "Worst Keyword Recall",
        "background_false_accept_rate": "Background FAR",
        "background_rejection_rate": "Background Rejection Rate",
    }
    lines = ["| 模型 | 阶段 | " + " | ".join(labels.values()) + " |",
             "|---|---|" + "---:|" * len(labels)]
    for model in ("bcresnet", "convmixer"):
        for phase in ("float", "int8"):
            metric = runs[model][phase]
            lines.append(
                f"| {model} | {'Float' if phase == 'float' else 'Full INT8'} | "
                + " | ".join(_pct(metric[key]) for key in labels) + " |"
            )
    return lines


def _keyword_table(runs: dict[str, dict[str, Any]], phase: str) -> list[str]:
    lines = [
        "| 关键词 | BC Recall | BC Precision | BC F1 | Conv Recall | Conv Precision | Conv F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for keyword in KEYWORDS:
        bc = runs["bcresnet"][phase]["per_class"][keyword]
        conv = runs["convmixer"][phase]["per_class"][keyword]
        lines.append(
            f"| {keyword} / {DISPLAY[keyword]} | {_pct(bc['recall'])} | {_pct(bc['precision'])} | {_pct(bc['f1'])} | "
            f"{_pct(conv['recall'])} | {_pct(conv['precision'])} | {_pct(conv['f1'])} |"
        )
    return lines


def _source_table(runs: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| 模型 | 阶段 | 关键词 | Kokoro Recall | VoxCPM1.5 Recall |",
        "|---|---|---|---:|---:|",
    ]
    for model in ("bcresnet", "convmixer"):
        for phase in ("float", "int8"):
            sources = runs[model][phase]["per_source_per_keyword_recall"]
            for keyword in KEYWORDS:
                lines.append(
                    f"| {model} | {'Float' if phase == 'float' else 'Full INT8'} | {keyword} | "
                    f"{_pct(sources['kokoro'][keyword])} | {_pct(sources['voxcpm15'][keyword])} |"
                )
    return lines


def _confusion_section(runs: dict[str, dict[str, Any]]) -> list[str]:
    class_names = list(runs["bcresnet"]["report"]["class_names"])
    lines = [
        "| 模型 | 阶段 | true → predicted | count |",
        "|---|---|---|---:|",
    ]
    for model in ("bcresnet", "convmixer"):
        for phase in ("float", "int8"):
            for row in runs[model][phase]["top_confusion_pairs"][:8]:
                lines.append(f"| {model} | {phase} | {row['true']} → {row['predicted']} | {row['count']} |")
    lines.extend(["", "关键词 → background（最终输出第 0 类）的拒识统计：", "",
                  "| 模型 | 阶段 | qingxiaojia | doudou | diandian | xiaorui | duoduo | jizhiwa |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for model in ("bcresnet", "convmixer"):
        for phase in ("float", "int8"):
            metric = runs[model][phase]
            counts = [_confusion_count(metric, class_names, keyword, "background") for keyword in KEYWORDS]
            lines.append(f"| {model} | {phase} | " + " | ".join(str(value) for value in counts) + " |")
    lines.extend(["", "近音关键词 doudou / diandian / duoduo 的定向混淆：", "",
                  "| 模型 | 阶段 | doudou→diandian | doudou→duoduo | diandian→doudou | diandian→duoduo | duoduo→doudou | duoduo→diandian |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    pairs = (("doudou", "diandian"), ("doudou", "duoduo"), ("diandian", "doudou"),
             ("diandian", "duoduo"), ("duoduo", "doudou"), ("duoduo", "diandian"))
    for model in ("bcresnet", "convmixer"):
        for phase in ("float", "int8"):
            metric = runs[model][phase]
            counts = [_confusion_count(metric, class_names, truth, guess) for truth, guess in pairs]
            lines.append(f"| {model} | {phase} | " + " | ".join(str(value) for value in counts) + " |")
    lines.append("")
    for model in ("bcresnet", "convmixer"):
        for phase in ("float", "int8"):
            keyword, count = _background_top_false_accept(runs[model][phase], class_names)
            lines.append(f"- {model} {phase}：background 最常被误识别为 `{keyword}`（{count} 条）。")
    return lines


def build_outputs(
    bc_run: Path, conv_run: Path, dataset_info_path: Path,
) -> tuple[str, dict[str, Any]]:
    runs = {
        "bcresnet": _load_run(bc_run, "bcresnet"),
        "convmixer": _load_run(conv_run, "convmixer"),
    }
    dataset = _read(dataset_info_path.resolve())
    if dataset.get("TEST_READ") is not False:
        raise ValueError("Dataset artifact does not preserve TEST_READ=false")
    if runs["bcresnet"]["report"]["class_names"] != runs["convmixer"]["report"]["class_names"]:
        raise ValueError("Model vocabularies differ")
    selection_models: dict[str, Any] = {}
    for model, run in runs.items():
        export = run["report"]["int8_export"]
        selection_models[model] = {
            "role": ROLES[model], "run": _relative(run["run_dir"]),
            "tflite": {"path": _relative(run["tflite"]), "sha256": run["tflite_sha256"],
                       "bytes": int(export["bytes"]), "full_int8": True},
            "frozen_int8_threshold": float(run["freeze"]["top1_threshold"]),
            "frozen_int8_margin_threshold": float(run["freeze"]["margin_threshold"]),
            "validation_summary": {"float": _summary(run["float"]), "int8": _summary(run["int8"]),
                                   "sample_count": int(run["int8"]["sample_count"])},
        }
    selection = {
        "schema": "wakeword-studio.multikws-model-selection-validation/v1",
        "dataset": {
            "id": dataset["dataset_id"], "dataset_sha256": dataset["dataset_sha256"],
            "manifest_sha256": dataset["manifest_sha256"],
            "manifest_file_sha256": dataset["manifest_file_sha256"],
            "split_counts": dataset["split_counts"],
        },
        "class_names": list(runs["bcresnet"]["report"]["class_names"]),
        "models": selection_models, "selection_source": "validation_only",
        "TEST_READ": False, "98PCT": "NOT_ACHIEVED",
    }

    lines = [
        "# Teacher Six Multi-KWS：Validation 阶段报告",
        "",
        "> 本报告由现有正式 run artifact 自动生成；未重新训练、未重新推理、未重新校准。",
        "> `selection_source=validation_only`，`TEST_READ=false`，`98PCT=NOT_ACHIEVED`。",
        "",
        "## 1. 实验目标",
        "",
        "一个 7-class softmax 模型同时识别六个提示词；第 0 类是 `background`，不是六个独立 binary model。",
        "",
        "| class | keyword_id | 显示文本 |", "|---:|---|---|", "| 0 | background | 背景/拒识 |",
    ]
    lines.extend(f"| {index} | {key} | {DISPLAY[key]} |" for index, key in enumerate(KEYWORDS, 1))
    lines.extend([
        "", "## 2. Dataset 与公平协议", "",
        f"- Dataset ID：`{dataset['dataset_id']}`；dataset SHA256：`{dataset['dataset_sha256']}`。",
        f"- Train / Validation / Test：{dataset['split_counts']['train']} / {dataset['split_counts']['validation']} / {dataset['split_counts']['test']}。",
        f"- 来源计数：Kokoro {dataset['source_counts']['kokoro']}、VoxCPM1.5 {dataset['source_counts']['voxcpm15']}、procedural ambient {dataset['source_counts']['procedural_ambient']}。",
        "- 两模型使用同一 dataset、split、deterministic epoch sampler、seed 与 Validation calibration/ranking protocol；PTQ representative split 均为 `train`。",
        "- 当前 Test 从未用于特征提取、训练、校准、选择或本报告：`TEST_READ=false`。",
        "- 数据是 multi-source / multi-speaker；`AGE_VERIFIED=false`，multi-speaker 不等于 multi-age。",
        "", "## 3. 模型训练与部署概况", "",
        "`parameter_count` 严格采用 `TRAINING_REPORT.json` 的可训练参数统计口径。",
        "",
        "| 模型 | Architecture | Steps / Epochs | Early stop | Params | Estimated MACs | TFLite bytes / KiB | SHA256 | Full INT8 | Hardware verified |",
        "|---|---|---:|---|---:|---:|---:|---|---|---|",
    ])
    for model in ("bcresnet", "convmixer"):
        run = runs[model]; report = run["report"]; export = report["int8_export"]
        epochs = float(report["completed_steps"]) / float(report["sampler"]["steps_per_epoch"])
        architecture = json.dumps(report["architecture_config"], ensure_ascii=False, separators=(",", ":"))
        lines.append(
            f"| {model} | `{architecture}` | {report['completed_steps']} / {epochs:.1f} | "
            f"{str(report['stopped_early']).lower()} | {report['parameter_count']} | {report['estimated_macs']} | "
            f"{export['bytes']} / {export['KiB']:.2f} | `{run['tflite_sha256']}` | true | "
            f"{str(report['HARDWARE_RUNTIME_VERIFIED']).lower()} |"
        )
    lines.extend(["", "## 4. BC-ResNet vs ConvMixer 总体 Validation 对比", ""])
    lines.extend(_overall_table(runs))
    lines.extend(["", "## 5. 六关键词逐项对比", "", "### Float", ""])
    lines.extend(_keyword_table(runs, "float"))
    lines.extend(["", "### Full INT8（最终部署候选）", ""])
    lines.extend(_keyword_table(runs, "int8"))
    lines.extend(["", "## 6. Float → INT8 量化稳定性", "",
                  "下表为 `INT8 - Float`；正值表示数值升高，负值表示下降。", "",
                  "| 模型 | Macro Recall | Macro F1 | Worst Keyword Recall | Background FAR |",
                  "|---|---:|---:|---:|---:|"])
    for model in ("bcresnet", "convmixer"):
        deg = _degradation(runs[model])
        lines.append(f"| {model} | {_pp(deg['macro_recall_pp'])} | {_pp(deg['macro_f1_pp'])} | {_pp(deg['worst_keyword_recall_pp'])} | {_pp(deg['background_far_pp'])} |")
    lines.extend([
        "", "BC-ResNet 存在明显 PTQ sensitivity，尤其 `doudou`；ConvMixer 的 Recall 量化稳定性明显更好，但其 Background FAR 在量化后仍明显升高。INT8 的个别指标高于 Float 只表示该冻结 operating point 和有限 Validation 样本上的观测差异，不表示量化使模型本质变强。",
        "", "## 7. Confusion / Error Analysis", "",
        "`关键词 → background` 是 false reject/rejection；`关键词 → 另一个关键词` 才是 keyword-to-keyword confusion。最终输出为 background 可能来自 top-1 本来就是 background，也可能来自非 background top-1 未通过 threshold/margin。",
        "",
    ])
    lines.extend(_confusion_section(runs))
    lines.extend(["", "## 8. Source Generalization", ""])
    lines.extend(_source_table(runs))
    lines.extend([
        "", "Kokoro 上接近满分不能外推为真实系统接近 98%；VoxCPM1.5 明显更差，表明跨 source / reference speaker 泛化仍是主要瓶颈。procedural ambient 不是 speech source，因此没有 per-keyword Recall。",
        "", "## 9. 部署成本", "",
        f"BC-ResNet 的 TRAINING_REPORT estimated MACs 为 {runs['bcresnet']['report']['estimated_macs']:,}，ConvMixer 为 {runs['convmixer']['report']['estimated_macs']:,}（约 {runs['convmixer']['report']['estimated_macs']/runs['bcresnet']['report']['estimated_macs']:.2f}×）。BC-ResNet 计算量更小；ConvMixer TFLite 更小且 INT8 Validation 更稳，但计算量明显更高。实际 ESP32-S3 latency 尚未验证。若 exporter 日志另有 MAC 数字，它与 TRAINING_REPORT 的静态 `estimated_macs` 属于不同统计口径，不应混用。",
        "", "## 10. 指标与 operating point 说明", "",
        "- **Worst Keyword Recall**：六个关键词 Recall 的最小值。",
        "- **Background FAR**：真实 background 在冻结决策逻辑后被输出为任一关键词的比例。",
        "- **Background Rejection Rate**：真实 background 最终输出第 0 类的比例，与同一集合上的 FAR 互补。",
        "- **Per-source Per-keyword Recall**：在指定 speech source 且真实标签为该关键词的子集上，最终正确输出该关键词的比例。",
        "- **Float→INT8 degradation**：同一 Validation 与冻结 operating point 下 INT8 指标减 Float 指标；`pp` 是百分点。",
        "- **Threshold / margin threshold**：运行时代码先稳定降序取 top-1/top-2。只有 top-1 不是 background、top-1 score ≥ threshold 且 top-1−top-2 ≥ margin threshold 时才接受关键词；否则输出第 0 类。",
        "- **MACs**：模型一次前向的静态乘加次数估算，不等同于真实硬件 latency。",
        "- **Full INT8/PTQ**：用 Train representative samples 做训练后量化，TFLite 输入、输出及受支持算子均为 INT8；未做量化感知训练。",
        "", "## 11. 当前阶段结论", "",
        "- BC-ResNet：`ROLE=COMPUTE_LIGHT_BASELINE`。",
        "- ConvMixer：`ROLE=PRIMARY_CANDIDATE`；Final INT8 overall / worst-keyword 更均衡、PTQ stability 更好且文件更小，但 MACs 明显更高。",
        "- `98PCT=NOT_ACHIEVED`；不能声称六关键词达到 98%。",
        "", "## 12. 当前限制", "",
        "- Validation 是合成 benchmark；real microphone acceptance 未完成。",
        "- Test 尚未打开：`TEST_READ=false`。",
        "- ESP32-S3 hardware runtime 未验证。",
        "- `AGE_VERIFIED=false`。",
        "- source / speaker 泛化仍不足。",
    ])
    return "\n".join(lines) + "\n", selection


def write_outputs(markdown: str, selection: dict[str, Any], output: Path, selection_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    selection_output.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--convmixer-run", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    args = parser.parse_args()
    markdown, selection = build_outputs(args.bc_run, args.convmixer_run, args.dataset_info)
    write_outputs(markdown, selection, args.output, args.selection_output)
    print(json.dumps({"report": str(args.output.resolve()), "selection": str(args.selection_output.resolve()),
                      "TEST_READ": selection["TEST_READ"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
