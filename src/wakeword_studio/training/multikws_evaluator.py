"""Validation calibration, confusion analysis, and runtime decisions for Multi-KWS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class MultiKWSDecision:
    accepted: bool
    state: str
    class_index: int
    top1_score: float
    top2_index: int
    top2_score: float
    margin: float


def runtime_decision(scores: Sequence[float], *, threshold: float, margin_threshold: float) -> MultiKWSDecision:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("scores must be one-dimensional with at least two classes")
    order = np.argsort(-values, kind="stable")
    top1, top2 = int(order[0]), int(order[1])
    margin = float(values[top1] - values[top2])
    accepted = top1 != 0 and float(values[top1]) >= threshold and margin >= margin_threshold
    state = "WAKE" if accepted else "BACKGROUND" if top1 == 0 else "AMBIGUOUS"
    return MultiKWSDecision(accepted, state, top1 if accepted else 0, float(values[top1]), top2, float(values[top2]), margin)


def _predictions(scores: np.ndarray, threshold: float, margin: float) -> np.ndarray:
    return np.asarray([runtime_decision(row, threshold=threshold, margin_threshold=margin).class_index for row in scores], np.int32)


def confusion_matrix(targets: Sequence[int], predictions: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, predicted in zip(targets, predictions):
        matrix[int(target), int(predicted)] += 1
    return matrix


def metrics_from_predictions(
    scores: np.ndarray, targets: Sequence[int], predictions: Sequence[int],
    class_names: Sequence[str], sources: Sequence[str], threshold: float, margin: float,
) -> dict[str, Any]:
    target = np.asarray(targets, np.int32); predicted = np.asarray(predictions, np.int32)
    count = len(class_names); matrix = confusion_matrix(target, predicted, count)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=np.float64), where=row_sum != 0)
    per_class: dict[str, Any] = {}; recalls=[]; precisions=[]; f1s=[]
    for index, name in enumerate(class_names):
        tp=int(matrix[index,index]); fn=int(matrix[index,:].sum()-tp); fp=int(matrix[:,index].sum()-tp)
        recall=tp/(tp+fn) if tp+fn else 0.0; precision=tp/(tp+fp) if tp+fp else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[name]={"tp":tp,"fp":fp,"fn":fn,"recall":recall,"precision":precision,"f1":f1}
        if index: recalls.append(recall); precisions.append(precision); f1s.append(f1)
    source_array=np.asarray(sources,object); per_source={}
    for source in sorted(set(sources)):
        mask=source_array==source; rows={}
        for index,name in enumerate(class_names[1:],start=1):
            selected=mask & (target==index); rows[name]=float(np.mean(predicted[selected]==index)) if np.any(selected) else None
        per_source[source]=rows
    pairs=[]
    for truth in range(1,count):
        for guess in range(count):
            if truth!=guess and matrix[truth,guess]: pairs.append({"true":class_names[truth],"predicted":class_names[guess],"count":int(matrix[truth,guess])})
    pairs.sort(key=lambda row:(-row["count"],row["true"],row["predicted"]))
    topk=[]
    for row in scores:
        order=np.argsort(-row,kind="stable")[:min(3,count)]
        topk.append([{"class":class_names[int(i)],"score":float(row[int(i)])} for i in order])
    bg=target==0
    return {
        "threshold":float(threshold),"margin_threshold":float(margin),
        "confusion_matrix":matrix.tolist(),"normalized_confusion_matrix":normalized.tolist(),
        "per_class":per_class,"macro_recall":float(np.mean(recalls)),"macro_precision":float(np.mean(precisions)),
        "macro_f1":float(np.mean(f1s)),"micro_accuracy":float(np.mean(target==predicted)),
        "worst_keyword_recall":float(min(recalls)),"per_source_per_keyword_recall":per_source,
        "background_false_accept_rate":float(np.mean(predicted[bg]!=0)) if np.any(bg) else 0.0,
        "background_rejection_rate":float(np.mean(predicted[bg]==0)) if np.any(bg) else 0.0,
        "top_confusion_pairs":pairs[:20],"top_k_scores":topk,"sample_count":len(target),"test_loaded":False,
    }


def calibrate_validation(scores: np.ndarray, targets: Sequence[int], class_names: Sequence[str], sources: Sequence[str]) -> dict[str, Any]:
    scores=np.asarray(scores,np.float64); target=np.asarray(targets,np.int32)
    best=None; best_rank=None
    for threshold in np.linspace(0.0,0.9,10):
        for margin in np.linspace(0.0,0.5,6):
            pred=_predictions(scores,float(threshold),float(margin))
            metrics=metrics_from_predictions(scores,target,pred,class_names,sources,float(threshold),float(margin))
            rank=(metrics["macro_f1"],metrics["worst_keyword_recall"],-metrics["background_false_accept_rate"],metrics["micro_accuracy"])
            if best_rank is None or rank>best_rank: best_rank=rank; best=metrics
    assert best is not None
    best["calibration_source"]="validation_only"; best["selection_formula"]="maximize (macro_f1, worst_keyword_recall, -background_far, accuracy)"
    return best

