"""Metrics identical to sentinel.evaluate so LLM and XGBoost numbers are comparable."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> Dict:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(labels)) < 2:
        return {"method": "platt", "a": 1.0, "b": 0.0}
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(logits).reshape(-1, 1), labels)
    return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_platt(logits, cal: Dict) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(cal["a"] * np.asarray(logits, dtype=np.float64) + cal["b"])))


def threshold_for_recall(probs, labels, target):
    pos = np.sort(probs[labels == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    return float(pos[max(1, math.ceil(target * len(pos))) - 1])


def threshold_for_precision(probs, labels, target):
    order = np.argsort(-probs)
    p_sorted, y_sorted = probs[order], labels[order]
    prec = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return float(p_sorted[int(np.argmax(prec))])
    return float(p_sorted[ok[-1]])


def at_threshold(probs, labels, thr) -> Dict:
    pred = probs >= thr
    tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum()); neg = int((labels == 0).sum())
    return {"threshold": float(thr), "flagged": int(pred.sum()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_positive_rate": fp / neg if neg else 0.0}


def ece(probs, labels, bins=15) -> float:
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def _thin(xs, ys, n=200):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [round(float(v), 5) for v in np.asarray(xs)[idx]], [round(float(v), 5) for v in np.asarray(ys)[idx]]


def curves(probs, labels) -> Dict:
    from sklearn.metrics import precision_recall_curve, roc_curve

    if len(np.unique(labels)) < 2:
        return {}
    p, r, _ = precision_recall_curve(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    rx, py = _thin(r, p); fx, ty = _thin(fpr, tpr)
    return {"pr": {"recall": rx, "precision": py}, "roc": {"fpr": fx, "tpr": ty}}


def classification_metrics(probs, labels, review_thr, block_thr) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    both = len(np.unique(labels)) == 2
    return {"curves": curves(probs, labels), "rows": int(len(labels)), "fraud_rows": int(labels.sum()),
            "pr_auc": float(average_precision_score(labels, probs)) if both else None,
            "roc_auc": float(roc_auc_score(labels, probs)) if both else None,
            "ece": ece(probs, labels),
            "at_review": at_threshold(probs, labels, review_thr),
            "at_block": at_threshold(probs, labels, block_thr)}


def calibrate_and_pick_thresholds(logits, labels, target_recall=0.85, target_precision=0.70) -> Dict:
    cal = fit_platt(logits, labels)
    probs = apply_platt(logits, cal)
    review = threshold_for_recall(probs, labels, target_recall)
    block = max(threshold_for_precision(probs, labels, target_precision), review)
    return {**cal, "review_threshold": review, "block_threshold": block,
            "target_recall": target_recall, "target_precision": target_precision}
