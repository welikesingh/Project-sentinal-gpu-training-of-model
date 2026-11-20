"""Metrics, Platt calibration and threshold selection.

    python -m sentinel.evaluate --run output/<run> --split holdout [--max-rows 20000]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- calibration
def fit_platt(logits: np.ndarray, labels: np.ndarray) -> Dict:
    """Fit p = sigmoid(a*logit + b) by logistic regression on one feature."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(labels)) < 2:
        return {"method": "platt", "a": 1.0, "b": 0.0}
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logits.reshape(-1, 1), labels)
    return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_platt(logits: np.ndarray, cal: Dict) -> np.ndarray:
    z = cal["a"] * np.asarray(logits, dtype=np.float64) + cal["b"]
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- thresholds
def threshold_for_recall(probs: np.ndarray, labels: np.ndarray, target: float) -> float:
    """Largest threshold whose recall >= target."""
    pos = np.sort(probs[labels == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = max(1, math.ceil(target * len(pos)))
    return float(pos[k - 1])


def threshold_for_precision(probs: np.ndarray, labels: np.ndarray, target: float) -> float:
    """Smallest threshold whose precision >= target (max-precision threshold if never reached)."""
    order = np.argsort(-probs)
    p_sorted, y_sorted = probs[order], labels[order]
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, len(y_sorted) + 1)
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return float(p_sorted[int(np.argmax(prec))])
    return float(p_sorted[ok[-1]])


def at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    pred = probs >= thr
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    neg = int((labels == 0).sum())
    return {
        "threshold": float(thr),
        "flagged": int(pred.sum()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / neg if neg else 0.0,
    }


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo) & (probs <= hi)
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def _thin(xs, ys, n: int = 200):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [round(float(v), 5) for v in np.asarray(xs)[idx]], [round(float(v), 5) for v in np.asarray(ys)[idx]]


def curves(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """Down-sampled PR and ROC curve points (for dashboards)."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    if len(np.unique(labels)) < 2:
        return {}
    p, r, _ = precision_recall_curve(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    rx, py = _thin(r, p)
    fx, ty = _thin(fpr, tpr)
    return {"pr": {"recall": rx, "precision": py}, "roc": {"fpr": fx, "tpr": ty}}


def classification_metrics(probs: np.ndarray, labels: np.ndarray,
                           review_thr: float, block_thr: float) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    both = len(np.unique(labels)) == 2
    return {
        "curves": curves(probs, labels),
        "rows": int(len(labels)),
        "fraud_rows": int(labels.sum()),
        "pr_auc": float(average_precision_score(labels, probs)) if both else None,
        "roc_auc": float(roc_auc_score(labels, probs)) if both else None,
        "ece": ece(probs, labels),
        "at_review": at_threshold(probs, labels, review_thr),
        "at_block": at_threshold(probs, labels, block_thr),
    }


def reason_metrics(reason_logits: np.ndarray, reason_labels: np.ndarray) -> Dict:
    from sklearn.metrics import f1_score

    from sentinel import REASON_CODES

    pred = reason_logits > 0
    per = {c: float(f1_score(reason_labels[:, i], pred[:, i], zero_division=0))
           for i, c in enumerate(REASON_CODES)}
    return {"micro_f1": float(f1_score(reason_labels, pred, average="micro", zero_division=0)),
            "per_code_f1": per}


# ---------------------------------------------------------------- end to end
def score_rows(model, tok, rows: List[dict], max_length: int, batch_size: int):
    from sentinel.data import reason_vector
    from sentinel.model import predict_logits

    logits = predict_logits(model, tok, [r["prompt"] for r in rows], max_length, batch_size).numpy()
    labels = np.array([r["fraud_label"] for r in rows])
    reasons = np.array([reason_vector(r["reasons"]) for r in rows])
    return logits, labels, reasons


def calibrate_and_pick_thresholds(fraud_logits: np.ndarray, labels: np.ndarray,
                                  target_recall: float = 0.85,
                                  target_precision: float = 0.70) -> Dict:
    cal = fit_platt(fraud_logits, labels)
    probs = apply_platt(fraud_logits, cal)
    review = threshold_for_recall(probs, labels, target_recall)
    block = max(threshold_for_precision(probs, labels, target_precision), review)
    return {**cal, "review_threshold": review, "block_threshold": block,
            "target_recall": target_recall, "target_precision": target_precision}


def evaluate_run(run_dir: str, split: str = "holdout", max_rows: int = 0,
                 batch_size: int = 64, data_dir: str = "data/sft") -> Dict:
    from sentinel.data import load_split, stratified_sample
    from sentinel.model import load_trained

    run = Path(run_dir)
    model, tok, meta = load_trained(run_dir)
    cal = json.loads((run / "calibration.json").read_text())
    rows = stratified_sample(load_split(split, Path(data_dir)), max_rows)
    logits, labels, reasons = score_rows(model, tok, rows, meta["max_length"], batch_size)
    probs = apply_platt(logits[:, 0], cal)
    res = {
        "split": split,
        "calibrated": classification_metrics(probs, labels, cal["review_threshold"], cal["block_threshold"]),
        "reasons": reason_metrics(logits[:, 1:], reasons),
    }
    out = run / f"metrics_{split}.json"
    out.write_text(json.dumps(res, indent=2))
    logger.info("Wrote %s", out)
    return res


def main(argv: Optional[List[str]] = None):
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Evaluate a trained SENTINEL run")
    p.add_argument("--run", required=True, help="Run directory (output/<name>)")
    p.add_argument("--split", default="holdout", choices=["train", "validation", "holdout"])
    p.add_argument("--max-rows", type=int, default=0, help="Stratified subsample (0 = all rows)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-dir", default="data/sft")
    a = p.parse_args(argv)
    res = evaluate_run(a.run, a.split, a.max_rows, a.batch_size, a.data_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
