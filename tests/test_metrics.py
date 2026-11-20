import numpy as np

from sentinel.evaluate import (apply_platt, at_threshold, calibrate_and_pick_thresholds,
                               classification_metrics, fit_platt, threshold_for_precision,
                               threshold_for_recall)


def _data(seed=0, n=2000, rate=0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    logits = rng.normal(0, 1, n) + 3 * y
    return logits, y


def test_platt_monotonic_and_calibrated():
    logits, y = _data()
    cal = fit_platt(logits, y)
    p = apply_platt(logits, cal)
    assert cal["a"] > 0
    assert np.all(np.diff(p[np.argsort(logits)]) >= -1e-12)
    assert abs(p.mean() - y.mean()) < 0.01


def test_recall_threshold_hits_target():
    logits, y = _data()
    p = apply_platt(logits, fit_platt(logits, y))
    thr = threshold_for_recall(p, y, 0.85)
    assert at_threshold(p, y, thr)["recall"] >= 0.85


def test_precision_threshold_hits_target_when_reachable():
    logits, y = _data()
    p = apply_platt(logits, fit_platt(logits, y))
    thr = threshold_for_precision(p, y, 0.70)
    assert at_threshold(p, y, thr)["precision"] >= 0.70


def test_perfect_separation_metrics():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = classification_metrics(p, y, review_thr=0.5, block_thr=0.85)
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    assert m["at_review"]["recall"] == 1.0 and m["at_review"]["precision"] == 1.0
    assert m["at_block"]["flagged"] == 1


def test_calibrate_block_not_below_review():
    logits, y = _data(rate=0.01)
    c = calibrate_and_pick_thresholds(logits, y)
    assert c["block_threshold"] >= c["review_threshold"]
