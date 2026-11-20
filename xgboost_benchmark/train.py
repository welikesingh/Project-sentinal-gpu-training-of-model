"""Train the XGBoost benchmark on the same splits / rules as the LLM.

    python train.py                       # from inside xgboost_benchmark/
    python train.py --device cuda         # GPU (Hyperstack)

Same protocol as sentinel.train:
  train on `train`, early-stop + Platt-calibrate + pick thresholds on `validation`
  (REVIEW: recall >= 0.85, BLOCK: precision >= 0.70), report frozen-threshold metrics on `holdout`.
Writes artifacts/: model.json, meta.json, metrics_validation.json, metrics_holdout.json
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_split  # noqa: E402
from features import FEATURES, parse_prompt, to_frame  # noqa: E402
from metrics import apply_platt, calibrate_and_pick_thresholds, classification_metrics  # noqa: E402

logger = logging.getLogger("xgb.train")
ART = Path(__file__).resolve().parent / "artifacts"


def frame(rows):
    return to_frame([parse_prompt(r["prompt"]) for r in rows]), np.array([r["fraud_label"] for r in rows])


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Train the SENTINEL XGBoost benchmark")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default=str(ART))
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--early-stop", type=int, default=200)
    p.add_argument("--scale-pos-weight", type=float, default=0.0, help="0 = sqrt(neg/pos)")
    p.add_argument("--target-recall", type=float, default=0.85)
    p.add_argument("--target-precision", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    import xgboost as xgb

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dd = Path(a.data_dir) if a.data_dir else None
    Xtr, ytr = frame(load_split("train", dd))
    Xva, yva = frame(load_split("validation", dd))
    Xho, yho = frame(load_split("holdout", dd))

    spw = a.scale_pos_weight or float(np.sqrt((ytr == 0).sum() / max(1, (ytr == 1).sum())))
    params = {"objective": "binary:logistic", "eval_metric": ["aucpr", "logloss"],
              "tree_method": "hist", "device": a.device, "max_depth": a.max_depth, "eta": a.eta,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 1.0,
              "scale_pos_weight": spw, "max_cat_to_onehot": 16, "seed": a.seed}
    dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
    t0 = time.time()
    booster = xgb.train(params, dtr, a.rounds, evals=[(dtr, "train"), (dva, "validation")],
                        early_stopping_rounds=a.early_stop, verbose_eval=200)
    train_s = time.time() - t0
    best = booster.best_iteration
    logger.info("Trained in %.1fs, best_iteration=%d", train_s, best)

    def margins(X):
        return booster.predict(xgb.DMatrix(X, enable_categorical=True), output_margin=True,
                               iteration_range=(0, best + 1))

    cal = calibrate_and_pick_thresholds(margins(Xva), yva, a.target_recall, a.target_precision)
    res = {}
    for split, X, y in (("validation", Xva, yva), ("holdout", Xho, yho)):
        probs = apply_platt(margins(X), cal)
        res[split] = {"split": split, "calibrated": classification_metrics(
            probs, y, cal["review_threshold"], cal["block_threshold"])}
        (out / f"metrics_{split}.json").write_text(json.dumps(res[split], indent=2))
        m = res[split]["calibrated"]
        logger.info("%s: PR-AUC=%.4f ROC-AUC=%.4f review=%s block=%s", split, m["pr_auc"], m["roc_auc"],
                    {k: round(v, 3) for k, v in m["at_review"].items()}, {k: round(v, 3) for k, v in m["at_block"].items()})

    gain = booster.get_score(importance_type="total_gain")
    tot = sum(gain.values()) or 1.0
    booster.save_model(out / "model.json")
    meta = {"kind": "xgboost", "features": FEATURES, "best_iteration": best, "params": params,
            "calibration": cal, "train_rows": int(len(ytr)), "train_fraud": int(ytr.sum()),
            "train_runtime_s": round(train_s, 1), "host": socket.gethostname(), "xgboost": xgb.__version__,
            "feature_importance": dict(sorted(((k, v / tot) for k, v in gain.items()), key=lambda kv: -kv[1]))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Artifacts in %s", out)


if __name__ == "__main__":
    main()
