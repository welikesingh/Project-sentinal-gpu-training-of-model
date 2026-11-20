"""XGBoost benchmark API (Hyperstack GPU via hyperstack/xgb.sh, or your machine + ngrok).

    uvicorn server:app --host 0.0.0.0 --port 8001        # from inside xgboost_benchmark/

Endpoints (all except /health need header X-API-Key when XGB_API_KEY is set):
  GET  /health
  GET  /stats                       model info, calibration, metrics (+ curves), feature importance
  POST /score                       {"prompt"} | {structured fields} | {"transactions": [...]}  (same contract as the LLM)
  GET  /samples?split=holdout&n=50&fraud_only=false&fraud_ratio=0.3&seed=1   labelled rows for live benchmarking
  GET  /scenarios                   curated test scenarios (structured transactions)
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from features import REASON_MAP, build_prompt, parse_prompt, to_frame  # noqa: E402
from metrics import apply_platt  # noqa: E402

logger = logging.getLogger("xgb.server")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")
if not os.getenv("XGB_API_KEY"):
    logger.warning("XGB_API_KEY not set - anyone with the URL (e.g. your ngrok link) can call this API")
REASON_TEXT = {
    "R01_NEW_DEVICE": "new device", "R02_GEO_MISMATCH": "location differs from merchant/home city",
    "R03_ODD_HOUR": "unusual hour", "R04_FAILED_ATTEMPTS": "recent failed attempts",
    "R05_AMOUNT_SPIKE": "amount far above the customer's average", "R06_NEW_ACCOUNT": "new account",
    "R07_VELOCITY": "high transaction velocity", "R08_HIGH_RISK_MCC": "high-risk merchant category",
}


class XGBScorer:
    def __init__(self, art_dir: Path):
        import xgboost as xgb

        self.xgb = xgb
        self.art = Path(art_dir)
        self.meta = json.loads((self.art / "meta.json").read_text())
        self.cal = self.meta["calibration"]
        self.booster = xgb.Booster()
        self.booster.load_model(self.art / "model.json")
        self.device = os.getenv("XGB_DEVICE", "cpu")      # "cuda" on the Hyperstack GPU VM
        if self.device != "cpu":
            try:
                self.booster.set_param({"device": self.device})
            except self.xgb.core.XGBoostError as e:
                logger.warning("XGB_DEVICE=%s not usable (%s) - using cpu", self.device, e)
                self.device = "cpu"
        self.features = self.meta["features"]
        self.rng = (0, self.meta["best_iteration"] + 1)
        self.code_idx = {c: [self.features.index(f) for f in fs if f in self.features] for c, fs in REASON_MAP.items()}

    def score(self, items: List) -> List[Dict]:
        prompts, ids = [], []
        for it in items:
            if isinstance(it, str):
                prompts.append(it); ids.append(None)
            else:
                prompts.append(it["prompt"] if "prompt" in it else build_prompt(it))
                ids.append(it.get("transaction_id"))
        dm = self.xgb.DMatrix(to_frame([parse_prompt(p) for p in prompts]), enable_categorical=True)
        margin = self.booster.predict(dm, output_margin=True, iteration_range=self.rng)
        contrib = self.booster.predict(dm, pred_contribs=True, iteration_range=self.rng)
        probs = apply_platt(margin, self.cal)
        out = []
        for i, p in enumerate(probs):
            p = float(p)
            decision = ("BLOCK" if p >= self.cal["block_threshold"]
                        else "REVIEW" if p >= self.cal["review_threshold"] else "ALLOW")
            code_contrib = {c: float(contrib[i, idx].sum()) for c, idx in self.code_idx.items()}
            reasons = [c for c, v in sorted(code_contrib.items(), key=lambda kv: -kv[1]) if v > 0.25][:4]
            lead = {"ALLOW": "Allowed despite", "REVIEW": "Sent to review:", "BLOCK": "Blocked:"}[decision]
            expl = (f"{lead} {', '.join(REASON_TEXT[r] for r in reasons)}." if reasons else
                    "No strong risk signals observed." if decision == "ALLOW" else
                    "Risk score elevated by the combination of transaction context and history.")
            out.append({"transaction_id": ids[i], "fraud_probability": round(p, 6), "decision": decision,
                        "confidence": round(max(p, 1 - p), 4), "reason_codes": reasons, "explanation": expl})
        return out

    def stats(self) -> Dict:
        def read(n):
            f = self.art / n
            return json.loads(f.read_text()) if f.exists() else None
        keep = ("best_iteration", "train_rows", "train_fraud", "train_runtime_s", "host", "xgboost")
        return {"kind": "xgboost", "model": f"XGBoost {self.meta.get('xgboost', '')} (hist, depth {self.meta['params']['max_depth']})",
                "run": self.art.name, "info": {**{k: self.meta.get(k) for k in keep}, "device": self.device},
                "calibration": self.cal,
                "metrics": {"validation": read("metrics_validation.json"), "holdout": read("metrics_holdout.json")},
                "feature_importance": self.meta.get("feature_importance", {})}


app = FastAPI(title="SENTINEL XGBoost benchmark", version="1.0")
_state: Dict = {}


def scorer() -> XGBScorer:
    if "scorer" not in _state:
        _state["scorer"] = XGBScorer(Path(os.getenv("XGB_ARTIFACTS", HERE / "artifacts")))
    return _state["scorer"]


def auth(x_api_key: Optional[str] = Header(default=None)):
    key = os.getenv("XGB_API_KEY")
    if key and not hmac.compare_digest(x_api_key or "", key):
        raise HTTPException(401, "missing or invalid X-API-Key")


def _split_rows(split: str) -> List[dict]:
    cache = _state.setdefault("rows", {})
    if split not in cache:
        from data import load_split
        cache[split] = load_split(split)
    return cache[split]


@app.get("/health")
def health():
    ok = (Path(os.getenv("XGB_ARTIFACTS", HERE / "artifacts")) / "model.json").exists()
    return {"status": "ok" if ok else "no-model", "kind": "xgboost", "run": Path(os.getenv("XGB_ARTIFACTS", HERE / "artifacts")).name,
            "device": os.getenv("XGB_DEVICE", "cpu"), "auth_required": bool(os.getenv("XGB_API_KEY"))}


@app.get("/stats", dependencies=[Depends(auth)])
def stats():
    return scorer().stats()


@app.post("/score", dependencies=[Depends(auth)])
def score(body: Dict):
    items = body.get("transactions") or [body]
    if len(items) > 1024:
        raise HTTPException(413, "max 1024 transactions per request")
    t0 = time.perf_counter()
    try:
        res = scorer().score(items)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, f"bad request: {e}")
    return {"results": res, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/samples", dependencies=[Depends(auth)])
def samples(split: str = Query("holdout", pattern="^(train|validation|holdout)$"),
            n: int = Query(50, ge=1, le=2000), fraud_only: bool = False,
            fraud_ratio: float = Query(0.3, ge=0, le=1), seed: int = 1):
    rows = _split_rows(split)
    rnd = random.Random(seed)
    fraud = [r for r in rows if r["fraud_label"] == 1]
    genuine = [r for r in rows if r["fraud_label"] == 0]
    if fraud_only:
        pick = rnd.sample(fraud, min(n, len(fraud)))
    else:
        nf = min(len(fraud), int(round(n * fraud_ratio)))
        pick = rnd.sample(fraud, nf) + rnd.sample(genuine, min(len(genuine), n - nf))
        rnd.shuffle(pick)
    return {"split": split, "n": len(pick),
            "rows": [{"transaction_id": r["transaction_id"], "prompt": r["prompt"],
                      "fraud_label": r["fraud_label"], "reasons": r["reasons"]} for r in pick]}


@app.get("/scenarios", dependencies=[Depends(auth)])
def scenarios():
    return json.loads((HERE / "scenarios.json").read_text())
