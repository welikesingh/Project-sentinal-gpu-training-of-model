"""Score transactions with a trained run and return SENTINEL JSON decisions.

    # score N sample rows from a split (shows label next to prediction)
    python -m sentinel.infer --run output/<run> sample --split holdout --n 5
    # score a raw fmt-v1 prompt
    python -m sentinel.infer --run output/<run> prompt "[TXN] id=... [TASK] ..."
    # score a JSON / JSONL file of transactions ({"prompt": ...} or structured fields)
    python -m sentinel.infer --run output/<run> file txns.jsonl
    # HTTP:  POST /score  {"prompt": ...} | {"transactions": [...]}   GET /health
    python -m sentinel.infer --run output/<run> serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from sentinel import REASON_CODES

logger = logging.getLogger(__name__)

REASON_TEXT = {
    "R01_NEW_DEVICE": "new device",
    "R02_GEO_MISMATCH": "location differs from merchant/home city",
    "R03_ODD_HOUR": "unusual hour",
    "R04_FAILED_ATTEMPTS": "recent failed attempts",
    "R05_AMOUNT_SPIKE": "amount far above the customer's average",
    "R06_NEW_ACCOUNT": "new account",
    "R07_VELOCITY": "high transaction velocity",
    "R08_HIGH_RISK_MCC": "high-risk merchant category",
}
DECISIONS = ("ALLOW", "REVIEW", "BLOCK")


# ------------------------------------------------------------------ prompt
def build_prompt(t: Dict) -> str:
    """Serialise a structured transaction into the fmt-v1 prompt used in training.

    Keys (all optional except amount): transaction_id, amount, mcc, channel, hour, weekday,
    customer_age, account_age_days, avg_90d, txns_30d, fails_24h, new_device (bool),
    city, merchant_city, new_city (bool), dist_km (bucket str), amt_ratio, txns_1h,
    amount_24h, device_users_7d, history: [{days_ago, channel, amount, mcc, city}, ...]
    """
    yn = lambda v: "yes" if v else "no"  # noqa: E731
    num = lambda k, fmt: "unk" if t.get(k) is None else format(float(t[k]), fmt)  # noqa: E731
    hist = t.get("history") or []
    hist_s = " | ".join(
        f"-{h.get('days_ago', 0)}d:{h.get('channel', 'UPI')} {float(h.get('amount', 0)):.1f} "
        f"{h.get('mcc', 'unknown')} {h.get('city', 'unknown')}" for h in hist[:20]
    )
    return (
        f"[TXN] id={t.get('transaction_id', 'TX_NEW')} amt={float(t['amount']):.1f} "
        f"mcc={t.get('mcc', 'unknown')} channel={t.get('channel', 'UPI')} "
        f"time={int(t.get('hour', 12)):02d}h {t.get('weekday', 'Mon')}\n"
        f"[CUST] age={t.get('customer_age', 35)} acct_age_days={t.get('account_age_days', 365)} "
        f"avg_90d={num('avg_90d', '.2f')} txns_30d={t.get('txns_30d', 0)} "
        f"fails_24h={t.get('fails_24h', 0)}\n"
        f"[CTX] new_device={yn(t.get('new_device'))} city={t.get('city', 'unknown')} "
        f"merchant_city={t.get('merchant_city', 'unknown')} new_city={yn(t.get('new_city'))} "
        f"dist_km={t.get('dist_km', '0-50')} amt_ratio={num('amt_ratio', '.1f')} "
        f"txns_1h={t.get('txns_1h', 0)} amount_24h={float(t.get('amount_24h', 0)):.2f} "
        f"device_users_7d={t.get('device_users_7d', 0)}\n"
        f"[HIST] {hist_s if hist_s else 'none observed'}\n"
        "[TASK] Assess fraud risk using only the observed context."
    )


# ------------------------------------------------------------------ decision
def decide(prob: float, review_thr: float, block_thr: float) -> str:
    if prob >= block_thr:
        return "BLOCK"
    if prob >= review_thr:
        return "REVIEW"
    return "ALLOW"


def explain(decision: str, reasons: List[str]) -> str:
    if not reasons:
        return ("No strong risk signals observed." if decision == "ALLOW"
                else "Risk score elevated by the combination of transaction context and history.")
    parts = [REASON_TEXT.get(r, r) for r in reasons]
    lead = {"ALLOW": "Allowed despite", "REVIEW": "Sent to review:", "BLOCK": "Blocked:"}[decision]
    return f"{lead} {', '.join(parts)}."


def to_response(prob: float, reason_logits: np.ndarray, cal: Dict,
                transaction_id: Optional[str] = None) -> Dict:
    decision = decide(prob, cal["review_threshold"], cal["block_threshold"])
    reason_p = 1 / (1 + np.exp(-reason_logits))
    reasons = [c for c, p in zip(REASON_CODES, reason_p) if p >= 0.5]
    return {
        "transaction_id": transaction_id,
        "fraud_probability": round(float(prob), 6),
        "decision": decision,
        "confidence": round(float(max(prob, 1 - prob)), 4),
        "reason_codes": reasons,
        "explanation": explain(decision, reasons),
    }


RESPONSE_SCHEMA_KEYS = {"transaction_id", "fraud_probability", "decision", "confidence",
                        "reason_codes", "explanation"}


def validate_response(r: Dict) -> bool:
    return (set(r) == RESPONSE_SCHEMA_KEYS
            and 0.0 <= r["fraud_probability"] <= 1.0
            and r["decision"] in DECISIONS
            and 0.0 <= r["confidence"] <= 1.0
            and all(c in REASON_CODES for c in r["reason_codes"])
            and isinstance(r["explanation"], str) and r["explanation"] != "")


# ------------------------------------------------------------------ scorer
class SentinelScorer:
    def __init__(self, run_dir: str, review_threshold: Optional[float] = None,
                 block_threshold: Optional[float] = None):
        from sentinel.model import load_trained

        self.run = Path(run_dir)
        self.model, self.tok, self.meta = load_trained(run_dir)
        self.cal = json.loads((self.run / "calibration.json").read_text())
        if review_threshold is not None:
            self.cal["review_threshold"] = review_threshold
        if block_threshold is not None:
            self.cal["block_threshold"] = block_threshold

    def score(self, items: List[Union[str, Dict]], batch_size: int = 32) -> List[Dict]:
        from sentinel.evaluate import apply_platt
        from sentinel.model import predict_logits

        prompts, ids = [], []
        for it in items:
            if isinstance(it, str):
                prompts.append(it); ids.append(None)
            else:
                prompts.append(it["prompt"] if "prompt" in it else build_prompt(it))
                ids.append(it.get("transaction_id"))
        logits = predict_logits(self.model, self.tok, prompts, self.meta["max_length"], batch_size).numpy()
        probs = apply_platt(logits[:, 0], self.cal)
        return [to_response(p, lg[1:], self.cal, i) for p, lg, i in zip(probs, logits, ids)]

    def stats(self) -> Dict:
        """Everything the dashboard shows: run info, calibration, metrics, loss curve."""
        def read(name):
            p = self.run / name
            return json.loads(p.read_text()) if p.exists() else None

        log = read("train_log.json") or []
        keep = ("base_model", "method", "device", "host", "train_runtime_s", "train_loss",
                "train_rows", "max_length", "reason_codes")
        return {
            "kind": "llm",
            "model": f"{self.meta['base_model']} + {self.meta['method'].upper()}",
            "run": self.run.name,
            "info": {k: self.meta.get(k) for k in keep},
            "calibration": self.cal,
            "metrics": {"validation": read("metrics_validation.json"), "holdout": read("metrics_holdout.json")},
            "loss_curve": [{"step": e["step"], "loss": e["loss"]} for e in log if "loss" in e],
        }

    def latency_ms(self, prompt: str, n: int = 20) -> Dict[str, float]:
        self.score([prompt])  # warm-up
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.score([prompt])
            ts.append((time.perf_counter() - t0) * 1000)
        return {"p50_ms": float(np.percentile(ts, 50)), "p99_ms": float(np.percentile(ts, 99))}


# ------------------------------------------------------------------ server
TEST_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SENTINEL test</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:15px system-ui;max-width:760px;margin:24px auto;padding:0 16px}
textarea,input{width:100%;box-sizing:border-box;font:13px monospace}textarea{height:190px}
button{margin:8px 0;padding:8px 16px}pre{background:#f4f4f4;padding:12px;white-space:pre-wrap}</style></head>
<body><h2>SENTINEL fraud scoring</h2>
<label>API key <input id="k" type="password" placeholder="X-API-Key (if the server requires one)"></label>
<p>Transaction JSON (structured fields, or {"prompt": "..."}):</p>
<textarea id="t">{"transaction_id":"TX_DEMO","amount":75000,"mcc":"electronics","channel":"card","hour":3,
 "weekday":"Sun","customer_age":24,"account_age_days":6,"avg_90d":1800,"txns_30d":2,"fails_24h":4,
 "new_device":true,"city":"Pune","merchant_city":"Delhi","new_city":true,"dist_km":"1000+",
 "amt_ratio":41.7,"txns_1h":3,"amount_24h":12000,"device_users_7d":2}</textarea>
<button onclick="go()">Score</button><pre id="o"></pre>
<script>async function go(){const o=document.getElementById('o');o.textContent='...';
try{const r=await fetch('score',{method:'POST',headers:{'Content-Type':'application/json',
'X-API-Key':document.getElementById('k').value},body:document.getElementById('t').value});
o.textContent=r.status+' '+JSON.stringify(await r.json(),null,2)}catch(e){o.textContent=e}}</script>
</body></html>"""


def make_server(scorer: SentinelScorer, host: str, port: int, api_key: Optional[str] = None):
    """Build (not start) the HTTP server.

    GET  /        browser test page
    GET  /health  status + thresholds (no auth)
    POST /score   {"prompt": ...} | {structured fields} | {"transactions": [...]}
    If api_key is set, POST /score requires header  X-API-Key: <key>.
    """
    import hmac
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    lock = threading.Lock()
    max_body = 1 << 20

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, TEST_PAGE, "text/html; charset=utf-8")
            elif self.path == "/health":
                self._send(200, {"status": "ok", "run": scorer.run.name, "auth_required": bool(api_key),
                                 "device": str(next(scorer.model.parameters()).device),
                                 "thresholds": {k: scorer.cal[k] for k in ("review_threshold", "block_threshold")}})
            elif self.path == "/stats":
                if api_key and not hmac.compare_digest(self.headers.get("X-API-Key", ""), api_key):
                    return self._send(401, {"error": "missing or invalid X-API-Key"})
                self._send(200, scorer.stats())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            if api_key and not hmac.compare_digest(self.headers.get("X-API-Key", ""), api_key):
                return self._send(401, {"error": "missing or invalid X-API-Key"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > max_body:
                    return self._send(413, {"error": "body too large"})
                body = json.loads(self.rfile.read(n) or b"{}")
                items = body.get("transactions") or [body]
                if len(items) > 256:
                    return self._send(413, {"error": "max 256 transactions per request"})
                t0 = time.perf_counter()
                with lock:
                    res = scorer.score(items)
                ms = (time.perf_counter() - t0) * 1000
                self._send(200, {"results": res, "latency_ms": round(ms, 2)})
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                self._send(400, {"error": f"bad request: {e}"})

        def log_message(self, format, *args):  # noqa: A002
            logger.info("%s %s", self.address_string(), format % args)

    return ThreadingHTTPServer((host, port), H)


def serve(scorer: SentinelScorer, host: str, port: int, api_key: Optional[str] = None):
    if host not in ("127.0.0.1", "localhost") and not api_key:
        logger.warning("Serving on %s WITHOUT an API key - anyone who can reach this port can call it. "
                       "Set SENTINEL_API_KEY or --api-key.", host)
    srv = make_server(scorer, host, port, api_key)
    logger.info("Serving on http://%s:%d  (GET / test page, GET /health, POST /score)", host, port)
    srv.serve_forever()


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="SENTINEL inference")
    p.add_argument("--run", required=True)
    p.add_argument("--review-threshold", type=float)
    p.add_argument("--block-threshold", type=float)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--split", default="holdout"); s.add_argument("--n", type=int, default=5)
    s.add_argument("--fraud-only", action="store_true"); s.add_argument("--data-dir", default="data/sft")
    s = sub.add_parser("prompt"); s.add_argument("text")
    s = sub.add_parser("file"); s.add_argument("path")
    s = sub.add_parser("latency"); s.add_argument("--n", type=int, default=50); s.add_argument("--data-dir", default="data/sft")
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--api-key", default=os.getenv("SENTINEL_API_KEY"), help="Require X-API-Key (default: $SENTINEL_API_KEY)")
    a = p.parse_args(argv)

    scorer = SentinelScorer(a.run, a.review_threshold, a.block_threshold)
    if a.cmd == "serve":
        return serve(scorer, a.host, a.port, a.api_key)
    if a.cmd == "prompt":
        print(json.dumps(scorer.score([a.text])[0], indent=2))
    elif a.cmd == "file":
        text = Path(a.path).read_text(encoding="utf-8").strip()
        items = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        for r in scorer.score(items):
            print(json.dumps(r))
    elif a.cmd == "sample":
        from sentinel.data import load_split

        rows = load_split(a.split, Path(a.data_dir))
        if a.fraud_only:
            rows = [r for r in rows if r["fraud_label"] == 1]
        rows = rows[: a.n]
        for row, res in zip(rows, scorer.score([{"prompt": r["prompt"], "transaction_id": r["transaction_id"]} for r in rows])):
            print(json.dumps({**res, "actual_fraud_label": row["fraud_label"]}))
    elif a.cmd == "latency":
        from sentinel.data import load_split

        prompt = load_split("holdout", Path(a.data_dir), limit=1)[0]["prompt"]
        print(json.dumps(scorer.latency_ms(prompt, a.n), indent=2))


if __name__ == "__main__":
    main()
