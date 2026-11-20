"""fmt-v1 prompt <-> tabular features.

The XGBoost baseline sees exactly the information the LLM sees: every feature is
parsed from the same fmt-v1 prompt text (current transaction, customer, context
and the serialized history). `build_prompt` mirrors sentinel.infer.build_prompt so
structured test transactions produce identical prompts for both models.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import pandas as pd

CATEGORIES = {
    "mcc": ["grocery", "quick_commerce", "food", "electronics", "gaming", "gift_cards",
            "healthcare", "travel", "education", "other"],
    "channel": ["UPI", "card", "wallet", "net_banking"],
    "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}
DIST = {"<5": 0, "5-50": 1, "0-50": 1, "50-200": 2, "200-500": 3, "500-1000": 4, "1000+": 5}
UNIT_DAYS = {"m": 1 / 1440, "h": 1 / 24, "d": 1.0}
NAN = float("nan")

NUMERIC = [
    "amt", "hour", "night", "age", "acct_age_days", "avg_90d", "txns_30d", "fails_24h",
    "new_device", "new_city", "geo_mismatch", "dist_km", "amt_ratio", "txns_1h", "amount_24h",
    "device_users_7d", "hist_n", "hist_mean_amt", "hist_max_amt", "hist_recency_days",
    "hist_n_24h", "hist_n_cities", "hist_city_seen", "hist_same_mcc", "amt_vs_hist_max",
]
FEATURES = NUMERIC + list(CATEGORIES)

# Feature -> reason code, used to turn SHAP contributions into SENTINEL reason codes.
REASON_MAP = {
    "R01_NEW_DEVICE": ["new_device", "device_users_7d"],
    "R02_GEO_MISMATCH": ["new_city", "geo_mismatch", "dist_km", "hist_city_seen", "hist_n_cities"],
    "R03_ODD_HOUR": ["hour", "night", "weekday"],
    "R04_FAILED_ATTEMPTS": ["fails_24h"],
    "R05_AMOUNT_SPIKE": ["amt", "amt_ratio", "amt_vs_hist_max", "avg_90d", "hist_mean_amt", "hist_max_amt"],
    "R06_NEW_ACCOUNT": ["acct_age_days", "age"],
    "R07_VELOCITY": ["txns_1h", "amount_24h", "hist_n_24h", "txns_30d", "hist_recency_days", "hist_n"],
    "R08_HIGH_RISK_MCC": ["mcc", "channel", "hist_same_mcc"],
}

_KV = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")
_HIST = re.compile(r"-(\d+)([mhd]):(\S+) ([\d.]+) (\S+) (.+)$")


def _num(v: Optional[str]) -> float:
    if v is None or v in ("unk", ""):
        return NAN
    if v == "10L+":
        return 1_000_000.0
    try:
        return float(v)
    except ValueError:
        return NAN


def _flag(v: Optional[str]) -> float:
    return {"yes": 1.0, "no": 0.0}.get(v, NAN)


def parse_prompt(prompt: str) -> Dict:
    kv: Dict[str, str] = {}
    hist: List[tuple] = []
    for line in prompt.split("\n"):
        if line.startswith("[HIST]"):
            body = line[7:].strip()
            if body and body not in ("none observed", "(none)"):
                for e in body.split(" | "):
                    m = _HIST.match(e.strip())
                    if m:
                        hist.append((int(m.group(1)) * UNIT_DAYS[m.group(2)], m.group(3),
                                     float(m.group(4)), m.group(5), m.group(6)))
        elif line.startswith("[") and "]" in line and not line.startswith("[TASK]"):
            kv.update(_KV.findall(line[line.index("]") + 1:].strip()))

    hour, weekday = NAN, None
    tm = re.match(r"(\d+)h\s*(\w+)?", kv.get("time", ""))
    if tm:
        hour, weekday = float(tm.group(1)), tm.group(2)
    amt = _num(kv.get("amt"))
    city, mcity = kv.get("city", "unknown"), kv.get("merchant_city", "unknown")
    h_amts = [h[2] for h in hist]
    h_cities = {h[4] for h in hist}
    f = {
        "amt": amt,
        "hour": hour,
        "night": 1.0 if not math.isnan(hour) and hour < 6 else 0.0,
        "age": _num(kv.get("age")),
        "acct_age_days": _num(kv.get("acct_age_days")),
        "avg_90d": _num(kv.get("avg_90d")),
        "txns_30d": _num(kv.get("txns_30d")),
        "fails_24h": _num(kv.get("fails_24h")),
        "new_device": _flag(kv.get("new_device")),
        "new_city": _flag(kv.get("new_city")),
        "geo_mismatch": NAN if "unknown" in (city, mcity) else float(city != mcity),
        "dist_km": float(DIST[kv["dist_km"]]) if kv.get("dist_km") in DIST else NAN,
        "amt_ratio": _num(kv.get("amt_ratio")),
        "txns_1h": _num(kv.get("txns_1h")),
        "amount_24h": _num(kv.get("amount_24h")),
        "device_users_7d": _num(kv.get("device_users_7d")),
        "hist_n": float(len(hist)),
        "hist_mean_amt": sum(h_amts) / len(h_amts) if h_amts else NAN,
        "hist_max_amt": max(h_amts) if h_amts else NAN,
        "hist_recency_days": min(h[0] for h in hist) if hist else NAN,
        "hist_n_24h": float(sum(1 for h in hist if h[0] <= 1)),
        "hist_n_cities": float(len(h_cities)),
        "hist_city_seen": (1.0 if city in h_cities else 0.0) if hist else NAN,
        "hist_same_mcc": float(sum(1 for h in hist if h[3] == kv.get("mcc"))),
        "amt_vs_hist_max": amt / max(h_amts) if h_amts and max(h_amts) > 0 else NAN,
        "mcc": kv.get("mcc"),
        "channel": kv.get("channel"),
        "weekday": weekday,
    }
    return f


def to_frame(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FEATURES)
    for c, cats in CATEGORIES.items():
        df[c] = pd.Categorical(df[c], categories=cats)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def build_prompt(t: Dict) -> str:
    """Structured transaction -> fmt-v1 prompt (same as sentinel.infer.build_prompt)."""
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
