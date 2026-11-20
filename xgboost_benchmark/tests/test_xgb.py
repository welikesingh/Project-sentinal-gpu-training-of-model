import gzip
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import FEATURES, build_prompt, parse_prompt, to_frame  # noqa: E402

REAL = ("[TXN] id=TX_0000092051 amt=3200.0 mcc=electronics channel=UPI time=06h Sun\n"
        "[CUST] age=70 acct_age_days=999 avg_90d=459.89 txns_30d=3 fails_24h=0\n"
        "[CTX] new_device=yes city=Ahmedabad merchant_city=Hyderabad new_city=no dist_km=500-1000 "
        "amt_ratio=7.0 txns_1h=0 amount_24h=0.00 device_users_7d=0\n"
        "[HIST] -12d:UPI 120.0 grocery Ahmedabad | -2h:card 820.0 electronics Ahmedabad | -30m:wallet 360.0 food Pune\n"
        "[TASK] Assess fraud risk using only the observed context.")


def test_parse_real_prompt():
    f = parse_prompt(REAL)
    assert set(f) == set(FEATURES)
    assert f["amt"] == 3200 and f["hour"] == 6 and f["weekday"] == "Sun" and f["night"] == 0
    assert f["new_device"] == 1 and f["new_city"] == 0 and f["geo_mismatch"] == 1 and f["dist_km"] == 4
    assert f["hist_n"] == 3 and f["hist_max_amt"] == 820 and f["hist_n_24h"] == 2
    assert abs(f["hist_recency_days"] - 30 / 1440) < 1e-9 and f["hist_city_seen"] == 1
    assert f["hist_same_mcc"] == 1 and f["mcc"] == "electronics"


def test_unknowns_become_nan():
    p = REAL.replace("avg_90d=459.89", "avg_90d=unk").replace("new_device=yes", "new_device=unk") \
            .replace("amt=3200.0", "amt=10L+").replace("dist_km=500-1000", "dist_km=unk")
    f = parse_prompt(p)
    assert math.isnan(f["avg_90d"]) and math.isnan(f["new_device"]) and math.isnan(f["dist_km"])
    assert f["amt"] == 1_000_000


def test_build_prompt_round_trip():
    t = {"amount": 50000, "hour": 3, "weekday": "Sun", "new_device": True, "mcc": "gaming",
         "history": [{"days_ago": 2, "channel": "UPI", "amount": 120, "mcc": "grocery", "city": "Pune"}]}
    f = parse_prompt(build_prompt(t))
    assert f["amt"] == 50000 and f["hour"] == 3 and f["new_device"] == 1 and f["hist_n"] == 1
    df = to_frame([f])
    assert list(df.columns) == FEATURES and str(df["mcc"].dtype) == "category"


def _write_split(d: Path, n, seed):
    rng = random.Random(seed)
    d.mkdir(parents=True)
    with gzip.open(d / "part-00001.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i in range(n):
            fraud = int(i % 5 == 0)
            t = {"transaction_id": f"TX_{seed}_{i}", "amount": rng.uniform(20000, 90000) if fraud else rng.uniform(50, 3000),
                 "hour": 3 if fraud else 14, "new_device": bool(fraud), "fails_24h": 4 if fraud else 0,
                 "mcc": "electronics" if fraud else "grocery"}
            fh.write(json.dumps({"transaction_id": t["transaction_id"], "prompt": build_prompt(t),
                                 "fraud_label": fraud, "reason_targets": []}) + "\n")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("xgb")
    for s, n, seed in (("train", 400, 1), ("validation", 200, 2), ("holdout", 200, 3)):
        _write_split(root / "sft" / s, n, seed)
    import train
    train.main(["--data-dir", str(root / "sft"), "--out", str(root / "art"), "--rounds", "50", "--early-stop", "10"])
    return root


def test_train_writes_artifacts(trained):
    art = trained / "art"
    for f in ("model.json", "meta.json", "metrics_validation.json", "metrics_holdout.json"):
        assert (art / f).exists()
    m = json.loads((art / "metrics_holdout.json").read_text())["calibrated"]
    assert m["roc_auc"] > 0.95 and "pr" in m["curves"]


def test_server_endpoints(trained, monkeypatch):
    monkeypatch.setenv("XGB_ARTIFACTS", str(trained / "art"))
    monkeypatch.setenv("XGB_API_KEY", "k")
    monkeypatch.setenv("DATA_DIR", str(trained / "sft"))
    from fastapi.testclient import TestClient

    import server
    server._state.clear()
    c = TestClient(server.app)
    H = {"X-API-Key": "k"}
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/stats").status_code == 401
    st = c.get("/stats", headers=H).json()
    assert st["kind"] == "xgboost" and st["metrics"]["holdout"]["calibrated"]["rows"] == 200
    risky = {"transaction_id": "R", "amount": 80000, "hour": 3, "new_device": True, "fails_24h": 4, "mcc": "electronics"}
    safe = {"transaction_id": "S", "amount": 300, "hour": 14, "mcc": "grocery"}
    r = c.post("/score", json={"transactions": [risky, safe]}, headers=H).json()["results"]
    assert r[0]["fraud_probability"] > r[1]["fraud_probability"]
    assert r[0]["decision"] in ("REVIEW", "BLOCK") and r[1]["decision"] == "ALLOW"
    assert set(r[0]) == {"transaction_id", "fraud_probability", "decision", "confidence", "reason_codes", "explanation"}
    assert c.post("/score", json={"x": 1}, headers=H).status_code == 400
    s = c.get("/samples?n=20&fraud_ratio=0.5", headers=H).json()
    assert s["n"] == 20 and sum(x["fraud_label"] for x in s["rows"]) == 10
    sc = c.get("/scenarios", headers=H).json()
    assert len(sc) >= 5 and all("transaction" in x for x in sc)
    out = c.post("/score", json={"transactions": [x["transaction"] for x in sc]}, headers=H)
    assert out.status_code == 200
