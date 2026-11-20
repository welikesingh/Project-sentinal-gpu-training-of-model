from sentinel import REASON_CODES
from sentinel.data import balance, load_split, reason_vector, stratified_sample


def test_load_split_normalizes(fake_data_dir):
    rows = load_split("train", fake_data_dir, download=False)
    assert len(rows) == 120
    r = rows[0]
    assert set(r) == {"transaction_id", "prompt", "fraud_label", "reasons", "eligible"}
    assert r["prompt"].startswith("[TXN]") and r["fraud_label"] in (0, 1)


def test_reason_vector_order():
    v = reason_vector(["R03_ODD_HOUR", "UNKNOWN"])
    assert len(v) == len(REASON_CODES) and v[2] == 1.0 and sum(v) == 1.0


def test_balance_keeps_all_fraud(fake_data_dir):
    rows = load_split("train", fake_data_dir, download=False)
    n_fraud = sum(r["fraud_label"] for r in rows)
    out = balance(rows, genuine_per_fraud=2)
    assert sum(r["fraud_label"] for r in out) == n_fraud
    assert len(out) == n_fraud * 3
    assert len(balance(rows, 0)) == len(rows)


def test_stratified_sample(fake_data_dir):
    rows = load_split("validation", fake_data_dir, download=False)
    n_fraud = sum(r["fraud_label"] for r in rows)
    out = stratified_sample(rows, 20)
    assert len(out) == 20 and sum(r["fraud_label"] for r in out) == n_fraud
