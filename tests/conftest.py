import gzip
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel import REASON_CODES  # noqa: E402
from sentinel.infer import build_prompt  # noqa: E402


def _row(i: int, fraud: int, rng: random.Random) -> dict:
    # Fraud rows get observable risk signals so the tiny model has something to learn.
    t = {
        "transaction_id": f"TX_{i:010d}",
        "amount": rng.uniform(5000, 90000) if fraud else rng.uniform(50, 3000),
        "mcc": "electronics" if fraud else "grocery",
        "hour": 3 if fraud else 14,
        "new_device": bool(fraud),
        "fails_24h": 4 if fraud else 0,
        "city": "Pune", "merchant_city": "Delhi" if fraud else "Pune",
    }
    reasons = ["R01_NEW_DEVICE", "R03_ODD_HOUR", "R04_FAILED_ATTEMPTS"] if fraud else []
    return {"example_id": t["transaction_id"], "transaction_id": t["transaction_id"],
            "prompt": build_prompt(t), "fraud_label": fraud, "reason_targets": reasons,
            "eligible_at_next_split_start": True}


@pytest.fixture(scope="session")
def fake_data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sft")
    rng = random.Random(0)
    i = 0
    for split, n in (("train", 120), ("validation", 60), ("holdout", 60)):
        d = root / split
        d.mkdir()
        with gzip.open(d / "part-00001.jsonl.gz", "wt", encoding="utf-8") as f:
            for k in range(n):
                f.write(json.dumps(_row(i, int(k % 6 == 0), rng)) + "\n")
                i += 1
    return root


assert len(REASON_CODES) == 8
