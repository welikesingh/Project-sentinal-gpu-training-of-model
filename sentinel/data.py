"""Download and load the prebuilt SFT splits from the Supabase storage bucket.

Bucket layout (read-only):
  training/sft/{train,validation,holdout}/part-*.jsonl.gz

Each row has: transaction_id, prompt (fmt-v1 text), fraud_label (0/1),
reason_targets (list of reason codes), eligible_at_next_split_start, ...
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from sentinel import REASON_CODES

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "holdout")
DEFAULT_DATA_DIR = Path("data/sft")


def _supabase() -> tuple[str, str, str]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    bucket = os.getenv("SUPABASE_BUCKET", "sentinel-data")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set (see .env.example)")
    return url, key, bucket


def _headers(key: str) -> Dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def list_parts(split: str) -> List[str]:
    url, key, bucket = _supabase()
    prefix = f"training/sft/{split}/"
    resp = requests.post(
        f"{url}/storage/v1/object/list/{bucket}",
        headers={**_headers(key), "Content-Type": "application/json"},
        json={"prefix": prefix, "limit": 1000},
        timeout=60,
    )
    resp.raise_for_status()
    names = sorted(o["name"] for o in resp.json() if o["name"].endswith(".jsonl.gz"))
    return [prefix + n for n in names]


def download_split(split: str, data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Download all parts of a split into data_dir/split (skips files already present)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    url, key, bucket = _supabase()
    out = Path(data_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    for obj in list_parts(split):
        dest = out / Path(obj).name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            continue
        logger.info("Downloading %s", obj)
        with requests.get(
            f"{url}/storage/v1/object/authenticated/{bucket}/{obj}",
            headers=_headers(key), stream=True, timeout=300,
        ) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
    return out


def reason_vector(reasons: Iterable[str]) -> List[float]:
    s = set(reasons or [])
    return [1.0 if c in s else 0.0 for c in REASON_CODES]


def _normalize(row: dict) -> dict:
    return {
        "transaction_id": row.get("transaction_id") or row.get("example_id"),
        "prompt": row["prompt"],
        "fraud_label": int(row["fraud_label"]),
        "reasons": list(row.get("reason_targets") or []),
        "eligible": bool(row.get("eligible_at_next_split_start", True)),
    }


def load_split(
    split: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    download: bool = True,
    limit: Optional[int] = None,
) -> List[dict]:
    """Load a split as a list of normalized rows (downloading it first if needed)."""
    d = Path(data_dir) / split
    if download and not any(d.glob("*.jsonl.gz")):
        download_split(split, data_dir)
    files = sorted(d.glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl.gz files in {d}")
    rows: List[dict] = []
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(_normalize(json.loads(line)))
                    if limit and len(rows) >= limit:
                        return rows
    n_fraud = sum(r["fraud_label"] for r in rows)
    logger.info("Loaded %s: %d rows, %d fraud (%.2f%%)", split, len(rows), n_fraud,
                100 * n_fraud / max(len(rows), 1))
    return rows


def balance(rows: List[dict], genuine_per_fraud: int, seed: int = 42) -> List[dict]:
    """Keep every fraud row plus `genuine_per_fraud` genuine rows per fraud row.

    genuine_per_fraud <= 0 keeps everything (use --pos-weight to handle imbalance).
    """
    if genuine_per_fraud <= 0:
        return list(rows)
    rng = random.Random(seed)
    fraud = [r for r in rows if r["fraud_label"] == 1]
    genuine = [r for r in rows if r["fraud_label"] == 0]
    k = min(len(genuine), max(1, len(fraud)) * genuine_per_fraud)
    out = fraud + rng.sample(genuine, k)
    rng.shuffle(out)
    return out


def stratified_sample(rows: List[dict], n: int, seed: int = 42) -> List[dict]:
    """Subsample to n rows keeping all fraud rows (for quick evals)."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    fraud = [r for r in rows if r["fraud_label"] == 1]
    genuine = [r for r in rows if r["fraud_label"] == 0]
    return fraud + rng.sample(genuine, max(0, min(len(genuine), n - len(fraud))))


if __name__ == "__main__":  # python -m sentinel.data  -> download all splits
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for s in SPLITS:
        download_split(s)
        load_split(s, download=False)
