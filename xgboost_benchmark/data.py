"""Load the same SFT splits the LLM uses (downloads from Supabase if missing)."""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)
SPLITS = ("train", "validation", "holdout")


def default_data_dir() -> Path:
    env = os.getenv("DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    shared = here.parent / "data" / "sft"          # repo layout: reuse the LLM's download
    return shared if shared.exists() else here / "data" / "sft"


def download_split(split: str, data_dir: Path) -> None:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    bucket = os.getenv("SUPABASE_BUCKET", "sentinel-data")
    if not url or not key:
        raise RuntimeError(f"No data in {data_dir}/{split} and SUPABASE_URL/SUPABASE_KEY not set")
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    prefix = f"training/sft/{split}/"
    r = requests.post(f"{url}/storage/v1/object/list/{bucket}", headers={**h, "Content-Type": "application/json"},
                      json={"prefix": prefix, "limit": 1000}, timeout=60)
    r.raise_for_status()
    out = data_dir / split
    out.mkdir(parents=True, exist_ok=True)
    for o in r.json():
        if not o["name"].endswith(".jsonl.gz"):
            continue
        dest = out / o["name"]
        if dest.exists() and dest.stat().st_size:
            continue
        logger.info("Downloading %s%s", prefix, o["name"])
        with requests.get(f"{url}/storage/v1/object/authenticated/{bucket}/{prefix}{o['name']}",
                          headers=h, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)


def load_split(split: str, data_dir: Optional[Path] = None, limit: Optional[int] = None) -> List[dict]:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    if not any((data_dir / split).glob("*.jsonl.gz")):
        download_split(split, data_dir)
    rows: List[dict] = []
    for fp in sorted((data_dir / split).glob("*.jsonl.gz")):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append({"transaction_id": r.get("transaction_id") or r.get("example_id"),
                                 "prompt": r["prompt"], "fraud_label": int(r["fraud_label"]),
                                 "reasons": list(r.get("reason_targets") or [])})
                    if limit and len(rows) >= limit:
                        return rows
    logger.info("Loaded %s: %d rows, %d fraud", split, len(rows), sum(r["fraud_label"] for r in rows))
    return rows


def stratified_sample(rows: List[dict], n: int, seed: int = 42) -> List[dict]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    fraud = [r for r in rows if r["fraud_label"] == 1]
    genuine = [r for r in rows if r["fraud_label"] == 0]
    return fraud + rng.sample(genuine, max(0, min(len(genuine), n - len(fraud))))
