"""HTTP clients for the two scoring services (same /score contract)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class Endpoint:
    name: str
    url: str
    key: str = ""
    timeout: float = 60.0

    @property
    def base(self) -> str:
        return (self.url or "").strip().rstrip("/")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
        if self.key:
            h["X-API-Key"] = self.key
        return h

    def _call(self, method: str, path: str, **kw) -> Tuple[bool, Any, float]:
        if not self.base:
            return False, "URL not set", 0.0
        t0 = time.perf_counter()
        try:
            r = requests.request(method, self.base + path, headers=self._headers(),
                                 timeout=kw.pop("timeout", self.timeout), **kw)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                try:
                    detail = r.json()
                except ValueError:
                    detail = r.text[:200]
                return False, f"HTTP {r.status_code}: {detail}", ms
            return True, r.json(), ms
        except requests.RequestException as e:
            return False, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000

    def health(self):
        return self._call("GET", "/health", timeout=8)

    def stats(self):
        return self._call("GET", "/stats", timeout=30)

    def scenarios(self):
        return self._call("GET", "/scenarios", timeout=15)

    def samples(self, split: str, n: int, fraud_ratio: float, seed: int, fraud_only: bool = False):
        return self._call("GET", "/samples", timeout=60, params={
            "split": split, "n": n, "fraud_ratio": fraud_ratio, "seed": seed, "fraud_only": str(fraud_only).lower()})

    def score(self, items: List[Dict], chunk: int = 64) -> Tuple[bool, Any, float]:
        """Score items in chunks; returns (ok, results | error, total round-trip ms)."""
        out: List[Dict] = []
        total = 0.0
        for i in range(0, len(items), chunk):
            ok, data, ms = self._call("POST", "/score", json={"transactions": items[i:i + chunk]}, timeout=300)
            total += ms
            if not ok:
                return False, data, total
            out.extend(data["results"])
        return True, out, total

    def score_one(self, item: Dict) -> Tuple[bool, Optional[Dict], float, Optional[float]]:
        ok, data, ms = self._call("POST", "/score", json=item)
        if not ok:
            return False, data, ms, None
        return True, data["results"][0], ms, data.get("latency_ms")
