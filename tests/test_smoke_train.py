"""End-to-end: train a tiny Qwen2 for a few steps -> calibrate -> evaluate -> infer.

Uses a tiny random Qwen2 checkpoint, so it runs in seconds on CPU or GPU.
Set SENTINEL_SMOKE_MODEL to override, or `-m "not smoke"` to skip.
"""

import json
import os

import pytest

TINY = os.getenv("SENTINEL_SMOKE_MODEL", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")


@pytest.mark.smoke
@pytest.mark.parametrize("method", ["lora", "qlora"])
def test_train_eval_infer(fake_data_dir, tmp_path, method):
    pytest.importorskip("transformers")
    if method == "qlora":
        import torch
        if not torch.cuda.is_available():
            pytest.skip("QLoRA needs CUDA")
    from sentinel.evaluate import evaluate_run
    from sentinel.infer import SentinelScorer, validate_response
    from sentinel.train import parse_args, run

    run_dir = run(parse_args([
        "--base-model", TINY, "--data-dir", str(fake_data_dir), "--output-root", str(tmp_path),
        "--run-name", f"smoke-{method}", "--method", method, "--max-steps", "4", "--batch-size", "8", "--genuine-per-fraud", "2",
        "--max-length", "256", "--eval-batch-size", "16", "--lr", "1e-3",
    ]))
    for f in ("adapter/adapter_config.json", "sentinel_config.json", "calibration.json",
              "metrics_validation.json", "metrics_holdout.json"):
        assert (run_dir / f).exists(), f

    cal = json.loads((run_dir / "calibration.json").read_text())
    assert cal["block_threshold"] >= cal["review_threshold"]

    res = evaluate_run(str(run_dir), "holdout", data_dir=str(fake_data_dir))
    assert res["calibrated"]["rows"] == 60

    scorer = SentinelScorer(str(run_dir))
    out = scorer.score([{"amount": 75000, "hour": 3, "new_device": True, "transaction_id": "TX_A"},
                        "[TXN] id=TX_B amt=120.0 mcc=grocery channel=UPI time=14h Mon\n[TASK] Assess."])
    assert len(out) == 2 and all(validate_response(r) for r in out)
    assert out[0]["transaction_id"] == "TX_A"

    # ---- HTTP server: auth, scoring, test page, health
    import threading

    import requests

    from sentinel.infer import make_server

    srv = make_server(scorer, "127.0.0.1", 0, api_key="secret")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        txn = {"transaction_id": "TX_HTTP", "amount": 75000, "hour": 3, "new_device": True}
        assert requests.post(f"{base}/score", json=txn, timeout=30).status_code == 401
        r = requests.post(f"{base}/score", json=txn, headers={"X-API-Key": "secret"}, timeout=30)
        assert r.status_code == 200 and validate_response(r.json()["results"][0])
        r = requests.post(f"{base}/score", json={"transactions": [txn, txn]},
                          headers={"X-API-Key": "secret"}, timeout=30)
        assert len(r.json()["results"]) == 2
        assert requests.post(f"{base}/score", json={"x": 1}, headers={"X-API-Key": "secret"},
                             timeout=30).status_code == 400
        assert "SENTINEL" in requests.get(base, timeout=30).text
        assert requests.get(f"{base}/health", timeout=30).json()["auth_required"] is True
    finally:
        srv.shutdown()
