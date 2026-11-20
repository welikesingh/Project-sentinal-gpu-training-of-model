import numpy as np

from sentinel.infer import build_prompt, decide, explain, to_response, validate_response

CAL = {"method": "platt", "a": 1.0, "b": 0.0, "review_threshold": 0.3, "block_threshold": 0.8}


def test_decide_bands():
    assert decide(0.1, 0.3, 0.8) == "ALLOW"
    assert decide(0.3, 0.3, 0.8) == "REVIEW"
    assert decide(0.95, 0.3, 0.8) == "BLOCK"


def test_response_is_valid_json_schema():
    logits = np.array([5.0, -5, 5, -5, -5, -5, -5, -5])  # R01 + R03
    r = to_response(0.9, logits, CAL, "TX_1")
    assert validate_response(r)
    assert r["decision"] == "BLOCK"
    assert r["reason_codes"] == ["R01_NEW_DEVICE", "R03_ODD_HOUR"]
    assert "new device" in r["explanation"]


def test_explain_without_reasons():
    assert explain("ALLOW", []) == "No strong risk signals observed."


def test_build_prompt_matches_fmt_v1_sections():
    p = build_prompt({"amount": 50000, "hour": 3, "new_device": True,
                      "history": [{"days_ago": 2, "channel": "UPI", "amount": 120, "mcc": "grocery", "city": "Pune"}]})
    for tag in ("[TXN]", "[CUST]", "[CTX]", "[HIST]", "[TASK]"):
        assert tag in p
    assert "amt=50000.0" in p and "time=03h" in p and "new_device=yes" in p
    assert "-2d:UPI 120.0 grocery Pune" in p
