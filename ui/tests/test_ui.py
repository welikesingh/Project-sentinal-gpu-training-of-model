"""Render every tab headlessly with Streamlit's AppTest.

- offline: endpoints unreachable -> app must still render without exceptions
- online : set UI_TEST_LLM_URL (LoRA) / UI_TEST_QLORA_URL / UI_TEST_XGB_URL (+ _KEY) to run against live services
"""

import os
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(env):
    from streamlit.testing.v1 import AppTest

    for k, v in env.items():
        os.environ[k] = v
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_renders_offline():
    at = _run({"LLM_URL": "http://127.0.0.1:9", "QLORA_URL": "http://127.0.0.1:9", "XGB_URL": "http://127.0.0.1:9",
               "LLM_API_KEY": "", "XGB_API_KEY": ""})
    assert len(at.tabs) == 6
    assert any("offline" in m.value for m in at.markdown)
    assert any("Decision pending" in m.value for m in at.markdown)


@pytest.mark.skipif(not os.getenv("UI_TEST_XGB_URL"), reason="set UI_TEST_* env vars to test live services")
def test_live_scoring_and_batch():
    at = _run({"LLM_URL": os.getenv("UI_TEST_LLM_URL", ""), "LLM_API_KEY": os.getenv("UI_TEST_LLM_KEY", ""),
               "QLORA_URL": os.getenv("UI_TEST_QLORA_URL", "http://127.0.0.1:9"),
               "XGB_URL": os.environ["UI_TEST_XGB_URL"], "XGB_API_KEY": os.getenv("UI_TEST_XGB_KEY", "")})
    text = " ".join(m.value for m in at.markdown)
    assert "online" in text
    # Live Test: submit the scenario form
    at.button[[b.label for b in at.button].index("Score with all models")].click().run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "fraud probability" in text
    # Batch Test: fetch + score samples
    at.button[[b.label for b in at.button].index("Fetch & score")].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Decision agreement" in m.value for m in at.markdown)
