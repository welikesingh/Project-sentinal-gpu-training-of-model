# SENTINEL dashboard (Streamlit, iOS style)

Tabs: **Overview** (holdout KPIs and CRD acceptance targets) · **LLM** (run info, loss curve, thresholds,
reason-code F1) · **Benchmark** (LLM vs XGBoost: PR/ROC curves, operating points, uplift, feature importance)
· **Live Test** (scenario presets or a custom form or a raw prompt, scored on both models side by side with latency)
· **Batch Test** (labelled samples from the XGBoost service or an uploaded JSONL/CSV, scored on both models with metrics and a CSV export).

It talks only HTTP to the two services, so it can run anywhere:

| Setting | Typical value |
|---|---|
| `LLM_URL` / `LLM_API_KEY` | `http://<HYPERSTACK_VM_IP>:8000` / key printed by `hyperstack/serve.sh` |
| `XGB_URL` / `XGB_API_KEY` | `https://<id>.ngrok-free.app` (or `http://localhost:8001`) / your `XGB_API_KEY` |

The sidebar can override all four at runtime.

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py                       # http://localhost:8501
python -m pytest -q ui/tests                  # headless render test
# Docker
docker build -t sentinel-ui ui
docker run --rm -p 8501:8501 -e LLM_URL=... -e LLM_API_KEY=... -e XGB_URL=... -e XGB_API_KEY=... sentinel-ui
```
