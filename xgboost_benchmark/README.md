# XGBoost benchmark (runs on your machine)

The gradient-boosting baseline the LLM must beat (CRD OBJ-04: PR-AUC uplift ≥ +0.03).
It uses the **same splits, the same fmt-v1 prompts, and the same protocol** as the LLM: every feature is
parsed from the prompt text (`features.py`), early-stopping + Platt calibration + thresholds come from
validation (REVIEW: recall ≥ 0.85, BLOCK: precision ≥ 0.70), and metrics are frozen-threshold on holdout.

```
features.py   prompt -> 25 numeric + 3 categorical features; build_prompt() mirrors the LLM's
train.py      trains, calibrates, writes artifacts/ (model.json, meta.json, metrics_*.json)
server.py     FastAPI: /health /stats /score /samples /scenarios (same /score contract as the LLM)
scenarios.json  curated test transactions served to the dashboard
```

## Run (Python)
```bash
cd xgboost_benchmark
pip install -r requirements.txt
python train.py                     # ~1–2 min on CPU (--device cuda on a GPU)
python -m pytest -q tests           # parser, training, every endpoint
export XGB_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(24))")   # PowerShell: $env:XGB_API_KEY="..."
uvicorn server:app --host 0.0.0.0 --port 8001
```
Data: reuses `../data/sft` when present. Otherwise it downloads from Supabase (`SUPABASE_URL` / `SUPABASE_KEY` in `.env`).

## Always-on service (recommended)
The trained model (`artifacts/`) is committed, so the service is ready without retraining.
This starts it as a Docker container with `--restart unless-stopped`, so it comes back by itself
after a reboot or Docker Desktop restart:
```powershell
powershell -ExecutionPolicy Bypass -File xgboost_benchmark\run_service.ps1            # Windows
bash xgboost_benchmark/run_service.sh                                                  # Linux/macOS/Git Bash
```
It prints the URL and the API key (kept in `xgboost_benchmark/.env`, git-ignored). After retraining
(`python train.py`), run it again with `-Rebuild` / `--rebuild`. Stop it with `-Stop` / `--stop`.
For Docker Desktop to come back after a reboot, enable *Settings → General → Start Docker Desktop when you sign in*.

## Run (Docker, manual)
```bash
docker build -t sentinel-xgb xgboost_benchmark
docker run --rm -p 8001:8001 -e XGB_API_KEY=$XGB_API_KEY -v "$PWD/data/sft:/data/sft" sentinel-xgb
```

## Expose with ngrok
```bash
ngrok http 8001          # copy the https://….ngrok-free.app URL into the dashboard (XGBoost URL)
```
Always set `XGB_API_KEY` before exposing the API; anyone who has the ngrok URL can reach it.
