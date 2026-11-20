# SENTINEL XGBoost benchmark API
#   docker build -t sentinel-xgb xgboost_benchmark
#   docker run --rm -p 8001:8001 -e XGB_API_KEY=... -v "$PWD/data/sft:/data/sft" sentinel-xgb
# artifacts/ (model.json, meta.json, metrics_*.json) is baked in; /data/sft is only needed for /samples
# (or pass SUPABASE_URL / SUPABASE_KEY and it downloads on first /samples call).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=300 DATA_DIR=/data/sft
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 --timeout 300 -r requirements.txt
COPY . .
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8001/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
