# SENTINEL dashboard (Streamlit)
#   docker build -t sentinel-ui ui
#   docker run --rm -p 8501:8501 -e LLM_URL=http://<VM_IP>:8000 -e LLM_API_KEY=... \
#       -e XGB_URL=https://<id>.ngrok-free.app -e XGB_API_KEY=... sentinel-ui
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=300
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 --timeout 300 -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
