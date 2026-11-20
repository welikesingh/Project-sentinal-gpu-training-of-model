#!/usr/bin/env bash
# Run ON the VM. XGBoost benchmark on the GPU: train it, then serve it on :8001 in tmux session "xgb".
#   bash hyperstack/xgb.sh train          # GPU training (~1 min), writes output/xgb/
#   bash hyperstack/xgb.sh serve          # API on 0.0.0.0:8001 (auto-restarts); uses output/xgb/ if trained,
#                                         # else the model committed in xgboost_benchmark/artifacts/
#   bash hyperstack/xgb.sh serve 8001
#   bash hyperstack/xgb.sh stop
# Uses the same API key as the LLMs (SENTINEL_API_KEY in .env), so the dashboard needs only one key.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run: bash hyperstack/setup_vm.sh first"; exit 1; }
.venv/bin/python -c "import xgboost" 2>/dev/null || .venv/bin/python -m pip install -q -r xgboost_benchmark/requirements.txt
CMD="${1:-serve}"
PORT="${2:-8001}"
mkdir -p output

case "$CMD" in
  train)
    echo "Training XGBoost on the GPU (same splits + thresholds protocol as the LLM) ..."
    .venv/bin/python xgboost_benchmark/train.py --device cuda --out output/xgb 2>&1 | tee output/xgb_train.log
    echo "Done. Artifacts in output/xgb   Next: bash hyperstack/xgb.sh serve" ;;

  serve)
    ART=output/xgb; [ -f "$ART/model.json" ] || ART=xgboost_benchmark/artifacts
    touch .env
    if ! grep -q '^SENTINEL_API_KEY=' .env && [ -z "${SENTINEL_API_KEY:-}" ]; then
      echo "SENTINEL_API_KEY=$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(24))')" >> .env
      echo "Generated SENTINEL_API_KEY in .env"
    fi
    KEY="${SENTINEL_API_KEY:-$(grep '^SENTINEL_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '\r')}"
    tmux kill-session -t xgb 2>/dev/null || true
    : > output/xgb.log
    tmux new-session -d -s xgb "bash hyperstack/_tmux_run.sh xgb '$ART' $PORT '$KEY' output/xgb.log"
    echo -n "Starting XGBoost server on :$PORT"
    for _ in $(seq 1 60); do curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1 && break; echo -n "."; sleep 1; done
    echo
    curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "Server did not come up:"; tail -n 30 output/xgb.log; exit 1; }
    IP="${PUBLIC_IP:-<VM_PUBLIC_IP>}"
    echo "Serving XGBoost ($ART)  ->  $(curl -s http://localhost:$PORT/health)"
    echo "  Health  : curl http://$IP:$PORT/health"
    echo "  API key : $KEY"
    echo "  Stop    : bash hyperstack/xgb.sh stop" ;;

  stop)
    tmux kill-session -t xgb 2>/dev/null && echo "Stopped xgb" || echo "xgb was not running" ;;

  *) echo "usage: bash hyperstack/xgb.sh train|serve|stop [port]"; exit 1 ;;
esac
