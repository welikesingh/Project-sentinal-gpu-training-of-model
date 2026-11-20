#!/usr/bin/env bash
# Keep the SENTINEL dashboard always running (Docker, --restart unless-stopped).
#   bash ui/run_service.sh                                   # start (build if needed)
#   LLM_URL=http://<VM_IP>:8000 LLM_API_KEY=<key> bash ui/run_service.sh
#   bash ui/run_service.sh --rebuild | --stop
# Settings persist in ui/.env (git-ignored); XGBoost key defaults to xgboost_benchmark/.env.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; NAME=sentinel-ui; PORT="${PORT:-8501}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
[ "${1:-}" = "--stop" ] && { echo "Stopped $NAME"; exit 0; }
get() { grep -s "^$1=" "$HERE/.env" | tail -1 | cut -d= -f2- | tr -d '\r'; }
L_URL="${LLM_URL:-$(get LLM_URL)}"; L_URL="${L_URL:-http://host.docker.internal:8000}"
L_KEY="${LLM_API_KEY:-$(get LLM_API_KEY)}"
X_URL="${XGB_URL:-$(get XGB_URL)}"; X_URL="${X_URL:-http://host.docker.internal:8001}"
X_KEY="${XGB_API_KEY:-$(get XGB_API_KEY)}"
[ -n "$X_KEY" ] || X_KEY="$(grep -s '^XGB_API_KEY=' "$ROOT/xgboost_benchmark/.env" | cut -d= -f2- | tr -d '\r')"
printf 'LLM_URL=%s\nLLM_API_KEY=%s\nXGB_URL=%s\nXGB_API_KEY=%s\n' "$L_URL" "$L_KEY" "$X_URL" "$X_KEY" > "$HERE/.env"
if [ "${1:-}" = "--rebuild" ] || [ -z "$(docker images -q $NAME)" ]; then docker build -t "$NAME" "$HERE"; fi
docker run -d --name "$NAME" --restart unless-stopped -p "$PORT:8501" --env-file "$HERE/.env" \
  --add-host host.docker.internal:host-gateway "$NAME" >/dev/null
for _ in $(seq 1 40); do curl -fs "localhost:$PORT/_stcore/health" >/dev/null && break; sleep 1; done
echo "Dashboard : http://localhost:$PORT  (restarts automatically)"
echo "LLM       : $L_URL"
echo "XGBoost   : $X_URL"
echo "Share     : ngrok http $PORT"
