#!/usr/bin/env bash
# Keep the XGBoost benchmark API always running (Docker, --restart unless-stopped).
#   bash xgboost_benchmark/run_service.sh            # build if needed + start
#   bash xgboost_benchmark/run_service.sh --rebuild  # after retraining
#   bash xgboost_benchmark/run_service.sh --stop
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; NAME=sentinel-xgb; PORT="${PORT:-8001}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
[ "${1:-}" = "--stop" ] && { echo "Stopped $NAME"; exit 0; }
[ -f "$HERE/.env" ] || echo "XGB_API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))' 2>/dev/null || python -c 'import secrets;print(secrets.token_urlsafe(24))')" > "$HERE/.env"
if [ "${1:-}" = "--rebuild" ] || [ -z "$(docker images -q $NAME)" ]; then docker build -t "$NAME" "$HERE"; fi
docker run -d --name "$NAME" --restart unless-stopped -p "$PORT:8001" --env-file "$HERE/.env" \
  -v "$ROOT/data/sft:/data/sft" "$NAME" >/dev/null
for _ in $(seq 1 30); do curl -fs "localhost:$PORT/health" >/dev/null && break; sleep 1; done
echo "XGBoost API: http://localhost:$PORT  $(curl -s localhost:$PORT/health)"
echo "API key   : $(grep '^XGB_API_KEY=' "$HERE/.env" | cut -d= -f2-)"
echo "Expose    : ngrok http $PORT"
