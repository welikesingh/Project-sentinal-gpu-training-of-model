#!/usr/bin/env bash
# Run ON the VM. Serves a trained LLM run on the GPU at 0.0.0.0:<port> in tmux (auto-restarts).
#   bash hyperstack/serve.sh                       # newest FINISHED run (any method), port 8000
#   bash hyperstack/serve.sh lora  8000            # newest finished LoRA run  on :8000
#   bash hyperstack/serve.sh qlora 8002            # newest finished QLoRA run on :8002
#   bash hyperstack/serve.sh output/<run> 8000     # a specific run
# tmux session: "serve" for port 8000, "serve-<port>" otherwise. Log: output/serve.log / output/serve-<port>.log
# Requires SENTINEL_API_KEY (in .env or the environment); one is generated into .env if missing.
# Testers open  http://<VM_IP>:<port>/   (needs a firewall rule for inbound TCP <port>).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run: bash hyperstack/setup_vm.sh first"; exit 1; }

# newest finished run; optional $1 = lora|qlora filters on the method in sentinel_config.json
latest_finished() {
  for d in $(ls -1dt output/*/ 2>/dev/null); do
    [ -f "${d}calibration.json" ] && [ -f "${d}adapter/adapter_config.json" ] || continue
    if [ -n "${1:-}" ]; then
      grep -q "\"method\": *\"$1\"" "${d}sentinel_config.json" 2>/dev/null || continue
    fi
    echo "${d%/}"; return
  done
}
ARG="${1:-}"
case "$ARG" in
  lora|qlora) RUN="$(latest_finished "$ARG" || true)"; LABEL="$ARG" ;;
  "")         RUN="$(latest_finished || true)";        LABEL="llm" ;;
  *)          RUN="$ARG";                               LABEL="llm" ;;
esac
RUN="${RUN%/}"
PORT="${2:-8000}"
[ -n "$RUN" ] && [ -f "$RUN/calibration.json" ] || {
  echo "No finished ${LABEL} run found in output/. Train first: bash hyperstack/train.sh${ARG:+ --method $ARG}"; exit 1; }

touch .env
if ! grep -q '^SENTINEL_API_KEY=' .env && [ -z "${SENTINEL_API_KEY:-}" ]; then
  echo "SENTINEL_API_KEY=$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(24))')" >> .env
  echo "Generated SENTINEL_API_KEY in .env"
fi
KEY="${SENTINEL_API_KEY:-$(grep '^SENTINEL_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '\r')}"

if [ "$PORT" = "8000" ]; then SESSION=serve; LOG=output/serve.log; else SESSION="serve-$PORT"; LOG="output/serve-$PORT.log"; fi
tmux kill-session -t "$SESSION" 2>/dev/null || true
mkdir -p output
: > "$LOG"
tmux new-session -d -s "$SESSION" "bash hyperstack/_tmux_run.sh serve '$RUN' $PORT '$KEY' '$LOG'"

echo -n "Starting $LABEL server on :$PORT (loading model on GPU)"
for _ in $(seq 1 120); do
  if curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  echo -n "."; sleep 2
done
echo
if ! curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "Server did not come up. Last log lines:"; tail -n 30 "$LOG"; exit 1
fi

IP="${PUBLIC_IP:-<VM_PUBLIC_IP>}"
echo "Serving $RUN  ->  $(curl -s http://localhost:$PORT/health)"
echo "  Test page : http://$IP:$PORT/"
echo "  Health    : curl http://$IP:$PORT/health"
echo "  API key   : $KEY   (share only with testers; header X-API-Key)"
echo "  Auto-restarts if it crashes. Stop with: tmux kill-session -t $SESSION"
