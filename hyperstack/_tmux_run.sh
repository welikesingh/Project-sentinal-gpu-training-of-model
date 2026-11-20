#!/usr/bin/env bash
# Internal: what train.sh / serve.sh / xgb.sh run inside tmux (always under bash, whatever tmux's default shell is).
#   bash hyperstack/_tmux_run.sh train [sentinel.train args...]
#   bash hyperstack/_tmux_run.sh serve <run_dir> <port> <api_key> [log_file]
#   bash hyperstack/_tmux_run.sh xgb   <artifacts_dir> <port> <api_key> [log_file]
cd "$(dirname "$0")/.."
mode="$1"; shift
case "$mode" in
  train)
    .venv/bin/python -m sentinel.train --tag hyperstack "$@" 2>&1 | tee output/last_train.log
    echo "[train.sh] finished with exit code ${PIPESTATUS[0]}" | tee -a output/last_train.log ;;
  serve)
    run="$1"; port="$2"; export SENTINEL_API_KEY="$3"; log="${4:-output/serve.log}"
    while true; do
      .venv/bin/python -m sentinel.infer --run "$run" serve --host 0.0.0.0 --port "$port" 2>&1 | tee -a "$log"
      echo "[serve.sh] server exited - restarting in 5s" | tee -a "$log"
      sleep 5
    done ;;
  xgb)
    export XGB_ARTIFACTS="$(realpath "$1")"; port="$2"; export XGB_API_KEY="$3"; log="${4:-output/xgb.log}"
    export XGB_DEVICE="${XGB_DEVICE:-cuda}"
    cd xgboost_benchmark
    while true; do
      ../.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port "$port" 2>&1 | tee -a "../$log"
      echo "[xgb.sh] server exited - restarting in 5s" | tee -a "../$log"
      sleep 5
    done ;;
esac
