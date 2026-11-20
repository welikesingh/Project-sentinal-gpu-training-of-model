#!/usr/bin/env bash
# Run ON the VM. Starts training in tmux session "train" (survives SSH disconnects).
#   bash hyperstack/train.sh                        # LoRA defaults (~10-20 min on one GPU)
#   bash hyperstack/train.sh --method qlora --epochs 3
#   bash hyperstack/train.sh --max-steps 30 --max-eval-rows 2000   # 2-minute demo run
#   FORCE=1 bash hyperstack/train.sh ...            # replace a training run that is still going
# Watch:  tail -f output/last_train.log      or  tmux attach -t train  (detach: Ctrl-B then D)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run: bash hyperstack/setup_vm.sh first"; exit 1; }
if tmux has-session -t train 2>/dev/null; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "A training run is already in progress (tmux session 'train')."
    echo "  watch it : tail -f output/last_train.log"
    echo "  replace  : FORCE=1 bash hyperstack/train.sh $*"
    exit 1
  fi
  tmux kill-session -t train
fi
mkdir -p output
: > output/last_train.log
ARGS=""; [ $# -gt 0 ] && ARGS=$(printf ' %q' "$@")
tmux new-session -d -s train "bash hyperstack/_tmux_run.sh train$ARGS"
sleep 8
if ! tmux has-session -t train 2>/dev/null || grep -q -E "Traceback|finished with exit code [1-9]" output/last_train.log 2>/dev/null; then
  echo "Training failed to start:"; tail -n 30 output/last_train.log; exit 1
fi
echo "Training started (tmux session 'train')."
echo "  follow : tail -f output/last_train.log      (done when it prints 'Artifacts in output/<run>')"
