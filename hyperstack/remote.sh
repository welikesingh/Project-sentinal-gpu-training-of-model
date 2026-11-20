#!/usr/bin/env bash
# Drive training on a Hyperstack GPU VM from your machine over SSH.
#
#   export HS_HOST=<vm public ip>        # from the Hyperstack console
#   export HS_USER=ubuntu                # default
#   export HS_KEY=~/.ssh/id_ed25519      # key pair attached to the VM
#
#   bash hyperstack/remote.sh push       # copy code + .env to ~/sentinel on the VM
#   bash hyperstack/remote.sh setup      # venv + CUDA torch + deps + data download
#   bash hyperstack/remote.sh train [train args]   # runs in tmux, survives disconnects
#   bash hyperstack/remote.sh logs       # follow training log
#   bash hyperstack/remote.sh status     # nvidia-smi + is training still running
#   bash hyperstack/remote.sh pull       # copy output/ runs back to ./output
#   bash hyperstack/remote.sh test       # run pytest on the VM
#   bash hyperstack/remote.sh ssh        # interactive shell
#   bash hyperstack/remote.sh all [train args]     # push + setup + train
#
# Remember to hibernate/delete the VM in the console when done - it bills hourly.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

: "${HS_HOST:?set HS_HOST to the VM public IP}"
HS_USER="${HS_USER:-ubuntu}"
HS_KEY="${HS_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-sentinel}"
SSH_OPTS=(-i "$HS_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
T="$HS_USER@$HS_HOST"

rsh() { ssh "${SSH_OPTS[@]}" "$T" "$@"; }

push() {
  echo ">> pushing code to $T:~/$REMOTE_DIR"
  rsh "mkdir -p ~/$REMOTE_DIR"
  tar czf - --exclude=.venv --exclude=.git --exclude=output --exclude=data \
      --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.docx' \
      sentinel tests hyperstack requirements.txt pytest.ini .env.example $( [ -f .env ] && echo .env ) \
    | rsh "tar xzf - -C ~/$REMOTE_DIR"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  push)   push ;;
  setup)  rsh "cd ~/$REMOTE_DIR && bash hyperstack/setup_vm.sh" ;;
  train)
    args="--tag hyperstack $*"
    rsh "cd ~/$REMOTE_DIR && mkdir -p output && tmux kill-session -t sentinel 2>/dev/null; \
         tmux new-session -d -s sentinel 'source .venv/bin/activate && python -m sentinel.train $args 2>&1 | tee output/last_train.log'"
    echo ">> training started in tmux session 'sentinel'. Follow with: bash hyperstack/remote.sh logs" ;;
  logs)   rsh "tail -n 50 -f ~/$REMOTE_DIR/output/last_train.log" ;;
  status) rsh "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv; \
               tmux has-session -t sentinel 2>/dev/null && echo 'training: RUNNING' || echo 'training: not running'; \
               ls -1 ~/$REMOTE_DIR/output 2>/dev/null" ;;
  pull)
    mkdir -p output
    rsh "cd ~/$REMOTE_DIR && tar czf - --exclude=checkpoints output" | tar xzf - -C .
    echo ">> runs now in ./output:"; ls -1 output ;;
  test)   rsh "cd ~/$REMOTE_DIR && source .venv/bin/activate && python -m pytest -q" ;;
  ssh)    ssh "${SSH_OPTS[@]}" "$T" ;;
  all)    push; rsh "cd ~/$REMOTE_DIR && bash hyperstack/setup_vm.sh"; bash "$SELF" train "$@" ;;
  *)      sed -n '2,20p' "$SELF" ;;
esac
