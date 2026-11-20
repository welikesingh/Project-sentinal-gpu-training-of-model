#!/usr/bin/env bash
# Run ON the VM any time: one-screen health check of everything the class needs.
#   bash hyperstack/check.sh            # quick checks
#   bash hyperstack/check.sh --full     # + GPU forward pass with Qwen and the test suite
cd "$(dirname "$0")/.."
PASS=0; FAILN=0
ok()   { echo -e "  \033[32m✔\033[0m $*"; PASS=$((PASS+1)); }
bad()  { echo -e "  \033[31m✘\033[0m $*"; FAILN=$((FAILN+1)); }
info() { echo -e "  \033[33m•\033[0m $*"; }
PY=.venv/bin/python

echo "SENTINEL VM check  ($(hostname), $(date '+%F %T'))"
echo "[system]"
if nvidia-smi >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -1)"; else bad "nvidia-smi failed (no GPU / driver)"; fi
FREE_GB=$(df -Pk . | awk 'NR==2{print int($4/1024/1024)}'); [ "$FREE_GB" -ge 5 ] && ok "disk free: ${FREE_GB} GB" || bad "disk free only ${FREE_GB} GB"
command -v tmux >/dev/null && ok "tmux installed" || bad "tmux missing (run setup_vm.sh)"

echo "[python]"
if [ -x "$PY" ]; then
  $PY -c "import torch,sys; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" >/tmp/sentinel_torch 2>&1 \
    && ok "torch + CUDA: $(cat /tmp/sentinel_torch)" || bad "torch cannot use the GPU: $(tail -1 /tmp/sentinel_torch)"
  $PY -c "import transformers, peft, accelerate, bitsandbytes, sklearn, xgboost; print('transformers', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__, '| xgboost', xgboost.__version__)" >/tmp/sentinel_pk 2>&1 \
    && ok "$(cat /tmp/sentinel_pk)" || bad "python packages: $(tail -1 /tmp/sentinel_pk)"
else bad ".venv missing (run: bash hyperstack/setup_vm.sh)"; fi

echo "[data + models]"
if [ -f .env ] && ! grep -q YOUR-PROJECT .env && grep -q '^SUPABASE_URL=' .env; then ok ".env has Supabase settings"; else bad ".env missing or not filled (cp .env.example .env && nano .env)"; fi
for s in train validation holdout; do
  n=$(ls data/sft/$s/*.jsonl.gz 2>/dev/null | wc -l); [ "$n" -gt 0 ] && ok "data/sft/$s: $n files" || bad "data/sft/$s missing (python -m sentinel.data)"
done
[ -x "$PY" ] && $PY - <<'EOF' && ok "Qwen/Qwen2.5-0.5B-Instruct cached locally" || bad "Qwen weights not cached (re-run setup_vm.sh)"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
EOF

echo "[runs + services]"
tmux has-session -t train 2>/dev/null && info "training RUNNING  -> tail -f output/last_train.log" || info "no training running"
LAST=""; for d in $(ls -1dt output/*/ 2>/dev/null); do [ -f "${d}calibration.json" ] && { LAST="${d%/}"; break; }; done
if [ -n "$LAST" ]; then
  ok "latest finished run: $LAST"
  $PY - "$LAST" <<'EOF' 2>/dev/null
import json, sys
m = json.load(open(sys.argv[1] + "/metrics_holdout.json"))["calibrated"]
print(f"      holdout PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f} | recall@REVIEW {m['at_review']['recall']:.2f} | precision@BLOCK {m['at_block']['precision']:.2f}")
EOF
else info "no finished run yet (bash hyperstack/train.sh)"; fi
for svc in "LoRA:8000:bash hyperstack/serve.sh lora 8000" "XGBoost:8001:bash hyperstack/xgb.sh serve" "QLoRA:8002:bash hyperstack/serve.sh qlora 8002"; do
  name=${svc%%:*}; rest=${svc#*:}; port=${rest%%:*}; how=${rest#*:}
  if curl -fs localhost:$port/health >/tmp/sentinel_h 2>/dev/null; then ok "$name API up on :$port -> $(cut -c1-110 /tmp/sentinel_h)"
  else info "$name API not running ($how)"; fi
done

if [ "${1:-}" = "--full" ] && [ -x "$PY" ]; then
  echo "[full]"
  $PY - <<'EOF' && ok "Qwen loads on GPU and scores a prompt" || bad "Qwen GPU forward pass failed"
import torch
from sentinel.model import build_model, predict_logits
m, t = build_model("Qwen/Qwen2.5-0.5B-Instruct", "lora")
x = predict_logits(m, t, ["[TXN] id=T amt=100.0 mcc=grocery channel=UPI time=12h Mon\n[TASK] Assess fraud risk using only the observed context."])
assert x.shape == (1, 9) and torch.isfinite(x).all()
EOF
  $PY -m pytest -q tests >/tmp/sentinel_pt 2>&1 && ok "tests: $(tail -1 /tmp/sentinel_pt)" || { bad "tests failed:"; tail -15 /tmp/sentinel_pt; }
fi

echo
[ "$FAILN" -eq 0 ] && echo -e "\033[32mALL GOOD\033[0m ($PASS checks passed)" || { echo -e "\033[31m$FAILN problem(s)\033[0m - fix the ✘ lines above"; exit 1; }
