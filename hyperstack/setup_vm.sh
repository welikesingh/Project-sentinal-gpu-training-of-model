#!/usr/bin/env bash
# Runs ON the Hyperstack VM (Ubuntu + NVIDIA driver image). Safe to re-run.
#  1. system packages (python3-venv, pip, tmux, curl)
#  2. ~/sentinel/.venv with CUDA PyTorch (falls back to an older CUDA build if the driver is old)
#  3. project requirements
#  4. SFT data from Supabase (needs .env) + Qwen and test-model weights, so class time needs no downloads
set -euo pipefail
cd "$(dirname "$0")/.."
ok()   { echo -e "\033[32m[ok]\033[0m $*"; }
fail() { echo -e "\033[31m[FAIL]\033[0m $*"; exit 1; }

command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null || fail "no NVIDIA driver/GPU visible - use a GPU flavor with an NVIDIA/CUDA image"
ok "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
FREE_GB=$(df -Pk . | awk 'NR==2{print int($4/1024/1024)}')
[ "$FREE_GB" -ge 15 ] || fail "only ${FREE_GB} GB free disk here; need >= 15 GB (torch + models)"
ok "disk: ${FREE_GB} GB free"

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
need=(); for p in python3-venv python3-pip tmux curl; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
if [ ${#need[@]} -gt 0 ]; then
  APT="$SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600"
  $APT update -y && $APT install -y "${need[@]}"
fi
ok "system packages"

[ -x .venv/bin/pip ] || { rm -rf .venv; python3 -m venv .venv; }
PY=.venv/bin/python
PIP="$PY -m pip --timeout 300 --retries 10"
$PIP install -q --upgrade pip

cuda_ok() { $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }
if ! cuda_ok; then
  for spec in "${TORCH_SPEC:-2.8.0 cu126}" "2.7.1 cu118"; do
    ver=${spec% *}; tag=${spec#* }
    echo "Installing torch $ver ($tag) ..."
    $PIP install -q "torch==$ver" --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/$tag}" && cuda_ok && break
    echo "torch $ver/$tag cannot use this driver, trying an older CUDA build ..."
    unset TORCH_INDEX
  done
fi
cuda_ok || fail "PyTorch cannot see the GPU (driver too old?). Pick an image with a newer NVIDIA driver."
ok "$($PY -c "import torch;print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))")"

$PIP install -q -r requirements.txt
$PIP install -q -r xgboost_benchmark/requirements.txt
ok "python requirements (LLM + XGBoost)"

[ -f .env ] || cp .env.example .env
if grep -q "YOUR-PROJECT" .env; then
  fail "edit .env first (SUPABASE_URL / SUPABASE_KEY), then re-run: nano .env"
fi
$PY -m sentinel.data >/dev/null
ok "data: $(ls data/sft/*/*.jsonl.gz | wc -l) files in data/sft"

$PY - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen2.5-0.5B-Instruct", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"):
    snapshot_download(repo)
    print("cached", repo)
EOF
ok "model weights cached"
echo -e "\n\033[32mVM ready.\033[0m  Next: bash hyperstack/check.sh   then   bash hyperstack/train.sh"
