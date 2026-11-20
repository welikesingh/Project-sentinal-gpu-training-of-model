#!/usr/bin/env bash
# Run ON the VM. Starts all three models on the GPU and prints the one command to run on your PC.
#   PUBLIC_IP=<VM_IP> bash hyperstack/serve_all.sh
#     LoRA    -> :8000  (newest finished LoRA run)
#     XGBoost -> :8001  (output/xgb if trained with xgb.sh train, else the committed model)
#     QLoRA   -> :8002  (newest finished QLoRA run)
# A model with no finished run is skipped (the dashboard just shows it offline).
# Firewall: open inbound TCP 8000, 8001, 8002.
cd "$(dirname "$0")/.."
export PUBLIC_IP="${PUBLIC_IP:-$(curl -fs --max-time 3 https://api.ipify.org || echo '<VM_PUBLIC_IP>')}"
UP=()
echo "=== LoRA :8000 ===";    bash hyperstack/serve.sh lora 8000  && UP+=("LoRA :8000")    || echo "(skipped LoRA)"
echo "=== QLoRA :8002 ===";   bash hyperstack/serve.sh qlora 8002 && UP+=("QLoRA :8002")   || echo "(skipped QLoRA)"
echo "=== XGBoost :8001 ==="; bash hyperstack/xgb.sh serve 8001   && UP+=("XGBoost :8001") || echo "(skipped XGBoost)"
KEY="${SENTINEL_API_KEY:-$(grep '^SENTINEL_API_KEY=' .env | tail -1 | cut -d= -f2- | tr -d '\r')}"

echo
echo "Running: ${UP[*]:-nothing}"
echo "-------------------------------------------------------------------------------"
echo "On your PC (PowerShell, in the repo folder) run this ONE line:"
echo
echo "  & ui\\run_service.ps1 -VmIp $PUBLIC_IP -Key $KEY"
echo
echo "then open http://localhost:8501"
echo "-------------------------------------------------------------------------------"
