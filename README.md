# SENTINEL — fraud-scoring LLM, XGBoost benchmark, and dashboard

## Showcase runbook: 3 models on the GPU, dashboard on your PC (run line by line)

```
 Hyperstack GPU VM                                  Your PC
 ┌──────────────────────────────┐                  ┌──────────────────────────────┐
 │ LoRA     API :8000  (GPU)    │◀────── HTTP ─────│ Dashboard http://localhost:8501│
 │ XGBoost  API :8001  (GPU)    │◀─────────────────│ Overview · LoRA vs QLoRA ·   │
 │ QLoRA    API :8002  (GPU)    │◀─────────────────│ Benchmark · Live · Batch ·   │
 └──────────────────────────────┘   one API key    │ Decision (GO / NO-GO)        │
                                                   └──────────────────────────────┘
```
XGBoost is the model in production today (baseline). LoRA and QLoRA are the new candidates.
Hyperstack firewall: open inbound TCP **22, 8000, 8001, 8002**.

**1. On the VM** (`ssh ubuntu@<VM_IP>`):
```bash
cd sentinel
git pull
bash hyperstack/setup_vm.sh                      # safe to re-run; now also installs XGBoost
bash hyperstack/check.sh --full                  # must end with ALL GOOD

bash hyperstack/train.sh                         # LoRA (~8 min)
tail -f output/last_train.log                    # wait for "Artifacts in output/...", then Ctrl-C
bash hyperstack/train.sh --method qlora          # QLoRA (~8 min)
tail -f output/last_train.log                    # wait again, then Ctrl-C
bash hyperstack/xgb.sh train                     # XGBoost on the GPU (~1 min) -> output/xgb

PUBLIC_IP=<VM_IP> bash hyperstack/serve_all.sh   # starts LoRA :8000, XGBoost :8001, QLoRA :8002
bash hyperstack/check.sh                         # all three lines say "API up"
```
`serve_all.sh` ends by printing the exact line to run on your PC (with the API key filled in).

**2. On your PC** (PowerShell, in `D:\finetuning`):
```powershell
git pull
& ui\run_service.ps1 -VmIp <VM_IP> -Key <key> -Rebuild   # -Rebuild only the first time after a pull
start http://localhost:8501                              # three green dots in the sidebar
```
No Docker? Run it directly instead:
```powershell
$env:LORA_URL="http://<VM_IP>:8000"; $env:QLORA_URL="http://<VM_IP>:8002"; $env:XGB_URL="http://<VM_IP>:8001"
$env:LLM_API_KEY="<key>"; $env:XGB_API_KEY="<key>"
streamlit run ui/app.py
```

**3. The demo, tab by tab:** Overview (KPIs for all three) → LoRA vs QLoRA (training time, loss, thresholds) →
Benchmark (PR/ROC curves, Δ vs XGBoost) → Live Test (score a scenario on all three; do **at least 5 calls** so the
latency check fills in) → Batch Test (200 labelled holdout rows on all three) → **Decision**: GO (deploy the best
candidate) or NO-GO (keep XGBoost), with the checklist below it. Click **Download decision report** for the meeting notes.
The targets (recall ≥ 85 %, precision at BLOCK ≥ 70 %, PR-AUC ≥ 0.80, uplift ≥ +0.03, no worse false-positive rate,
p99 ≤ 250 ms) can be changed in the sidebar under *Acceptance targets*.

**4. Optional: copy the trained models to your PC** (Git Bash or PowerShell):
```bash
scp -r ubuntu@<VM_IP>:sentinel/output ./output-from-vm   # LoRA + QLoRA run folders and output/xgb
```

**5. When done:** `tmux kill-server` on the VM, then **hibernate the VM**.

---

```
 Hyperstack GPU VM                 Your machine                         Anyone with the links
 ┌───────────────────────┐        ┌────────────────────────────┐       ┌──────────────────────┐
 │ LLM API  :8000        │        │ XGBoost API :8001 ──ngrok──┼──────▶│ Streamlit dashboard  │
 │ Qwen2.5-0.5B + LoRA   │◀───────┼────────────────────────────┼───────│ :8501 (local, Docker │
 │ sentinel/ + hyperstack│  HTTP  │ xgboost_benchmark/         │       │  or any host)  ui/   │
 └───────────────────────┘        └────────────────────────────┘       └──────────────────────┘
        Dockerfile.llm                xgboost_benchmark/Dockerfile          ui/Dockerfile
```

All three services share one contract: `POST /score` takes `{"prompt"}` or structured fields or
`{"transactions":[...]}` and returns
`{transaction_id, fraud_probability, decision: ALLOW|REVIEW|BLOCK, confidence, reason_codes, explanation}`.
`GET /stats` returns metrics (incl. PR/ROC curves), calibration and model info. Auth uses the `X-API-Key` header.

| Folder | What | Runs on |
|---|---|---|
| `sentinel/`, `hyperstack/`, `tests/`, `Dockerfile.llm` | LLM training (LoRA/QLoRA), eval, GPU inference API | Hyperstack GPU |
| `xgboost_benchmark/` | XGBoost baseline: same splits, prompts, and protocol; API with samples and scenarios | Your machine (via ngrok) |
| `ui/` | iOS-style Streamlit dashboard: stats, benchmark, live and batch tests | Anywhere |

Both models use the same protocol. They train on `train` (Jan–Oct 2025), then early-stop, calibrate, and pick
thresholds on `validation` (REVIEW: recall ≥ 0.85, BLOCK: precision ≥ 0.70). Holdout metrics (Jan–Feb 2026)
use those frozen thresholds. Data comes from the Supabase bucket `sentinel-data/training/sft/*`.

---

## A. LLM on the Hyperstack GPU

**Console:** single-GPU NVIDIA VM (any card with ≥ 16 GB works; training needs only about 2.5 GB), an Ubuntu image with
NVIDIA drivers/CUDA, your SSH key, public IP. Open inbound TCP **22** and **8000**.

```bash
ssh ubuntu@<VM_IP>
git clone https://github.com/welikesingh/Project-sentinal-gpu-training-of-model.git sentinel && cd sentinel
cp .env.example .env && nano .env                # SUPABASE_URL, SUPABASE_KEY
bash hyperstack/setup_vm.sh                      # apt, venv, CUDA torch (auto-fallback), deps, data, Qwen weights -> "VM ready."
bash hyperstack/check.sh --full                  # GPU, torch, data, model, 15 tests -> "ALL GOOD"

bash hyperstack/train.sh                         # LoRA, full run: ~5 min train + ~3 min eval on a 4090-class GPU
bash hyperstack/train.sh --method qlora          # 4-bit QLoRA variant
bash hyperstack/train.sh --max-steps 30 --max-eval-rows 2000   # 1-minute demo run
tail -f output/last_train.log                    # done at "Artifacts in output/<run>"

source .venv/bin/activate
RUN=$(ls -1dt output/*/ | head -1)
python -m sentinel.evaluate --run $RUN --split holdout
python -m sentinel.infer --run $RUN sample --split holdout --n 5 --fraud-only

PUBLIC_IP=<VM_IP> bash hyperstack/serve.sh       # GPU API on :8000 (auto-restarts); waits until healthy; prints URL + key
bash hyperstack/check.sh                         # any time: what is running, latest run metrics
```
Check from your PC: `curl http://<VM_IP>:8000/health` (should show `"device": "cuda:0"`).
`train.sh` will not start a second run while one is active (`FORCE=1` replaces it).
To update: `git pull && bash hyperstack/setup_vm.sh`, then retrain or re-serve. **Hibernate the VM when done.**

## B. XGBoost benchmark on your machine (Windows/Linux)

The model is already trained and committed (`xgboost_benchmark/artifacts/`, holdout PR-AUC 0.215, ROC-AUC 0.973).
To keep it running all the time (Docker, auto-restart; prints the URL and API key):
```powershell
powershell -ExecutionPolicy Bypass -File xgboost_benchmark\run_service.ps1
ngrok http 8001                                  # share the https URL + key with the dashboard
```
Or retrain or run it without Docker:
```bash
# from the repo root, with the venv active (or: pip install -r xgboost_benchmark/requirements.txt)
cd xgboost_benchmark
python train.py                                  # ~1 min CPU; writes artifacts/ (already committed)
python -m pytest -q tests                        # 5 passed
# PowerShell:  $env:XGB_API_KEY = "<pick a long random key>"     bash:  export XGB_API_KEY=...
uvicorn server:app --host 127.0.0.1 --port 8001
```
Expose it in a second terminal:
```bash
ngrok http 8001                                  # copy the https://<id>.ngrok-free.app URL
curl https://<id>.ngrok-free.app/health
```

## C. Dashboard

Always-on (Docker, auto-restart); settings persist in `ui/.env`:
```powershell
& ui
un_service.ps1 -LlmUrl http://<VM_IP>:8000 -LlmKey <key from serve.sh>     # http://localhost:8501
& ui
un_service.ps1 -XgbUrl https://<id>.ngrok-free.app                        # optional: XGBoost via ngrok
```
Or without Docker:
```bash
# PowerShell: $env:LLM_URL="http://<VM_IP>:8000"; $env:LLM_API_KEY="<from serve.sh>"
#             $env:XGB_URL="http://localhost:8001"; $env:XGB_API_KEY="<xgboost_benchmark/.env>"
streamlit run ui/app.py                          # URLs/keys can also be changed live in the sidebar
python -m pytest -q ui/tests                     # headless render test (set UI_TEST_* for the live variant)
```
Tabs: **Overview** (holdout KPIs and acceptance targets OBJ-01/02/03/04/10) · **LLM** (run info, loss, thresholds,
reason-code F1) · **Benchmark** (PR/ROC overlays, operating points, uplift, feature importance) · **Live Test**
(scenario presets, a custom form, or a raw prompt, scored on both models with latency) · **Batch Test** (labelled
samples from the XGBoost service or an uploaded JSONL/CSV, with sample metrics and a CSV export).

To share the dashboard with others: run `ngrok http 8501` too, or deploy the `ui/` Docker image anywhere.

## Docker (each service deploys on its own)

```bash
docker build -t sentinel-xgb xgboost_benchmark   # XGBoost API (model baked in)
docker run -d -p 8001:8001 -e XGB_API_KEY=... -v "$PWD/data/sft:/data/sft" sentinel-xgb

docker build -t sentinel-ui ui                   # dashboard
docker run -d -p 8501:8501 -e LLM_URL=http://<VM_IP>:8000 -e LLM_API_KEY=... \
           -e XGB_URL=https://<id>.ngrok-free.app -e XGB_API_KEY=... sentinel-ui

docker build -f Dockerfile.llm -t sentinel-llm . # LLM API, on a GPU host with NVIDIA Container Toolkit
docker run -d --gpus all -p 8000:8000 -e SENTINEL_API_KEY=... -v "$PWD/output:/app/output" \
           sentinel-llm --run output/<run> serve --host 0.0.0.0 --port 8000
```
Or use compose: `docker compose up -d --build xgboost ui` (your machine), then
`docker compose --profile gpu up -d --build llm` (GPU host, set `LLM_RUN=<run>` in `.env`).

## Class-day checklist
1. Your PC: `docker ps` shows `sentinel-xgb` and `sentinel-ui` as *healthy* (Docker Desktop: *Start when you sign in*).
2. VM: `bash hyperstack/check.sh --full` prints **ALL GOOD**.
3. VM: `bash hyperstack/train.sh` → `tail -f output/last_train.log` → `PUBLIC_IP=<VM_IP> bash hyperstack/serve.sh`.
4. PC: `& ui
un_service.ps1 -LlmUrl http://<VM_IP>:8000 -LlmKey <key>` → open http://localhost:8501. Both dots are green.
5. Sharing: `ngrok http 8501` for the dashboard (the XGBoost API needs its own `ngrok http 8001` only for remote dashboards).

## Troubleshooting
| Symptom | Fix |
|---|---|
| Dashboard shows LLM offline | VM firewall needs TCP 8000; `tmux ls` on the VM should list `serve`; check the key |
| XGBoost offline via ngrok | Is `uvicorn` still running? Is the ngrok URL current (it changes on restart)? Is `XGB_API_KEY` the same? |
| 401 | Wrong or missing API key in the sidebar |
| Dashboard shows QLoRA or XGBoost offline | VM firewall needs TCP 8002 / 8001; `tmux ls` should list `serve-8002` / `xgb`; `serve_all.sh` skips a model that has no finished run |
| Decision says *pending* | A model is offline, or fewer than 5 Live Test calls have been made (latency check) |
| `setup_vm.sh`: `cuda False` | Image driver too old. Use a newer image, or `TORCH_INDEX=https://download.pytorch.org/whl/cu128` |
| CUDA OOM while training | Gradient checkpointing is on by default (~2.5 GB). Still OOM: `--batch-size 8 --grad-accum 2` |
| `train.sh` says a run is in progress | `tail -f output/last_train.log`, or `FORCE=1 bash hyperstack/train.sh ...` to replace it |
| `serve.sh`: server did not come up | It prints the log tail; usually no finished run yet, or port 8000 already in use (`tmux kill-session -t serve`) |

Plain HTTP with shared keys is for testing only. Put the APIs behind HTTPS before using real data.
