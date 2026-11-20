# Keep the SENTINEL dashboard always running on your PC (Docker, auto-restarts with Docker Desktop).
#   & ui\run_service.ps1 -VmIp <VM_IP> -Key <key>        # all 3 models on the Hyperstack VM (LoRA :8000, XGBoost :8001, QLoRA :8002)
#   & ui\run_service.ps1                                  # start again with the saved settings
#   & ui\run_service.ps1 -Rebuild                         # after code changes
#   & ui\run_service.ps1 -Stop
# Per-model overrides: -LoraUrl -QloraUrl -XgbUrl -LlmKey -XgbKey  (-LlmUrl is an alias for -LoraUrl)
# Settings persist in ui\.env (git-ignored).
param([string]$VmIp, [string]$Key, [Alias("LlmUrl")][string]$LoraUrl, [string]$QloraUrl, [string]$XgbUrl,
      [string]$LlmKey, [string]$XgbKey, [switch]$Rebuild, [switch]$Stop, [int]$Port = 8501)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$name = "sentinel-ui"

if (docker ps -aq --filter "name=^${name}$") { docker rm -f $name | Out-Null }
if ($Stop) { Write-Host "Stopped $name"; exit 0 }

$envFile = Join-Path $here ".env"
$cfg = [ordered]@{ LORA_URL = "http://host.docker.internal:8000"; QLORA_URL = "http://host.docker.internal:8002"
                   XGB_URL = "http://host.docker.internal:8001"; LLM_API_KEY = ""; XGB_API_KEY = "" }
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) { if ($line -match '^\s*([A-Z_]+)=(.*)$') { $cfg[$matches[1]] = $matches[2].Trim() } }
}
if ($cfg.Contains("LLM_URL")) { if (-not $cfg.LORA_URL -or $cfg.LORA_URL -like "*host.docker.internal:8000") { $cfg.LORA_URL = $cfg.LLM_URL }; $cfg.Remove("LLM_URL") }
$xgbEnv = Join-Path $root "xgboost_benchmark\.env"
if (-not $cfg.XGB_API_KEY -and (Test-Path $xgbEnv)) {
    $m = Select-String -Path $xgbEnv -Pattern '^XGB_API_KEY=(.*)$'; if ($m) { $cfg.XGB_API_KEY = $m.Matches[0].Groups[1].Value.Trim() }
}
if ($VmIp) {
    $cfg.LORA_URL = "http://${VmIp}:8000"; $cfg.XGB_URL = "http://${VmIp}:8001"; $cfg.QLORA_URL = "http://${VmIp}:8002"
}
if ($Key) { $cfg.LLM_API_KEY = $Key; $cfg.XGB_API_KEY = $Key }
if ($LoraUrl) { $cfg.LORA_URL = $LoraUrl }
if ($QloraUrl) { $cfg.QLORA_URL = $QloraUrl }
if ($XgbUrl) { $cfg.XGB_URL = $XgbUrl }
if ($LlmKey) { $cfg.LLM_API_KEY = $LlmKey }
if ($XgbKey) { $cfg.XGB_API_KEY = $XgbKey }
($cfg.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) | Out-File -Encoding ascii $envFile

if ($Rebuild -or -not (docker images -q $name)) { docker build -t $name $here; if ($LASTEXITCODE) { exit 1 } }
docker run -d --name $name --restart unless-stopped -p "${Port}:8501" --env-file $envFile `
    --add-host "host.docker.internal:host-gateway" $name | Out-Null
for ($i = 0; $i -lt 40; $i++) {
    try { $h = Invoke-WebRequest "http://localhost:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 2; break } catch { Start-Sleep 1 }
}
Write-Host "Dashboard : http://localhost:$Port   (restarts automatically)"
Write-Host "LoRA      : $($cfg.LORA_URL)"
Write-Host "QLoRA     : $($cfg.QLORA_URL)"
Write-Host "XGBoost   : $($cfg.XGB_URL)"
Write-Host "Share     : ngrok http $Port"
