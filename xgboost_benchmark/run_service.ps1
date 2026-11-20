# Keep the XGBoost benchmark API always running (Docker, auto-restarts with Docker Desktop).
#   powershell -ExecutionPolicy Bypass -File xgboost_benchmark\run_service.ps1          # build if needed + start
#   powershell -ExecutionPolicy Bypass -File xgboost_benchmark\run_service.ps1 -Rebuild # after retraining
#   powershell -ExecutionPolicy Bypass -File xgboost_benchmark\run_service.ps1 -Stop
param([switch]$Rebuild, [switch]$Stop, [int]$Port = 8001)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$name = "sentinel-xgb"

if (docker ps -aq --filter "name=^${name}$") { docker rm -f $name | Out-Null }
if ($Stop) { Write-Host "Stopped $name"; exit 0 }

$envFile = Join-Path $here ".env"
if (-not (Test-Path $envFile)) {
    $bytes = New-Object byte[] 24; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    "XGB_API_KEY=" + [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_') | Out-File -Encoding ascii $envFile
    Write-Host "Generated XGB_API_KEY in $envFile"
}
if ($Rebuild -or -not (docker images -q $name)) { docker build -t $name $here; if ($LASTEXITCODE) { exit 1 } }

docker run -d --name $name --restart unless-stopped -p "${Port}:8001" --env-file $envFile `
    -v "${root}\data\sft:/data/sft" $name | Out-Null
for ($i = 0; $i -lt 30; $i++) {
    try { $h = Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 2; break } catch { Start-Sleep 1 }
}
Write-Host "XGBoost API: http://localhost:$Port  status=$($h.status)  (restarts automatically)"
Write-Host "API key   : $((Get-Content $envFile | Select-String '^XGB_API_KEY=').ToString().Split('=',2)[1])"
Write-Host "Expose    : ngrok http $Port"
