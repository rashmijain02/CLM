[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8700"
)

$ErrorActionPreference = "Stop"
$ClientRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ClientRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "The Windows client environment is missing. Run setup-windows.ps1 -Mode Client first."
}

$health = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/health" -TimeoutSec 15
if (-not $health.ok) {
    throw "The CLM API reports ok=false."
}
if (-not $health.embedder) {
    throw "The CLM API is running, but the embedding server is unavailable."
}
if ($health.mock) {
    throw "The endpoint is the mock server, not the real CLM server."
}

& $Python (Join-Path $ClientRoot "main.py") --base-url $BaseUrl --scenario all
if ($LASTEXITCODE -ne 0) {
    throw "Typed scenario verification failed (exit $LASTEXITCODE)."
}
& $Python (Join-Path $ClientRoot "main.py") --base-url $BaseUrl --scenario rank
if ($LASTEXITCODE -ne 0) {
    throw "Ranking verification failed (exit $LASTEXITCODE)."
}

Write-Host "Real CLM server and Windows client verification passed."
