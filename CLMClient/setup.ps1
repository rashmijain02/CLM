[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ClientRoot = $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ClientRoot
$VirtualEnvironment = Join-Path $ClientRoot ".venv"

$Launcher = Get-Command py -ErrorAction SilentlyContinue
if ($Launcher) {
    & $Launcher.Source -3.12 -m venv $VirtualEnvironment
}
else {
    $Launcher = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Launcher) {
        throw "Python 3.10+ was not found. Install Python 3.12, reopen PowerShell, and run this script again."
    }
    & $Launcher.Source -m venv $VirtualEnvironment
}
if ($LASTEXITCODE -ne 0) {
    throw "Python could not create the virtual environment. Confirm that Python 3.10+ is installed."
}

$VenvPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ClientRoot "requirements.txt")
& $VenvPython -m pip install --no-deps --editable $RepositoryRoot

$EnvironmentFile = Join-Path $ClientRoot ".env"
if (-not (Test-Path $EnvironmentFile)) {
    Copy-Item (Join-Path $ClientRoot ".env.example") $EnvironmentFile
}

Write-Host ""
Write-Host "CLM client environment is ready."
Write-Host "Mock server:"
Write-Host "  .\.venv\Scripts\python.exe ..\tools\playground_mock.py"
Write-Host "Examples (in a second PowerShell window):"
Write-Host "  .\.venv\Scripts\python.exe .\main.py --scenario all"

