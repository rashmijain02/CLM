[CmdletBinding()]
param(
    [ValidateSet("Client", "WSL")]
    [string]$Mode = "Client",
    [string]$Distribution = "Ubuntu-24.04",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ClientRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RepositoryRoot = Split-Path -Parent $ClientRoot

function Test-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Add-UserPythonToPath {
    $paths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Launcher"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\Scripts")
    )
    $existing = $paths | Where-Object { Test-Path $_ }
    if ($existing) {
        $env:Path = ($existing -join ";") + ";" + $env:Path
    }
}

function Test-Python {
    Add-UserPythonToPath
    if (Test-Command "py") {
        & py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        return $LASTEXITCODE -eq 0
    }
    if (Test-Command "python") {
        & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        return $LASTEXITCODE -eq 0
    }
    return $false
}

function Get-Distributions {
    if (-not (Test-Command "wsl.exe")) {
        return @()
    }
    $raw = & wsl.exe --list --quiet 2>$null
    return @($raw | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ })
}

Write-Host "CLM setup target: $Mode"
Write-Host "Repository: $RepositoryRoot"

if ($Mode -eq "Client") {
    $pythonReady = Test-Python
    Write-Host "Python 3.10+: $pythonReady"
    Write-Host "Client setup script: $(Test-Path (Join-Path $ClientRoot 'setup.ps1'))"

    if ($CheckOnly) {
        if (-not $pythonReady) {
            throw "Python 3.10+ is not installed. Run this script without -CheckOnly to install Python 3.12."
        }
        exit 0
    }

    if (-not $pythonReady) {
        if (-not (Test-Command "winget.exe")) {
            throw "Python is missing and winget is unavailable. Install Python 3.10+ and rerun this script."
        }
        & winget.exe install --id Python.Python.3.12 --exact --scope user `
            --accept-package-agreements --accept-source-agreements --silent --disable-interactivity
        if ($LASTEXITCODE -ne 0) {
            throw "winget could not install Python 3.12 (exit $LASTEXITCODE)."
        }
        Add-UserPythonToPath
        if (-not (Test-Python)) {
            throw "Python was installed but is not available yet. Open a new PowerShell window and rerun this script."
        }
    }

    & (Join-Path $ClientRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "CLMClient setup failed (exit $LASTEXITCODE)."
    }
    Write-Host "Windows CLM client installation complete."
    exit 0
}

# WSL mode provisions both halves: the native Windows client and the Linux server.
if ($CheckOnly) {
    & $PSCommandPath -Mode Client -CheckOnly
}
else {
    & $PSCommandPath -Mode Client
}

if (-not (Test-Command "wsl.exe")) {
    throw "WSL is unavailable. Enable the Windows Subsystem for Linux and rerun this script."
}

$distributions = Get-Distributions
$distributionReady = $distributions -contains $Distribution
Write-Host "WSL distribution $Distribution installed: $distributionReady"
if ($distributions.Count -gt 0) {
    Write-Host "Installed distributions: $($distributions -join ', ')"
}

if ($CheckOnly) {
    if (-not $distributionReady) {
        throw "$Distribution is not installed. Run this script without -CheckOnly to install it."
    }
}
elseif (-not $distributionReady) {
    & wsl.exe --install --distribution $Distribution
    if ($LASTEXITCODE -ne 0) {
        throw "WSL could not install $Distribution (exit $LASTEXITCODE). Run PowerShell as Administrator."
    }
    Write-Warning "WSL installation started. Restart Windows if requested, initialize $Distribution, then rerun this command."
    exit 0
}

$linuxRepository = (& wsl.exe -d $Distribution -- wslpath -a $RepositoryRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxRepository) {
    throw "Could not map the repository path into $Distribution."
}
$linuxScript = "$linuxRepository/CLMClient/scripts/clm-local-setup/setup-linux.sh"
$arguments = @($linuxScript)
if ($CheckOnly) {
    $arguments += "--check-only"
}

Write-Host "Delegating Linux server setup to $linuxScript"
& wsl.exe -d $Distribution -- bash @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Linux CLM setup failed inside $Distribution (exit $LASTEXITCODE)."
}

Write-Host "WSL CLM server installation complete."
