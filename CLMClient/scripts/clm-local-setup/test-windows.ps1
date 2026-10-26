[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ScriptsRoot = $PSScriptRoot
$PowerShellScripts = Get-ChildItem -Path $ScriptsRoot -Filter "*.ps1"

foreach ($script in $PowerShellScripts) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $script.FullName,
        [ref]$tokens,
        [ref]$errors
    ) | Out-Null
    if ($errors.Count -gt 0) {
        $messages = $errors | ForEach-Object { $_.Message }
        throw "$($script.Name) has syntax errors: $($messages -join '; ')"
    }
    Write-Host "$($script.Name): syntax OK"
}

& (Join-Path $ScriptsRoot "setup-windows.ps1") -Mode Client -CheckOnly
if ($LASTEXITCODE -ne 0) {
    throw "Windows client preflight failed (exit $LASTEXITCODE)."
}

Write-Host "Standalone Windows setup script tests passed."

