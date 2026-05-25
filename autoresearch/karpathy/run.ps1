#Requires -Version 5.1
<#
.SYNOPSIS
    Run a single autoresearch training experiment (~5 min).
.DESCRIPTION
    Launches train.py, captures output to run.log, and prints the summary.
.PARAMETER LogFile
    Path to the log file (default: run.log in script directory).
.PARAMETER NoLog
    Stream output to console instead of logging to file.
#>
[CmdletBinding()]
param(
    [string]$LogFile,
    [switch]$NoLog
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $LogFile) { $LogFile = Join-Path $ScriptDir 'run.log' }

Push-Location $ScriptDir
try {
    if ($NoLog) {
        uv run train.py
        exit $LASTEXITCODE
    }

    Write-Host "Training started — logging to $LogFile" -ForegroundColor Cyan
    Write-Host "This takes ~5 minutes. Ctrl+C to abort." -ForegroundColor DarkGray

    uv run train.py *> $LogFile
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "`nTraining FAILED (exit code $exitCode). Last 50 lines:" -ForegroundColor Red
        Get-Content $LogFile -Tail 50
        exit $exitCode
    }

    # Print summary
    Write-Host "`n--- Results ---" -ForegroundColor Green
    Select-String -Path $LogFile -Pattern '^(val_bpb|training_seconds|total_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps|num_params_M|depth):' |
        ForEach-Object { $_.Line }

} finally {
    Pop-Location
}
