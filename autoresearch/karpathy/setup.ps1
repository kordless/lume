#Requires -Version 5.1
<#
.SYNOPSIS
    One-time setup for autoresearch on Windows.
    Installs uv (if missing), syncs dependencies, downloads data & trains tokenizer.
#>
[CmdletBinding()]
param(
    [int]$NumShards = 10,
    [int]$DownloadWorkers = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# ── Install uv if not on PATH ──────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv package manager..." -ForegroundColor Cyan
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # Refresh PATH so uv is available in this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', 'User') + ';' +
                [System.Environment]::GetEnvironmentVariable('PATH', 'Machine')
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv installation succeeded but 'uv' not found on PATH. Restart your terminal and try again."
    }
}
Write-Host "uv: $(uv --version)" -ForegroundColor Green

# ── Sync dependencies ──────────────────────────────────────────────────────
Write-Host "`nSyncing Python dependencies..." -ForegroundColor Cyan
Push-Location $ScriptDir
try {
    uv sync
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit code $LASTEXITCODE)" }
} finally {
    Pop-Location
}

# ── Download data & train tokenizer ─────────────────────────────────────────
Write-Host "`nPreparing data and tokenizer..." -ForegroundColor Cyan
Push-Location $ScriptDir
try {
    uv run prepare.py --num-shards $NumShards --download-workers $DownloadWorkers
    if ($LASTEXITCODE -ne 0) { throw "prepare.py failed (exit code $LASTEXITCODE)" }
} finally {
    Pop-Location
}

Write-Host "`nSetup complete! Run .\run.ps1 to train." -ForegroundColor Green
