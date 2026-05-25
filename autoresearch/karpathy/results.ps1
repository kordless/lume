#Requires -Version 5.1
<#
.SYNOPSIS
    Extract key metrics from run.log (PowerShell replacement for grep).
.PARAMETER LogFile
    Path to the log file (default: run.log in script directory).
#>
[CmdletBinding()]
param(
    [string]$LogFile
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $LogFile) { $LogFile = Join-Path $ScriptDir 'run.log' }

if (-not (Test-Path $LogFile)) {
    Write-Error "Log file not found: $LogFile"
}

Select-String -Path $LogFile -Pattern '^(val_bpb|peak_vram_mb):' |
    ForEach-Object { $_.Line }
