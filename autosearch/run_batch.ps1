## Batch training run — survives terminal close on Windows
## Launch: powershell -ExecutionPolicy Bypass -File run_batch.ps1 -Runs 50
## Or detached: Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File run_batch.ps1 -Runs 50" -WindowStyle Minimized

param(
    [int]$Runs = 50
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$batchLog = Join-Path $PSScriptRoot "batch_6hr.log"
$shadowTsv = Join-Path $PSScriptRoot "weber_shadow.tsv"

"Batch started: $(Get-Date) ($Runs runs)" | Out-File $batchLog -Encoding utf8

for ($i = 1; $i -le $Runs; $i++) {
    $runLog = Join-Path $PSScriptRoot "run_batch_$i.log"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    "=== RUN $i === $timestamp" | Out-File $batchLog -Append -Encoding utf8

    # Run training — use Start-Process to avoid file locking issues
    $proc = Start-Process -FilePath "uv" -ArgumentList "run","train.py" `
        -RedirectStandardOutput $runLog -RedirectStandardError "$runLog.err" `
        -NoNewWindow -PassThru -WorkingDirectory $PSScriptRoot

    $proc.WaitForExit()

    # Merge stderr into log
    if (Test-Path "$runLog.err") {
        Get-Content "$runLog.err" | Add-Content $runLog
        Remove-Item "$runLog.err" -Force
    }

    # Extract and log results
    if (Test-Path $runLog) {
        $lines = Get-Content $runLog
        foreach ($line in $lines) {
            if ($line -match "^val_bpb:|^time_budget:|^shadow_|^weber_c_sq:|^checkpoint:") {
                $line | Out-File $batchLog -Append -Encoding utf8
            }
        }
    }

    $exitCode = $proc.ExitCode
    if ($exitCode -ne 0) {
        "  CRASHED (exit code $exitCode)" | Out-File $batchLog -Append -Encoding utf8
    }

    "--- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---" | Out-File $batchLog -Append -Encoding utf8
}

"Batch complete: $(Get-Date)" | Out-File $batchLog -Append -Encoding utf8
