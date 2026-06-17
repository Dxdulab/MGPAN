# pull_logs.ps1 — 从服务器拉取运行结果到本地 server_sync/
#
# 拉取目录:
#   远程 logs/       → 本地 server_sync/logs/
#   远程 Mp_Error/   → 本地 server_sync/Mp_Error/
#   远程 Mp_Output/  → 本地 server_sync/Mp_Output/
#
# 用法:
#   .\pull_logs.ps1              # 每 5 秒拉取
#   .\pull_logs.ps1 10           # 每 10 秒拉取
#   .\pull_logs.ps1 5 -Once      # 只拉一次
#   Ctrl+C 停止

param(
    [int]$Interval = 5,
    [switch]$Once = $false
)

$ErrorActionPreference = "Stop"
$SCP      = "C:\Windows\System32\OpenSSH\scp.exe"
$HostName = "10.250.0.240"
$RemoteBase = "/home/24031212340/MGPAN"
$LocalBase  = Join-Path $PSScriptRoot "server_sync"

# 要拉取的远程目录列表
$SyncDirs = @("logs", "Mp_Error", "Mp_Output", "saved_models")

foreach ($dir in $SyncDirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $LocalBase $dir) | Out-Null
}

function Write-Log {
    param([string]$Level, [string]$Msg)
    $time = Get-Date -Format "HH:mm:ss"
    $color = if ($Level -eq "ERR") { "Red" } elseif ($Level -eq "WARN") { "Yellow" } else { "Green" }
    Write-Host "[$time] $Msg" -ForegroundColor $color
}

function Invoke-PullOnce {
    $errors = @()
    foreach ($dir in $SyncDirs) {
        $remotePath = "$HostName`:$RemoteBase/$dir/*"
        $localPath  = Join-Path $LocalBase $dir
        $scpArgs = @(
            "-r", "-q",
            "-o", "StrictHostKeyChecking=accept-new",
            $remotePath,
            "$localPath/"
        )
        $out = & $SCP @scpArgs 2>&1
        if ($LASTEXITCODE -ne 0 -and $out) {
            $errors += "$dir : $out"
        }
    }
    if ($errors.Count -gt 0) {
        return ($errors -join " | ")
    }
    return ""
}

Write-Host "Server:  ${HostName}:${RemoteBase}/" -ForegroundColor Cyan
Write-Host "Local:   ${LocalBase}/" -ForegroundColor Cyan
Write-Host "Dirs:    $($SyncDirs -join ', ')" -ForegroundColor Cyan
Write-Host "Interval: ${Interval}s" -ForegroundColor Cyan
Write-Host ""

if ($Once) {
    Write-Log "INFO" "Single pull..."
    $err = Invoke-PullOnce
    if ($err) { Write-Log "WARN" $err }
    Write-Log "INFO" "Done."
    exit 0
}

Write-Log "INFO" "Pulling continuously (Ctrl+C to stop)..."
Write-Host ""

$failCount = 0

while ($true) {
    $startTime = Get-Date

    try {
        $errorMsg = Invoke-PullOnce
        if ([string]::IsNullOrEmpty($errorMsg)) {
            $failCount = 0
            $secs = [int]((Get-Date) - $startTime).TotalSeconds
            Write-Log "INFO" "OK (${secs}s)"
        } else {
            $failCount++
            Write-Log "WARN" "Fail #${failCount}: $errorMsg"
        }
    } catch {
        $failCount++
        Write-Log "ERR" "Fail #${failCount}: $_"
    }

    $secs = [int]((Get-Date) - $startTime).TotalSeconds
    $sleepTime = $Interval - $secs
    if ($sleepTime -gt 0) {
        Start-Sleep -Seconds $sleepTime
    }
}
