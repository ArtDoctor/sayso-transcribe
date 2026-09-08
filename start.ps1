Set-Location $PSScriptRoot

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Sayso - Desktop Launcher" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $null = python --version
} catch {
    Write-Host "[ERROR] Python not found on PATH." -ForegroundColor Red
    exit 1
}

# Check if Python backend is already active
$backendUp = $false
try {
    $res = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/status" -TimeoutSec 1 -ErrorAction SilentlyContinue
    if ($res.status -eq "online") { $backendUp = $true }
} catch {}

if (-not $backendUp) {
    Write-Host "[*] Starting Python backend in background (no console, logging to logs.txt)..." -ForegroundColor Yellow
    $pyExe = if (Get-Command pythonw -ErrorAction SilentlyContinue) { "pythonw" } else { "python" }
    Start-Process -FilePath $pyExe -ArgumentList "-m", "backend.main" -WorkingDirectory $PSScriptRoot -WindowStyle Hidden

    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $testRes = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/status" -TimeoutSec 1 -ErrorAction SilentlyContinue
            if ($testRes.status -eq "online") {
                $backendUp = $true
                Write-Host "[OK] Python backend is online on 127.0.0.1:8765." -ForegroundColor Green
                break
            }
        } catch {}
    }
} else {
    Write-Host "[OK] Python backend is already active on port 8765." -ForegroundColor Green
}

# Check node_modules
if (-not (Test-Path "node_modules")) {
    Write-Host "[*] Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "[*] Launching Tauri 2 Desktop App..." -ForegroundColor Green
Write-Host ""
npm run tauri dev
