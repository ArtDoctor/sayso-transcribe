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

# The Tauri host starts and owns the Python backend. Do not launch it here:
# this lets the host hard-kill stale Python/Vite processes when the window closes.
Write-Host "[*] Tauri will start and supervise the Python backend." -ForegroundColor Yellow

# Check node_modules
if (-not (Test-Path "node_modules")) {
    Write-Host "[*] Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "[*] Launching Tauri 2 Desktop App..." -ForegroundColor Green
Write-Host ""
npm run tauri dev
