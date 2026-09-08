@echo off
cd /d "%~dp0"

echo ===================================================
echo   Sayso - Desktop Launcher
echo ===================================================
echo.

REM Start Python backend in background (no console window) if not already running
curl.exe -s http://127.0.0.1:8765/api/status >nul 2>&1
if errorlevel 1 (
    echo [*] Starting Python backend in background (logging to logs.txt)...
    where pythonw >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        start "" pythonw -m backend.main
    ) else (
        powershell -NoProfile -WindowStyle Hidden -Command "Start-Process python -ArgumentList '-m', 'backend.main' -WorkingDirectory '%~dp0' -WindowStyle Hidden"
    )

    echo [*] Waiting for Python backend to initialize...
    for /L %%i in (1,1,20) do (
        curl.exe -s http://127.0.0.1:8765/api/status >nul 2>&1
        if not errorlevel 1 goto backend_ready
        ping 127.0.0.1 -n 2 >nul
    )
    echo [WARNING] Python backend taking longer to initialize, continuing...
)

:backend_ready
echo [OK] Python backend active on 127.0.0.1:8765.
echo.

REM Install dependencies if needed
if not exist node_modules call npm install

REM Launch desktop app
echo Launching Tauri 2 Desktop App...
echo.
call npm run tauri dev
