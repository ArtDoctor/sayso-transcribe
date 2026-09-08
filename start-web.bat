@echo off
cd /d "%~dp0"

echo ===================================================
echo   Sayso - Web Dev Launcher
echo ===================================================
echo.

:: 1. Ensure Python backend is running on port 8765
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
    echo [ERROR] Python backend failed to respond on http://127.0.0.1:8765 within 20s.
    pause
    exit /b 1
)

:backend_ready
echo [OK] Python backend is active and listening on port 8765.

:: 2. Verify node_modules
if not exist node_modules (
    echo [*] Installing frontend dependencies (npm install)...
    call npm install
)

:: 3. Launch Vite Dev Server
echo [*] Starting Vite frontend server at http://localhost:1420...
call npm run dev
exit /b %ERRORLEVEL%
