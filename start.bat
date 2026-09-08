@echo off
cd /d "%~dp0"

echo ===================================================
echo   Sayso - Desktop Launcher
echo ===================================================
echo.

REM The Tauri host starts and owns the Python backend. Keeping this launcher
REM from spawning it is important: the host can then hard-kill it on exit,
REM including stale processes left by an older launch.
echo [*] Tauri will start and supervise the Python backend.
echo.
REM Install dependencies if needed
if not exist node_modules call npm install

REM Launch desktop app
echo Launching Tauri 2 Desktop App...
echo.
call npm run tauri dev
