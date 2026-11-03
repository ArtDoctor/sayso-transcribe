@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===================================================
echo   Sayso - Production Build Script
echo ===================================================
echo.

if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="/?" goto show_help
if /i "%~1"=="help" goto show_help

if /i "%~1"=="--portable" goto do_portable
if /i "%~1"=="portable" goto do_portable
if /i "%~1"=="-portable" goto do_portable

if /i "%~1"=="--frontend" goto do_frontend
if /i "%~1"=="frontend" goto do_frontend
if /i "%~1"=="--web" goto do_frontend
if /i "%~1"=="web" goto do_frontend

goto do_desktop_build

:do_portable
shift
set "PORTABLE_ARGS="
:portable_args_loop
if "%~1"=="" goto run_portable
set "PORTABLE_ARGS=!PORTABLE_ARGS! %1"
shift
goto portable_args_loop

:run_portable
echo [*] Building portable distribution via package-portable.ps1...
powershell -ExecutionPolicy Bypass -File "%~dp0package-portable.ps1" !PORTABLE_ARGS!
exit /b %ERRORLEVEL%

:do_frontend
where npm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm was not found in PATH. Please install Node.js and npm.
    exit /b 1
)

if not exist node_modules (
    echo [*] Installing frontend dependencies: npm install
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] npm install failed.
        exit /b %ERRORLEVEL%
    )
)

echo [*] Building frontend static assets: npm run build
call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Frontend build failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo   Frontend build completed successfully!
echo   Output located in: dist\
echo ===================================================
exit /b 0

:do_desktop_build
where npm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm was not found in PATH. Please install Node.js v18+ and npm.
    exit /b 1
)

where cargo >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Cargo was not found in PATH. Please install Rust toolchain: https://rustup.rs/
    exit /b 1
)

if not exist node_modules (
    echo [*] Installing frontend dependencies: npm install
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] npm install failed.
        exit /b %ERRORLEVEL%
    )
)

echo [*] Building Sayso desktop release binary via Tauri...
if "%~1"=="" (
    call npm run tauri build
) else (
    call npm run tauri build -- %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAIL] Tauri build failed with error %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo   SAYSO DESKTOP BUILD COMPLETED SUCCESSFULLY!
echo ===================================================
echo.
echo Release output located in:
echo   Executable: src-tauri\target\release\Sayso.exe
echo   Bundles:    src-tauri\target\release\bundle\
echo.
echo Tip: To build a self-contained portable package with embedded Python:
echo   build.bat --portable
echo ===================================================
exit /b 0

:show_help
echo Usage:
echo   build.bat                    Builds release desktop binary and bundles via Tauri
echo   build.bat --no-bundle        Builds release desktop binary without installers
echo   build.bat --frontend         Builds web frontend dist only
echo   build.bat --portable [opts]  Builds self-contained portable distribution folder
echo   build.bat [tauri flags]      Passes flags directly to Tauri CLI e.g. --debug
echo.
echo Portable options passed to package-portable.ps1:
echo   -NoModel                     Omit bundled Cohere model cache
echo   -SkipBuild                   Reuse existing compiled binary without rebuilding
echo   -ValidateOnly                Validate prerequisites without building
exit /b 0
