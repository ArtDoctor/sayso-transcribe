@echo off
cd /d "%~dp0"

echo ===================================================
echo   Running Sayso Fast Test Suites
echo ===================================================
echo.

echo [1/3] Running Python Backend Unit Tests...
call python -m pytest tests/test_backend.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Python backend unit tests failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Running End-to-End Pipeline Tests (test_e2e.py)...
call python -m pytest tests/test_e2e.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Python E2E integration test failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Running Frontend Unit Tests (vitest)...
call npm test
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Frontend tests failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo   ALL UNIT AND E2E TESTS PASSED SUCCESSFULLY!
echo ===================================================
exit /b 0
