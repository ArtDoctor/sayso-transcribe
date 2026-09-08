[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [string]$PythonVersion = "3.11.9",
    [string]$PythonRuntime = "",
    [switch]$NoModel,
    [switch]$SkipBuild,
    # Kept for compatibility; normal runs now replace the package automatically.
    [switch]$Clean,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ScriptRoot "Sayso-Portable"
}

$AppName = "Sayso"
$AppDir = Join-Path $OutputRoot $AppName
$Requirements = Join-Path $ScriptRoot "requirements-portable.txt"
$BackendSource = Join-Path $ScriptRoot "backend"
$TauriExe = Join-Path $ScriptRoot "src-tauri\target\release\Sayso.exe"
$modelRepoName = "models--CohereLabs--cohere-transcribe-03-2026"

function Assert-Path([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description not found: $Path"
    }
}

function Invoke-Checked([string]$FilePath, [string[]]$ArgumentList) {
    Write-Host "> $FilePath $($ArgumentList -join ' ')" -ForegroundColor DarkGray
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

function Get-HuggingFaceHub {
    if ($env:HF_HUB_CACHE) {
        return $env:HF_HUB_CACHE
    }
    if ($env:HF_HOME) {
        return (Join-Path $env:HF_HOME "hub")
    }
    return (Join-Path $env:USERPROFILE ".cache\huggingface\hub")
}

function Find-ModelSource {
    $hub = Get-HuggingFaceHub
    $candidate = Join-Path $hub $modelRepoName
    if (-not (Test-Path -LiteralPath $candidate)) {
        return $null
    }

    $required = @("config.json", "tokenizer.model", "model.safetensors")
    $snapshots = Get-ChildItem -LiteralPath (Join-Path $candidate "snapshots") -Directory -ErrorAction SilentlyContinue
    foreach ($snapshot in $snapshots) {
        $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $snapshot.FullName $_)) })
        if ($missing.Count -eq 0) {
            return $candidate
        }
    }
    throw "Model cache exists but is incomplete: $candidate"
}

function Install-EmbeddedPython([string]$Destination) {
    if ($PythonRuntime) {
        Assert-Path $PythonRuntime "PythonRuntime"
        Assert-Path (Join-Path $PythonRuntime "python.exe") "PythonRuntime python.exe"
        Write-Host "Copying supplied embedded Python runtime..." -ForegroundColor Cyan
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        Copy-Item -Path (Join-Path $PythonRuntime "*") -Destination $Destination -Recurse -Force
        return
    }

    $zipName = "python-$PythonVersion-embed-amd64.zip"
    $url = "https://www.python.org/ftp/python/$PythonVersion/$zipName"
    $download = Join-Path ([IO.Path]::GetTempPath()) $zipName
    $extract = Join-Path ([IO.Path]::GetTempPath()) ("sayso-python-" + [guid]::NewGuid().ToString("N"))

    Write-Host "Downloading embedded Python $PythonVersion..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $download
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    Expand-Archive -LiteralPath $download -DestinationPath $extract -Force
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $extract "*") -Destination $Destination -Recurse -Force

    $pth = Get-ChildItem -LiteralPath $Destination -Filter "*._pth" -File | Select-Object -First 1
    if (-not $pth) {
        throw "The embedded Python archive did not contain a ._pth file."
    }
    $lines = @(Get-Content -LiteralPath $pth.FullName)
    if ($lines -notcontains "Lib\site-packages") { $lines += "Lib\site-packages" }
    if ($lines -notcontains "..\") { $lines += "..\" }
    if ($lines -notcontains "import site") { $lines += "import site" }
    Set-Content -LiteralPath $pth.FullName -Value $lines -Encoding ASCII

    $python = Join-Path $Destination "python.exe"
    $getPip = Join-Path $extract "get-pip.py"
    Write-Host "Bootstrapping pip in embedded Python..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
    Invoke-Checked $python @($getPip, "--disable-pip-version-check")
    Invoke-Checked $python @("-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip", "setuptools", "wheel")
    Write-Host "Installing Sayso backend dependencies..." -ForegroundColor Cyan
    Invoke-Checked $python @("-m", "pip", "install", "--disable-pip-version-check", "-r", $Requirements)

    Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $download -Force -ErrorAction SilentlyContinue
}

function Write-PortableFiles {
    $launcher = @'
@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0Sayso.exe" (
  echo Sayso.exe was not found beside this launcher.
  pause
  exit /b 1
)
start "Sayso" "%~dp0Sayso.exe"
endlocal
'@
    Set-Content -LiteralPath (Join-Path $AppDir "start.bat") -Value $launcher -Encoding ASCII

    $readme = @'
Sayso Portable
==============

Run Sayso.exe directly or double-click start.bat. This folder is self-contained
for 64-bit Windows 10/11 and does not require Python, Node.js, Rust, or npm.

The models folder contains the Cohere transcription model when this package was
built normally. It can be deleted to reduce the initial package size; Sayso will
show its model download screen and download the model into this folder on the
first run. Keep the folder writable.

Recordings and logs are stored beside the executable in recordings/ and logs.txt.
The bundled Python runtime is in python/ and should not be removed.

This application captures Windows microphone/WASAPI audio and may run slower on
CPU-only machines. NVIDIA CUDA acceleration is used automatically when present.
'@
    Set-Content -LiteralPath (Join-Path $AppDir "README-portable.txt") -Value $readme -Encoding UTF8
}

Write-Host "Sayso portable package" -ForegroundColor Cyan
Write-Host "Output: $OutputRoot"

Assert-Path $BackendSource "Backend source"
Assert-Path $Requirements "Portable requirements file"
Assert-Path (Join-Path $ScriptRoot "src-tauri\tauri.conf.json") "Tauri configuration"

if ($ValidateOnly) {
    Write-Host "Validation passed. Use without -ValidateOnly to build the package." -ForegroundColor Green
    exit 0
}

# Always rebuild the package from scratch. Keep the output root itself so a
# custom output root can contain other files or package directories.
if (Test-Path -LiteralPath $AppDir) {
    Write-Host "Removing existing package: $AppDir" -ForegroundColor Yellow
    Remove-Item -LiteralPath $AppDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null

if (-not $SkipBuild) {
    Assert-Path (Join-Path $ScriptRoot "package.json") "package.json"
    Invoke-Checked "npm.cmd" @("run", "tauri", "build")
}

if (-not (Test-Path -LiteralPath $TauriExe)) {
    $candidate = Get-ChildItem -LiteralPath (Join-Path $ScriptRoot "src-tauri\target\release") -Filter "*.exe" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "(?i)(uninstall|setup|nsis)" } |
        Select-Object -First 1
    if ($candidate) { $TauriExe = $candidate.FullName }
}
Assert-Path $TauriExe "Tauri release executable"
Copy-Item -LiteralPath $TauriExe -Destination (Join-Path $AppDir "Sayso.exe") -Force

Write-Host "Copying backend source..." -ForegroundColor Cyan
$backendDest = Join-Path $AppDir "backend"
New-Item -ItemType Directory -Force -Path $backendDest | Out-Null
Copy-Item -Path (Join-Path $BackendSource "*") -Destination $backendDest -Recurse -Force
Get-ChildItem -LiteralPath $backendDest -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $backendDest -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force

$runtimeDest = Join-Path $AppDir "python"
Install-EmbeddedPython $runtimeDest

New-Item -ItemType Directory -Force -Path (Join-Path $AppDir "models\huggingface\hub") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $AppDir "models\torch") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $AppDir "recordings") | Out-Null

if (-not $NoModel) {
    $modelSource = Find-ModelSource
    if (-not $modelSource) {
        throw "Cohere model cache not found. Download it first or rerun with -NoModel. Expected under $(Get-HuggingFaceHub)\$modelRepoName"
    }
    Write-Host "Copying Cohere model cache (this is about 4 GB)..." -ForegroundColor Cyan
    Copy-Item -LiteralPath $modelSource -Destination (Join-Path $AppDir "models\huggingface\hub") -Recurse -Force
} else {
    Write-Host "Leaving model directory empty (-NoModel)." -ForegroundColor Yellow
}

Write-PortableFiles
Write-Host "Portable package created: $AppDir" -ForegroundColor Green
Write-Host "Run $AppDir\Sayso.exe or $AppDir\start.bat" -ForegroundColor Green
