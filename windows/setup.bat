@echo off
setlocal enabledelayedexpansion
title Qwen3.8-27B TensorFold Setup — RTX 5060 Ti

:: Setup script: creates venv, installs TensorFold, downloads model
:: by Screwed Up Tech — screwedup.tech

cd /d "%~dp0\.."

echo ============================================================
echo   Qwen3.8 TensorFold Setup — RTX 5060 Ti ^(16 GB^)
echo   by Screwed Up Tech — screwedup.tech
echo ============================================================
echo.

:: Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    echo         and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

:: Check NVIDIA driver
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi not found. Install NVIDIA driver 570+.
    pause
    exit /b 1
)

:: Check GPU compute capability
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2^>nul') do set CC=%%G
echo [*] GPU compute capability: %CC%

:: Check MSVC Build Tools (needed for CUDA JIT at inference time)
where cl >nul 2>&1
if errorlevel 1 (
    set "MSVC_FOUND="
    for /f "tokens=*" %%D in ('dir /b /ad "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC" 2^>nul') do set "MSVC_FOUND=1"
    if not defined MSVC_FOUND (
        echo [WARN] Visual Studio Build Tools not found.
        echo        TensorFold needs cl.exe for CUDA kernel compilation.
        echo        Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo        Select "Desktop development with C++" workload.
        echo.
    ) else (
        echo [*] MSVC Build Tools found
    )
) else (
    echo [*] cl.exe found in PATH
)

:: Create venv
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating Python venv...
    python -m venv .venv
    echo [*] Upgrading pip...
    .venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
)

:: Install TensorFold
echo [*] Installing TensorFold...
.venv\Scripts\pip.exe install tensorfold 2>&1 | findstr /v "already satisfied"

:: Install huggingface_hub for model downloads
.venv\Scripts\pip.exe install huggingface_hub 2>&1 | findstr /v "already satisfied"

:: Load .env
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" if not "!line!"=="" (
            set "%%A=%%B"
        )
    )
)
if not defined MODEL set MODEL=dense

:: Download model
if /i "%MODEL%"=="flashnext" (
    echo [*] Downloading Qwen3.8 Flash Next EXL3 2.05bpw...
    echo     This is 62.8 GB — it will take a while.
    if not defined FLASHNEXT_MODEL_DIR set FLASHNEXT_MODEL_DIR=models\Qwen3.8-Flash-Next-exl3-2.05bpw
    .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('turboderp/Qwen3.8-Flash-Next-exl3', revision='2.05bpw_h4_ng4', local_dir='%FLASHNEXT_MODEL_DIR%')"
) else (
    echo [*] Downloading Qwen3.8-27B EXL3 2.0bpw...
    echo     This is about 6.8 GB.
    if not defined DENSE_MODEL_DIR set DENSE_MODEL_DIR=models\Qwen3.8-27B-EXL3-2.0bpw
    .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw', local_dir='%DENSE_MODEL_DIR%')"
)

echo.
echo ============================================================
echo   Setup complete!
echo   Run windows\start.bat to serve the model.
echo ============================================================
pause
