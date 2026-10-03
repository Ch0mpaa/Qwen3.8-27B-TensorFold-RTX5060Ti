@echo off
setlocal enabledelayedexpansion
title Qwen3.8-27B TensorFold — RTX 5060 Ti

:: Qwen3.8-27B / Flash Next on TensorFold for RTX 5060 Ti (16 GB)
:: by Screwed Up Tech — screwedup.tech

cd /d "%~dp0\.."

:: Load .env
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" if not "!line!"=="" (
            set "%%A=%%B"
        )
    )
)

:: Defaults
if not defined MODEL set MODEL=dense
if not defined PORT set PORT=8090
if not defined HOST set HOST=0.0.0.0
if not defined CONTEXT_SIZE set CONTEXT_SIZE=3072
if not defined THINKING set THINKING=off
if not defined DRAFT set DRAFT=none
if not defined GPU_MEM_GB set GPU_MEM_GB=14.7
if not defined NO_UPDATE_CHECK set NO_UPDATE_CHECK=1
if not defined MODEL_ID set MODEL_ID=qwen3.8-27b-exl3-2bpw-tensorfold
if not defined VISION set VISION=off
if not defined TF_VENV set TF_VENV=.venv

:: Check venv
if not exist "%TF_VENV%\Scripts\python.exe" (
    echo [ERROR] TensorFold venv not found at %TF_VENV%
    echo Run windows\setup.bat first.
    pause
    exit /b 1
)

:: Select model
if /i "%MODEL%"=="flashnext" (
    if not defined FLASHNEXT_MODEL_DIR set FLASHNEXT_MODEL_DIR=models\Qwen3.8-Flash-Next-exl3-2.05bpw
    set "MODEL_PATH=%FLASHNEXT_MODEL_DIR%"
    set "MODEL_ID=qwen3.8-flash-next-exl3-tensorfold"
    echo [*] Model: Qwen3.8 Flash Next ^(MoE + MTP^)
) else (
    if not defined DENSE_MODEL_DIR set DENSE_MODEL_DIR=models\Qwen3.8-27B-EXL3-2.0bpw
    set "MODEL_PATH=%DENSE_MODEL_DIR%"
    echo [*] Model: Qwen3.8-27B dense ^(EXL3 2.0bpw^)
)

:: Check model exists
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model not found at %MODEL_PATH%
    echo Run windows\setup.bat to download it.
    pause
    exit /b 1
)

:: Check VRAM
echo [*] Checking GPU...
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2^>nul') do set FREE_VRAM=%%G
if defined FREE_VRAM (
    echo [*] Free VRAM: %FREE_VRAM% MiB
) else (
    echo [WARN] Could not read VRAM — nvidia-smi not found
)

:: Build args
set "ARGS=-m tensorfold serve %MODEL_PATH% --host %HOST% --port %PORT% --name %MODEL_ID% --context %CONTEXT_SIZE%"

if /i "%THINKING%"=="off" set "ARGS=%ARGS% --no-thinking"
if /i "%DRAFT%"=="none" set "ARGS=%ARGS% --no-drafts"
if /i "%NO_UPDATE_CHECK%"=="1" set "ARGS=%ARGS% --no-update-check"

:: Flash Next specific flags
if /i "%MODEL%"=="flashnext" (
    if defined MTP_DRAFTS set "ARGS=%ARGS% --mtp-drafts %MTP_DRAFTS%"
    if defined MTP_CONFIDENCE set "ARGS=%ARGS% --mtp-confidence %MTP_CONFIDENCE%"
    if defined SSD_EXPERTS_GIB set "ARGS=%ARGS% --ssd-experts %SSD_EXPERTS_GIB%"
)

echo.
echo ============================================================
echo   Qwen3.8 on TensorFold — RTX 5060 Ti ^(16 GB^)
echo   by Screwed Up Tech — screwedup.tech
echo ============================================================
echo   Model:    %MODEL_PATH%
echo   Endpoint: http://%HOST%:%PORT%/v1
echo   Context:  %CONTEXT_SIZE% tokens
echo   Thinking: %THINKING%
echo   Draft:    %DRAFT%
echo ============================================================
echo.

echo [*] Starting TensorFold...
"%TF_VENV%\Scripts\python.exe" %ARGS%

pause
