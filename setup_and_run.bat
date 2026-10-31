@echo off
REM ==========================================================================
REM  setup_and_run.bat - one-click setup and start for the Voice Interpreter
REM
REM  Double-click this file. It:
REM    1. checks that Python is installed,
REM    2. installs the packages listed in requirements.txt,
REM    3. reports whether an NVIDIA GPU can be used (much faster),
REM    4. starts the app (interpreter.py; add --classic for the old window).
REM
REM  Note: this uses whichever "python" is on your PATH. For development the
REM  project uses a virtual environment instead (.venv) - see README.txt.
REM  Lines starting with REM are comments; "echo" prints text.
REM ==========================================================================
title Voice Interpreter Setup
color 0A
echo.
echo ============================================
echo  Voice Interpreter - Setup
echo ============================================
echo.

REM -- Step 1: is Python installed? (errorlevel is non-zero if the command failed)
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.9+ from https://python.org
    echo Tick Add Python to PATH during install.
    pause
    exit /b 1
)
echo [OK] Python found.
echo.

REM -- Step 2: install / update the required packages
echo Installing core packages...
pip install -r requirements.txt --quiet
REM faster-whisper pulls in the CPU-only onnxruntime, which hides the GPU
REM build used by music mode; swap it back.
pip uninstall -y onnxruntime >nul 2>&1
pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2 --quiet

REM -- Step 3: ask CTranslate2 (the Whisper engine) how many CUDA GPUs it sees
echo.
echo Checking for NVIDIA GPU...
python -c "import ctranslate2; n=ctranslate2.get_cuda_device_count(); print('[GPU] CUDA devices: '+str(n)) if n>0 else print('[CPU] No CUDA GPU - running on CPU')"

REM -- Step 4: start the app; "pause" keeps this window open afterwards
echo.
echo [OK] Setup complete. Starting Voice Interpreter...
echo.
echo Log file will be created at: interpreter.log.jsonl
echo.
python interpreter.py
pause
