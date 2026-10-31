@echo off
title Voice Interpreter Setup
color 0A
echo.
echo ============================================
echo  Voice Interpreter - Setup
echo ============================================
echo.

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

echo Installing core packages...
pip install -r requirements.txt --quiet

echo.
echo Checking for NVIDIA GPU...
python -c "import ctranslate2; n=ctranslate2.get_cuda_device_count(); print('[GPU] CUDA devices: '+str(n)) if n>0 else print('[CPU] No CUDA GPU - running on CPU')"

echo.
echo [OK] Setup complete. Starting Voice Interpreter...
echo.
echo Log file will be created at: interpreter.log.jsonl
echo.
python interpreter.py
pause
