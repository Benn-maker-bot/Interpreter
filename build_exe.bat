@echo off
title Voice Interpreter - Build EXE
color 0B
echo.
echo ============================================
echo  Voice Interpreter - Build Standalone EXE
echo ============================================
echo.
echo Estimated time: 3-5 minutes.
echo.
pause

echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [2/3] Building executable...
pyinstaller ^
  --onefile ^
  --windowed ^
  --name "VoiceInterpreter" ^
  --collect-all customtkinter ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  --hidden-import comtypes.client ^
  --hidden-import pyaudiowpatch ^
  --collect-binaries nvidia.cublas ^
  --collect-binaries nvidia.cudnn ^
  --hidden-import numpy ^
  interpreter.py

echo.
echo [3/3] Done.
if exist "dist\VoiceInterpreter.exe" (
    echo SUCCESS: dist\VoiceInterpreter.exe
    explorer dist
) else (
    echo Build may have had issues. Check output above.
)
pause
