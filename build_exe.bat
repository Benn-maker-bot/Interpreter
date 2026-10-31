@echo off
REM ==========================================================================
REM  build_exe.bat - turn the app into a single VoiceInterpreter.exe
REM
REM  PyInstaller bundles Python, interpreter.py and every package it uses into
REM  one file, so the app can run on a PC without Python installed. The result
REM  appears in the "dist" folder. AI models are NOT bundled: they download on
REM  first use, as in the normal app.
REM
REM  What the PyInstaller options mean:
REM    --onefile              produce one .exe instead of a folder of files
REM    --windowed             no black console window behind the app
REM    --name                 the .exe's name
REM    --collect-all X        include package X with all its data files
REM                           (customtkinter themes, faster-whisper's VAD model,
REM                           CTranslate2's native libraries)
REM    --hidden-import X      include X even though PyInstaller can't see it
REM                           being imported (it is loaded indirectly)
REM    --collect-binaries X   include X's DLLs (the NVIDIA CUDA libraries)
REM  Lines ending in ^ continue on the next line (one long command).
REM ==========================================================================
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

REM -- Step 1: make sure every package (including PyInstaller) is installed
echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet

REM -- Step 2: build (see the option list at the top of this file)
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
  --hidden-import sentencepiece ^
  --hidden-import pyaudiowpatch ^
  --collect-binaries nvidia.cublas ^
  --collect-binaries nvidia.cudnn ^
  --hidden-import numpy ^
  interpreter.py

REM -- Step 3: report the result and open the output folder
echo.
echo [3/3] Done.
if exist "dist\VoiceInterpreter.exe" (
    echo SUCCESS: dist\VoiceInterpreter.exe
    explorer dist
) else (
    echo Build may have had issues. Check output above.
)
pause
