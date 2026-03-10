@echo off
cd /d "%~dp0"

echo.
echo  ==========================================
echo   RAGBrain - Starting...
echo  ==========================================
echo.

REM Check venv exists
if not exist "venv\Scripts\python.exe" (
    echo  ERROR: venv not found!
    echo  Run: python -m venv venv
    echo  Then: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Kill existing Ollama and restart with NVIDIA GPU forced
echo  [1/3] Restarting Ollama with NVIDIA GPU...
taskkill /F /IM ollama.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul
set CUDA_VISIBLE_DEVICES=0
start /min "" ollama serve
timeout /t 4 /nobreak >nul
echo        Done.

REM Activate venv
echo  [2/3] Activating environment...
call venv\Scripts\activate.bat
echo        Done.

REM Start RAGBrain (opens browser automatically)
echo  [3/3] Starting RAGBrain...
echo.
echo  Browser will open automatically.
echo  Press Ctrl+C to stop.
echo.
venv\Scripts\python.exe backend\main.py

pause