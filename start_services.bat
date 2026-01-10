@echo off
REM Start Backend and Frontend for Legal Aid Assistant
REM This script activates the virtual environment and starts both services

cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo ========================================
echo Legal Aid Assistant - Starting Services
echo ========================================
echo.

REM Check if Ollama is running
echo Checking Ollama...
curl -s http://localhost:11434/api/tags > nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Ollama is not running!
    echo Please start Ollama first: ollama serve
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama is running

REM Start backend in a new window
echo.
echo Starting Backend (port 8000)...
start "Legal Aid Assistant - Backend" cmd /k "cd backend & python app.py"

REM Wait for backend to start
timeout /t 3 /nobreak

REM Start frontend in a new window
echo Starting Frontend (port 8501)...
start "Legal Aid Assistant - Frontend" cmd /k "streamlit run frontend/ui.py"

echo.
echo ========================================
echo Services started!
echo ========================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
echo.
pause
