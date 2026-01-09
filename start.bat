@echo off
REM Quick start script for Legal Aid Assistant
REM This script starts both backend and frontend in separate windows

echo.
echo ========================================
echo Legal Aid Assistant - Quick Start
echo ========================================
echo.

REM Check if Ollama is running
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ Ollama is not running!
    echo.
    echo Start Ollama first with: ollama serve
    echo.
    pause
    exit /b 1
)
echo ✅ Ollama is running

REM Get current directory
setlocal enabledelayedexpansion
set "DIR=%~dp0"

REM Start backend
echo.
echo Starting Backend (port 8000)...
start "Legal Aid Assistant - Backend" cmd /k cd /d "!DIR!" ^& python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 3 /nobreak

REM Start frontend
echo.
echo Starting Frontend (port 8501)...
start "Legal Aid Assistant - Frontend" cmd /k cd /d "!DIR!" ^& streamlit run frontend/ui.py

echo.
echo ========================================
echo Services Started!
echo ========================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
echo.
echo Browser should open automatically.
echo If not, visit: http://localhost:8501
echo.
echo Close either window to stop the service.
echo.
pause
