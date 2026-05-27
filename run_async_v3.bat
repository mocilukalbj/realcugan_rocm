@echo off
chcp 65001 >nul
title Real-CUGAN Async V3

set SCRIPT_DIR=%~dp0
set PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo Python not found at %PYTHON%
    echo Please run: python -m venv venv ^&^& venv\Scripts\pip install torch torchvision opencv-python numpy
    pause
    exit /b 1
)

"%PYTHON%" "%SCRIPT_DIR%run_async_launcher.py"
pause
