@echo off
REM Quick setup script for RealtimeSTT-ONNX using UV

echo ========================================
echo RealtimeSTT-ONNX Setup Script
echo ========================================
echo.

REM Check if UV is installed
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo UV is not installed. Installing UV...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo Please restart your terminal and run this script again.
    pause
    exit /b 1
)

echo Found UV: 
uv --version
echo.

echo Creating virtual environment...
uv venv
echo.

echo Activating virtual environment...
call .venv\Scripts\activate
echo.

echo Installing package in editable mode...
uv pip install -e .
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the virtual environment:
echo   .venv\Scripts\activate
echo.
echo To run examples:
echo   python examples/basic_transcription.py
echo.
echo For GPU support:
echo   uv pip install -e ".[gpu]"
echo.
echo For development tools:
echo   uv pip install -e ".[dev]"
echo.
pause
