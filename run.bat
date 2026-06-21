@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe launcher.py
) else (
    echo .venv 가 없습니다. setup.ps1 을 먼저 실행하세요.
    pause
    exit /b 1
)
