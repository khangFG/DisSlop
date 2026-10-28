@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Discord LLM Bot with MCP Integration...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    echo Error: .venv not found. Please create virtual environment first.
    pause
)
