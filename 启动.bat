@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting cad-agent GUI...
".venv\Scripts\python.exe" gui.py
if errorlevel 1 pause
