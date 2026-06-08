@echo off
cd /d "%~dp0"

:: Kill existing Flototext/pythonw instances running flototext.main
taskkill /f /im pythonw.exe 2>nul

:: Small delay to ensure process is fully terminated
timeout /t 1 /nobreak >nul

:: Launch Flototext using the dedicated venv (isolated from global packages)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m flototext.main
