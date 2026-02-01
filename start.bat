@echo off
cd /d "%~dp0"

:: Kill existing Flototext instances if running
taskkill /f /fi "WINDOWTITLE eq Flototext*" 2>nul
for /f "tokens=2" %%a in ('wmic process where "commandline like '%%flototext.main%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%a 2>nul
)

:: Launch Flototext
start "" pythonw -m flototext.main
