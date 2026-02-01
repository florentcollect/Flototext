@echo off
taskkill /f /im pythonw.exe /fi "MEMUSAGE gt 100000" >nul 2>&1
echo Flototext stopped.
timeout /t 2 >nul
