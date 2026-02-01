@echo off
title Stop Flototext
echo Stopping Flototext...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Flototext*" >nul 2>&1
wmic process where "commandline like '%%flototext.main%%'" call terminate >nul 2>&1
echo Flototext stopped.
timeout /t 2 >nul
