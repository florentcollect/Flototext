@echo off
cd /d "%~dp0"

:: Kill existing Flototext instances only. Matching on the command line keeps
:: unrelated pythonw.exe apps (other Python GUI tools) alive.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'pythonw.exe' -and $_.CommandLine -like '*flototext.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

:: Small delay to ensure process is fully terminated. `timeout` aborts when stdin
:: is redirected (e.g. launched from another shell), so wait with ping instead.
ping -n 2 127.0.0.1 >nul 2>&1

:: Launch Flototext using the dedicated venv (isolated from global packages)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m flototext.main
