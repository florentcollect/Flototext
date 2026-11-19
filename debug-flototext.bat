@echo off
cd /d "%~dp0"

:: Kill existing Flototext instances only (pythonw or a previous debug
:: console), matched on the command line so other Python scripts survive.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'pythonw.exe' -or $_.Name -eq 'python.exe') -and $_.CommandLine -like '*flototext.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

:: Small delay to ensure process is fully terminated
timeout /t 1 /nobreak >nul

echo ============================================================
echo  FLOTOTEXT - MODE DEBUG (console visible)
echo  Attends le message "Model loaded successfully" (icone verte)
echo  puis appuie sur F2 et regarde les messages ci-dessous.
echo ============================================================
echo.

:: Launch Flototext WITH visible console, using the dedicated venv
"%~dp0.venv\Scripts\python.exe" -m flototext.main

echo.
echo ============================================================
echo  Flototext s'est arrete. Appuie sur une touche pour fermer.
echo ============================================================
pause >nul
