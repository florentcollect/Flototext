@echo off
cd /d "%~dp0"

:: Kill existing Flototext (pythonw) instances. On ne tue PAS python.exe pour
:: ne pas couper d'autres scripts en cours.
taskkill /f /im pythonw.exe 2>nul

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
