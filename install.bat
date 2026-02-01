@echo off
title Flototext - Installation
cd /d "%~dp0"

echo ========================================
echo   Installation de Flototext
echo ========================================
echo.

:: Vérifier les droits administrateur
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Relancement en mode administrateur...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

set "APP_NAME=Flototext"
set "APP_PATH=%~dp0"
set "APP_EXE=%APP_PATH%start.bat"
set "UNINSTALL_EXE=%APP_PATH%uninstall.bat"
set "ICON_PATH=%APP_PATH%assets\icon.ico"

:: 1. Créer le raccourci dans le dossier Startup (démarrage automatique)
echo [1/2] Configuration du demarrage automatique...
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Flototext.lnk'); $SC.TargetPath = '%APP_EXE%'; $SC.WorkingDirectory = '%APP_PATH%'; $SC.IconLocation = '%ICON_PATH%'; $SC.Description = 'Flototext - Voice Recognition'; $SC.Save()"

:: 2. Ajouter l'entrée dans le registre Windows (Applications et fonctionnalités)
echo [2/2] Enregistrement dans Windows...
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "DisplayName" /t REG_SZ /d "Flototext" /f >nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "DisplayIcon" /t REG_SZ /d "%ICON_PATH%" /f >nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "UninstallString" /t REG_SZ /d "\"%UNINSTALL_EXE%\"" /f >nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "InstallLocation" /t REG_SZ /d "%APP_PATH%" /f >nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "Publisher" /t REG_SZ /d "Flototext" /f >nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "DisplayVersion" /t REG_SZ /d "1.0.0" /f >nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "NoModify" /t REG_DWORD /d 1 /f >nul
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /v "NoRepair" /t REG_DWORD /d 1 /f >nul

echo.
echo ========================================
echo   Installation terminee !
echo ========================================
echo.
echo - Flototext se lancera au demarrage de Windows
echo - Vous pouvez desinstaller depuis Parametres Windows
echo - Appuyez sur F2 pour dicter du texte
echo.
pause
