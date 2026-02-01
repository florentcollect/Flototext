@echo off
title Flototext - Desinstallation
cd /d "%~dp0"

echo ========================================
echo   Desinstallation de Flototext
echo ========================================
echo.

:: Vérifier les droits administrateur
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Relancement en mode administrateur...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: 1. Arrêter l'application si elle tourne
echo [1/4] Arret de l'application...
taskkill /f /im pythonw.exe /fi "MEMUSAGE gt 100000" >nul 2>&1

:: 2. Supprimer le raccourci du dossier Startup
echo [2/4] Suppression du demarrage automatique...
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Flototext.lnk" >nul 2>&1

:: 3. Supprimer l'entrée du registre Windows
echo [3/4] Suppression de l'enregistrement Windows...
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Flototext" /f >nul 2>&1

:: 4. Demander si l'utilisateur veut supprimer les fichiers
echo [4/4] Nettoyage...
echo.
set /p DELETE_FILES="Supprimer tous les fichiers de Flototext ? (O/N): "
if /i "%DELETE_FILES%"=="O" (
    echo Suppression des fichiers...
    cd /d "%TEMP%"
    rmdir /s /q "%~dp0" >nul 2>&1
    echo Fichiers supprimes.
) else (
    echo Fichiers conserves dans: %~dp0
)

echo.
echo ========================================
echo   Desinstallation terminee !
echo ========================================
echo.
pause
