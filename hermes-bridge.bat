@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

rem ============================================================================
rem  hermes-bridge.bat - lance le pont de transcription Hermes -> Flototext.
rem
rem  Surveille F:\Projets-IA\Hermes\hermes-data\stt-bridge\in\ et transcrit
rem  chaque vocal recu par Hermes (WhatsApp, Discord) avec le moteur ASR de
rem  Flototext et son dictionnaire de correction.
rem
rem  Usage :
rem    hermes-bridge.bat          fenetre visible, journal a l'ecran
rem    hermes-bridge.bat --auto   silencieux (raccourci de demarrage Windows)
rem
rem  Journal : F:\Projets-IA\Hermes\hermes-data\stt-bridge\watcher.log
rem  Arret   : fermer la fenetre, ou Ctrl+C
rem ============================================================================

set "RACINE=%~dp0"
set "PY=%RACINE%.venv\Scripts\python.exe"
set "SCRIPT=%RACINE%hermes_bridge.py"
set "PONT=F:\Projets-IA\Hermes\hermes-data\stt-bridge"

set "AUTO=0"
if /I "%~1"=="--auto" set "AUTO=1"
if /I "%~1"=="/auto" set "AUTO=1"

if not exist "%PY%" (
    echo [X] Environnement Python introuvable : %PY%
    echo     Installer Flototext d'abord ^(install.bat^).
    if "%AUTO%"=="0" pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [X] Script introuvable : %SCRIPT%
    if "%AUTO%"=="0" pause
    exit /b 1
)

if not exist "%PONT%\in" (
    echo [!] Dossier d'echange absent : %PONT%\in
    echo     Hermes le cree au demarrage. Verifier que le conteneur tourne.
)

if "%AUTO%"=="0" (
    echo Pont de transcription Hermes ^<- Flototext
    echo   surveille : %PONT%\in
    echo   journal   : %PONT%\watcher.log
    echo.
    echo Le premier chargement du modele ASR prend une a deux minutes.
    echo Laisser cette fenetre ouverte. Ctrl+C pour arreter.
    echo.
)

"%PY%" "%SCRIPT%"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    if "%AUTO%"=="0" (
        echo.
        echo [X] Le pont s'est arrete avec le code %RC%. Detail : %PONT%\watcher.log
        pause
    )
)

endlocal
exit /b %RC%
