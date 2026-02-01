@echo off
title Flototext - Voice Recognition
cd /d "%~dp0"
echo Starting Flototext...
echo Press F2 to record, release to transcribe.
echo.
python -m flototext.main
pause
