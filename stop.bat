@echo off
:: Only Flototext: matching the command line spares every other Python app
:: (the old "any pythonw.exe over 100 MB" rule killed unrelated tools).
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'pythonw.exe' -or $_.Name -eq 'python.exe') -and $_.CommandLine -like '*flototext.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
echo Flototext stopped.
timeout /t 2 >nul
