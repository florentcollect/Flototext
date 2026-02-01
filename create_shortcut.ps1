$WshShell = New-Object -ComObject WScript.Shell
$StartupPath = [Environment]::GetFolderPath('Startup')
$Shortcut = $WshShell.CreateShortcut("$StartupPath\Flototext.lnk")
$Shortcut.TargetPath = "F:\Flototext\start-flototext.bat"
$Shortcut.WorkingDirectory = "F:\Flototext"
$Shortcut.Description = "Flototext - Voice Recognition"
$Shortcut.Save()
Write-Host "Raccourci cree dans: $StartupPath"
