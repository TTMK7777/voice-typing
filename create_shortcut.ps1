$ws = New-Object -ComObject WScript.Shell
$dir = $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop 'VoiceTyping.lnk'
$sc = $ws.CreateShortcut($lnk)
$pyw = Join-Path $dir '.venv\Scripts\pythonw.exe'
$sc.TargetPath = $pyw
$sc.Arguments = 'app_rt.py'
$sc.WorkingDirectory = $dir
$icon = Join-Path $dir 'icon.ico'
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Description = 'voice-typing realtime'
$sc.Save()
Write-Output ('pythonw exists: ' + (Test-Path $pyw))
Write-Output ('icon exists: ' + (Test-Path $icon))
Write-Output ('shortcut created: ' + $lnk)
