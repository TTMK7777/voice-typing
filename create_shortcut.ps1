# デスクトップに VoiceTyping ショートカットを作る。
# -Startup を付けると Windows サインイン時の自動起動(スタートアップフォルダ)にも入れる。
param([switch]$Startup)

$ws = New-Object -ComObject WScript.Shell
$dir = $PSScriptRoot
$pyw = Join-Path $dir '.venv\Scripts\pythonw.exe'
$icon = Join-Path $dir 'icon.ico'

function New-VoiceTypingShortcut($folder) {
    $lnk = Join-Path $folder 'VoiceTyping.lnk'
    $sc = $ws.CreateShortcut($lnk)
    $sc.TargetPath = $pyw
    $sc.Arguments = 'app_rt.py'
    $sc.WorkingDirectory = $dir
    if (Test-Path $icon) { $sc.IconLocation = $icon }
    $sc.Description = 'voice-typing realtime'
    $sc.Save()
    return $lnk
}

Write-Output ('pythonw exists: ' + (Test-Path $pyw))
Write-Output ('icon exists: ' + (Test-Path $icon))
$desktop = [Environment]::GetFolderPath('Desktop')
Write-Output ('shortcut created: ' + (New-VoiceTypingShortcut $desktop))
if ($Startup) {
    $startup = [Environment]::GetFolderPath('Startup')
    Write-Output ('startup shortcut created: ' + (New-VoiceTypingShortcut $startup))
}
