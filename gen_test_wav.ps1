Add-Type -AssemblyName System.Speech
$text = [System.IO.File]::ReadAllText("$PSScriptRoot\test_text.txt", [System.Text.Encoding]::UTF8)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Haruka Desktop")
$s.SetOutputToWaveFile("$PSScriptRoot\test.wav")
$s.Speak($text)
$s.Dispose()
Write-Output "wav generated"
