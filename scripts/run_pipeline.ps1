# Pipeline starten (nur dieser Befehl — keinen erklärenden Text in die Konsole einfügen).
Set-Location $PSScriptRoot\..
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}
& $py "lib\stock_rally_v10\pipeline_runner.py" @args
exit $LASTEXITCODE
