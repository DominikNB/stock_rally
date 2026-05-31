# KI-Analyse neu (Gemini) + docs/index.html KI-Block patchen — ohne vollen Pipeline-Lauf.
Set-Location $PSScriptRoot\..
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}
& $py "scripts\run_website_analysis_gemini.py" @args
exit $LASTEXITCODE
