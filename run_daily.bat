@echo off
setlocal EnableExtensions

REM ──────────────────────────────────────────────────────────────────────────────
REM  run_daily.bat  —  täglich ausführen via Windows Task Scheduler
REM  Führt stock_rally_v10.ipynb aus und pusht docs/ nach GitHub Pages.
REM ──────────────────────────────────────────────────────────────────────────────

set PROJECT_DIR=C:\Python projects\stock_rally
set VENV_ACTIVATE=%PROJECT_DIR%\.venv\Scripts\activate.bat
set NOTEBOOK=%PROJECT_DIR%\stock_rally_v10.ipynb
set LOGFILE=%PROJECT_DIR%\run_daily.log

echo. >> "%LOGFILE%"
echo ========================================= >> "%LOGFILE%"
echo Run started: %DATE% %TIME% >> "%LOGFILE%"
echo ========================================= >> "%LOGFILE%"

REM Activate virtual environment
call "%VENV_ACTIVATE%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate venv >> "%LOGFILE%"
    exit /b 1
)

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Execute notebook (timeout 3 hours)
echo Executing notebook... >> "%LOGFILE%"
jupyter nbconvert --to notebook --execute ^
    --ExecutePreprocessor.timeout=10800 ^
    --inplace ^
    "%NOTEBOOK%" >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Notebook execution failed >> "%LOGFILE%"
    exit /b 1
)

echo Notebook done. Pushing docs/ to GitHub... >> "%LOGFILE%"

REM Commit and push docs/
git add docs/
git diff --cached --quiet
if %ERRORLEVEL% NEQ 0 (
    git commit -m "Daily signals %DATE%"
    git push >> "%LOGFILE%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: git push failed >> "%LOGFILE%"
        exit /b 1
    )
    echo Push successful. >> "%LOGFILE%"
) else (
    echo No changes in docs/ to push. >> "%LOGFILE%"
)

echo Run finished: %DATE% %TIME% >> "%LOGFILE%"
endlocal
