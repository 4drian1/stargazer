@echo off
REM prepare_env.bat - C3 pack runtime env (monitoring machine)
REM Pure ASCII. Uses .runtime\Scripts\python.exe if present (already has
REM torch/pandas); otherwise creates .c3venv with py -3.13.
setlocal
cd /d "%~dp0"

set "PY=.runtime\Scripts\python.exe"
if not exist "%PY%" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] no .runtime venv and no py launcher found.
        echo Install Python 3.13 or provide .runtime venv.
        exit /b 1
    )
    echo [INFO] .runtime not found, creating .c3venv with py -3.13 ...
    py -3.13 -m venv .c3venv || exit /b 1
    set "PY=.c3venv\Scripts\python.exe"
)

echo [INFO] using python: %PY%
"%PY%" -m pip install --upgrade pip || exit /b 1
"%PY%" -m pip install akshare lightgbm scikit-learn pyarrow msgpack || exit /b 1
"%PY%" -c "import akshare, lightgbm, sklearn, pyarrow, msgpack, pandas, numpy; print('[OK] env ready')"
if errorlevel 1 exit /b 1
echo [DONE] now run: backfill then training, see README.md
