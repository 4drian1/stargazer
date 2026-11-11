@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\_dev_venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Project Python is missing. Restore the verified environment first.
    exit /b 1
)
"%PY%" -B -X utf8 ..\release_audit_v1\verify_live_release.py
if errorlevel 1 exit /b 1
REM Build a candidate only. Never overwrite a running production application.
"%PY%" -B -X utf8 -m PyInstaller --noconfirm --distpath dist_candidate Stargazer_demo_safe.spec
if errorlevel 1 exit /b 1
"%PY%" -B -X utf8 ..\release_audit_v1\verify_frozen_package_v1.py dist_candidate\Stargazer.exe
if errorlevel 1 exit /b 1
echo Candidate verified: dist_candidate\Stargazer.exe
echo Install only after runtime self-checks pass and Stargazer is closed.
