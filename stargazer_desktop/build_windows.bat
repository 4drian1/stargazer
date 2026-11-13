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
"%PY%" -B -X utf8 ..\release_audit_v1\replay_closure_package_v1.py dist_candidate\Stargazer.exe ..\patrol_repair_20261111_v1\test_closure_failures_v1.py
if errorlevel 1 exit /b 1
"%PY%" -B -X utf8 ..\release_audit_v1\replay_closure_package_v1.py dist_candidate\Stargazer.exe ..\patrol_repair_20261111_v1\test_flow_recovery_v1.py
if errorlevel 1 exit /b 1
"%PY%" -B -X utf8 ..\release_audit_v1\replay_closure_package_v1.py dist_candidate\Stargazer.exe ..\patrol_repair_20261111_v1\test_pool_restart_recovery_v219.py
if errorlevel 1 exit /b 1
"%PY%" -B -X utf8 ..\release_audit_v1\replay_closure_package_v1.py dist_candidate\Stargazer.exe ..\patrol_repair_20261111_v1\test_live_patrol_v220.py
if errorlevel 1 exit /b 1
echo Candidate verified: dist_candidate\Stargazer.exe
echo Install only after runtime self-checks pass and Stargazer is closed.
