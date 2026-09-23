@echo off
setlocal
cd /d "%~dp0"
set "PY="
py -3.13 --version >nul 2>&1 && set "PY=py -3.13"
if not defined PY py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY (
  echo Python 3.11-3.13 not found. Install 64-bit Python, then run this file again.
  pause
  exit /b 1
)
%PY% -m venv .runtime
if errorlevel 1 goto :fail
.runtime\Scripts\python.exe -m pip install --upgrade pip
.runtime\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
.runtime\Scripts\python.exe -m pip install -r requirements_model.txt
if errorlevel 1 goto :fail
echo Runtime installation complete.
pause
exit /b 0
:fail
echo Runtime installation failed. Check the network and error above.
pause
exit /b 1
