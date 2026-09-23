@echo off
setlocal
cd /d "%~dp0"
set "STARGAZER_HOME=%~dp0"
if "%STARGAZER_HOME:~-1%"=="\" set "STARGAZER_HOME=%STARGAZER_HOME:~0,-1%"
set "STOCKDB_ROOT=C:\Users\10757\trading\free-stockdb-windows-v0.3.2-more-power\stockdb"
if not exist "%STOCKDB_ROOT%\stockdb.exe" (
  echo stockdb.exe not found: %STOCKDB_ROOT%
  pause
  exit /b 1
)
start "StockDB" /min "%STOCKDB_ROOT%\stockdb.exe"
timeout /t 5 /nobreak >nul
if not exist ".runtime\Scripts\python.exe" (
  echo Run install_runtime.bat first.
  pause
  exit /b 1
)
.runtime\Scripts\python.exe stargazer_ml\v5_9_portable_refresh.py
if errorlevel 1 goto :fail
.runtime\Scripts\python.exe stargazer_ml\v5_9_live_batch.py --output-dir data\model --top-n 50 --device cpu
if errorlevel 1 goto :fail
echo Daily data and dual-strategy advice are ready. Start Stargazer.exe.
pause
exit /b 0
:fail
echo Refresh or scoring failed. Keep this window and send the error log for diagnosis.
pause
exit /b 1
