@echo off
setlocal

REM 本地刷新+打分脚本：收盘后更新日线并生成本地模型建议

cd /d "D:\stargazer"

set "STARGAZER_HOME=D:\stargazer"
set "STOCKDB_ROOT=D:\a股数据\free-stockdb-win\stockdb"

if not exist "%STOCKDB_ROOT%\stockdb.exe" (
  echo stockdb.exe not found: %STOCKDB_ROOT%
  echo 请修改本脚本里的 STOCKDB_ROOT 为你本机的 free-stockdb 路径
  pause
  exit /b 1
)

start "StockDB" /min "%STOCKDB_ROOT%\stockdb.exe"
timeout /t 5 /nobreak >nul

if not exist "%STARGAZER_HOME%\.venv\Scripts\python.exe" (
  echo 本地虚拟环境不存在: %STARGAZER_HOME%\.venv
  pause
  exit /b 1
)

"%STARGAZER_HOME%\.venv\Scripts\python.exe" stargazer_ml\v5_9_portable_refresh.py
if errorlevel 1 goto :fail

"%STARGAZER_HOME%\.venv\Scripts\python.exe" stargazer_ml\v5_9_live_batch.py --output-dir data\model --top-n 50 --device cpu
if errorlevel 1 goto :fail

echo 数据与本地建议已更新，可以启动 Stargazer.exe 了。
pause
exit /b 0

:fail
echo 刷新或打分失败，请保留窗口并把错误信息发给我。
pause
exit /b 1
