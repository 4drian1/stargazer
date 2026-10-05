@echo off
title Stargazer build
echo ============================================
echo   Stargazer - Windows build script
echo ============================================
echo.

REM --- find a Python >= 3.10 ---
set PY=

REM 1) Python Install Manager runtimes: %LOCALAPPDATA%\Python\pythoncore-3.xx-64
for /d %%D in ("%LOCALAPPDATA%\Python\pythoncore-3.*-64") do (
    if exist "%%D\python.exe" set PY=%%D\python.exe
)

REM 2) py launcher with 3.12+
if not defined PY (
    py -3.14 --version >nul 2>&1
    if not errorlevel 1 set PY=py -3.14
)
if not defined PY (
    py -3.13 --version >nul 2>&1
    if not errorlevel 1 set PY=py -3.13
)
if not defined PY (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 set PY=py -3.12
)

REM 3) plain python on PATH
if not defined PY set PY=python

%PY% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No usable Python found.
    pause
    exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is too old, need 3.10+. Found:
    %PY% --version
    pause
    exit /b 1
)

echo Using interpreter: %PY%
%PY% --version
echo.

REM 1. install deps
echo [1/2] installing dependencies...
%PY% -m pip install -r requirements.txt pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)

REM 2. build onefile exe
echo [2/2] building exe...
REM resolve akshare package dir (file_fold/calendar.json must ship in-bundle)
for /f "delims=" %%i in ('%PY% -c "import akshare,os;print(os.path.dirname(akshare.__file__))"') do set AKSITE=%%i
%PY% -m PyInstaller --noconfirm --clean --onefile --windowed --name "Stargazer" --icon app.ico --add-data "web;web" --add-data "STRATEGY.md;." --add-data "config.yaml;." --add-data "data;data" --add-data "%AKSITE%/file_fold;akshare/file_fold" --collect-all py_mini_racer --hidden-import yaml --hidden-import services --hidden-import notifier --hidden-import paths --hidden-import ml_bridge --hidden-import plan_bridge --hidden-import intraday_monitor --hidden-import agent.brain --hidden-import tools.financials --hidden-import tools.lhb --hidden-import tools.market --hidden-import tools.news --hidden-import tools.research --hidden-import strategy.exits --hidden-import strategy.funnel --hidden-import strategy.oversold --hidden-import strategy.scorer --hidden-import info_panel --hidden-import psycopg --hidden-import pyarrow --hidden-import akshare.stock_feature.stock_info --hidden-import akshare.index.index_research_sw --hidden-import akshare.index.index_stock_us_sina --hidden-import akshare.futures.futures_foreign app.py
if errorlevel 1 (
    echo [ERROR] build failed, send the message above to me
    pause
    exit /b 1
)

echo.
echo ============================================
echo   DONE: dist\Stargazer.exe
echo   Double-click it to run. config.yaml and
echo   data folder will be created beside exe.
echo ============================================
pause
