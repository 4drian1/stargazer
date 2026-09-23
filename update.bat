@echo off
setlocal
cd /d "%~dp0"
set "DIST_DIR=%~dp0stargazer_desktop\dist"
if not exist "%DIST_DIR%" set "DIST_DIR=%~dp0dist"
echo === Stargazer update ===
if not exist .git (
  echo [0/2] init git repo ...
  git init -b main
  git remote add origin https://github.com/1075751936a-netizen/stargazer.git
  git fetch origin
  git reset origin/main
  git branch -M main
) else (
  echo [1/2] git pull ...
  git pull --ff-only
)
if errorlevel 1 (echo [ERROR] git sync failed & pause & exit /b 1)
echo [2/2] get latest release tag and download Stargazer.exe ...
set "TAG="
for /f "tokens=2" %%r in ('git ls-remote --tags --refs --sort=-v:refname origin 2^>nul') do (
  if not defined TAG set "TAG=%%~nxr"
)
if not defined TAG (echo [ERROR] no release tag found on remote & pause & exit /b 1)
echo latest tag: %TAG%
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'https://github.com/1075751936a-netizen/stargazer/releases/download/%TAG%/Stargazer.exe' -OutFile $env:DIST_DIR'\Stargazer.exe' } catch { Write-Host $_; exit 2 }"
if errorlevel 1 (echo [ERROR] exe download failed & pause & exit /b 1)
echo === done. config and data are NOT touched. ===
pause
