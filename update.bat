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
powershell -NoProfile -Command "$urls = @('%TAG%/Stargazer.exe', '%TAG%/stargazer_test', '%TAG%/stargazer_test.exe'); $ok = $false; foreach ($u in $urls) { try { Invoke-WebRequest -Uri ('https://github.com/1075751936a-netizen/stargazer/releases/download/' + $u) -OutFile $env:DIST_DIR'\Stargazer.exe'; Write-Host ('downloaded ' + $u); $ok = $true; break } catch { Write-Host ('miss: ' + $u) } }; if (-not $ok) { exit 2 }"
if errorlevel 1 (echo [ERROR] exe download failed & pause & exit /b 1)
echo === done. config and data are NOT touched. ===
pause
