@echo off
setlocal
cd /d "%~dp0"
set "DIST_DIR=%~dp0stargazer_desktop\dist"
if not exist "%DIST_DIR%" set "DIST_DIR=%~dp0dist"
echo === Stargazer update ===
if not exist .git git init -b main
git remote get-url origin >nul 2>nul
if errorlevel 1 git remote add origin https://github.com/1075751936a-netizen/stargazer.git
set "SYNC_OK="
for /l %%i in (1,1,3) do (
  echo [1/2] git fetch origin ... (attempt %%i)
  git fetch origin && set "SYNC_OK=1" && goto synced
  echo retry in 5s ...
  ping -n 6 127.0.0.1 >nul
)
goto syncfail
:synced
git branch -M main
git reset origin/main
if errorlevel 1 goto syncfail
echo [1/2] git sync done
goto step2
:syncfail
echo [ERROR] git sync failed (network to github.com unstable; check proxy/VPN and rerun)
pause
exit /b 1
:step2
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
