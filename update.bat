@echo off
setlocal
cd /d "%~dp0"
set "DIST_DIR=%~dp0stargazer_desktop\dist"
if not exist "%DIST_DIR%" set "DIST_DIR=%~dp0dist"
echo === Stargazer update ===
echo [1/2] git pull ...
git pull --ff-only
if errorlevel 1 (echo [ERROR] git pull failed & pause & exit /b 1)
echo [2/2] download latest Stargazer.exe from GitHub Releases ...
powershell -NoProfile -Command "try { $r = Invoke-RestMethod -Uri 'https://api.github.com/repos/1075751936a-netizen/stargazer/releases/latest'; $a = $r.assets | Where-Object { $_.name -eq 'Stargazer.exe' } | Select-Object -First 1; if (-not $a) { Write-Host '[ERROR] no Stargazer.exe asset in latest release'; exit 2 }; Invoke-WebRequest -Uri $a.browser_download_url -OutFile $env:DIST_DIR'\Stargazer.exe'; Write-Host ('downloaded ' + $a.name + ' (' + [math]::Round($a.size/1MB,1) + ' MB) -> ' + $env:DIST_DIR) } catch { Write-Host $_; exit 2 }"
if errorlevel 1 (echo [ERROR] exe download failed & pause & exit /b 1)
echo === done. config and data are NOT touched. ===
pause
