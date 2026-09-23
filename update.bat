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
echo [2/2] get latest release and download Stargazer.exe ...
powershell -NoProfile -Command "$creds = 'url=https://github.com' | git credential fill 2>$null; $tok = ($creds | Select-String '^password=').Line -replace '^password='; if (-not $tok) { Write-Host '[ERROR] no git credential for github.com'; exit 2 }; $h = @{ Authorization = ('Bearer ' + $tok); 'Accept' = 'application/vnd.github+json' }; try { $rel = Invoke-RestMethod -Headers $h -Uri 'https://api.github.com/repos/1075751936a-netizen/stargazer/releases/latest'; $asset = $rel.assets | Where-Object { $_.name -eq 'Stargazer.exe' } | Select-Object -First 1; if (-not $asset) { Write-Host '[ERROR] Stargazer.exe not in latest release'; exit 2 }; Invoke-WebRequest -Headers (@{ Authorization = ('Bearer ' + $tok); Accept = 'application/octet-stream' }) -Uri $asset.url -OutFile $env:DIST_DIR'\Stargazer.exe'; Write-Host ('downloaded ' + $rel.tag_name + ' -> ' + $env:DIST_DIR) } catch { Write-Host $_; exit 2 }"
if errorlevel 1 (echo [ERROR] exe download failed & pause & exit /b 1)
echo === done. config and data are NOT touched. ===
pause
