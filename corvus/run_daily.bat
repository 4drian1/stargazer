@echo off
cd /d D:\stargazer\corvus
set PYTHONIOENCODING=utf-8
set FRED_API_KEY=dda54fa3f47e310c9dc89331702c64e8
D:\stargazer\_dev_venv\Scripts\python.exe -m corvus.daily_run --mode all >> data\health\daily_cron.log 2>&1
