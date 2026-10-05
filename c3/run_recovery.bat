@echo off
REM run_recovery.bat — 修复 windows_of 漏年后，补齐缺失年份并重训。
REM 现有 raw 已含 2019/2022/2025，这里只补 2020-2021、2023-2024（各两遍以重试瞬时失败）。
cd /d c:\Users\10757\trading\stargazer\c3_pack
set PY=.c3venv\Scripts\python.exe

%PY% -X utf8 backfill_pipeline.py --phase fetch --start 20200101 --end 20211231 --out out
%PY% -X utf8 backfill_pipeline.py --phase fetch --start 20230101 --end 20241231 --out out
REM 重试两遍，捕获上一遍未 done 的窗口（瞬时网络失败）
%PY% -X utf8 backfill_pipeline.py --phase fetch --start 20200101 --end 20211231 --out out
%PY% -X utf8 backfill_pipeline.py --phase fetch --start 20230101 --end 20241231 --out out

REM 用全量 raw 重算标签（复用 daily_cache，秒级）
%PY% -X utf8 backfill_pipeline.py --phase label --out out

REM C3 烟基线重训
%PY% -X utf8 c3_smoke_train.py --events out/event_cluster.parquet --out out
