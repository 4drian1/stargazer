cd c:\Users\10757\trading\stargazer\c3_pack
$py = ".\.c3venv\Scripts\python.exe"
& $py -X utf8 backfill_pipeline.py --phase fetch --start 20200101 --end 20211231 --out out
& $py -X utf8 backfill_pipeline.py --phase fetch --start 20230101 --end 20241231 --out out
# 重试两遍，捕获上一遍未 done 的窗口（瞬时网络失败）
& $py -X utf8 backfill_pipeline.py --phase fetch --start 20200101 --end 20211231 --out out
& $py -X utf8 backfill_pipeline.py --phase fetch --start 20230101 --end 20241231 --out out
# 用全量 raw 重算标签（复用 daily_cache）
& $py -X utf8 backfill_pipeline.py --phase label --out out
# C3 烟基线重训
& $py -X utf8 c3_smoke_train.py --events out/event_cluster.parquet --out out
