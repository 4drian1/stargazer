"""probe_stockdb_now — 确认 stockdb HTTP 当前可达 (分钟/复权/日线)。"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from data_source_provenance import get_minute_http, get_adjustment_http, get_daily

t0 = time.time()
try:
    d = get_daily("600519")
    print(f"[probe] get_daily 600519 rows={len(d)} max_date={int(d['date'].max()) if not d.empty else None}  ({time.time()-t0:.2f}s)")
except Exception as e:
    print(f"[probe] get_daily FAIL {type(e).__name__}: {str(e)[:200]}")

t1 = time.time()
try:
    m = get_minute_http("600519", "20260916")
    print(f"[probe] get_minute 600519@20260916 rows={len(m)}  ({time.time()-t1:.2f}s)")
except Exception as e:
    print(f"[probe] get_minute FAIL {type(e).__name__}: {str(e)[:200]}")

t2 = time.time()
try:
    a = get_adjustment_http("600519")
    print(f"[probe] get_adjustment 600519 events={len(a)}  ({time.time()-t2:.2f}s)")
except Exception as e:
    print(f"[probe] get_adjustment FAIL {type(e).__name__}: {str(e)[:200]}")

print("[probe] DONE")
