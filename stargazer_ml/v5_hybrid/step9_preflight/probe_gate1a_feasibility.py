"""Gate 1A Final Verification - 可行性探活 A。

目的：判断"全库聚合 QA"是否可用 bulk-by-date 查询实现，以及本地是否有分钟 parquet。
- A1: 分钟 bulk-by-date (分钟k:*:YYYYMMDD) 是否返回当日全部股票, 单日耗时/体量
- A2: 本地是否存在分钟 parquet 缓存 (若有则全库扫描极易)
- A3: 复权 bulk (复权:*:*) 是否一次返回全库事件
"""
import time
import glob
import os
import data_availability_audit as a

print("== A1: bulk minute by date ==")
t = time.time()
recs = a.http_get("分钟k", "*", "20260916", timeout=300)
el = round(time.time() - t, 1)
print("elapsed_s", el, "type", type(recs))
if isinstance(recs, list):
    print("n_records", len(recs))
    if recs:
        e0 = recs[0]
        print("elem0 type", type(e0), "keys", (list(e0.keys()) if isinstance(e0, dict) else e0))
        # 找 symbol/code 字段
        if isinstance(e0, dict):
            for k in e0:
                if k.lower() in ("code", "symbol", "sym"):
                    vals = set()
                    for r in recs[:200000]:
                        if isinstance(r, dict) and r.get(k) is not None:
                            vals.add(r.get(k))
                    print(f"  field={k} distinct_in_first_200k={len(vals)}")
elif isinstance(recs, dict) and "__error__" in recs:
    print("ERROR", recs["__error__"])

print("== A2: minute parquet cache ==")
roots = [r"D:/a股数据", r"D:/a股数据/free-stockdb-win/stockdb", r"D:/a股数据/data_daily_parts"]
for root in roots:
    parts = glob.glob(os.path.join(root, "**", "*minute*.parquet"), recursive=True)
    print(f"  {root}: minute parquet = {len(parts)}")

print("== A3: bulk adjustment ==")
t = time.time()
adj = a.http_get("复权", "*", "*", timeout=120)
el = round(time.time() - t, 1)
print("elapsed_s", el, "type", type(adj),
      "len", (len(adj) if hasattr(adj, "__len__") else "n/a"))
if isinstance(adj, list) and adj:
    e0 = adj[0]
    print("elem0", (list(e0.keys()) if isinstance(e0, dict) else str(e0))[:200])
elif isinstance(adj, dict) and "__error__" in adj:
    print("ERROR", adj["__error__"])
