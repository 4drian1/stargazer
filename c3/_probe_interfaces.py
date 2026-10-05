"""接口实测探针（一次性，不入交付管线）：
A. akshare 巨潮个股公告接口：列名/分页/速率/参数行为
B. stockdb HTTP：universe 取法、日k 字段（市值/换手有无）、单日全市场行数
"""
from __future__ import annotations
import json
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:7899"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_get_raw(table, code="*", date="*", timeout=120):
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{BASE}/?cmd=get&t={t}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener.open(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        import msgpack
        return msgpack.unpackb(raw, raw=False, strict_map_key=False)


def recs_of(obj):
    if isinstance(obj, list):
        out = [it[1] for it in obj if isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], dict)]
        if not out and obj and isinstance(obj[0], dict):
            out = obj
        return out
    return []


def probe_stockdb():
    print("=== B1. 单日全市场 日k (universe 候选) ===")
    t0 = time.time()
    obj = http_get_raw("日k", "*", "20260916", timeout=300)
    recs = recs_of(obj)
    codes = sorted({str(r.get("code")) for r in recs})
    print(f"  {time.time()-t0:.1f}s rows={len(recs)} symbols={len(codes)} first={codes[:3]}")
    print(f"  cols={sorted(recs[0].keys())}")
    print(f"  sample={ {k: recs[0][k] for k in sorted(recs[0].keys())} }")

    print("\n=== B2. 单票日k全历史 ===")
    t0 = time.time()
    recs2 = recs_of(http_get_raw("日k", "600519", "*"))
    dates = [r.get("date") for r in recs2]
    print(f"  {time.time()-t0:.1f}s rows={len(recs2)} date=[{min(dates)},{max(dates)}]")
    print(f"  cols={sorted(recs2[0].keys())}")

    print("\n=== B3. 还有哪些表？ ===")
    for tbl in ["表", "list", "tables", "info"]:
        try:
            obj = http_get_raw(tbl, "*", "*", timeout=10)
            s = str(obj)[:300]
            print(f"  t={tbl}: {s}")
        except Exception as e:
            print(f"  t={tbl}: FAIL {type(e).__name__} {str(e)[:80]}")


def probe_cninfo():
    print("\n=== A1. akshare 巨潮个股公告 ===")
    import akshare as ak
    t0 = time.time()
    df = ak.stock_zh_a_disclosure_report_cninfo(
        symbol="600519", market="沪深京", start_date="20240101", end_date="20241231", keyword="")
    dt = time.time() - t0
    print(f"  600519 2024全年: {dt:.1f}s rows={len(df)}")
    print(f"  cols={list(df.columns)}")
    print(df.head(3).to_string())
    print(f"  dtypes:\n{df.dtypes}")

    print("\n=== A2. 半年窗口 + 另一票 ===")
    for code, s, e in [("000001", "20240101", "20240630"), ("300750", "20240101", "20241231")]:
        t0 = time.time()
        d = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code, market="沪深京", start_date=s, end_date=e, keyword="")
        print(f"  {code} {s}~{e}: {time.time()-t0:.1f}s rows={len(d)}")
        if len(d):
            print(f"    cols={list(d.columns)}")
            print(f"    first row: {d.iloc[0].to_dict()}")

    print("\n=== A3. 速率实测：10 次连发 ===")
    t0 = time.time()
    for i in range(10):
        ak.stock_zh_a_disclosure_report_cninfo(
            symbol="600519", market="沪深京", start_date="20240101", end_date="20240131", keyword="")
    print(f"  10 reqs: {time.time()-t0:.1f}s -> {(time.time()-t0)/10:.2f}s/req")


if __name__ == "__main__":
    probe_stockdb()
    probe_cninfo()
