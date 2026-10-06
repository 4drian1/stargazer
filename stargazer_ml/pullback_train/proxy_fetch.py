# -*- coding: utf-8 -*-
"""代理篮子成分分钟数据补拉（走新管线：raw只读+量纲验收+bar级/100修正）。

用法: python proxy_fetch.py proxy_missing_codes.json [--workers 12]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location("fm",
                                                os.path.join(
                                                    os.path.dirname(os.path.abspath(__file__)),
                                                    "fetch_minute_2026.py"))
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)

PREFIXES = ("2025*", "202601*", "202602*", "202603*", "202604*", "202605*",
            "202606*", "202607*", "202608*", "202609*")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_code(code):
    rp = os.path.join(fm.RAW, f"{code}.parquet")
    old = (pd.DataFrame(columns=["date"]) if not os.path.exists(rp)
           else pd.read_parquet(rp))
    rows, retries = [], 0
    for pf in PREFIXES:
        for attempt in range(fm.MAX_RETRY):
            try:
                rows.extend(fm.fetch_prefix(code, pf))
                break
            except Exception:  # noqa: BLE001
                retries += 1
                if attempt == fm.MAX_RETRY - 1:
                    return code, "error", {"retries": retries}
                time.sleep(1.5 * (attempt + 1))
    new = pd.DataFrame(rows)
    if not len(new):
        return code, "no_data", {"retries": retries}
    new = new.drop_duplicates(subset=["date"])
    for c in fm.COLS:
        if c in new.columns:
            new[c] = pd.to_numeric(new[c], errors="coerce")
    new["date"] = new["date"].astype(np.int64)
    new = new.sort_values("date")
    os.makedirs(fm.RAW, exist_ok=True)
    raw_all = (pd.concat([old, new], ignore_index=True)
               .drop_duplicates(subset=["date"]).sort_values("date"))
    raw_all.to_parquet(rp, index=False)
    fixed, quar_days, stats = fm.accept_batch(new)
    cp = os.path.join(fm.CLEAN, f"{code}.parquet")
    cold = (pd.DataFrame(columns=["date"]) if not os.path.exists(cp)
            else pd.read_parquet(cp))
    if len(cold):
        for c, dv in (("volume_scale_factor", 1), ("volume_repaired", False)):
            if c not in cold.columns:
                cold[c] = dv
        mask = ~(cold["date"].astype(np.int64) // 1000000).isin(
            set((new["date"] // 1000000).tolist()))
        merged = pd.concat([cold[mask], fixed], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"])
    else:
        merged = fixed.sort_values("date")
    if quar_days and len(merged):
        merged = merged[~((merged["date"].astype(np.int64) // 1000000)
                          .isin(quar_days))]
    os.makedirs(fm.CLEAN, exist_ok=True)
    merged.to_parquet(cp, index=False)
    status = "unit_anomaly" if (stats["quar_days"] or stats["dropped_bars"]
                                or stats["batch_unit_anomaly"]) else "ok"
    return code, status, {"retries": retries, **stats,
                          "rows": len(merged)}


def main():
    workers = 12
    args = sys.argv[1:]
    codes = json.load(open(args[0], encoding="utf-8"))
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    log(f"codes={len(codes)} workers={workers}")
    stats, done = {}, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_code, c): c for c in codes}
        for fut in as_completed(futs):
            code, status, info = fut.result()
            stats[status] = stats.get(status, 0) + 1
            done += 1
            if status == "unit_anomaly":
                log(f"warn {code}: {info}")
            if done % 25 == 0 or done == len(codes):
                log(f"{done}/{len(codes)} {time.time()-t0:.0f}s {stats}")
    log(f"status: {stats}")


if __name__ == "__main__":
    main()
