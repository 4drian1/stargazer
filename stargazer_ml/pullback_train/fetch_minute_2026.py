# -*- coding: utf-8 -*-
"""2026分钟数据增量下载入口（raw/clean 分离 + 量纲自动验收）。

目录结构（评审要求）：
  minute_cache_raw/     原始响应，追加写入，永不原地修改
  minute_cache_clean/   通过验收并做 bar 级量纲修正后的数据（标签引擎读这里）
  quarantine_2026.csv   验收不合格/量纲不确定的 (code,day)，引擎自动剔除

每次新批次写入 clean 前自动验收（评审要求）：
  r = amount/(volume*close) 的 P1/P50/P99 与中位、按(code,day)异常bar比例、
  bar-VWAP 是否落在 [low,high]；与2025正常批单位分布（P1≈0.995/P50=1.0/
  P99≈1.005）比对，异常批次进隔离区，不直接被标签引擎读取。

bar 级修正规则与 refetch_clean_2026.py 一致：
  r<0.5（100倍污染，VWAP被压100倍）→ volume/100，保留真实量纲与VWAP结构；
  r>上限（量纲无法确定）→ 剔除该bar；
  单日异常bar/越界比例>10% 或批量中位r偏离[0.9,1.1] → 整日/整批隔离。
禁止用 amount/close 反推 volume（会使VWAP机械等于收盘价，抹掉量结构）。

断点续拉：raw 已含 202608 即视为本轮拉过（9月数据未完整前以8月为完成标志）。

用法： python fetch_minute_2026.py [--workers 16]
"""
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
RAW = os.path.join(OUT_DIR, "minute_cache_raw")
CLEAN = os.path.join(OUT_DIR, "minute_cache_clean")
QUAR_PATH = os.path.join(OUT_DIR, "quarantine_2026.csv")
META_PATH = os.path.join(RAW, "meta_2026.json")
CAND = os.path.join(OUT_DIR, "minute_candidates_2026.parquet")
BASE = os.environ.get("STOCKDB_URL", "http://127.0.0.1:7899")
PREFIXES = ("202602*", "202603*", "202604*", "202605*", "202606*",
            "202607*", "202608*", "202609*")
MAX_RETRY = 3
TIMEOUT = 120
R_LO, R_HI = 0.5, 2.0      # r 正常区间（2025正常批P1=0.9945/P99=1.0048，留宽余）
DAY_BAD_MAX = 0.1          # 单日异常bar/越界比例上限，超过整日隔离
BATCH_MED_RANGE = (0.9, 1.1)  # 新批中位r允许范围，超出整批隔离
COLS = ("open", "high", "low", "close", "volume", "amount")

_lock = threading.Lock()
_meta = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_prefix(code, prefix):
    t = urllib.parse.quote(f"分钟k:{code}:{prefix}")
    with urllib.request.urlopen(f"{BASE}/?cmd=get&t={t}", timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    if isinstance(data, dict):  # 服务端过载时HTTP 200但返回{"error":...}
        raise RuntimeError(f"server error: {data}")
    return [it[1] for it in data
            if isinstance(it, list) and len(it) >= 2 and isinstance(it[1], dict)]


def r_ratio(df):
    v = pd.to_numeric(df["volume"], errors="coerce")
    a = pd.to_numeric(df["amount"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    with np.errstate(all="ignore"):
        r = a / (v * c)
    return r.replace([np.inf, -np.inf], np.nan)


def accept_batch(new):
    """量纲自动验收 + bar 级修正。

    返回 (fixed: 可入clean的DataFrame, quar_days: set, stats: dict)。
    """
    df = new.copy()
    r = r_ratio(df)
    bad100 = (r < R_LO).fillna(False)
    bad_und = (r > R_HI).fillna(False)
    df["volume"] = np.where(bad100, df["volume"] / 100, df["volume"])
    df["volume_scale_factor"] = np.where(bad100, 100, 1).astype(np.int16)
    df["volume_repaired"] = bad100.values
    df = df[~bad_und]
    df["__day"] = df["date"].astype(np.int64) // 1000000
    # 修正后再算残余异常（NaN零成交bar不算异常）
    r2 = r_ratio(df)
    bad_ratio = r2.groupby(df["__day"]).apply(
        lambda s: float(((s < R_LO) | (s > R_HI)).mean()))
    ok = (df["volume"] > 0) & (df["amount"] > 0)
    vwap = pd.Series(np.nan, index=df.index)
    vwap[ok] = df["amount"][ok] / df["volume"][ok]
    oor = ok & ((vwap < df["low"] * 0.999) | (vwap > df["high"] * 1.001))
    vwap_bad = oor.groupby(df["__day"]).mean()
    quar = set(bad_ratio[bad_ratio > DAY_BAD_MAX].index) \
        | set(vwap_bad[vwap_bad > DAY_BAD_MAX].index)
    df = df[~df["__day"].isin(quar)].drop(columns=["__day"])
    r_all = r_ratio(df)
    stats = {"n_bars": int(len(df)),
             "repaired_bars": int(bad100.sum()),
             "dropped_bars": int(bad_und.sum()),
             "quar_days": len(quar),
             "r_med": (float(np.nanmedian(r_all)) if len(df) else np.nan),
             "r_p1": (float(np.nanpercentile(r_all, 1)) if len(df) else np.nan),
             "r_p99": (float(np.nanpercentile(r_all, 99)) if len(df) else np.nan)}
    # 批量中位r偏离正常单位分布 → 整批隔离（量纲不确定）
    if len(df) and not (BATCH_MED_RANGE[0] <= stats["r_med"]
                        <= BATCH_MED_RANGE[1]):
        quar = set((new["date"].astype(np.int64) // 1000000).tolist())
        df = df.iloc[0:0]
        stats["quar_days"] = len(quar)
        stats["batch_unit_anomaly"] = True
    else:
        stats["batch_unit_anomaly"] = False
    return df, quar, stats


def fetch_code(code):
    rp = os.path.join(RAW, f"{code}.parquet")
    old = (pd.DataFrame(columns=["date"]) if not os.path.exists(rp)
           else pd.read_parquet(rp))
    have = set(old["date"].astype(np.int64).tolist()) if len(old) else set()
    if any(str(d).startswith("202608") for d in have):
        # 已含8月数据说明本轮拉过（9月数据未完整前以8月为完成标志）
        return code, "cached", 0, {}, set()
    rows, retries = [], 0
    for pf in PREFIXES:
        for attempt in range(MAX_RETRY):
            try:
                rows.extend(fetch_prefix(code, pf))
                break
            except Exception as e:  # noqa: BLE001
                retries += 1
                if attempt == MAX_RETRY - 1:
                    return code, "error", retries, \
                        {"error": f"{pf}: {type(e).__name__}"}, set()
                time.sleep(1.5 * (attempt + 1))
    new = pd.DataFrame(rows)
    if not len(new):
        return code, "no_new", retries, {}, set()
    new = new.drop_duplicates(subset=["date"])
    for c in COLS:
        if c in new.columns:
            new[c] = pd.to_numeric(new[c], errors="coerce")
    new["date"] = new["date"].astype(np.int64)
    new = new.sort_values("date")
    # 1) raw：追加写入，永不原地修改
    os.makedirs(RAW, exist_ok=True)
    raw_all = (pd.concat([old, new], ignore_index=True)
               .drop_duplicates(subset=["date"]).sort_values("date"))
    raw_all.to_parquet(rp, index=False)
    # 2) 量纲自动验收 + bar 级修正
    fixed, quar_days, stats = accept_batch(new)
    # 3) clean：剔除本次重拉的日期与隔离日后并入修正版
    cp = os.path.join(CLEAN, f"{code}.parquet")
    cold = (pd.DataFrame(columns=["date"]) if not os.path.exists(cp)
            else pd.read_parquet(cp))
    new_days = set((new["date"].astype(np.int64) // 1000000).tolist())
    if len(cold):
        for c, dv in (("volume_scale_factor", 1), ("volume_repaired", False)):
            if c not in cold.columns:
                cold[c] = dv
        mask = ~(cold["date"].astype(np.int64) // 1000000).isin(new_days)
        merged = pd.concat([cold[mask], fixed], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"])
    else:
        merged = fixed.sort_values("date")
    if quar_days and len(merged):
        merged = merged[~((merged["date"].astype(np.int64) // 1000000)
                          .isin(quar_days))]
    os.makedirs(CLEAN, exist_ok=True)
    merged.to_parquet(cp, index=False)
    status = ("unit_anomaly" if (stats["quar_days"] or stats["dropped_bars"]
                                 or stats["batch_unit_anomaly"])
              else "ok")
    return code, status, retries, stats, quar_days


def main():
    workers = 16
    args = sys.argv[1:]
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            _meta.update(json.load(f))
    cand = pd.read_parquet(CAND)
    codes = sorted(c for c in cand["code"].unique()
                   if not c.startswith(("8", "4", "92")))
    log(f"codes total={len(codes)} workers={workers}")
    t0 = time.time()
    done = 0
    quar_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_code, c): c for c in codes}
        for fut in as_completed(futs):
            code, status, retries, stats, quar_days = fut.result()
            for d in quar_days:
                quar_rows.append({"code": code, "day": int(d),
                                  "reason": "download_acceptance_fail"})
            with _lock:
                _meta[code] = {"status": status, "retries": retries,
                               **stats,
                               "pulled_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                done += 1
                if done % 100 == 0 or done == len(codes):
                    with open(META_PATH, "w", encoding="utf-8") as f:
                        json.dump(_meta, f, ensure_ascii=False)
                    el = time.time() - t0
                    log(f"{done}/{len(codes)} elapsed={el:.0f}s "
                        f"eta={(el/done)*(len(codes)-done):.0f}s")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_meta, f, ensure_ascii=False)
    # 隔离区追加（引擎 load_code 自动读取）
    if quar_rows:
        qnew = pd.DataFrame(quar_rows)
        qold = (pd.read_csv(QUAR_PATH, dtype={"code": str})
                if os.path.exists(QUAR_PATH)
                else pd.DataFrame(columns=["code", "day", "reason"]))
        qall = (pd.concat([qold, qnew], ignore_index=True)
                .drop_duplicates(["code", "day"]))
        qall.to_csv(QUAR_PATH, index=False)
        log(f"quarantine 新增 {len(qnew)} (code,day)，累计 {len(qall)}")
    st = pd.Series({k: v["status"] for k, v in _meta.items()})
    log(f"status: {st.value_counts().to_dict()}")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
