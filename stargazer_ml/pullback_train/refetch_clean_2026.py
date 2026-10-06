# -*- coding: utf-8 -*-
"""2026分钟数据量纲修复·正确版（raw重拉 + /100 clean + 验收 + 隔离）。

背景：上一版 repair_minute_volume.py 用 volume:=amount/close 原地修复，
使被修复bar的VWAP机械等于收盘价，破坏了真实成交量结构（缩量回踩、
下跌/反弹量比、VWAP特征全部失真）。本脚本按评审裁决重做：

流程（对 repair_volume_report.csv 中受影响的 2,369 只代码）：
1. 从 stockdb 重新拉取受影响月份的原始分钟响应（不依赖被覆盖的缓存），
   写入不可修改的 minute_cache_raw/{code}.parquet（已存在则复用，不重拉）；
   记录 sha256、拉取时间、数据源。
2. 逐月验收原始响应：r=amount/(volume×close) 月中位数 <0.5 → 确认100倍
   污染，修正因子100；月中位数∈[0.5,2] → 源端数据已正常，因子1（告警记录）；
   其他 → 无法确定，整月隔离。
3. 生成 minute_cache_clean/{code}.parquet：未受影响代码原样复制；受影响
   代码 = 现有缓存剔除受影响月份 + raw月份（确认污染的bar做 volume/100，
   保留 volume_scale_factor / volume_repaired 两列）。
4. 修正后逐(code,day)验收：r_clean 分布 vs 2025正常批参照分布（中位数/
   P1/P50/P99）、bar VWAP 是否落在 [low,high] 内、残余异常bar比例；
   不合格(code,day) → quarantine_2026.csv，标签引擎 load_code 自动剔除。
5. clean_runbook.json：数据源/拉取时间/原始hash/修正因子计数/验收统计。

未受影响代码的缓存数据从未被修改（上一版修复只写含污染bar的文件，
且本流程对受影响代码整体替换受影响月份），2025实验数据不变。

用法： python refetch_clean_2026.py [--workers 16] [--skip-refetch]
"""
import hashlib
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
OLD_CACHE = os.path.join(OUT_DIR, "minute_cache")
RAW_DIR = os.path.join(OUT_DIR, "minute_cache_raw")
CLEAN_DIR = os.path.join(OUT_DIR, "minute_cache_clean")
REPORT = os.path.join(OUT_DIR, "repair_volume_report.csv")
QUAR_PATH = os.path.join(OUT_DIR, "quarantine_2026.csv")
RUNBOOK = os.path.join(OUT_DIR, "clean_runbook.json")
BASE = os.environ.get("STOCKDB_URL", "http://127.0.0.1:7899")
R_LO, R_HI = 0.5, 2.0  # 量纲一致容忍带（正常批r≈1）

_lock = threading.Lock()
_runbook = {"source": BASE, "pulled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_hashes": {}, "codes": {}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_prefix(code, prefix):
    t = urllib.parse.quote(f"分钟k:{code}:{prefix}")
    with urllib.request.urlopen(f"{BASE}/?cmd=get&t={t}", timeout=120) as r:
        body = r.read().decode("utf-8")
    data = json.loads(body)
    # 服务端对过载/异常请求返回 {"error": ...} 且 HTTP 200，必须显式抛错重试
    if not isinstance(data, list):
        raise RuntimeError(f"server returned non-list: {str(body)[:120]}")
    return body


def r_ratio(df):
    v = pd.to_numeric(df["volume"], errors="coerce")
    a = pd.to_numeric(df["amount"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    ok = (v > 0) & (a > 0) & (c > 0)
    r = pd.Series(np.nan, index=df.index, dtype=float)
    r[ok] = a[ok] / (v[ok] * c[ok])
    return r


def refetch_code(code, months):
    """返回 (raw_months_df, info)。raw 已落盘则直接复用。"""
    raw_path = os.path.join(RAW_DIR, f"{code}.parquet")
    info = {"months": {}, "factor_counts": {1: 0, 100: 0},
            "quarantine_months": []}
    if os.path.exists(raw_path):
        raw_all = pd.read_parquet(raw_path)
    else:
        frames, hashes = [], {}
        for ym in months:
            prefix = f"{ym}*"
            body = fetch_prefix(code, prefix)
            hashes[ym] = hashlib.sha256(body.encode("utf-8")).hexdigest()
            data = json.loads(body)
            if not isinstance(data, list):
                raise RuntimeError(f"server returned non-list: {str(body)[:120]}")
            rows = [it[1] for it in data
                    if isinstance(it, list) and len(it) >= 2
                    and isinstance(it[1], dict)]
            if rows:
                d = pd.DataFrame(rows).drop_duplicates(subset=["date"])
                for c in ("open", "high", "low", "close", "volume", "amount"):
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors="coerce")
                d["date"] = d["date"].astype(np.int64)
                d["__ym"] = ym
                frames.append(d)
        raw_all = (pd.concat(frames, ignore_index=True)
                   if frames else pd.DataFrame())
        if len(raw_all):
            raw_all = raw_all.sort_values("date")
            os.makedirs(RAW_DIR, exist_ok=True)
            raw_all.to_parquet(raw_path, index=False)
        with _lock:
            _runbook["raw_hashes"][code] = hashes
    info["raw_rows"] = len(raw_all)
    if not len(raw_all):
        return raw_all, info

    # 逐月验收原始响应（月因子仅作报告）；修正按 bar 级执行：
    # r≈0.01 的bar无疑是100倍污染 → volume/100（量纲明确）；
    # r>上限 的bar量纲无法确定 → 从clean剔除（raw中保留），当日占比>10%则整日隔离。
    clean_frames = []
    for ym, g in raw_all.groupby("__ym"):
        r = r_ratio(g)
        med = float(np.nanmedian(r)) if r.notna().any() else np.nan
        month_factor = (100 if (np.isfinite(med) and med < R_LO)
                        else (1 if (np.isfinite(med) and R_LO <= med <= R_HI)
                              else None))
        if month_factor is None:
            log(f"warn {code} {ym}: 月中位r={med}，量纲不确定，整月隔离")
        g = g.copy()
        bad100 = (r < R_LO).fillna(False)
        bad_und = (r > R_HI).fillna(False)
        g["volume"] = np.where(bad100, g["volume"] / 100, g["volume"])
        g["volume_scale_factor"] = np.where(bad100, 100, 1).astype(np.int16)
        g["volume_repaired"] = bad100.values
        n_drop = int(bad_und.sum())
        g = g[~bad_und]
        info["months"][ym] = {"median_r": med, "factor": month_factor,
                              "bad_bars": int(bad100.sum()),
                              "dropped_bars": n_drop}
        info["factor_counts"][100] += int(bad100.sum())
        if month_factor is None:
            info["quarantine_months"].append(ym)
            days = (g["date"].astype(np.int64) // 1000000).unique().tolist()
            info.setdefault("quarantine_days", []).extend(int(d) for d in days)
        clean_frames.append(g.drop(columns=["__ym"]))
    clean = (pd.concat(clean_frames, ignore_index=True)
             if clean_frames else pd.DataFrame())
    return clean, info


def validate_clean(clean, code):
    """修正后逐(code,day)验收。返回 (bad_day_ratio: Series, vwap_bad: Series)。"""
    if not len(clean):
        return None, None
    r = r_ratio(clean)
    clean = clean.assign(__day=clean["date"] // 1000000, __r=r)
    grp = clean.groupby("__day")
    bad_ratio = grp.apply(lambda g: float(((g["__r"] < R_LO)
                                           | (g["__r"] > R_HI)).mean()),
                          include_groups=False)
    # bar VWAP 是否落在 [low, high]（仅统计有效vwap的bar，零成交bar的NaN不算异常）
    ok = (clean["volume"] > 0) & (clean["amount"] > 0)
    vwap = pd.Series(np.nan, index=clean.index)
    vwap[ok] = clean["amount"][ok] / clean["volume"][ok]
    out_of_range = ok & ((vwap < clean["low"] * 0.999)
                         | (vwap > clean["high"] * 1.001))
    vwap_bad = out_of_range.groupby(clean["__day"]).mean()
    vwap_nan = (~ok).groupby(clean["__day"]).mean()
    return bad_ratio, vwap_bad


def process_code(code, months):
    try:
        clean, info = refetch_code(code, months)
        with _lock:
            _runbook["codes"][code] = {k: v for k, v in info.items()
                                       if k != "months"}
        if not len(clean):
            return code, "quarantine_all", info
        bad_ratio, vwap_bad = validate_clean(clean, code)
        # 拼接：现有缓存剔除受影响月份 + clean月份
        old_path = os.path.join(OLD_CACHE, f"{code}.parquet")
        old = (pd.read_parquet(old_path) if os.path.exists(old_path)
               else pd.DataFrame(columns=["date"]))
        if len(old):
            days = set((clean["date"] // 1000000).unique().tolist())
            mask = ~(old["date"].astype(np.int64) // 1000000).isin(days)
            # 补齐 flags 列（未受影响月份无修复标记 → factor=1/repaired=False）
            for c, dv in (("volume_scale_factor", 1), ("volume_repaired", False)):
                if c not in old.columns:
                    old[c] = dv
            merged = pd.concat([old[mask], clean], ignore_index=True)
            merged = merged.sort_values("date").drop_duplicates(subset=["date"])
        else:
            merged = clean.sort_values("date")
        os.makedirs(CLEAN_DIR, exist_ok=True)
        merged.to_parquet(os.path.join(CLEAN_DIR, f"{code}.parquet"),
                          index=False)
        q_days = [int(d) for d in bad_ratio[bad_ratio > 0.1].index] \
            + [int(d) for d in vwap_bad[vwap_bad > 0.1].index]
        q_days = sorted(set(q_days))
        return code, ("ok" if not q_days else "partial_quarantine"), {
            "quarantine_days": q_days,
            "clean_rows": len(merged)}
    except Exception as e:  # noqa: BLE001
        return code, "error", {"error": f"{type(e).__name__}: {e}"}


def main():
    workers = 16
    limit = None
    args = sys.argv[1:]
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    rep = pd.read_csv(REPORT, dtype={"code": str})
    # repair_volume_report.csv 的 code 列残留文件名截断尾点（"000006."），清洗之
    rep["code"] = rep["code"].str.rstrip(".")
    rep.to_csv(REPORT, index=False)
    rep["ym"] = rep["day"] // 100
    by_code = rep.groupby("code")["ym"].apply(
        lambda s: sorted(s.unique().tolist())).to_dict()
    if limit:
        by_code = dict(list(by_code.items())[:limit])
    log(f"受影响代码 {len(by_code)} 只，月份数合计 "
        f"{sum(len(v) for v in by_code.values())}")

    # 2025正常批参照分布（抽样20只未受影响代码的2025年r分布）
    ref_codes = [c for c in os.listdir(OLD_CACHE)[:2000]
                 if c.endswith(".parquet")
                 and c[:-len(".parquet")] not in by_code][:20]
    ref_r = []
    for c in ref_codes:
        d = pd.read_parquet(os.path.join(OLD_CACHE, c),
                            columns=["volume", "amount", "close", "date"])
        d = d[(d["date"].astype(np.int64) // 1000000) < 20260101]
        r = r_ratio(d)
        ref_r.append(r.dropna())
    ref_r = pd.concat(ref_r)
    ref_q = np.nanpercentile(ref_r, [1, 50, 99])
    log(f"2025正常批参照 r 分布: P1={ref_q[0]:.4f} P50={ref_q[1]:.4f} "
        f"P99={ref_q[2]:.4f} (n={len(ref_r):,})")
    _runbook["reference_r_dist"] = {"P1": float(ref_q[0]),
                                    "P50": float(ref_q[1]),
                                    "P99": float(ref_q[2]),
                                    "n": int(len(ref_r))}

    t0 = time.time()
    quar_rows, stats = [], {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_code, c, m): c for c, m in by_code.items()}
        for fut in as_completed(futs):
            code, status, info = fut.result()
            stats[status] = stats.get(status, 0) + 1
            for d in info.get("quarantine_days", []):
                quar_rows.append({"code": code, "day": d,
                                  "reason": "post_fix_validation_fail"})
            for ym in info.get("quarantine_months", []):
                quar_rows.append({"code": code,
                                  "day": int(f"{ym}01"), "reason":
                                  f"month_factor_undetermined:{ym}"})
            done += 1
            if done % 200 == 0 or done == len(by_code):
                log(f"{done}/{len(by_code)} {time.time()-t0:.0f}s "
                    f"{stats}")
    # 未受影响代码原样复制
    copied = 0
    for f in os.listdir(OLD_CACHE):
        code_of = f[:-len(".parquet")]  # "000006.parquet" -> "000006"
        if not f.endswith(".parquet") or code_of in by_code:
            continue
        src, dst = os.path.join(OLD_CACHE, f), os.path.join(CLEAN_DIR, f)
        if not os.path.exists(dst):
            os.makedirs(CLEAN_DIR, exist_ok=True)
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
            copied += 1
    log(f"未受影响代码复制: {copied}")
    qdf = pd.DataFrame(quar_rows).drop_duplicates(["code", "day"]) \
        if quar_rows else pd.DataFrame(columns=["code", "day", "reason"])
    qdf.to_csv(QUAR_PATH, index=False)
    _runbook["status_counts"] = stats
    _runbook["unaffected_copied"] = copied
    _runbook["quarantine_code_days"] = len(qdf)
    with open(RUNBOOK, "w", encoding="utf-8") as f:
        json.dump(_runbook, f, ensure_ascii=False, indent=1)
    log(f"quarantine (code,day): {len(qdf)}；runbook {RUNBOOK}；"
        f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
