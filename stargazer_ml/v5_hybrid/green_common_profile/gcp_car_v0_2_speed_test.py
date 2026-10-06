"""CAR v0.2 扩样首批 20 条测速 (2026-09-29).

从冻结清单前 20 条候选, 抓取所需分钟日和复权链, 测实际耗时.
不新增候选, 不改状态机, 不训练; 失败保留原因, 不找替补.
缓存读取与新请求耗时分开记录.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import (get_adjustment_http,  # noqa: E402
                                    get_daily_parquet, get_minute_http)
import gcp_round_b_engine as E  # noqa: E402
import gcp_car_v0_1_labels as L  # noqa: E402

CONTRACTS = HERE / "contracts"
SNAP_CAR = HERE / "gcp_round_b_small_sample_v1" / "minute_car"
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"
ADJ_CACHE = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"
OUT = HERE / "gcp_round_b_small_sample_v1"

MAX_RETRY = 3
DISK_CAP_BYTES = 50 * 1024 * 1024


def main():
    SNAP_CAR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 读冻结清单
    cand_lines = (CONTRACTS / "car_v0_2_expansion_candidate_list.txt").read_text(encoding="utf-8").strip().split("\n")
    registry = json.loads((CONTRACTS / "car_v0_2_expansion_registry.json").read_text(encoding="utf-8"))

    # 前 20 条
    test_candidates = []
    for line in cand_lines[:20]:
        code, setup, p_str, e_str = line.split(",")
        test_candidates.append({"code": code, "setup": setup, "P": int(p_str), "E": int(e_str)})

    print("=" * 60)
    print(f"CAR v0.2 首批 {len(test_candidates)} 条测速")
    print("=" * 60)

    # 交易日历
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    # 网格
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    # 加载已有分钟缓存
    print("\n[1/4] 加载已有缓存...")
    t_cache = time.time()
    minute_cache = {}
    for snap_dir in (SNAP_OLD, SNAP_EXT, SNAP_CAR):
        if not snap_dir.exists():
            continue
        for pf in snap_dir.glob("*.parquet"):
            code = pf.stem
            try:
                m = pd.read_parquet(pf)
                if "date" in m and not m.empty:
                    days = m["date"].astype(str).str[:8].astype(int).unique()
                    for d in days:
                        sub = m[m["date"].astype(str).str[:8].astype(int) == d].copy()
                        minute_cache.setdefault((code, int(d)), []).append(sub)
            except Exception:
                pass
    t_cache_done = time.time()
    print(f"  缓存加载耗时: {t_cache_done - t_cache:.1f}s, 命中 (code,date): {len(minute_cache)}")

    # 复权缓存
    adj_data = json.loads(ADJ_CACHE.read_text(encoding="utf-8")) if ADJ_CACHE.exists() else {}

    # ---- 收集所需请求 ----
    minute_needs = set()
    adj_needs = set()
    for c in test_candidates:
        code = c["code"]
        e_date = c["E"]
        minute_needs.add((code, e_date))
        i_e = cal_idx.get(e_date, -1)
        if i_e >= 0 and i_e + 1 < len(cal):
            minute_needs.add((code, cal[i_e + 1]))
        adj_needs.add(code)

    # ---- 复权请求 ----
    print(f"\n[2/4] 复权链请求 ({len(adj_needs)} 股)...")
    t_adj = time.time()
    adj_new = 0
    adj_hit = 0
    adj_fail = 0
    for code in sorted(adj_needs):
        if code in adj_data and adj_data[code].get("ok"):
            adj_hit += 1
            continue
        # 新请求
        result = None
        for attempt in range(MAX_RETRY):
            try:
                raw = get_adjustment_http(code)
                if raw:
                    adj_data[code] = {"ok": True, "chain": raw}
                    result = "SUCCESS"
                    break
                else:
                    result = "EMPTY"
            except Exception as e:
                result = f"ERROR:{type(e).__name__}"
                if attempt < MAX_RETRY - 1:
                    time.sleep(0.5)
        if result == "SUCCESS":
            adj_new += 1
        else:
            adj_data[code] = {"ok": False, "reason": result}
            adj_fail += 1
        print(f"  {code}: {result}")
    ADJ_CACHE.write_text(json.dumps(adj_data, ensure_ascii=False), encoding="utf-8")
    t_adj_done = time.time()
    print(f"  复权耗时: {t_adj_done - t_adj:.1f}s (命中={adj_hit}, 新增={adj_new}, 失败={adj_fail})")

    # ---- 分钟请求 ----
    print(f"\n[3/4] 分钟数据请求 ({len(minute_needs)} 条)...")
    t_min = time.time()
    min_hit = 0
    min_new = 0
    min_fail = 0
    min_source_missing = 0
    disk_used = 0

    for code, date in sorted(minute_needs):
        key = (code, date)
        if key in minute_cache:
            min_hit += 1
            continue
        # 磁盘预算
        if disk_used >= DISK_CAP_BYTES:
            print(f"  DISK_CAPPED at {disk_used/1024/1024:.1f}MB")
            break
        date_str = str(date)
        result = None
        df = pd.DataFrame()
        for attempt in range(MAX_RETRY):
            try:
                df = get_minute_http(code, date_str)
                if not df.empty:
                    result = "SUCCESS"
                    break
                else:
                    result = "SOURCE_MISSING"
            except Exception as e:
                result = f"ERROR:{type(e).__name__}"
                if attempt < MAX_RETRY - 1:
                    time.sleep(0.3)
        if result == "SUCCESS":
            out_f = SNAP_CAR / f"{code}.parquet"
            if out_f.exists():
                old = pd.read_parquet(out_f)
                df = pd.concat([old, df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
            df.to_parquet(out_f, index=False)
            size = out_f.stat().st_size
            disk_used += size
            min_new += 1
            minute_cache[key] = [df[df["date"].astype(str).str[:8].astype(int) == date].copy()]
        elif result == "SOURCE_MISSING":
            min_source_missing += 1
        else:
            min_fail += 1
        print(f"  {code}@{date}: {result}  [disk={disk_used/1024/1024:.1f}MB]")
    t_min_done = time.time()
    print(f"  分钟耗时: {t_min_done - t_min:.1f}s (命中={min_hit}, 新增={min_new}, 源缺={min_source_missing}, 错误={min_fail})")

    # ---- 标签生成 ----
    print(f"\n[4/4] 标签生成 ({len(test_candidates)} 候选)...")
    t_label = time.time()
    results = []
    for c in test_candidates:
        code = c["code"]
        e_date = c["E"]
        setup = c["setup"]
        result = {"code": code, "E": e_date, "setup": setup, "signal_found": False}

        # 复权
        adj_entry = adj_data.get(code, {})
        result["adj_ok"] = adj_entry.get("ok", False)

        # E日分钟
        e_key = (code, e_date)
        if e_key not in minute_cache or not minute_cache[e_key]:
            result["replay_error"] = "E_DAY_MISSING"
            results.append(result)
            continue
        e_df = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)

        # 构建 zones
        i_e = cal_idx.get(e_date, -1)
        if i_e < 0:
            result["replay_error"] = "E_NOT_IN_CAL"
            results.append(result)
            continue
        lookback = cal[max(0, i_e - 250):i_e]
        d_sub = panel[(panel["code"].astype(str).str.zfill(6) == code)
                      & (panel["date"].astype(int).isin(lookback))].sort_values("date").reset_index(drop=True)
        if len(d_sub) < 25:
            result["replay_error"] = f"DAILY_INSUFFICIENT n={len(d_sub)}"
            results.append(result)
            continue
        atr_ref = float(E.build_atr_ref(d_sub))
        zones = E.build_zones(d_sub, atr_ref)
        ref_price = float(d_sub["close"].iloc[-1])

        # v0.2 信号检测
        from gcp_car_v0_2_extractor import detect_signal_v02
        sig = detect_signal_v02(e_df, zones, atr_ref, ref_price)
        if sig is None:
            result["replay_error"] = "NO_SIGNAL"
            results.append(result)
            continue

        result["signal_found"] = True
        result["signal_ts"] = sig["signal_ts"]
        result["sig_type"] = sig["signal_type"]

        # RECLAIMED_AT_S 不进主标签
        if sig["signal_type"] == "RECLAIMED_AT_S":
            result["reclaim_status"] = "RECLAIMED_AT_S"
            results.append(result)
            continue

        # 标签计算
        fields = L.init_car_label_fields()
        sig_idx = e_df[e_df["date"].astype(int) == sig["signal_ts"]].index
        if len(sig_idx) == 0:
            result["replay_error"] = "SIGNAL_TS_NOT_IN_DF"
            results.append(result)
            continue
        sig_idx = sig_idx[0]
        ts_arr = e_df["date"].astype(int).to_numpy()
        lo_arr = e_df["low"].astype(float).to_numpy()
        cl_arr = e_df["close"].astype(float).to_numpy()

        L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx, atr_ref, grid, e_date, 30, "30m", fields, cal)
        L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx, atr_ref, grid, e_date, 60, "60m", fields, cal)

        result["reclaim_30m"] = fields.get("future_reclaim_30m")
        result["reclaim_60m"] = fields.get("future_reclaim_60m")
        result["reclaim_status"] = fields.get("future_reclaim_status")
        result["window_complete_60m"] = fields.get("future_window_complete_60m")
        result["label_end_ts"] = fields.get("future_label_end_ts")
        results.append(result)

    t_label_done = time.time()

    # ---- 汇总 ----
    t_total = time.time() - t0
    print("\n" + "=" * 60)
    print("[测速汇总]")
    print("=" * 60)
    print(f"  总耗时: {t_total:.1f}s")
    print(f"  缓存加载: {t_cache_done - t_cache:.1f}s")
    print(f"  复权请求: {t_adj_done - t_adj:.1f}s (命中={adj_hit}, 新增={adj_new}, 失败={adj_fail})")
    print(f"  分钟请求: {t_min_done - t_min:.1f}s (命中={min_hit}, 新增={min_new}, 源缺={min_source_missing}, 错误={min_fail})")
    print(f"  标签生成: {t_label_done - t_label:.1f}s")
    print(f"  磁盘占用: {disk_used/1024/1024:.1f} MB")

    # 标签分流
    n_sig = sum(1 for r in results if r.get("signal_found"))
    n_reclaimed_at_s = sum(1 for r in results if r.get("reclaim_status") == "RECLAIMED_AT_S")
    n_eligible = sum(1 for r in results if r.get("reclaim_status") in ("RECLAIMED", "NOT_RECLAIMED"))
    n_reclaimed = sum(1 for r in results if r.get("reclaim_60m") == 1)
    n_not_reclaimed = sum(1 for r in results if r.get("reclaim_60m") == 0)
    n_no_signal = sum(1 for r in results if r.get("replay_error") == "NO_SIGNAL")
    n_missing = sum(1 for r in results if r.get("replay_error") == "E_DAY_MISSING")

    print(f"\n[标签分流]")
    print(f"  有信号: {n_sig}, RECLAIMED_AT_S: {n_reclaimed_at_s}, ELIGIBLE: {n_eligible}")
    print(f"  收回: {n_reclaimed}, 未收回: {n_not_reclaimed}")
    print(f"  无信号: {n_no_signal}, E日缺数据: {n_missing}")

    # 整批估算
    if min_new > 0:
        avg_new_req = (t_min_done - t_min) / max(min_new + min_fail + min_source_missing, 1)
        est_full_minute = avg_new_req * registry["requests"]["minute_new"]
        print(f"\n[整批估算]")
        print(f"  分钟新增请求: {registry['requests']['minute_new']} 条 × ~{avg_new_req:.1f}s/条 = ~{est_full_minute:.0f}s ({est_full_minute/60:.1f}min)")
        print(f"  复权新增请求: {registry['requests']['adj_new']} 股 × ~{(t_adj_done-t_adj)/max(adj_new+adj_fail,1):.1f}s/股")
        print(f"  标签生成: ~{(t_label_done-t_label)/len(test_candidates):.1f}s/条 × {registry['unique_candidates']} = ~{(t_label_done-t_label)/len(test_candidates)*registry['unique_candidates']/60:.1f}min")

    # 写报告
    report = {
        "test_size": len(test_candidates),
        "timing": {
            "total": round(t_total, 1),
            "cache_load": round(t_cache_done - t_cache, 1),
            "adj_request": round(t_adj_done - t_adj, 1),
            "minute_request": round(t_min_done - t_min, 1),
            "label_gen": round(t_label_done - t_label, 1),
        },
        "requests": {
            "adj_hit": adj_hit, "adj_new": adj_new, "adj_fail": adj_fail,
            "minute_hit": min_hit, "minute_new": min_new,
            "minute_source_missing": min_source_missing, "minute_fail": min_fail,
        },
        "disk_bytes": disk_used,
        "labels": {
            "signal_found": n_sig, "reclaimed_at_s": n_reclaimed_at_s,
            "eligible": n_eligible, "reclaimed": n_reclaimed,
            "not_reclaimed": n_not_reclaimed, "no_signal": n_no_signal,
            "e_day_missing": n_missing,
        },
        "per_candidate": results,
    }
    out_f = OUT / "car_v0_2_speed_test_report.json"
    out_f.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告: {out_f.name}")


if __name__ == "__main__":
    main()
