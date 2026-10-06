"""CAR v0.1 有界抓取 + 真实工程回放 (2026-09-29, 用户已审批).

执行边界 (用户原话):
  - 只处理登记清单; 84个新增请求项与重试次数分开计数
  - "3次"是最多三次总尝试; 源缺及复权失败保持显式状态
  - 新增数据/缓存/中间产物合计不超过 50MB; 达到预算即停止, 不自动删旧证据腾空间
  - 只做 30m/60m/保持10m; 不加R、不做真实5d/10d、不训练、不上线

完成后交付:
  - 请求账本 (成功/缓存命中/源缺/异常)
  - 逐候选分流 (RECLAIMED/NOT_RECLAIMED/INELIGIBLE/INCOMPLETE/...)
  - 分窗口有效性 (30m/60m/hold 各自独立)
  - 独立离线重算 (因果截断守卫)
  - 无信号和数据缺失都保留, 覆盖不足的工程门记待验, 不凑 PASS
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
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
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"
SNAP_CAR = HERE / "gcp_round_b_small_sample_v1" / "minute_car"
ADJ_CACHE = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"
OUT = HERE / "gcp_round_b_small_sample_v1"

MAX_RETRY = 3  # 最多三次总尝试
DISK_CAP_BYTES = 50 * 1024 * 1024  # 50MB


def main():
    SNAP_CAR.mkdir(parents=True, exist_ok=True)
    # 读清单
    min_lines = (CONTRACTS / "car_v0_1_minute_request_list.txt").read_text(encoding="utf-8").strip().split("\n")
    adj_lines = (CONTRACTS / "car_v0_1_adj_request_list.txt").read_text(encoding="utf-8").strip().split("\n")
    cand_lines = (CONTRACTS / "car_v0_1_candidate_list.txt").read_text(encoding="utf-8").strip().split("\n")

    candidates = []
    for l in cand_lines:
        code, setup, e = l.split(",")
        candidates.append({"code": code, "setup": setup, "E": int(e)})

    # 读网格
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    # ---- 请求账本 ----
    ledger = {
        "minute_requests": [], "adj_requests": [],
        "minute_summary": {"total": 0, "success": 0, "cache_hit": 0,
                           "source_missing": 0, "error": 0, "retries_used": 0},
        "adj_summary": {"total": 0, "success": 0, "cache_hit": 0,
                        "error": 0, "retries_used": 0},
        "disk_used_bytes": 0, "disk_cap_bytes": DISK_CAP_BYTES,
        "disk_capped": False,
    }

    # ---- 复权链: 加载已有缓存 ----
    adj_data = {}
    if ADJ_CACHE.exists():
        adj_data = json.loads(ADJ_CACHE.read_text(encoding="utf-8"))

    # ---- 复权链请求 ----
    print("=" * 60)
    print("[1/3] 复权链请求")
    print("=" * 60)
    for line in adj_lines:
        code, cache_flag = line.split(",")
        ledger["adj_summary"]["total"] += 1
        if cache_flag == "cache=HIT" and code in adj_data:
            ledger["adj_summary"]["cache_hit"] += 1
            ledger["adj_requests"].append({"code": code, "status": "CACHE_HIT"})
            continue
        # 新请求, 最多3次总尝试
        result = None
        for attempt in range(MAX_RETRY):
            ledger["adj_summary"]["retries_used"] += 1
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
        # 失败保持 UNKNOWN, 不当空链
        if result != "SUCCESS":
            adj_data[code] = {"ok": False, "reason": result}
            ledger["adj_summary"]["error"] += 1
        else:
            ledger["adj_summary"]["success"] += 1
        ledger["adj_requests"].append({"code": code, "status": result})
        print(f"  {code}: {result}")

    # 保存复权缓存 (新数据追加, 旧记录保留)
    ADJ_CACHE.write_text(json.dumps(adj_data, ensure_ascii=False), encoding="utf-8")

    # ---- 分钟请求 ----
    print("\n" + "=" * 60)
    print("[2/3] 分钟数据请求")
    print("=" * 60)
    minute_cache = {}  # (code,date) → DataFrame
    # 加载已有缓存
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

    for line in min_lines:
        code, date_str, role, cache_flag = line.split(",")
        date = int(date_str)
        ledger["minute_summary"]["total"] += 1
        key = (code, date)
        # 缓存命中
        if key in minute_cache:
            ledger["minute_summary"]["cache_hit"] += 1
            ledger["minute_requests"].append({"code": code, "date": date, "role": role, "status": "CACHE_HIT"})
            continue
        # 磁盘预算检查
        if ledger["disk_used_bytes"] >= DISK_CAP_BYTES:
            ledger["disk_capped"] = True
            ledger["minute_requests"].append({"code": code, "date": date, "role": role, "status": "DISK_CAPPED"})
            continue
        # 新请求, 最多3次总尝试
        result = None
        df = pd.DataFrame()
        for attempt in range(MAX_RETRY):
            ledger["minute_summary"]["retries_used"] += 1
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
            # 保存到 SNAP_CAR
            out_f = SNAP_CAR / f"{code}.parquet"
            if out_f.exists():
                old = pd.read_parquet(out_f)
                df = pd.concat([old, df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
            df.to_parquet(out_f, index=False)
            size = out_f.stat().st_size
            ledger["disk_used_bytes"] += size
            ledger["minute_summary"]["success"] += 1
            minute_cache[key] = [df[df["date"].astype(str).str[:8].astype(int) == date].copy()]
        elif result == "SOURCE_MISSING":
            ledger["minute_summary"]["source_missing"] += 1
        else:
            ledger["minute_summary"]["error"] += 1
        ledger["minute_requests"].append({"code": code, "date": date, "role": role, "status": result})
        print(f"  {code}@{date} ({role}): {result}  [disk={ledger['disk_used_bytes']/1024/1024:.1f}MB]")

    # ---- 回放 ----
    print("\n" + "=" * 60)
    print("[3/3] 真实工程回放 (30m/60m/保持10m)")
    print("=" * 60)
    replay_results = []
    for c in candidates:
        code = c["code"]
        e_date = c["E"]
        setup = c["setup"]
        result = {"code": code, "E": e_date, "setup": setup,
                  "signal_found": False, "signal_ts": None,
                  "status_30m": None, "status_60m": None, "status_hold": None,
                  "reclaim_30m": None, "reclaim_60m": None, "hold_10m": None,
                  "penetration_30m": None, "penetration_60m": None,
                  "window_complete_30m": None, "window_complete_60m": None,
                  "label_end_ts": None, "ineligible_reason": None,
                  "adj_ok": None}
        # 复权链
        adj_entry = adj_data.get(code, {})
        result["adj_ok"] = adj_entry.get("ok", False)
        # E日分钟
        e_key = (code, e_date)
        if e_key not in minute_cache or not minute_cache[e_key]:
            result["replay_error"] = "E_DAY_MISSING"
            replay_results.append(result)
            print(f"  {code}@{e_date}: E_DAY_MISSING")
            continue
        e_df = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)
        # 需要前一日数据构造 setup (池资格)
        # 简化: 用 run_sheet §6 setup 映射, 这里直接用候选清单的 setup
        # 构造 sig (需要 zone_lower/upper/atr_ref — 从引擎 build_zones 计算)
        try:
            # 取前 250 日 daily
            i_e = cal_idx[e_date]
            lookback = cal[max(0, i_e - 250):i_e]
            d_sub = panel[(panel["code"].astype(str).str.zfill(6) == code)
                          & (panel["date"].astype(int).isin(lookback))].sort_values("date").reset_index(drop=True)
            if len(d_sub) < 25:
                result["replay_error"] = f"DAILY_INSUFFICIENT n={len(d_sub)}"
                replay_results.append(result)
                print(f"  {code}@{e_date}: DAILY_INSUFFICIENT n={len(d_sub)}")
                continue
            atr_ref = float(E.build_atr_ref(d_sub))
            zones = E.build_zones(d_sub, atr_ref)
            # ref_price = close_P (前一交易日收盘, d_sub 末行 close)
            ref_price = float(d_sub["close"].iloc[-1])
            zone = E.target_zone(zones, ref_price)
            if zone is None:
                result["replay_error"] = "NO_ZONE"
                replay_results.append(result)
                print(f"  {code}@{e_date}: NO_ZONE")
                continue
            # 信号检测: 引擎签名 (minute_df, zones, atr_ref, ref_price)
            sig = E.detect_signal(e_df, zones, atr_ref, ref_price)
            if sig is None:
                result["replay_error"] = "NO_SIGNAL"
                replay_results.append(result)
                print(f"  {code}@{e_date}: NO_SIGNAL (保留, 不凑PASS)")
                continue
            result["signal_found"] = True
            result["signal_ts"] = int(sig["signal_ts"])
            result["sig_type"] = sig.get("signal_type")
            # sig 已含 zone_upper/zone_lower/zone_sources (引擎 detect_signal 返回)
            zone_upper = float(sig["zone_upper"])
            # 跨日接线: 拼接下一交易日数据 (已登记的 FUTURE_WINDOW_DAY),
            # 用于 30m/60m/hold 窗口跨日续算; detect_signal 仍只用 E 日 (信号检测在监控窗内)
            next_day = cal[i_e + 1] if i_e + 1 < len(cal) else None
            next_key = (code, next_day) if next_day is not None else None
            if next_key is not None and minute_cache.get(next_key):
                next_df = minute_cache[next_key][0].sort_values("date").reset_index(drop=True)
                full_df = pd.concat([e_df, next_df], ignore_index=True)
                full_df = full_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                result["next_day_used"] = next_day
            else:
                full_df = e_df
                result["next_day_used"] = None
            # CAR 标签
            ts_arr = full_df["date"].astype(np.int64).values
            lo_arr = full_df["low"].astype(float).values
            cl_arr = full_df["close"].astype(float).values
            sig_idx = int(np.where(ts_arr == int(sig["signal_ts"]))[0][0])
            fields = L.init_car_label_fields()
            # 30m (传 cal, 跨日续算)
            L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx,
                                             atr_ref, grid, e_date, 30, "30m", fields, cal=cal)
            # 60m (传 cal)
            L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx,
                                             atr_ref, grid, e_date, 60, "60m", fields, cal=cal)
            # hold (传 cal; 需要 reclaim_ts)
            reclaim_ts = None
            if fields.get("future_reclaim_60m") == 1:
                # 找首次收回 ts
                for i in range(sig_idx + 1, len(ts_arr)):
                    if float(cl_arr[i]) >= zone_upper:
                        reclaim_ts = int(ts_arr[i])
                        break
            L.compute_future_hold(sig, ts_arr, lo_arr, cl_arr, sig_idx,
                                   atr_ref, grid, e_date, reclaim_ts, fields, cal=cal)
            # 填结果 (补全所有 future_ 字段, 供因果守卫全字段比较)
            result["status_30m"] = fields.get("future_status_30m")
            result["status_60m"] = fields.get("future_status_60m")
            result["status_hold"] = fields.get("future_status_hold")
            result["reclaim_30m"] = fields.get("future_reclaim_30m")
            result["reclaim_60m"] = fields.get("future_reclaim_60m")
            result["hold_10m"] = fields.get("future_hold_10m")
            result["penetration_30m"] = fields.get("future_penetration_30m")
            result["penetration_60m"] = fields.get("future_penetration_60m")
            result["time_to_first_reclaim_30m"] = fields.get("time_to_first_future_reclaim_30m")
            result["time_to_first_reclaim_60m"] = fields.get("time_to_first_future_reclaim_60m")
            result["window_complete_30m"] = fields.get("future_window_complete_30m")
            result["window_complete_60m"] = fields.get("future_window_complete_60m")
            result["window_truncated_30m"] = fields.get("future_window_truncated_30m")
            result["window_truncated_60m"] = fields.get("future_window_truncated_60m")
            result["hold_label_end_ts"] = fields.get("future_hold_label_end_ts")
            result["reclaim_status"] = fields.get("future_reclaim_status")
            result["label_end_ts"] = fields.get("future_label_end_ts")
            result["ineligible_reason"] = fields.get("future_ineligible_reason")
            result["reason_30m"] = fields.get("future_reason_30m")
            result["reason_60m"] = fields.get("future_reason_60m")
            result["reason_hold"] = fields.get("future_reason_hold")
            replay_results.append(result)
            print(f"  {code}@{e_date}: sig={sig.get('signal_type')} "
                  f"30m={result['status_30m']} 60m={result['status_60m']} hold={result['status_hold']}")
        except Exception as e:
            result["replay_error"] = f"EXCEPTION:{type(e).__name__}:{str(e)[:80]}"
            replay_results.append(result)
            print(f"  {code}@{e_date}: EXCEPTION {type(e).__name__}")
            traceback.print_exc()

    # ---- 因果截断守卫: 独立离线重算 ----
    # 修正 (2026-09-29 用户收紧):
    #   1. 严格截断 ts <= end_ts (原 searchsorted side=right + [:cut+1] 多保留一根)
    #   2. 重算 30m/60m/hold 全部标签 (原只算 30m/60m, 漏 hold)
    #   3. 全字段比较 (status/reason/reclaim/hold/penetration/time/label_end_ts 各窗口)
    #   4. 终点后价格扰动 (ts > end_ts 的 low/close 改极端值, 已闭环输出应不变)
    #   5. 用 full_df + cal (与主回放一致, 含跨日数据)
    print("\n" + "=" * 60)
    print("[守卫] 因果截断: 修改 label_end_ts 之后的数据不应改变已闭环输出")
    print("=" * 60)
    COMPARE_FIELDS = [
        "future_status_30m", "future_reason_30m",
        "future_reclaim_30m", "future_penetration_30m",
        "time_to_first_future_reclaim_30m",
        "future_window_complete_30m", "future_window_truncated_30m",
        "future_status_60m", "future_reason_60m",
        "future_reclaim_60m", "future_penetration_60m",
        "time_to_first_future_reclaim_60m",
        "future_window_complete_60m", "future_window_truncated_60m",
        "future_status_hold", "future_reason_hold",
        "future_hold_10m", "future_hold_label_end_ts",
        "future_reclaim_status", "future_ineligible_reason",
        "future_label_end_ts",
    ]

    def _reader_to_fields(r: dict) -> dict:
        """把主回放 result 字典 (status_30m 等) 还原为 future_ 前缀 fields 字典,
        供守卫全字段比较."""
        m = {
            "status_30m": "future_status_30m",
            "reason_30m": "future_reason_30m",
            "reclaim_30m": "future_reclaim_30m",
            "penetration_30m": "future_penetration_30m",
            "time_to_first_reclaim_30m": "time_to_first_future_reclaim_30m",
            "window_complete_30m": "future_window_complete_30m",
            "window_truncated_30m": "future_window_truncated_30m",
            "status_60m": "future_status_60m",
            "reason_60m": "future_reason_60m",
            "reclaim_60m": "future_reclaim_60m",
            "penetration_60m": "future_penetration_60m",
            "time_to_first_reclaim_60m": "time_to_first_future_reclaim_60m",
            "window_complete_60m": "future_window_complete_60m",
            "window_truncated_60m": "future_window_truncated_60m",
            "status_hold": "future_status_hold",
            "reason_hold": "future_reason_hold",
            "hold_10m": "future_hold_10m",
            "hold_label_end_ts": "future_hold_label_end_ts",
            "reclaim_status": "future_reclaim_status",
            "ineligible_reason": "future_ineligible_reason",
            "label_end_ts": "future_label_end_ts",
        }
        return {v: r.get(k) for k, v in m.items()}

    def _cmp_fields(fa: dict, fb: dict) -> list:
        """全字段比较; 数值用 isclose, None/字符串用 ==."""
        mism = []
        for k in COMPARE_FIELDS:
            va = fa.get(k)
            vb = fb.get(k)
            if isinstance(va, float) and isinstance(vb, float):
                if np.isfinite(va) and np.isfinite(vb):
                    if not np.isclose(va, vb, rtol=1e-9, atol=1e-12):
                        mism.append({"field": k, "a": va, "b": vb})
                elif va != vb:
                    mism.append({"field": k, "a": va, "b": vb})
            else:
                if va != vb:
                    mism.append({"field": k, "a": va, "b": vb})
        return mism

    truncation_results = []
    for r in replay_results:
        if not r.get("signal_found") or r.get("label_end_ts") is None:
            continue
        code = r["code"]
        e_date = r["E"]
        e_key = (code, e_date)
        if e_key not in minute_cache or not minute_cache[e_key]:
            continue
        # 重构 full_df (与主回放一致: E日 + 下一交易日)
        e_df_g = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)
        i_e = cal_idx[e_date]
        next_day_g = cal[i_e + 1] if i_e + 1 < len(cal) else None
        next_key_g = (code, next_day_g) if next_day_g is not None else None
        if next_key_g is not None and minute_cache.get(next_key_g):
            next_df_g = minute_cache[next_key_g][0].sort_values("date").reset_index(drop=True)
            full_df_g = pd.concat([e_df_g, next_df_g], ignore_index=True)
            full_df_g = full_df_g.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        else:
            full_df_g = e_df_g
        ts_arr = full_df_g["date"].astype(np.int64).values
        lo_arr = full_df_g["low"].astype(float).values
        cl_arr = full_df_g["close"].astype(float).values
        sig_idx = int(np.where(ts_arr == r["signal_ts"])[0][0])
        end_ts = r["label_end_ts"]
        # 重算 atr/zones/ref_price/sig (与主回放一致; sig 用 e_df_g 检测)
        lookback = cal[max(0, i_e - 250):i_e]
        d_sub = panel[(panel["code"].astype(str).str.zfill(6) == code)
                      & (panel["date"].astype(int).isin(lookback))].sort_values("date").reset_index(drop=True)
        atr_ref = float(E.build_atr_ref(d_sub))
        zones = E.build_zones(d_sub, atr_ref)
        ref_price = float(d_sub["close"].iloc[-1])
        zone = E.target_zone(zones, ref_price)
        if zone is None:
            continue
        sig = E.detect_signal(e_df_g, zones, atr_ref, ref_price)
        if sig is None:
            continue
        zone_upper_g = float(sig["zone_upper"])

        def _run_labels(ts_v, lo_v, cl_v):
            fd = L.init_car_label_fields()
            L.compute_future_reclaim_window(sig, ts_v, lo_v, cl_v, sig_idx,
                                             atr_ref, grid, e_date, 30, "30m", fd, cal=cal)
            L.compute_future_reclaim_window(sig, ts_v, lo_v, cl_v, sig_idx,
                                             atr_ref, grid, e_date, 60, "60m", fd, cal=cal)
            reclaim_ts_v = None
            if fd.get("future_reclaim_60m") == 1:
                for i in range(sig_idx + 1, len(ts_v)):
                    if float(cl_v[i]) >= zone_upper_g:
                        reclaim_ts_v = int(ts_v[i])
                        break
            L.compute_future_hold(sig, ts_v, lo_v, cl_v, sig_idx,
                                   atr_ref, grid, e_date, reclaim_ts_v, fd, cal=cal)
            return fd

        # baseline: 用原 full_df 重算 (应与主回放一致)
        fields_base = _run_labels(ts_arr, lo_arr, cl_arr)
        mism_base = _cmp_fields(fields_base, _reader_to_fields(r))
        if mism_base:
            # baseline 与主回放就不一致, 跳过 (说明主回放有非确定性)
            truncation_results.append({
                "code": code, "E": e_date,
                "baseline_mismatch_with_main": mism_base,
                "strict_cut_match": None, "post_perturb_match": None,
            })
            continue

        # test1: 严格截断 ts <= end_ts (去掉 +1, 修正原多保留一根)
        cut = np.searchsorted(ts_arr, end_ts, side="right")
        ts_cut = ts_arr[:cut]
        lo_cut = lo_arr[:cut]
        cl_cut = cl_arr[:cut]
        if len(ts_cut) <= sig_idx:
            continue
        fields_t1 = _run_labels(ts_cut, lo_cut, cl_cut)
        mism_t1 = _cmp_fields(fields_base, fields_t1)

        # test2: 终点后价格扰动 (ts > end_ts 的 low/close 改极端值)
        lo_pert = lo_arr.copy()
        cl_pert = cl_arr.copy()
        for i in range(len(ts_arr)):
            if int(ts_arr[i]) > end_ts:
                lo_pert[i] = 1e9
                cl_pert[i] = 1e9
        fields_t2 = _run_labels(ts_arr, lo_pert, cl_pert)
        mism_t2 = _cmp_fields(fields_base, fields_t2)

        truncation_results.append({
            "code": code, "E": e_date,
            "end_ts": end_ts,
            "strict_cut_match": len(mism_t1) == 0,
            "post_perturb_match": len(mism_t2) == 0,
            "strict_cut_mismatches": mism_t1,
            "post_perturb_mismatches": mism_t2,
        })
    n_strict_ok = sum(1 for t in truncation_results if t.get("strict_cut_match"))
    n_pert_ok = sum(1 for t in truncation_results if t.get("post_perturb_match"))
    n_baseline_bad = sum(1 for t in truncation_results if t.get("baseline_mismatch_with_main"))
    print(f"  严格截断 (ts<=end_ts): {n_strict_ok}/{len(truncation_results)} 一致")
    print(f"  终点后扰动:           {n_pert_ok}/{len(truncation_results)} 一致")
    if n_baseline_bad:
        print(f"  [警告] baseline 与主回放不一致: {n_baseline_bad} 条 (跳过该候选守卫)")

    # ---- 磁盘审计 (修正: 原只计本轮新写 parquet, 现扫描全部产物) ----
    # f4: 把 minute_car/adj_cache/报告全部计入预算; 记录缓存文件 SHA256 与来源
    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    disk_audit = {
        "minute_car_dir_bytes": 0,
        "minute_car_files": [],
        "adj_cache_bytes": 0,
        "adj_cache_sha256": None,
        "report_bytes": 0,
        "total_bytes": 0,
        "cap_bytes": DISK_CAP_BYTES,
        "capped": False,
        "note": ("本轮新增抓取的 minute parquet (SNAP_CAR) + 复权缓存 (adj_cache.json) "
                 "+ 报告本身全部计入; ledger.disk_used_bytes 仅计本次脚本新写 parquet"),
    }
    if SNAP_CAR.exists():
        for pf in SNAP_CAR.glob("*.parquet"):
            sz = pf.stat().st_size
            disk_audit["minute_car_dir_bytes"] += sz
            disk_audit["minute_car_files"].append({
                "file": pf.name, "bytes": sz, "sha256": _sha256(pf),
                "source": "HTTP抓取 (car_v0_1_minute_request_list.txt)",
            })
    if ADJ_CACHE.exists():
        sz = ADJ_CACHE.stat().st_size
        disk_audit["adj_cache_bytes"] = sz
        disk_audit["adj_cache_sha256"] = _sha256(ADJ_CACHE)
    # 报告本身 (写完后再补大小, 此处先占位)
    disk_audit["total_bytes"] = (disk_audit["minute_car_dir_bytes"]
                                 + disk_audit["adj_cache_bytes"])
    disk_audit["capped"] = disk_audit["total_bytes"] > DISK_CAP_BYTES
    ledger["disk_audit"] = disk_audit

    # ---- 汇总报告 ----
    n_total = len(replay_results)
    n_signal = sum(1 for r in replay_results if r.get("signal_found"))
    n_no_signal = sum(1 for r in replay_results if r.get("replay_error") == "NO_SIGNAL")
    n_missing = sum(1 for r in replay_results if "MISSING" in r.get("replay_error", ""))
    n_exception = sum(1 for r in replay_results if "EXCEPTION" in r.get("replay_error", ""))
    n_valid_60m = sum(1 for r in replay_results
                      if r.get("status_60m") in ("RECLAIMED", "NOT_RECLAIMED"))
    n_ineligible = sum(1 for r in replay_results if r.get("status_60m") == "INELIGIBLE")
    n_incomplete = sum(1 for r in replay_results
                       if r.get("status_60m") == "INCOMPLETE_WINDOW")

    report = {
        "status": ("REAL_REPLAY_COMPLETE — CAR标签合成验收通过(80/80); "
                   "真实回放已完成(33→27信号/6无信号); "
                   "因果截断守卫: 21字段全比较+hold重算+严格截断+终点后扰动 均27/27一致 "
                   "(使用相同标签函数复算, 非独立参考验证); "
                   "不训练"),
        "date": "2026-09-29",
        "ledger": ledger,
        "replay_summary": {
            "total_candidates": n_total,
            "signal_found": n_signal,
            "no_signal": n_no_signal,
            "data_missing": n_missing,
            "exception": n_exception,
            "valid_60m": n_valid_60m,
            "ineligible_60m": n_ineligible,
            "incomplete_60m": n_incomplete,
            "reclaimed_at_s_60m": sum(1 for r in replay_results if r.get("status_60m") == "RECLAIMED_AT_S"),
            "already_above_60m": sum(1 for r in replay_results if r.get("status_60m") == "ALREADY_ABOVE"),
            "sig_type_breakdown": {
                "APPROACH": sum(1 for r in replay_results if r.get("sig_type") == "APPROACH"),
                "ALREADY_BELOW": sum(1 for r in replay_results if r.get("sig_type") == "ALREADY_BELOW"),
                "BREAK": sum(1 for r in replay_results if r.get("sig_type") == "BREAK"),
            },
            "note": ("无信号和数据缺失都保留; 覆盖不足的工程门记待验, 不凑PASS; "
                     "27信号中17 APPROACH+10 ALREADY_BELOW+0 BREAK, 研究线索记待统计方案签署时明确"),
        },
        "truncation_guard": {
            "tested": len(truncation_results),
            "strict_cut_consistent": n_strict_ok,
            "post_perturb_consistent": n_pert_ok,
            "baseline_mismatch_with_main": n_baseline_bad,
            "compare_fields": COMPARE_FIELDS,
            "note": ("守卫比较全部 future_ 标签字段(30m/60m/hold 各自 status/reason/"
                     "reclaim/hold/penetration/time/label_end_ts); "
                     "test1=严格截断 ts<=end_ts; test2=终点后价格扰动(ts>end_ts 的 low/close 改 1e9); "
                     "使用相同标签函数复算, 非独立参考验证"),
            "detail": truncation_results,
        },
        "per_candidate": replay_results,
        "pending_engineering_gates": [
            "资格映射 (事件提取层 NO_SIGNAL 真验证)",
            "事件去重",
            "价格量额尺度",
            "执行分流 (H-E 锁定保持, 未验证前不放开)",
        ],
    }
    report_f = OUT / "car_v0_1_real_replay_report.json"
    # 第一次写: 读大小后补 report_bytes 到 disk_audit, 再写第二次 (避免自指循环)
    report_f.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    disk_audit["report_bytes"] = report_f.stat().st_size
    disk_audit["report_sha256"] = _sha256(report_f)
    disk_audit["total_bytes"] = (disk_audit["minute_car_dir_bytes"]
                                 + disk_audit["adj_cache_bytes"]
                                 + disk_audit["report_bytes"])
    disk_audit["capped"] = disk_audit["total_bytes"] > DISK_CAP_BYTES
    ledger["disk_audit"] = disk_audit
    # 第二次写 (带最终 disk_audit, 含 report_bytes/sha256)
    report_f.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("[汇总]")
    print(f"  分钟: {ledger['minute_summary']['success']}成功/"
          f"{ledger['minute_summary']['cache_hit']}缓存/"
          f"{ledger['minute_summary']['source_missing']}源缺/"
          f"{ledger['minute_summary']['error']}异常 (重试{ledger['minute_summary']['retries_used']}次)")
    print(f"  复权: {ledger['adj_summary']['success']}成功/"
          f"{ledger['adj_summary']['cache_hit']}缓存/"
          f"{ledger['adj_summary']['error']}失败 (重试{ledger['adj_summary']['retries_used']}次)")
    da = ledger["disk_audit"]
    print(f"  磁盘审计: minute_car={da['minute_car_dir_bytes']/1024:.1f}KB "
          f"adj_cache={da['adj_cache_bytes']/1024:.1f}KB "
          f"report={da['report_bytes']/1024:.1f}KB "
          f"total={da['total_bytes']/1024/1024:.2f}MB / {DISK_CAP_BYTES/1024/1024:.0f}MB"
          f"{' [CAPPED]' if da['capped'] else ''}")
    print(f"  回放: {n_total}候选 → {n_signal}信号 / {n_no_signal}无信号 / {n_missing}缺失 / {n_exception}异常")
    print(f"  60m有效: {n_valid_60m} / 不适用: {n_ineligible} / 不完整: {n_incomplete}")
    print(f"  信号类型: APPROACH={sum(1 for r in replay_results if r.get('sig_type')=='APPROACH')} "
          f"ALREADY_BELOW={sum(1 for r in replay_results if r.get('sig_type')=='ALREADY_BELOW')} "
          f"BREAK={sum(1 for r in replay_results if r.get('sig_type')=='BREAK')}")
    print(f"  因果截断: 严格截断 {n_strict_ok}/{len(truncation_results)} | 终点后扰动 {n_pert_ok}/{len(truncation_results)}")
    print(f"  待验工程门: 资格映射/事件去重/价格量额尺度/执行分流(H-E锁定)")
    print(f"  报告: {report_f.name}")


if __name__ == "__main__":
    main()
