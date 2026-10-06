
"""CAR v0.2 扩样整批 400 条覆盖 (2026-09-29).

运行边界:
  - 复用已完成的 20 条及缓存, 只处理冻结清单剩余部分
  - 保留无信号/缺数据/量纲异常/公司行为隔离, 不找替补候选
  - 有限重试(3次), 50MB 总预算; 异常和超预算停止报告
  - 整批完成后暂停, 先核对分段样本与质量门, 再决定是否训练
  - 0 请求失败/0 源缺只证明采集成功, 不等于全部质量门通过
  - 整批仍需运行已冻结的价格尺度/量额单位/窗口完整性/公司行为检查

交付三张表:
  1. 各阶段耗时及请求量
  2. 质量分流 (工程门明细)
  3. 训练/验证/测试各段的有效日期与正负样本数
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
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
from gcp_car_v0_1_engineering_gates import check_gap_propagation, check_corp_action  # noqa: E402
from gcp_car_v0_2_extractor import detect_signal_v02  # noqa: E402

CONTRACTS = HERE / "contracts"
SNAP_CAR = HERE / "gcp_round_b_small_sample_v1" / "minute_car"
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"
ADJ_CACHE = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"
OUT = HERE / "gcp_round_b_small_sample_v1"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"

MAX_RETRY = 3
DISK_CAP_BYTES = 50 * 1024 * 1024
CKPT = HERE / "gcp_round_b_small_sample_v1" / "car_v0_2_label_checkpoint.jsonl"


def _np_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not serializable: {type(obj)}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    SNAP_CAR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 读冻结清单 ----
    cand_lines = (CONTRACTS / "car_v0_2_expansion_candidate_list.txt").read_text(
        encoding="utf-8").strip().split("\n")
    registry = json.loads((CONTRACTS / "car_v0_2_expansion_registry.json").read_text(
        encoding="utf-8"))

    all_candidates = []
    for line in cand_lines:
        code, setup, p_str, e_str = line.split(",")
        all_candidates.append({"code": code, "setup": setup,
                               "P": int(p_str), "E": int(e_str)})

    print("=" * 60)
    print(f"CAR v0.2 整批覆盖: {len(all_candidates)} 候选")
    print("=" * 60)

    # ---- 交易日历 ----
    panel = get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    cal_set = set(cal)

    # 日线面板按代码预索引 (避免逐候选全表扫描)
    daily_by_code = {c: g.reset_index(drop=True) for c, g in panel.groupby("code")}

    def daily_lookback(code, lookback_dates):
        dc = daily_by_code.get(code)
        if dc is None:
            return pd.DataFrame()
        return dc[dc["date"].isin(lookback_dates)].sort_values("date").reset_index(drop=True)

    # ---- 网格 ----
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    # ---- 收集所需请求 (先于缓存加载, 用于裁剪加载范围) ----
    minute_needs = set()
    adj_needs = set()
    for c in all_candidates:
        code = c["code"]
        e_date = c["E"]
        minute_needs.add((code, e_date))
        i_e = cal_idx.get(e_date, -1)
        if i_e >= 0 and i_e + 1 < len(cal):
            minute_needs.add((code, cal[i_e + 1]))
        adj_needs.add(code)
    needs_by_code: dict = {}
    for c2, d2 in minute_needs:
        needs_by_code.setdefault(c2, set()).add(d2)

    # ---- 加载已有分钟缓存 (仅加载所需日期) ----
    print("\n[1/5] 加载已有缓存...")
    t_cache = time.time()
    minute_cache = {}
    for snap_dir in (SNAP_OLD, SNAP_EXT, SNAP_CAR):
        if not snap_dir.exists():
            continue
        for pf in snap_dir.glob("*.parquet"):
            code = pf.stem
            want = needs_by_code.get(code)
            if want is None:
                continue
            try:
                m = pd.read_parquet(pf)
                if "date" in m and not m.empty:
                    day_int = m["date"].astype(str).str[:8].astype(int)
                    for d in pd.unique(day_int):
                        if int(d) not in want:
                            continue
                        minute_cache.setdefault((code, int(d)), []).append(
                            m[day_int == int(d)].copy())
            except Exception:
                pass
    t_cache_done = time.time()
    print(f"  缓存加载耗时: {t_cache_done - t_cache:.1f}s, 命中 (code,date): {len(minute_cache)}")

    # ---- 复权缓存 ----
    adj_data = json.loads(ADJ_CACHE.read_text(encoding="utf-8")) if ADJ_CACHE.exists() else {}

    # ---- 复权请求 ----
    print(f"\n[2/5] 复权链请求 ({len(adj_needs)} 股)...")
    t_adj = time.time()
    adj_new = 0
    adj_hit = 0
    adj_fail = 0
    for code in sorted(adj_needs):
        if code in adj_data and adj_data[code].get("ok"):
            adj_hit += 1
            continue
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
            except Exception as exc:
                result = f"ERROR:{type(exc).__name__}"
                if attempt < MAX_RETRY - 1:
                    time.sleep(0.5)
        if result == "SUCCESS":
            adj_new += 1
        else:
            adj_data[code] = {"ok": False, "reason": result}
            adj_fail += 1
    ADJ_CACHE.write_text(json.dumps(adj_data, ensure_ascii=False), encoding="utf-8")
    t_adj_done = time.time()
    print(f"  复权耗时: {t_adj_done - t_adj:.1f}s (命中={adj_hit}, 新增={adj_new}, 失败={adj_fail})")

    # ---- 分钟请求 ----
    print(f"\n[3/5] 分钟数据请求 ({len(minute_needs)} 条)...")
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
        if disk_used >= DISK_CAP_BYTES:
            print(f"  DISK_CAPPED at {disk_used / 1024 / 1024:.1f}MB")
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
            except Exception as exc:
                result = f"ERROR:{type(exc).__name__}"
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
    t_min_done = time.time()
    print(f"  分钟耗时: {t_min_done - t_min:.1f}s (命中={min_hit}, 新增={min_new}, "
          f"源缺={min_source_missing}, 错误={min_fail})")

    # ---- 标签生成 ----
    print(f"\n[4/5] 标签生成 ({len(all_candidates)} 候选)...")
    t_label = time.time()

    # 断点: 已完成候选直接复用
    done_map: dict = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done_map[(r["code"], r["E"])] = r
    ckpt_handle = open(CKPT, "a", encoding="utf-8")

    results = []

    def append_result(r):
        results.append(r)
        ckpt_handle.write(json.dumps(r, ensure_ascii=False, default=_np_default) + "\n")
        ckpt_handle.flush()
        if len(results) % 50 == 0:
            print(f"  标签进度: {len(results)}/{len(all_candidates)} "
                  f"({time.time() - t_label:.0f}s)", flush=True)

    for c in all_candidates:
        code = c["code"]
        e_date = c["E"]
        setup = c["setup"]
        if (code, e_date) in done_map:
            results.append(done_map[(code, e_date)])
            continue
        result = {"code": code, "E": e_date, "setup": setup,
                  "P": c["P"], "signal_found": False}

        adj_entry = adj_data.get(code, {})
        result["adj_ok"] = adj_entry.get("ok", False)

        e_key = (code, e_date)
        if e_key not in minute_cache or not minute_cache[e_key]:
            result["replay_error"] = "E_DAY_MISSING"
            append_result(result)
            continue
        e_df = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)

        i_e = cal_idx.get(e_date, -1)
        if i_e < 0:
            result["replay_error"] = "E_NOT_IN_CAL"
            append_result(result)
            continue
        lookback = cal[max(0, i_e - 250):i_e]
        d_sub = daily_lookback(code, lookback)
        if len(d_sub) < 25:
            result["replay_error"] = f"DAILY_INSUFFICIENT n={len(d_sub)}"
            append_result(result)
            continue
        atr_ref = float(E.build_atr_ref(d_sub))
        zones = E.build_zones(d_sub, atr_ref)
        ref_price = float(d_sub["close"].iloc[-1])

        sig = detect_signal_v02(e_df, zones, atr_ref, ref_price)
        if sig is None:
            result["replay_error"] = "NO_SIGNAL"
            append_result(result)
            continue

        result["signal_found"] = True
        result["signal_ts"] = sig["signal_ts"]
        result["sig_type"] = sig["signal_type"]

        if sig["signal_type"] == "RECLAIMED_AT_S":
            result["reclaim_status"] = "RECLAIMED_AT_S"
            append_result(result)
            continue

        fields = L.init_car_label_fields()
        sig_idx = e_df[e_df["date"].astype(int) == sig["signal_ts"]].index
        if len(sig_idx) == 0:
            result["replay_error"] = "SIGNAL_TS_NOT_IN_DF"
            append_result(result)
            continue
        sig_idx = sig_idx[0]
        ts_arr = e_df["date"].astype(int).to_numpy()
        lo_arr = e_df["low"].astype(float).to_numpy()
        cl_arr = e_df["close"].astype(float).to_numpy()

        L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx,
                                        atr_ref, grid, e_date, 30, "30m", fields, cal)
        L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx,
                                        atr_ref, grid, e_date, 60, "60m", fields, cal)

        # hold 标签
        reclaim_ts_60m = None
        if fields.get("future_reclaim_60m") == 1 and fields.get("time_to_first_future_reclaim_60m"):
            ttr = fields["time_to_first_future_reclaim_60m"]
            future_slots = L._build_future_window_slots(
                int(str(sig["signal_ts"])[-6:]), e_date, 60, grid, cal)
            if 0 < ttr <= len(future_slots):
                reclaim_ts_60m = int(f"{future_slots[ttr - 1][0]}{future_slots[ttr - 1][1]:06d}")
        L.compute_future_hold(sig, ts_arr, lo_arr, cl_arr, sig_idx, atr_ref,
                              grid, e_date, reclaim_ts_60m, fields, cal)

        result.update({
            "reclaim_30m": fields.get("future_reclaim_30m"),
            "reclaim_60m": fields.get("future_reclaim_60m"),
            "hold_10m": fields.get("future_hold_10m"),
            "penetration_30m": fields.get("future_penetration_30m"),
            "penetration_60m": fields.get("future_penetration_60m"),
            "time_to_first_reclaim_30m": fields.get("time_to_first_future_reclaim_30m"),
            "time_to_first_reclaim_60m": fields.get("time_to_first_future_reclaim_60m"),
            "window_complete_30m": fields.get("future_window_complete_30m"),
            "window_complete_60m": fields.get("future_window_complete_60m"),
            "window_truncated_30m": fields.get("future_window_truncated_30m"),
            "window_truncated_60m": fields.get("future_window_truncated_60m"),
            "label_end_ts": fields.get("future_label_end_ts"),
            "hold_label_end_ts": fields.get("future_hold_label_end_ts"),
            "reclaim_status": fields.get("future_reclaim_status"),
            "ineligible_reason": fields.get("future_ineligible_reason"),
            "status_30m": fields.get("future_status_30m"),
            "status_60m": fields.get("future_status_60m"),
            "status_hold": fields.get("future_status_hold"),
            "reason_30m": fields.get("future_reason_30m"),
            "reason_60m": fields.get("future_reason_60m"),
            "reason_hold": fields.get("future_reason_hold"),
        })
        append_result(result)
    ckpt_handle.close()
    t_label_done = time.time()

    # ---- 工程门检查 ----
    print(f"\n[5/5] 工程门检查 ({len(all_candidates)} 候选)...")
    t_gate = time.time()

    # 冻结池
    pool_a = pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")
    pool_a_set = set(zip(pool_a["code"].astype(str).str.zfill(6),
                         pool_a["date"].astype(int),
                         pool_a["setup_type"].astype(str)))
    pool_b_set = set(zip(pool_b["code"].astype(str).str.zfill(6),
                         pool_b["date"].astype(int),
                         pool_b["setup_type"].astype(str)))
    pool_set = pool_a_set | pool_b_set

    gates_report = {
        "gate1_eligibility": {"name": "资格映射", "results": [], "summary": {}},
        "gate2_dedup": {"name": "事件去重", "results": [], "summary": {}},
        "gate3_scale": {"name": "价格量额尺度", "results": [], "summary": {}},
    }

    # 门1
    g1 = gates_report["gate1_eligibility"]
    n_pass_g1 = 0
    for c in all_candidates:
        code = c["code"]
        p_date = c["P"]
        e_date = c["E"]
        setup = c["setup"]
        reasons = []
        if p_date not in cal_set:
            reasons.append(f"P_NOT_IN_CAL ({p_date})")
        if e_date not in cal_set:
            reasons.append(f"E_NOT_IN_CAL ({e_date})")
        if p_date in cal_idx and e_date in cal_idx:
            expected_e = cal[cal_idx[p_date] + 1] if cal_idx[p_date] + 1 < len(cal) else None
            if expected_e != e_date:
                reasons.append(f"E_NOT_NEXT_DAY (expected={expected_e}, got={e_date})")
        in_pool = (code, p_date, setup) in pool_set
        if not in_pool:
            in_a = (code, p_date, "A_SDP") in pool_a_set
            in_b = (code, p_date, "B_LVB") in pool_b_set
            if in_a and setup != "A_SDP":
                reasons.append(f"POOL_MISMATCH (in A_SDP but setup={setup})")
            elif in_b and setup != "B_LVB":
                reasons.append(f"POOL_MISMATCH (in B_LVB but setup={setup})")
            else:
                reasons.append("NOT_IN_ANY_POOL")
        ok = len(reasons) == 0
        if ok:
            n_pass_g1 += 1
        g1["results"].append({
            "code": code, "setup": setup, "P": p_date, "E": e_date,
            "in_pool": in_pool, "pass": ok, "reasons": reasons,
        })
    g1["summary"] = {"total": len(all_candidates), "pass": n_pass_g1,
                     "fail": len(all_candidates) - n_pass_g1}

    # 门2
    g2 = gates_report["gate2_dedup"]
    ce_pairs = [(c["code"], c["E"]) for c in all_candidates]
    ce_counts = Counter(ce_pairs)
    dup_candidates = [(k, v) for k, v in ce_counts.items() if v > 1]
    g2["summary"] = {
        "candidate_ce_unique": len(dup_candidates) == 0,
        "dup_count": len(dup_candidates),
        "total_pairs": len(ce_pairs),
    }

    # 门3
    g3 = gates_report["gate3_scale"]
    scale_results = []
    sub_pass = {"price": 0, "volume": 0, "grid": 0, "adj": 0}

    for n_gate, c in enumerate(all_candidates, start=1):
        code = c["code"]
        e_date = c["E"]
        p_date = c["P"]
        item = {"code": code, "E": e_date, "sub_checks": {}, "pass": True}

        dc = daily_by_code.get(code)
        daily_e = dc[dc["date"] == e_date] if dc is not None else pd.DataFrame()
        e_key = (code, e_date)
        if e_key not in minute_cache or not minute_cache[e_key] or daily_e.empty:
            item["sub_checks"]["data_missing"] = True
            item["pass"] = False
            scale_results.append(item)
            if n_gate % 50 == 0:
                print(f"  门3进度: {n_gate}/{len(all_candidates)}", flush=True)
            continue
        e_df = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)
        d_close = float(daily_e["close"].iloc[0])
        d_high = float(daily_e["high"].iloc[0])
        d_low = float(daily_e["low"].iloc[0])
        d_open = float(daily_e["open"].iloc[0])
        d_vol = float(daily_e["volume"].iloc[0])
        d_amt = float(daily_e["amount"].iloc[0])

        # 3a 价格尺度
        m_open = float(e_df["open"].iloc[0])
        m_high = float(e_df["high"].max())
        m_low = float(e_df["low"].min())
        m_close = float(e_df["close"].iloc[-1])

        def _rdiff(a, b):
            return abs(a - b) / (max(abs(a), abs(b)) + 1e-12)
        ohlc_rdiffs = {
            "open": _rdiff(d_open, m_open), "high": _rdiff(d_high, m_high),
            "low": _rdiff(d_low, m_low), "close": _rdiff(d_close, m_close),
        }
        price_ok = all(v < 0.01 for v in ohlc_rdiffs.values())
        try:
            i_e = cal_idx[e_date]
            lookback = cal[max(0, i_e - 250):i_e]
            d_sub = daily_lookback(code, lookback)
            atr_ref_g = float(E.build_atr_ref(d_sub))
            atr_ratio = atr_ref_g / d_close
            atr_ok = 0.001 < atr_ratio < 0.30
        except Exception:
            atr_ref_g, atr_ratio, atr_ok = None, None, False
        price_pass = price_ok and atr_ok
        item["sub_checks"]["price_scale"] = {
            "ohlc_rdiffs": ohlc_rdiffs, "atr_ref": atr_ref_g,
            "atr_ratio": atr_ratio, "pass": price_pass,
        }
        if price_pass:
            sub_pass["price"] += 1
        else:
            item["pass"] = False

        # 3b 量额单位
        ratio_arr = e_df["amount"].astype(float) / (
            e_df["volume"].astype(float) * e_df["close"].astype(float) + 1e-12)
        valid = ratio_arr[e_df["volume"] > 0]
        vol_amt_med = float(np.median(valid)) if len(valid) > 0 else None
        amt_ratio = float(e_df["amount"].sum()) / (d_amt + 1e-12)
        if not (0.8 < amt_ratio < 1.2):
            vol_status = "AMOUNT_MISMATCH"
            vol_pass = False
        elif vol_amt_med is not None and 0.5 < vol_amt_med < 1.5:
            vol_status = "CONSISTENT"
            vol_pass = True
        elif vol_amt_med is not None and 0.005 < vol_amt_med < 0.05:
            vol_status = "VOLUME_SCALE_UNKNOWN"
            vol_pass = False
        else:
            vol_status = "VOLUME_SCALE_UNKNOWN"
            vol_pass = False
        item["sub_checks"]["volume_scale"] = {
            "amt_vol_close_median": vol_amt_med, "amount_ratio": amt_ratio,
            "status": vol_status, "pass": vol_pass,
        }
        if vol_pass:
            sub_pass["volume"] += 1
        else:
            item["pass"] = False

        # 3c 窗口网格
        grid_check = E.validate_minute_day(e_df, grid, e_date)
        grid_pass = grid_check["ok"]
        gap_in_window = None
        # 找到回放结果匹配项
        replay_match = next((r for r in results
                             if r["code"] == code and r["E"] == e_date
                             and r.get("signal_found")), None)
        if replay_match and replay_match.get("signal_ts"):
            sig_ts_val = int(replay_match["signal_ts"])
            ttr_val = replay_match.get("time_to_first_reclaim_60m")
            next_df_val = None
            nd_idx_val = cal_idx.get(e_date, -1) + 1
            if nd_idx_val < len(cal):
                nd_val = cal[nd_idx_val]
                nd_key = (code, nd_val)
                if nd_key in minute_cache and minute_cache[nd_key]:
                    next_df_val = minute_cache[nd_key][0]
            gap_in_window = check_gap_propagation(
                e_df, next_df_val, sig_ts_val, ttr_val, grid, cal, e_date)
        if gap_in_window is not None:
            if gap_in_window["30m"] or gap_in_window["60m"] or gap_in_window["hold_10m"]:
                grid_pass = False
            nd = gap_in_window.get("next_day")
            if nd and (nd.get("30m") or nd.get("60m") or nd.get("hold_10m")):
                grid_pass = False
        item["sub_checks"]["window_grid"] = {
            "grid_ok": grid_check["ok"], "gap_in_window": gap_in_window,
            "pass": grid_pass,
        }
        if grid_pass:
            sub_pass["grid"] += 1
        else:
            item["pass"] = False

        # 3d 公司行为
        label_end_ts = replay_match.get("label_end_ts") if replay_match else None
        adj_entry = adj_data.get(code, {})
        adj_result = check_corp_action(adj_entry, p_date, label_end_ts)
        adj_pass = adj_result["pass"]
        item["sub_checks"]["corp_action"] = {
            "status": adj_result["status"], "pass": adj_pass,
        }
        if adj_pass:
            sub_pass["adj"] += 1
        else:
            item["pass"] = False

        scale_results.append(item)
        if n_gate % 50 == 0:
            print(f"  门3进度: {n_gate}/{len(all_candidates)}", flush=True)
    g3["results"] = scale_results
    n_scale_pass = sum(1 for s in scale_results if s["pass"])
    g3["summary"] = {
        "total": len(scale_results), "pass": n_scale_pass,
        "fail": len(scale_results) - n_scale_pass, "sub_pass": sub_pass,
    }
    t_gate_done = time.time()

    # ---- 时间切分 ----
    train_dates = set(range(registry["time_split"]["train"][0],
                            registry["time_split"]["train"][1] + 1))
    val_dates = set(range(registry["time_split"]["val"][0],
                          registry["time_split"]["val"][1] + 1))
    test_dates = set(range(registry["time_split"]["test"][0],
                           registry["time_split"]["test"][1] + 1))
    # 实际存在的 E 日
    all_e_dates = sorted(set(c["E"] for c in all_candidates))
    train_dates = set(d for d in all_e_dates if d in train_dates)
    val_dates = set(d for d in all_e_dates if d in val_dates)
    test_dates = set(d for d in all_e_dates if d in test_dates)

    def _segment_stats(dates_set):
        seg = {"total": 0, "signal": 0, "reclaimed_at_s": 0,
               "eligible": 0, "reclaimed": 0, "not_reclaimed": 0,
               "no_signal": 0, "missing": 0, "incomplete": 0,
               "ineligible": 0}
        for r in results:
            if r["E"] not in dates_set:
                continue
            seg["total"] += 1
            if not r.get("signal_found"):
                if r.get("replay_error") == "NO_SIGNAL":
                    seg["no_signal"] += 1
                else:
                    seg["missing"] += 1
                continue
            seg["signal"] += 1
            if r.get("reclaim_status") == "RECLAIMED_AT_S":
                seg["reclaimed_at_s"] += 1
                continue
            status = r.get("reclaim_status", "")
            if status == "RECLAIMED":
                seg["eligible"] += 1
                seg["reclaimed"] += 1
            elif status == "NOT_RECLAIMED":
                seg["eligible"] += 1
                seg["not_reclaimed"] += 1
            elif status in ("INCOMPLETE_WINDOW", "NO_WINDOW_BARS"):
                seg["incomplete"] += 1
            else:
                seg["ineligible"] += 1
        return seg

    train_seg = _segment_stats(train_dates)
    val_seg = _segment_stats(val_dates)
    test_seg = _segment_stats(test_dates)

    # ---- 训练就绪判定: 两种固定质量口径, 口径 A 为合同决策口径 ----
    # 口径 A (strict_all_gates): 价格/量额/窗口网格/公司行为全过, 数据在;
    #   不通过整条排除. 批次状态只由 A 决定, 不得在看到样本数后切换.
    # 口径 B (volume_isolated_price_retained): 仅量额一项不过、其余全过的候选,
    #   隔离量特征但保留价格标签 (统计方案 L134/L141 既定口径), 仅供对照披露.
    gate_by_key = {(it["code"], it["E"]): it for it in scale_results}

    def _include(row, caliber):
        it = gate_by_key.get((row["code"], row["E"]))
        if it is None:
            return False
        sc = it["sub_checks"]
        if sc.get("data_missing"):
            return False
        if caliber == "A":
            return bool(it["pass"])
        if it["pass"]:
            return True
        # 仅量额失败, 其余 (价格/网格/公司行为) 全过
        return (not sc["volume_scale"]["pass"]
                and sc["price_scale"]["pass"]
                and sc["window_grid"]["pass"]
                and sc["corp_action"]["pass"])

    seg_order = [("train", train_dates), ("val", val_dates), ("test", test_dates)]
    first_next_e = {"train": min(val_dates), "val": min(test_dates), "test": None}
    THRESHOLDS = {"train": (40, 30, 30), "val": (15, 10, 10), "test": (15, 10, 10)}

    caliber_report = {}
    for caliber in ("A", "B"):
        per_seg = {}
        for seg_name, date_set in seg_order:
            included_rows = []
            eff_dates = set()
            pos = neg = 0
            crossings = []
            max_end = None
            for r in results:
                if r["E"] not in date_set:
                    continue
                if r.get("reclaim_status") not in ("RECLAIMED", "NOT_RECLAIMED"):
                    continue
                if not _include(r, caliber):
                    continue
                included_rows.append(r)
                eff_dates.add(r["E"])
                if r.get("reclaim_60m") == 1:
                    pos += 1
                elif r.get("reclaim_60m") == 0:
                    neg += 1
                end_ts = r.get("label_end_ts")
                if end_ts:
                    end_date = int(str(int(end_ts))[:8])
                    max_end = end_date if max_end is None else max(max_end, end_date)
                    nxt = first_next_e[seg_name]
                    if nxt is not None and end_date >= nxt:
                        crossings.append({"code": r["code"], "E": r["E"],
                                          "label_end_ts": int(end_ts),
                                          "next_seg_first_e": nxt})
            min_dates, min_pos, min_neg = THRESHOLDS[seg_name]
            checks = {
                "effective_dates_ok": len(eff_dates) >= min_dates,
                "positive_ok": pos >= min_pos,
                "negative_ok": neg >= min_neg,
            }
            per_seg[seg_name] = {
                "effective_dates": len(eff_dates),
                "included": len(included_rows),
                "positive": pos, "negative": neg,
                "thresholds": {"min_dates": min_dates, "min_positive": min_pos,
                               "min_negative": min_neg},
                "checks": checks,
                "all_checks_pass": all(checks.values()),
                "boundary_crossings": crossings,
                "max_label_end_date": max_end,
            }
        caliber_report[caliber] = per_seg

    readiness_a = all(v["all_checks_pass"] for v in caliber_report["A"].values())
    batch_status = "READY_FOR_TRAINING" if readiness_a else "INSUFFICIENT_FOR_TRAINING"

    # ---- 磁盘实测 (同次运行): 快照 footprint 与本次新增字节分开 ----
    CKPT.unlink(missing_ok=True)  # 先清断点, 使 footprint 对应最终交付状态

    def _measure_footprint():
        return sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())

    def _disk_report_with(footprint_bytes):
        return {
            "footprint_bytes": footprint_bytes,
            "footprint_mb": round(footprint_bytes / 1024 / 1024, 2),
            "cap_bytes": DISK_CAP_BYTES,
            "within_cap": footprint_bytes <= DISK_CAP_BYTES,
            "new_bytes_this_run": disk_used,
            "note": ("new_bytes_this_run=0 表示本次运行全部复用缓存; "
                     "footprint 为快照目录最终全部数据与产物"),
        }

    disk_report = _disk_report_with(_measure_footprint())

    # ---- 汇总输出 ----
    t_total = time.time() - t0
    print("\n" + "=" * 60)
    print("[整批汇总]")
    print("=" * 60)
    print(f"  总耗时: {t_total:.1f}s")
    print(f"  缓存加载: {t_cache_done - t_cache:.1f}s")
    print(f"  复权请求: {t_adj_done - t_adj:.1f}s (命中={adj_hit}, 新增={adj_new}, 失败={adj_fail})")
    print(f"  分钟请求: {t_min_done - t_min:.1f}s (命中={min_hit}, 新增={min_new}, 源缺={min_source_missing}, 错误={min_fail})")
    print(f"  标签生成: {t_label_done - t_label:.1f}s")
    print(f"  工程门: {t_gate_done - t_gate:.1f}s")
    print(f"  磁盘本次新增: {disk_report['new_bytes_this_run'] / 1024 / 1024:.2f} MB "
          "(最终 footprint 见收尾行)")

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

    print(f"\n[工程门]")
    print(f"  门1 资格映射: {n_pass_g1}/{len(all_candidates)} PASS")
    print(f"  门2 事件去重: {'PASS' if len(dup_candidates) == 0 else 'FAIL'} ({len(dup_candidates)} 重复)")
    print(f"  门3 价格量额尺度: {n_scale_pass}/{len(scale_results)} PASS")
    print(f"    价格尺度: {sub_pass['price']}/{len(scale_results)}")
    print(f"    量额单位: {sub_pass['volume']}/{len(scale_results)}")
    print(f"    窗口网格: {sub_pass['grid']}/{len(scale_results)}")
    print(f"    公司行为: {sub_pass['adj']}/{len(scale_results)}")

    print(f"\n[分段样本 (标签口径, 未叠加质量门)]")
    for name, seg in (("训练", train_seg), ("验证", val_seg), ("测试", test_seg)):
        pos = seg["reclaimed"]
        neg = seg["not_reclaimed"]
        print(f"  {name}: 日期={len(train_dates if name == '训练' else val_dates if name == '验证' else test_dates)} "
              f"候选={seg['total']} 信号={seg['signal']} ELIGIBLE={seg['eligible']} "
              f"收={pos} 未收={neg} 无信号={seg['no_signal']} 缺数据={seg['missing']} "
              f"不完整={seg['incomplete']} 不适={seg['ineligible']}")

    print(f"\n[批次状态] {batch_status} (决策口径=A 严格全门)")
    for cal_name, cal_label in (("A", "A 严格全门通过"),
                                ("B", "B 量纲隔离量特征/保留价格标签")):
        print(f"  口径 {cal_label}:")
        for seg_name, seg_label2 in (("train", "训练"), ("val", "验证"), ("test", "测试")):
            cs = caliber_report[cal_name][seg_name]
            print(f"    {seg_label2}: 有效日={cs['effective_dates']} "
                  f"正={cs['positive']} 负={cs['negative']} "
                  f"跨边界={len(cs['boundary_crossings'])} "
                  f"门槛={'PASS' if cs['all_checks_pass'] else 'FAIL'}")

    # ---- 写报告 ----
    report = {
        "version": "v0.2_full_batch",
        "batch_date": "2026-09-29",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_status": batch_status,
        "status_basis": {
            "decision_caliber": "A_strict_all_gates",
            "rule": "批次状态只由口径 A 按冻结门槛生成; 口径 B 仅对照披露, "
                    "不得在看到样本数后切换口径凑门槛",
            "frozen_candidate_list_unchanged": True,
            "he_locked": True,
            "buy_advice_locked": True,
        },
        "quality_calibers": {
            "A_strict_all_gates": {
                "description": "价格/量额/窗口网格/公司行为全过, 数据在; 不通过整条排除",
                "per_segment": caliber_report["A"],
            },
            "B_volume_isolated_price_retained": {
                "description": "仅量额一项不过、其余全过: 隔离量特征, 保留价格标签 "
                               "(统计方案既定口径); 仅供对照, 不用于批次状态",
                "per_segment": caliber_report["B"],
            },
        },
        "timing": {
            "total": round(t_total, 1),
            "cache_load": round(t_cache_done - t_cache, 1),
            "adj_request": round(t_adj_done - t_adj, 1),
            "minute_request": round(t_min_done - t_min, 1),
            "label_gen": round(t_label_done - t_label, 1),
            "engineering_gates": round(t_gate_done - t_gate, 1),
        },
        "requests": {
            "adj_hit": adj_hit, "adj_new": adj_new, "adj_fail": adj_fail,
            "minute_hit": min_hit, "minute_new": min_new,
            "minute_source_missing": min_source_missing, "minute_fail": min_fail,
        },
        "disk": disk_report,
        "labels": {
            "signal_found": n_sig, "reclaimed_at_s": n_reclaimed_at_s,
            "eligible": n_eligible, "reclaimed": n_reclaimed,
            "not_reclaimed": n_not_reclaimed, "no_signal": n_no_signal,
            "e_day_missing": n_missing,
        },
        "engineering_gates": {
            "gate1_eligibility": {"name": g1["name"], "summary": g1["summary"],
                                   "results": g1["results"]},
            "gate2_dedup": {"name": gates_report["gate2_dedup"]["name"],
                            "summary": g2["summary"]},
            "gate3_scale": {"name": g3["name"], "summary": g3["summary"],
                            "results": g3["results"]},
        },
        "segment_samples": {
            "train": train_seg, "val": val_seg, "test": test_seg,
            "train_dates": len(train_dates),
            "val_dates": len(val_dates),
            "test_dates": len(test_dates),
        },
        "per_candidate": results,
    }
    out_f = OUT / "car_v0_2_full_batch_report.json"

    def _serialize():
        return json.dumps(report, ensure_ascii=False, indent=2, default=str)

    out_f.write_text(_serialize(), encoding="utf-8")
    # 写入后 footprint 才包含最终报告自身; 校正一次并复写 (数字位数稳定即收敛)
    final_footprint = _measure_footprint()
    if final_footprint != disk_report["footprint_bytes"]:
        disk_report.update(_disk_report_with(final_footprint))
        out_f.write_text(_serialize(), encoding="utf-8")
        corrected = _measure_footprint()
        disk_report.update(_disk_report_with(corrected))
        out_f.write_text(_serialize(), encoding="utf-8")

    print(f"\n报告: {out_f.name}")
    print(f"最终磁盘 footprint: {disk_report['footprint_mb']} MB "
          f"(≤50MB: {disk_report['within_cap']})")
    print(f"批次: {batch_status}; H-E 与买入建议继续锁定; 冻结清单未改动.")
    print("如需推进, 下一步是签署不按标签结果挑选的追加抽样方案 "
          "(明确新增预算与固定质量口径, 不重抽测试集).")


if __name__ == "__main__":
    main()
