"""CAR v0.2 提取器: 用新状态机重新扫描 33 候选的现有缓存, 生成新标签.

不增加请求, 不训练, 不调整阈值. 直接调用标签函数生成 v0.2 标签.
与 v0.1 逐候选对账, 输出差异报告.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import get_daily_parquet  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
import gcp_car_v0_1_labels as L  # noqa: E402

CONTRACTS = HERE / "contracts"
SNAP_CAR = HERE / "gcp_round_b_small_sample_v1" / "minute_car"
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"
ADJ_CACHE = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"
OUT = HERE / "gcp_round_b_small_sample_v1"


def detect_signal_v02(minute_df: pd.DataFrame, zones: list, atr_ref: float,
                      ref_price: float) -> dict | None:
    """v0.2 信号检测: APPROACH 仅观察, 跌破且未收回才触发.

    与 v0.1 差异: APPROACH 不返回事件; 首个 low<=zone_lower 的 bar 才触发.
    """
    tz = E.target_zone(zones, ref_price)
    if minute_df is None or len(minute_df) == 0 or tz is None or not np.isfinite(atr_ref):
        return None
    df = minute_df.sort_values("date")
    ts = df["date"].astype(int).to_numpy()
    o = df["open"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    cl = df["close"].astype(float).to_numpy()
    band = E.APPROACH_ATR * atr_ref
    first_in_window = True
    for i in range(len(df)):
        h = E.hhmm(ts[i])
        if not (E.MONITOR_START <= h <= E.MONITOR_END):
            continue
        if first_in_window and o[i] <= tz["zone_lower"]:
            # 跳空已在下方
            if cl[i] >= tz["zone_upper"]:
                # 跳空即收回 → RECLAIMED_AT_S (v0.2 不触发, 返回标记)
                return {"signal_ts": int(ts[i]), "signal_type": "RECLAIMED_AT_S",
                        "session": "AM" if h <= 1130 else "PM",
                        "zone_sources": tz["zone_sources"], "zone_id": tz["zone_id"],
                        "zone_upper": tz["zone_upper"], "zone_lower": tz["zone_lower"],
                        "signal_close": float(cl[i]),
                        "d_atr": (cl[i] - tz["zone_upper"]) / atr_ref,
                        "pen_atr_bar": max(0.0, (tz["zone_lower"] - lo[i]) / atr_ref)}
            sig_type = "ALREADY_BELOW"
        elif lo[i] <= tz["zone_lower"]:
            # 跌破 (含同 bar 收回)
            if cl[i] >= tz["zone_upper"]:
                # 同 bar 跌破收回 → RECLAIMED_AT_S
                return {"signal_ts": int(ts[i]), "signal_type": "RECLAIMED_AT_S",
                        "session": "AM" if h <= 1130 else "PM",
                        "zone_sources": tz["zone_sources"], "zone_id": tz["zone_id"],
                        "zone_upper": tz["zone_upper"], "zone_lower": tz["zone_lower"],
                        "signal_close": float(cl[i]),
                        "d_atr": (cl[i] - tz["zone_upper"]) / atr_ref,
                        "pen_atr_bar": max(0.0, (tz["zone_lower"] - lo[i]) / atr_ref)}
            sig_type = "BREAK"
        elif lo[i] <= tz["zone_upper"] + band and lo[i] > tz["zone_lower"]:
            # APPROACH: v0.2 仅观察, 不触发
            first_in_window = False
            continue
        else:
            first_in_window = False
            continue
        return {"signal_ts": int(ts[i]), "signal_type": sig_type,
                "session": "AM" if h <= 1130 else "PM",
                "zone_sources": tz["zone_sources"], "zone_id": tz["zone_id"],
                "zone_upper": tz["zone_upper"], "zone_lower": tz["zone_lower"],
                "signal_close": float(cl[i]),
                "d_atr": (cl[i] - tz["zone_upper"]) / atr_ref,
                "pen_atr_bar": max(0.0, (tz["zone_lower"] - lo[i]) / atr_ref)}
    return None


def main():
    # ---- 加载数据 ----
    sampling = json.loads(
        (CONTRACTS / "car_v0_1_sampling_summary.json").read_text(encoding="utf-8")
    )
    candidates = sampling["selected"]

    # 交易日历
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    # 分钟缓存
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

    # 网格
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    # v0.1 回放结果 (对账基准)
    replay_v01 = json.loads(
        (OUT / "car_v0_1_real_replay_report.json").read_text(encoding="utf-8")
    )
    v01_map = {}
    for r in replay_v01["per_candidate"]:
        key = (str(r["code"]).zfill(6), int(r["E"]))
        v01_map[key] = r

    # ---- v0.2 重新扫描 ----
    results_v02 = []
    diffs = []

    for c in candidates:
        code = str(c["code"]).zfill(6)
        e_date = int(c["E"])
        p_date = int(c["P"])
        setup = c["setup"]
        key = (code, e_date)

        # 加载分钟数据
        e_df = minute_cache.get(key, [None])[0]
        if e_df is None:
            results_v02.append({
                "code": code, "E": e_date, "setup": setup,
                "signal_found": False, "replay_error": "SOURCE_MISSING",
            })
            continue
        e_df = e_df.sort_values("date").reset_index(drop=True)

        # 构建 zones
        i_e = cal_idx.get(e_date, -1)
        if i_e < 0:
            results_v02.append({
                "code": code, "E": e_date, "setup": setup,
                "signal_found": False, "replay_error": "E_NOT_IN_CAL",
            })
            continue
        lookback = cal[max(0, i_e - 250):i_e]
        d_sub = panel[(panel["code"].astype(str).str.zfill(6) == code)
                      & (panel["date"].astype(int).isin(lookback))].sort_values("date").reset_index(drop=True)
        if len(d_sub) < 25:
            results_v02.append({
                "code": code, "E": e_date, "setup": setup,
                "signal_found": False, "replay_error": "INSUFFICIENT_HISTORY",
            })
            continue
        atr_ref = float(E.build_atr_ref(d_sub))
        zones = E.build_zones(d_sub, atr_ref)
        ref_price = float(d_sub["close"].iloc[-1])

        # v0.2 信号检测
        sig_v02 = detect_signal_v02(e_df, zones, atr_ref, ref_price)
        v01_r = v01_map.get(key, {})

        # v0.1 信号 (复现)
        sig_v01 = E.detect_signal(e_df, zones, atr_ref, ref_price)

        # 记录差异
        v01_type = sig_v01.get("signal_type") if sig_v01 else None
        v02_type = sig_v02.get("signal_type") if sig_v02 else None
        v01_ts = sig_v01.get("signal_ts") if sig_v01 else None
        v02_ts = sig_v02.get("signal_ts") if sig_v02 else None

        diff = {
            "code": code, "E": e_date, "setup": setup,
            "v01_type": v01_type, "v01_ts": v01_ts,
            "v02_type": v02_type, "v02_ts": v02_ts,
            "changed": (v01_type != v02_type) or (v01_ts != v02_ts),
        }

        if sig_v02 is None:
            results_v02.append({
                "code": code, "E": e_date, "setup": setup,
                "signal_found": False, "replay_error": "NO_SIGNAL",
            })
            diffs.append(diff)
            continue

        # RECLAIMED_AT_S: 事件状态, 不传入标签函数计算主标签
        if sig_v02.get("signal_type") == "RECLAIMED_AT_S":
            results_v02.append({
                "code": code, "E": e_date, "setup": setup,
                "signal_found": True, "signal_ts": sig_v02["signal_ts"],
                "sig_type": "RECLAIMED_AT_S",
                "reclaim_status": "RECLAIMED_AT_S",
                "ineligible_reason": "S 时已收回, 不进主标签分母",
            })
            diffs.append(diff)
            continue

        # v0.2 标签计算 (仅对触发事件: ALREADY_BELOW / BREAK)
        fields = L.init_car_label_fields()
        sig_idx = e_df[e_df["date"].astype(int) == sig_v02["signal_ts"]].index
        if len(sig_idx) == 0:
            results_v02.append({
                "code": code, "E": e_date, "setup": setup,
                "signal_found": True, "signal_ts": sig_v02["signal_ts"],
                "sig_type": sig_v02["signal_type"],
                "replay_error": "SIGNAL_TS_NOT_IN_DF",
            })
            diffs.append(diff)
            continue
        sig_idx = sig_idx[0]

        ts_arr = e_df["date"].astype(int).to_numpy()
        lo_arr = e_df["low"].astype(float).to_numpy()
        cl_arr = e_df["close"].astype(float).to_numpy()

        # 计算 30m / 60m 标签
        L.compute_future_reclaim_window(
            sig_v02, ts_arr, lo_arr, cl_arr, sig_idx, atr_ref,
            grid, e_date, 30, "30m", fields, cal)
        L.compute_future_reclaim_window(
            sig_v02, ts_arr, lo_arr, cl_arr, sig_idx, atr_ref,
            grid, e_date, 60, "60m", fields, cal)

        # hold 标签
        reclaim_ts_60m = None
        if fields.get("future_reclaim_60m") == 1 and fields.get("time_to_first_future_reclaim_60m"):
            ttr = fields["time_to_first_future_reclaim_60m"]
            future_slots = L._build_future_window_slots(
                int(str(sig_v02["signal_ts"])[-6:]), e_date, 60, grid, cal)
            if 0 < ttr <= len(future_slots):
                reclaim_ts_60m = int(f"{future_slots[ttr-1][0]}{future_slots[ttr-1][1]:06d}")
        L.compute_future_hold(
            sig_v02, ts_arr, lo_arr, cl_arr, sig_idx, atr_ref,
            grid, e_date, reclaim_ts_60m, fields, cal)

        result = {
            "code": code, "E": e_date, "setup": setup,
            "signal_found": True,
            "signal_ts": sig_v02["signal_ts"],
            "sig_type": sig_v02["signal_type"],
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
        }
        results_v02.append(result)
        diffs.append(diff)

    # ---- 对账报告 ----
    print("=" * 60)
    print("CAR v0.2 提取器: 重新扫描 33 候选缓存")
    print("=" * 60)

    n_changed = sum(1 for d in diffs if d["changed"])
    print(f"信号变化: {n_changed}/{len(diffs)} 条")

    # 统计 v0.2 分流
    v02_stats = {"A_SDP": {"total": 0, "signal": 0, "no_signal": 0,
                            "RECLAIMED_AT_S": 0, "ALREADY_BELOW": 0, "BREAK": 0,
                            "eligible": 0, "reclaimed": 0, "not_reclaimed": 0},
                 "B_LVB": {"total": 0, "signal": 0, "no_signal": 0,
                            "RECLAIMED_AT_S": 0, "ALREADY_BELOW": 0, "BREAK": 0,
                            "eligible": 0, "reclaimed": 0, "not_reclaimed": 0}}

    for r in results_v02:
        setup = r["setup"]
        s = v02_stats[setup]
        s["total"] += 1
        if not r.get("signal_found"):
            s["no_signal"] += 1
            continue
        s["signal"] += 1
        sig_t = r.get("sig_type", "")
        if sig_t in ("RECLAIMED_AT_S",):
            s["RECLAIMED_AT_S"] += 1
        elif sig_t in ("ALREADY_BELOW", "BREAK"):
            s[sig_t] += 1
            # 主标签
            status = r.get("reclaim_status", "")
            if status == "RECLAIMED":
                s["eligible"] += 1
                s["reclaimed"] += 1
            elif status == "NOT_RECLAIMED":
                s["eligible"] += 1
                s["not_reclaimed"] += 1

    print("\n[v0.2 分流统计]")
    for setup in ("A_SDP", "B_LVB"):
        s = v02_stats[setup]
        print(f"  {setup}: 候选={s['total']}, 信号={s['signal']}, 无信号={s['no_signal']}")
        print(f"    RECLAIMED_AT_S={s['RECLAIMED_AT_S']}, ALREADY_BELOW={s['ALREADY_BELOW']}, BREAK={s['BREAK']}")
        print(f"    ELIGIBLE={s['eligible']} (收回={s['reclaimed']}, 未收回={s['not_reclaimed']})")

    # 对账明细
    print("\n[逐候选对账 (差异项)]")
    for d in diffs:
        if d["changed"]:
            print(f"  {d['code']}@{d['E']}: v01={d['v01_type']}({d['v01_ts']}) → v02={d['v02_type']}({d['v02_ts']})")

    # 写报告
    report = {
        "version": "v0.2",
        "date": "2026-09-29",
        "per_candidate": results_v02,
        "diffs": diffs,
        "stats": v02_stats,
        "note": "v0.2 状态机: APPROACH 仅观察; 跌破且未收回才触发; 不增加请求不训练",
    }
    out_f = OUT / "car_v0_2_replay_report.json"
    out_f.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")
    print(f"\n报告已写入: {out_f.name}")


if __name__ == "__main__":
    main()
