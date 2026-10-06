"""CAR v0.1 三道行为工程门核销 (2026-09-29, 用户审批后执行).

用户原话要求 (不增加请求, 使用现有缓存):
  1. 资格映射: 33 条逐一核对 (code,P,setup) 与冻结池, 确认 E 是下一交易日;
     6 条 NO_SIGNAL 也保留资格核验.
  2. 事件去重: 验证每 (code,E) 至多一个物理事件, 多锚点/多周期不能增加独立样本数.
  3. 价格量额尺度: 日线锚点/ATR 与分钟价格同尺度; 检查量额单位、零量/缺槽及公司行为处理.
     仅保存复权链不代表已经正确应用.

执行边界:
  - 只读现有缓存, 不增加请求, 不训练
  - H-E 锁定保持, 执行分流未验证前不放开
  - BREAK=0 是研究覆盖问题, 不改阈值
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import get_daily_parquet  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402

CONTRACTS = HERE / "contracts"
SNAP_CAR = HERE / "gcp_round_b_small_sample_v1" / "minute_car"
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"
ADJ_CACHE = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"
OUT = HERE / "gcp_round_b_small_sample_v1"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"


# ================================================================
# 生产小函数: 可注入合成输入, 供 main() 和合成测试共同调用
# ================================================================

def check_gap_propagation(
    e_df: pd.DataFrame,
    next_df: "pd.DataFrame | None",
    sig_ts: int,
    ttr_60m: "int | None",
    grid: list,
    cal: list,
    e_date: int,
) -> dict:
    """检查缺槽在标签窗口内的传播 (逐窗口实查, 使用完整冻结网格).

    使用 _build_future_window_slots 构造跨日 (day, slot) 窗口,
    不再手写跨日偏移公式; 下一日网格用 next_day 自身的 expected_grid.

    所有返回键始终存在 (不触发 KeyError).
    """
    from gcp_car_v0_1_labels import _build_future_window_slots, _slot_of, _day_of

    sig_slot = _slot_of(sig_ts)
    sig_day = _day_of(sig_ts)

    # ---- E日网格校验 ----
    e_check = E.validate_minute_day(e_df, grid, e_date)
    e_missing_slots = set(e_check.get("missing", []))
    e_missing_pairs = {(e_date, s) for s in e_missing_slots}

    # ---- 下日网格校验 (用 next_day 自身的 expected_grid, 非 E日) ----
    next_day = None
    next_missing_slots: set = set()
    next_missing_pairs: set = set()
    if next_df is not None and e_date in cal:
        nd_idx = cal.index(e_date) + 1
        if 0 <= nd_idx < len(cal):
            next_day = cal[nd_idx]
            nd_check = E.validate_minute_day(next_df, grid, next_day)
            next_missing_slots = set(nd_check.get("missing", []))
            next_missing_pairs = {(next_day, s) for s in next_missing_slots}

    all_missing = e_missing_pairs | next_missing_pairs

    # ---- 构造窗口 (day, slot) 对, 跨日正确 ----
    # 特征前缀: 开盘至 S (含 S), 仅 E日
    exp_e = E.expected_grid(grid, e_date)
    prefix_pairs: set = set()
    if sig_slot in exp_e:
        sig_gi = exp_e.index(sig_slot)
        prefix_pairs = {(e_date, s) for s in exp_e[:sig_gi + 1]}

    # 30m / 60m 窗口 (跨日, 用 _build_future_window_slots)
    win60_list = _build_future_window_slots(sig_slot, sig_day, 60, grid, cal)
    win60_set = set(win60_list)
    win30_set = set(win60_list[:30])

    # hold 窗口: 首次收回后 10 个交易分钟 (跨日)
    hold_set: set = set()
    hold_start_slot = None
    if ttr_60m is not None and win60_list:
        idx = int(ttr_60m) - 1  # ttr 是 1-indexed
        if 0 <= idx < len(win60_list):
            reclaim_day, reclaim_slot = win60_list[idx]
            hold_start_slot = reclaim_slot
            hold_set = set(_build_future_window_slots(
                reclaim_slot, reclaim_day, 10, grid, cal))

    # ---- 逐窗口检查缺槽交集 ----
    gap_prefix = sorted(prefix_pairs & all_missing)
    gap_30m = sorted(win30_set & all_missing)
    gap_60m = sorted(win60_set & all_missing)
    gap_hold = sorted(hold_set & all_missing)

    # 下日细分
    nd_30m = sorted(next_missing_pairs & win30_set) if next_day else []
    nd_60m = sorted(next_missing_pairs & win60_set) if next_day else []
    nd_hold = sorted(next_missing_pairs & hold_set) if next_day else []

    return {
        "e_day_missing": sorted(e_missing_slots),
        "next_day_missing": sorted(next_missing_slots) if next_day else [],
        "prefix_to_S": gap_prefix,
        "30m": gap_30m,
        "60m": gap_60m,
        "hold_10m": gap_hold,
        "hold_start_slot": hold_start_slot,
        "time_to_first_reclaim": ttr_60m,
        "next_day": (
            {
                "date": next_day,
                "missing": sorted(next_missing_slots),
                "30m": nd_30m,
                "60m": nd_60m,
                "hold_10m": nd_hold,
            } if next_day is not None else None
        ),
    }


def check_corp_action(
    adj_entry: dict,
    p_date: int,
    label_end_ts: "int | None",
) -> dict:
    """公司行为三态判定 (链可用无事件 / 链不可得UNKNOWN / 有事件隔离).

    研究窗口 = [P+1, label_end_date]; 窗口外除权不误伤当前标签.
    链不可得 → UNKNOWN (不能当无事件); 窗口内除权 → 隔离.
    """
    label_end_date = int(str(int(label_end_ts))[:8]) if label_end_ts else None

    adj_ok = adj_entry.get("ok", False)
    chain = adj_entry.get("chain", {}) if adj_ok else None

    if not adj_ok or chain is None:
        return {
            "adj_ok": False,
            "status": "ADJ_UNKNOWN",
            "pass": False,
            "events_in_window": [],
            "events_outside_window": [],
            "label_end_date": label_end_date,
            "note": "复权链不可得; 隔离收益修正, 不能当无事件",
        }

    events_in: list = []
    events_out: list = []
    upper = label_end_date if label_end_date is not None else p_date
    if isinstance(chain, dict):
        for d_str, factor in chain.items():
            d_int = int(d_str)
            if d_int > p_date and d_int <= upper:
                events_in.append({"date": d_str, "factor": factor})
            elif d_int > upper:
                events_out.append({"date": d_str, "factor": factor})

    if events_in:
        status = "HAS_POST_P_EVENTS"
        adj_pass = False
        note = "研究窗口内跨除权事件; 隔离收益修正, 合成正例已通过, 真实样本未覆盖窗口内除权"
    else:
        status = "NO_EVENT_IN_WINDOW"
        adj_pass = True
        note = ("链可用且研究窗口无事件; 本样本无需转换; 合成正例已通过, 真实样本未覆盖窗口内除权"
                + (f"; 窗口后有{len(events_out)}个除权事件不误伤当前标签"
                   if events_out else ""))

    return {
        "adj_ok": True,
        "status": status,
        "pass": adj_pass,
        "events_in_window": events_in,
        "events_outside_window": events_out,
        "label_end_date": label_end_date,
        "note": note,
    }


def main():
    # ---- 加载数据 ----
    sampling = json.loads(
        (CONTRACTS / "car_v0_1_sampling_summary.json").read_text(encoding="utf-8")
    )
    candidates = sampling["selected"]  # 33 条, 每条有 code/setup/P/E/...
    report_replay = json.loads(
        (OUT / "car_v0_1_real_replay_report.json").read_text(encoding="utf-8")
    )
    replay_results = report_replay["per_candidate"]

    # 冻结池成员
    pool_a = pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")
    # 构建查找集合: (code, date, setup_type)
    pool_a_set = set(
        zip(pool_a["code"].astype(str).str.zfill(6),
            pool_a["date"].astype(int),
            pool_a["setup_type"].astype(str))
    )
    pool_b_set = set(
        zip(pool_b["code"].astype(str).str.zfill(6),
            pool_b["date"].astype(int),
            pool_b["setup_type"].astype(str))
    )
    pool_set = pool_a_set | pool_b_set

    # 交易日历
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_set = set(cal)
    cal_idx = {d: i for i, d in enumerate(cal)}

    # 分钟缓存 (与回放脚本一致: 三目录合并)
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

    # 复权链
    adj_data = json.loads(ADJ_CACHE.read_text(encoding="utf-8"))

    # 网格
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    gates_report = {
        "gate1_eligibility": {"name": "资格映射", "results": [], "summary": {}},
        "gate2_dedup": {"name": "事件去重", "results": [], "summary": {}},
        "gate3_scale": {"name": "价格量额尺度", "results": [], "summary": {}},
    }

    # ================================================================
    # 第一道门: 资格映射
    # ================================================================
    print("=" * 60)
    print("[门1] 资格映射: 33 条逐一核对 (code,P,setup) 与冻结池, E=下一交易日")
    print("=" * 60)
    g1 = gates_report["gate1_eligibility"]
    n_pass = 0
    n_fail = 0
    for c in candidates:
        code = str(c["code"]).zfill(6)
        setup = c["setup"]
        p_date = int(c["P"])
        e_date = int(c["E"])
        reasons = []

        # 1a. P 在交易日历中
        if p_date not in cal_set:
            reasons.append(f"P_NOT_IN_CAL ({p_date})")
        # 1b. E 在交易日历中
        if e_date not in cal_set:
            reasons.append(f"E_NOT_IN_CAL ({e_date})")
        # 1c. E = cal[cal_idx[P]+1] (E 是 P 的下一交易日)
        if p_date in cal_idx and e_date in cal_idx:
            expected_e = cal[cal_idx[p_date] + 1] if cal_idx[p_date] + 1 < len(cal) else None
            if expected_e != e_date:
                reasons.append(f"E_NOT_NEXT_DAY (expected={expected_e}, got={e_date})")
        else:
            if p_date not in cal_idx:
                reasons.append("P_NOT_IN_CAL_IDX")
            if e_date not in cal_idx:
                reasons.append("E_NOT_IN_CAL_IDX")

        # 1d. (code, P, setup) 在冻结池中
        in_pool = (code, p_date, setup) in pool_set
        if not in_pool:
            # 检查是否在另一个池中 (setup 不匹配)
            in_a = (code, p_date, "A_SDP") in pool_a_set
            in_b = (code, p_date, "B_LVB") in pool_b_set
            if in_a and setup != "A_SDP":
                reasons.append(f"POOL_MISMATCH (in A_SDP but setup={setup})")
            elif in_b and setup != "B_LVB":
                reasons.append(f"POOL_MISMATCH (in B_LVB but setup={setup})")
            else:
                reasons.append("NOT_IN_ANY_POOL")

        # 1e. 核对该候选在回放中的状态 (NO_SIGNAL 也核验资格)
        replay_match = next(
            (r for r in replay_results if str(r.get("code","")).zfill(6) == code
             and r.get("E") == e_date), None
        )
        replay_status = "NOT_IN_REPLAY"
        if replay_match:
            if replay_match.get("signal_found"):
                replay_status = replay_match.get("sig_type", "SIGNAL_FOUND")
            elif replay_match.get("replay_error") == "NO_SIGNAL":
                replay_status = "NO_SIGNAL"
            else:
                replay_status = replay_match.get("replay_error", "UNKNOWN")

        ok = len(reasons) == 0
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        g1["results"].append({
            "code": code, "setup": setup, "P": p_date, "E": e_date,
            "in_pool": in_pool, "e_is_next_day": (expected_e == e_date) if p_date in cal_idx and e_date in cal_idx else False,
            "replay_status": replay_status,
            "pass": ok,
            "reasons": reasons,
        })
        status = "PASS" if ok else "FAIL"
        print(f"  {code} {setup} P={p_date} E={e_date}: {status} (pool={in_pool}, replay={replay_status})"
              + (f"  [{', '.join(reasons)}]" if reasons else ""))

    g1["summary"] = {"total": len(candidates), "pass": n_pass, "fail": n_fail}
    print(f"  --> 资格映射: {n_pass}/{len(candidates)} PASS")
    if n_fail:
        print(f"  [警告] {n_fail} 条资格映射失败")

    # ================================================================
    # 第二道门: 事件去重
    # ================================================================
    print("\n" + "=" * 60)
    print("[门2] 事件去重: 每 (code,E) 至多一个物理事件")
    print("=" * 60)
    g2 = gates_report["gate2_dedup"]

    # 2a. 候选清单中 (code, E) 唯一性
    ce_pairs = [(str(c["code"]).zfill(6), int(c["E"])) for c in candidates]
    ce_counts = Counter(ce_pairs)
    dup_candidates = [(k, v) for k, v in ce_counts.items() if v > 1]

    # 2b. 回放中每 (code, E) 信号唯一性 (detect_signal 每日一信号)
    signal_results = []
    n_dup_signal = 0
    for r in replay_results:
        if not r.get("signal_found"):
            continue
        code = str(r.get("code", "")).zfill(6)
        e_date = r["E"]
        sig_type = r.get("sig_type")
        sig_ts = r.get("signal_ts")
        zone_sources = None
        # 重新运行 detect_signal 获取 zone_sources
        try:
            e_key = (code, e_date)
            if e_key in minute_cache and minute_cache[e_key]:
                e_df = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)
                i_e = cal_idx[e_date]
                lookback = cal[max(0, i_e - 250):i_e]
                d_sub = panel[(panel["code"].astype(str).str.zfill(6) == code)
                              & (panel["date"].astype(int).isin(lookback))].sort_values("date").reset_index(drop=True)
                if len(d_sub) >= 25:
                    atr_ref = float(E.build_atr_ref(d_sub))
                    zones = E.build_zones(d_sub, atr_ref)
                    ref_price = float(d_sub["close"].iloc[-1])
                    sig = E.detect_signal(e_df, zones, atr_ref, ref_price)
                    if sig is not None:
                        zone_sources = sig.get("zone_sources", [])
        except Exception:
            pass
        signal_results.append({
            "code": code, "E": e_date, "sig_type": sig_type,
            "sig_ts": sig_ts, "zone_sources": zone_sources,
        })

    # 2c. 检查信号时间戳唯一性 (同一 (code,E) 不应有多个 sig_ts)
    sig_ts_per_ce = {}
    for s in signal_results:
        key = (s["code"], s["E"])
        sig_ts_per_ce.setdefault(key, []).append(s["sig_ts"])
    multi_sig = {k: v for k, v in sig_ts_per_ce.items() if len(v) > 1}

    # 2d. 多锚点不增加独立样本 (zone_sources 有多个但只产生一个事件)
    multi_anchor = [s for s in signal_results if s["zone_sources"] and len(s["zone_sources"]) > 1]

    g2["results"] = signal_results
    g2["summary"] = {
        "candidate_dup_ce": dup_candidates,
        "candidate_ce_unique": len(dup_candidates) == 0,
        "signal_count": len(signal_results),
        "multi_signal_per_ce": multi_sig,
        "signal_unique_per_ce": len(multi_sig) == 0,
        "multi_anchor_events": len(multi_anchor),
        "multi_anchor_note": "多锚点合并为单事件 (合同§3.6), 不增加独立样本数",
    }
    print(f"  候选 (code,E) 唯一性: {'PASS' if len(dup_candidates)==0 else 'FAIL'} ({len(ce_pairs)} 条, {len(dup_candidates)} 重复)")
    print(f"  信号唯一性 (每(code,E)至多一个 sig_ts): {'PASS' if len(multi_sig)==0 else 'FAIL'}")
    print(f"  多锚点合并事件: {len(multi_anchor)} 条 (合并为单事件, 不增加样本)")
    print(f"  --> 事件去重: {'PASS' if len(dup_candidates)==0 and len(multi_sig)==0 else 'FAIL'}")

    # ================================================================
    # 第三道门: 价格量额尺度 (拆分四项独立判定)
    #   3a 价格尺度: E日日线OHLC vs 分钟聚合OHLC对拍
    #   3b 量额单位: amount/(volume*close)中位数检测量纲异常
    #   3c 窗口网格: 引擎validate_minute_day + 内部缺槽逐窗口传播
    #   3d 公司行为: 复权链登记 + 合成除权案例验证
    # ================================================================
    print("\n" + "=" * 60)
    print("[门3] 价格量额尺度 (拆分四项: 价格尺度/量额单位/窗口网格/公司行为)")
    print("=" * 60)
    g3 = gates_report["gate3_scale"]
    scale_results = []
    # 四项各自判定
    sub_pass = {"price": 0, "volume": 0, "grid": 0, "adj": 0}

    for c in candidates:
        code = str(c["code"]).zfill(6)
        setup = c["setup"]
        e_date = int(c["E"])
        p_date = int(c["P"])
        item = {"code": code, "E": e_date, "sub_checks": {}, "pass": True}

        # 加载 E 日日线和分钟
        daily_e = panel[(panel["code"].astype(str).str.zfill(6) == code)
                        & (panel["date"].astype(int) == e_date)]
        e_key = (code, e_date)
        if e_key not in minute_cache or not minute_cache[e_key] or daily_e.empty:
            item["sub_checks"]["data_missing"] = True
            item["pass"] = False
            scale_results.append(item)
            print(f"  {code}@{e_date}: SKIP (data missing)")
            continue
        e_df = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)
        d_close = float(daily_e["close"].iloc[0])
        d_high = float(daily_e["high"].iloc[0])
        d_low = float(daily_e["low"].iloc[0])
        d_open = float(daily_e["open"].iloc[0])
        d_vol = float(daily_e["volume"].iloc[0])
        d_amt = float(daily_e["amount"].iloc[0])

        # ---- 3a 价格尺度: E日日线OHLC vs 分钟聚合OHLC对拍 ----
        m_open = float(e_df["open"].iloc[0])
        m_high = float(e_df["high"].max())
        m_low = float(e_df["low"].min())
        m_close = float(e_df["close"].iloc[-1])
        # OHLC 各项相对差异
        def _rdiff(a, b):
            return abs(a - b) / (max(abs(a), abs(b)) + 1e-12)
        ohlc_rdiffs = {
            "open": _rdiff(d_open, m_open),
            "high": _rdiff(d_high, m_high),
            "low": _rdiff(d_low, m_low),
            "close": _rdiff(d_close, m_close),
        }
        # 通过条件: 每项差异 < 1% (允许微小聚合误差)
        price_ok = all(v < 0.01 for v in ohlc_rdiffs.values())
        # 也检查历史日线尺度 (ATR_ref 合理性)
        try:
            i_e = cal_idx[e_date]
            lookback = cal[max(0, i_e - 250):i_e]
            d_sub = panel[(panel["code"].astype(str).str.zfill(6) == code)
                          & (panel["date"].astype(int).isin(lookback))].sort_values("date").reset_index(drop=True)
            atr_ref = float(E.build_atr_ref(d_sub))
            atr_ratio = atr_ref / d_close
            atr_ok = 0.001 < atr_ratio < 0.30
        except Exception:
            atr_ref, atr_ratio, atr_ok = None, None, False
        price_pass = price_ok and atr_ok
        item["sub_checks"]["price_scale"] = {
            "daily_ohlc": {"open": d_open, "high": d_high, "low": d_low, "close": d_close},
            "minute_agg_ohlc": {"open": m_open, "high": m_high, "low": m_low, "close": m_close},
            "ohlc_rdiffs": ohlc_rdiffs,
            "atr_ref": atr_ref, "atr_ratio": atr_ratio,
            "pass": price_pass,
        }
        if price_pass:
            sub_pass["price"] += 1
        else:
            item["pass"] = False

        # ---- 3b 量额单位: amount/(volume*close)中位数检测 ----
        # 正常应 ≈1.0; 远离1.0说明volume或amount量纲异常
        ratio_arr = e_df["amount"].astype(float) / (
            e_df["volume"].astype(float) * e_df["close"].astype(float) + 1e-12)
        # 排除零量bar
        valid = ratio_arr[e_df["volume"] > 0]
        vol_amt_med = float(np.median(valid)) if len(valid) > 0 else None
        # amount 汇总一致性
        amt_ratio = float(e_df["amount"].sum()) / (d_amt + 1e-12)
        # volume 汇总一致性
        vol_ratio = float(e_df["volume"].sum()) / (d_vol + 1e-12)
        # 判定逻辑: amount汇总一致 + 量价关系正常 => PASS
        #           amount汇总一致但量价关系异常 => VOLUME_SCALE_UNKNOWN
        #           amount汇总不一致 => FAIL
        if not (0.8 < amt_ratio < 1.2):
            vol_status = "AMOUNT_MISMATCH"
            vol_pass = False
        elif vol_amt_med is not None and 0.5 < vol_amt_med < 1.5:
            vol_status = "CONSISTENT"
            vol_pass = True
        elif vol_amt_med is not None and 0.005 < vol_amt_med < 0.05:
            # volume 放大约100倍
            vol_status = "VOLUME_SCALE_UNKNOWN"
            vol_pass = False
        else:
            vol_status = "VOLUME_SCALE_UNKNOWN"
            vol_pass = False
        item["sub_checks"]["volume_scale"] = {
            "amt_vol_close_median": vol_amt_med,
            "amount_ratio": amt_ratio,
            "volume_ratio": vol_ratio,
            "status": vol_status,
            "volume_scale_valid": vol_pass,
            "pass": vol_pass,
            "note": ("量价关系正常" if vol_status == "CONSISTENT" else
                     "量纲异常: volume可能放大100x; volume_scale_valid=False, 量及量价衍生字段(VWAP等)置为不可用; 价格行为标签保留" if vol_status == "VOLUME_SCALE_UNKNOWN" else
                     "成交额汇总不一致"),
        }
        # 量特征有效性标志落地: 后续特征入口消费此标志
        item["volume_scale_valid"] = vol_pass
        if not vol_pass:
            item["volume_invalid_fields"] = ["volume", "amount", "vwap", "vol_ratio", "amount_ratio"]
        if vol_pass:
            sub_pass["volume"] += 1
        else:
            item["pass"] = False

        # ---- 3c 窗口网格: 引擎validate_minute_day + 缺槽逐窗口传播 ----
        grid_check = E.validate_minute_day(e_df, grid, e_date)
        zero_vol_bars = int((e_df["volume"] == 0).sum())
        replay_match = next(
            (r for r in replay_results if str(r.get("code", "")).zfill(6) == code
             and r.get("E") == e_date), None
        )
        # 缺槽逐窗口实查: 调用生产小函数 check_gap_propagation
        # (使用 _build_future_window_slots 构造跨日窗口, 下日网格用自身 expected_grid)
        gap_in_window = None
        if replay_match and replay_match.get("signal_ts"):
            sig_ts_val = int(replay_match["signal_ts"])
            ttr_val = replay_match.get("time_to_first_reclaim_60m")
            # 加载下一交易日分钟数据
            next_df_val = None
            nd_idx_val = cal_idx.get(e_date, -1) + 1
            if nd_idx_val < len(cal):
                nd_val = cal[nd_idx_val]
                nd_key = (code, nd_val)
                if nd_key in minute_cache and minute_cache[nd_key]:
                    next_df_val = minute_cache[nd_key][0]
            gap_in_window = check_gap_propagation(
                e_df, next_df_val, sig_ts_val, ttr_val, grid, cal, e_date)
        # 零量 bar 登记; 检查是否在信号窗口内 (影响量特征)
        zero_vol_in_window = None
        if replay_match and replay_match.get("signal_ts") and zero_vol_bars > 0:
            sig_ts_zv = int(replay_match["signal_ts"])
            sig_slot_zv = int(str(sig_ts_zv)[-6:])
            exp_zv = E.expected_grid(grid, e_date)
            if sig_slot_zv in exp_zv:
                sig_gi_zv = exp_zv.index(sig_slot_zv)
                win30_zv = set(exp_zv[sig_gi_zv + 1: sig_gi_zv + 31])
                win60_zv = set(exp_zv[sig_gi_zv + 1: sig_gi_zv + 61])
                zero_slots = set(
                    int(str(int(t))[-6:]) for t, v in zip(e_df["date"], e_df["volume"]) if int(v) == 0
                )
                zero_vol_in_window = {
                    "30m": sorted(win30_zv & zero_slots),
                    "60m": sorted(win60_zv & zero_slots),
                }
        # 网格通过判定: 缺槽不在标签窗口内 (30m/60m/hold_10m)
        # 缺槽在特征前缀 → 登记, 需后续检查前缀特征是否受影响
        # 零量 bar 有效但不等于缺槽, 不一律使行为标签失效
        grid_pass = grid_check["ok"]
        gap_flags = []
        if gap_in_window is not None:
            if gap_in_window["30m"]:
                grid_pass = False
                gap_flags.append("GAP_IN_30M")
            if gap_in_window["60m"]:
                grid_pass = False
                gap_flags.append("GAP_IN_60M")
            if gap_in_window["hold_10m"]:
                grid_pass = False
                gap_flags.append("GAP_IN_HOLD_10M")
            if gap_in_window["prefix_to_S"]:
                gap_flags.append("GAP_IN_PREFIX")
            nd = gap_in_window.get("next_day")
            if nd is not None:
                if nd.get("30m"):
                    grid_pass = False
                    gap_flags.append("GAP_IN_NEXT_DAY_30M")
                if nd.get("60m"):
                    grid_pass = False
                    gap_flags.append("GAP_IN_NEXT_DAY_60M")
                if nd.get("hold_10m"):
                    grid_pass = False
                    gap_flags.append("GAP_IN_NEXT_DAY_HOLD")
                if nd.get("missing") and not (nd.get("30m") or nd.get("60m") or nd.get("hold_10m")):
                    gap_flags.append("GAP_IN_NEXT_DAY_OUTSIDE_WINDOW")
        if zero_vol_bars > 0 and zero_vol_in_window is not None:
            if zero_vol_in_window["30m"] or zero_vol_in_window["60m"]:
                gap_flags.append("ZERO_VOL_IN_WINDOW")
                # 零量 bar 在窗口内: 量特征失效, 但行为标签(价格)可保留
        if gap_flags:
            item["sub_checks"]["grid_note"] = "; ".join(gap_flags)
        item["sub_checks"]["window_grid"] = {
            "grid_ok": grid_check["ok"],
            "grid_reason": grid_check["reason"],
            "missing_slots": grid_check.get("missing", []),
            "n_missing": len(grid_check.get("missing", [])),
            "dups": grid_check.get("dups", []),
            "zero_vol_bars": zero_vol_bars,
            "zero_vol_in_signal_window": zero_vol_in_window,
            "gap_in_signal_window": gap_in_window,
            "pass": grid_pass,
        }
        if grid_pass:
            sub_pass["grid"] += 1
        else:
            item["pass"] = False

        # ---- 3d 公司行为: 三态判定 (链可用无事件/链不可得UNKNOWN/有事件隔离) ----
        # 研究窗口 = [P+1, label_end_date]; 窗口外除权不误伤当前标签
        label_end_ts = replay_match.get("label_end_ts") if replay_match else None
        adj_entry = adj_data.get(code, {})
        # 调用生产小函数 check_corp_action (三态判定 + 窗口上界)
        adj_result = check_corp_action(adj_entry, p_date, label_end_ts)
        adj_status = adj_result["status"]
        adj_pass = adj_result["pass"]
        adj_events_in_window = adj_result["events_in_window"]
        adj_events_outside = adj_result["events_outside_window"]
        label_end_date = adj_result["label_end_date"]
        adj_note = adj_result["note"]
        chain = adj_entry.get("chain", {}) if adj_entry.get("ok") else None
        item["sub_checks"]["corp_action"] = {
            "adj_ok": adj_entry.get("ok", False),
            "n_events": len(chain) if isinstance(chain, dict) else 0,
            "events_in_window": adj_events_in_window,
            "events_outside_window": adj_events_outside,
            "label_end_date": label_end_date,
            "status": adj_status,
            "pass": adj_pass,
            "note": adj_note,
        }
        if adj_pass:
            sub_pass["adj"] += 1
        else:
            item["pass"] = False  # adj 隔离传播到候选整体

        scale_results.append(item)
        flags = []
        if not price_pass:
            flags.append("PRICE")
        if not vol_pass:
            flags.append(vol_status)
        if not grid_pass:
            flags.append("GRID")
        if not adj_pass:
            flags.append(adj_status)
        status = "PASS" if item["pass"] else "FAIL"
        print(f"  {code}@{e_date}: {status}" + (f"  [{', '.join(flags)}]" if flags else ""))

    g3["results"] = scale_results
    n_scale_pass = sum(1 for s in scale_results if s["pass"])
    n_scale_fail = sum(1 for s in scale_results if not s["pass"])
    g3["summary"] = {
        "total": len(scale_results), "pass": n_scale_pass, "fail": n_scale_fail,
        "sub_pass": sub_pass,
        "note": ("门3拆分为四项: 价格尺度/量额单位/窗口网格/公司行为; "
                 "301122@20260306量纲异常记VOLUME_SCALE_UNKNOWN; "
                 "公司行为合成正例已通过, 真实样本未覆盖窗口内除权"),
    }
    print(f"  --> 价格量额尺度: {n_scale_pass}/{len(scale_results)} PASS")
    print(f"      价格尺度: {sub_pass['price']}/{len(scale_results)}")
    print(f"      量额单位: {sub_pass['volume']}/{len(scale_results)}")
    print(f"      窗口网格: {sub_pass['grid']}/{len(scale_results)}")
    print(f"      公司行为: {sub_pass['adj']}/{len(scale_results)}")
    if n_scale_fail:
        print(f"  [警告] {n_scale_fail} 条尺度检查失败")

    # ================================================================
    # 汇总
    # ================================================================
    all_pass = (g1["summary"].get("fail", 0) == 0
                and g2["summary"].get("candidate_ce_unique", False)
                and g2["summary"].get("signal_unique_per_ce", False)
                and n_scale_fail == 0)

    gates_report["overall"] = {
        "all_pass": all_pass,
        "gate1_pass": g1["summary"].get("fail", 0) == 0,
        "gate2_pass": g2["summary"].get("candidate_ce_unique", False) and g2["summary"].get("signal_unique_per_ce", False),
        "gate3_pass": n_scale_fail == 0,
        "note": ("三道门通过后才签统计方案; H-E锁定保持; "
                 "BREAK=0 是研究覆盖问题不改阈值; 不增加请求不训练"),
    }

    # 写报告
    out_f = OUT / "car_v0_1_engineering_gates_report.json"
    out_f.write_text(json.dumps(gates_report, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")
    print("\n" + "=" * 60)
    print("[汇总]")
    print(f"  门1 资格映射: {g1['summary']['pass']}/{g1['summary']['total']} PASS")
    print(f"  门2 事件去重: {'PASS' if g2['summary']['candidate_ce_unique'] and g2['summary']['signal_unique_per_ce'] else 'FAIL'}")
    print(f"  门3 价格量额尺度: {n_scale_pass}/{len(scale_results)} PASS")
    print(f"  总体: {'ALL PASS' if all_pass else 'HAS FAILURES'}")
    print(f"  报告: {out_f.name}")
    print(f"  待办: 三道门通过后签统计方案 → 才训练共享基线")
    print(f"  H-E 锁定保持, 执行分流未验证前不放开")


if __name__ == "__main__":
    main()
