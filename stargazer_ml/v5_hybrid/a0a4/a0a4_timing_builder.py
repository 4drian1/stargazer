"""a0a4_timing_builder — 因果 Timing 母表构建器 (PATH A: 不改引擎, 事件驱动因果重建).

设计 (用户 2026-09-19 红队审计 + 路径A决策):
  - 直接复用现有 a14_universe.parquet 的 15:00 基线行作为 P_CLOSE-safe base (白名单,
    绝不读入旧 D 日动态字段作为重建输入).
  - signal_ts = FIRST_ARMED enter_ts (冻结引擎的 MODEL_SCORE_REQUEST 仅是收盘终态标签,
    enter_ts=15:00, 不是真实盘中事件; 真正因果可打分的盘中时刻是 first ARMED).
  - 以 snapshot_ts = S 重跑 run_state_engine (frozen) -> as-of-S 的 A4 特征;
    以分钟窗口 [open, S] 重算 as-of-S 的 A3 特征; 以 execute(S, m) 得 as-of-S actual_fill.
  - timing 个股收益用解析 ratio rescale (无需重载日线):
        ret_new = (1+ret_old)*fill_old/fill_new - 1
    Market Proxy alpha 复用行内 timing_market_alpha_h1 (D-open->H1-close, 与成交时刻无关,
    属 S 不变量) —— 不重新扫全市场, 不触发 proxy 全市场加载.
  - 输出完整母表 (所有候选):
        timing_eligible=1 -> 真正 Timing 训练样本 (含 as-of-S A3/A4 + exec + label)
        timing_eligible=0 -> NEVER_ARMED / 其他异常, timing 特征为 NaN, 仍保留 P_CLOSE base
    这样可同时报告 Trigger Coverage 与 Conditional Timing Quality.
  - 保留 missing semantics: 绝不 NaN->0 (A3/A4 在 09:30 触发时本就是 unavailable).
  - feature_asof_max_ts = S (<= signal_ts); 全量 3 个硬 Gate 校验 (仅对 eligible 子集).

产物: datasets/timing_universe_causal_v1.parquet  (绝不覆盖旧 A0A4_FIRST_PASS_LEAKED)

S0: 不重建. a14_universe.parquet 直接作为 S0 base, 训练端 track-aware selector 只取 A1/A2.
资源保护: 每 100 候选打印 C/D 剩余空间与 RSS; C<15GB / D<30GB / 异常 -> STOP.
临时目录由 run_py.ps1 重定向到 D 盘, 本脚本不写 C 盘任何文件.
"""
from __future__ import annotations
import json
import sys
import time
import shutil
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:  # noqa
    psutil = None

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))
from zone_state_engine import Zone, run_state_engine  # noqa  (冻结: 仅调用)
from execution_engine import execute  # ⑥ (冻结)
import latr_builder as L  # noqa  (仅用 load_minute / load_adjustment_chain)
import trading_calendar as TC  # noqa

# ---- 旧 D 日动态字段 (重建时一律重新计算 as-of-S, 不作为 base 输入) ----
EXCLUDE_DYNAMIC = {
    "close_position_15m", "recovery_atr", "new_low_flag",
    "dip_depth_5m_atr", "dip_speed_5m", "dip_depth_30m_atr", "dip_speed_30m", "dip_speed_15m",
    "zone_break_depth_atr", "zone_break_duration", "recovery_from_zone_low_atr", "dip_recovery_ratio",
    "reclaim_zone_lower", "reclaim_zone_mid", "reclaim_zone_upper", "reclaim_failure_count",
    "second_test_flag", "second_test_higher_low", "time_since_zone_touch", "time_below_zone",
    "lower_wick_ratio", "sell_volume_ratio", "rebound_volume_ratio", "vwap_reclaim",
    "cum_volume_vs_hist_same_time", "cum_turnover_vs_hist_same_time",
    "mae_30m", "mfe_30m", "mae_day", "mfe_day", "mae_h1", "mfe_h1", "failed_rebound",
    "final_state", "state_transitions_json", "zones_json",
    # A3 基础也重算 (更诚实的 as-of-S 值)
    "open_dip_atr", "total_dip_atr",
}

A4_KEYS = [
    "zone_break_depth_atr", "zone_break_duration", "recovery_from_zone_low_atr",
    "recovery_atr", "dip_recovery_ratio", "reclaim_zone_lower", "reclaim_zone_mid",
    "reclaim_zone_upper", "reclaim_failure_count", "second_test_flag",
    "second_test_higher_low", "time_since_zone_touch", "time_below_zone",
    "lower_wick_ratio", "sell_volume_ratio", "rebound_volume_ratio", "vwap_reclaim",
]
NON_FEATURE = {
    "symbol", "plan_date", "effective_trade_date", "d_date", "sample_id",
    "primary_setup", "dip_stage", "signal_time", "signal_ts", "fill_ts", "actual_fill",
    "feature_asof_max_ts", "pclose_feature_asof_ts", "label_end_ts", "timing_return_h1",
    "timing_alpha_h1", "s0_return_h3", "s0_alpha_h3", "timing_eligible",
    "timing_ineligible_reason", "first_armed_ts", "exec_status", "snapshot_time",
    "final_state", "state_transitions_json", "zones_json", "zone_id",
}
OUT_NAME = "timing_universe_causal_v1.parquet"


def _first_armed(transitions):
    best = None
    for t in transitions:
        if t.get("state") == "ARMED":
            ts = int(t.get("state_enter_ts"))
            if best is None or ts < best:
                best = ts
    return best


def build_timing_row(row, chains, minute_cache):
    """从 a14 基线行重建 as-of-S 的 timing 样本; 始终返回完整行 (含 timing_eligible).

    不触发全市场 proxy. NEVER_ARMED / 异常 -> timing_eligible=0, timing 特征为 NaN.
    """
    code = str(row.get("symbol") or row.get("code")).zfill(6)
    d_date = int(row["effective_trade_date"])
    plan_date = int(row["plan_date"])
    atr_ref = float(row["atr_ref"])
    setup = str(row.get("primary_setup", ""))
    dip_stage = row.get("dip_stage")

    # ---- base whitelist 拷贝 (P_CLOSE-safe; 旧 D 日动态字段已排除) ----
    base = {}
    for c in row.index:
        if c in EXCLUDE_DYNAMIC:
            continue
        base[c] = row[c]
    base["timing_eligible"] = 0
    base["timing_ineligible_reason"] = None
    base["first_armed_ts"] = None
    base["signal_ts"] = None
    base["signal_time"] = None
    base["fill_ts"] = None
    base["actual_fill"] = None
    base["feature_asof_max_ts"] = None
    base["final_state"] = row.get("final_state")
    base["exec_status"] = None
    base["pclose_feature_asof_ts"] = int(f"{plan_date}150000")
    h1_date = TC.next_trading_day(d_date)
    if h1_date is None:  # 尾部右删失 (日历末端算不出 H1) -> 同 build_one_sample 的 CENSORED_TAIL
        base["timing_ineligible_reason"] = "TAIL_CENSORED"
        base["label_end_ts"] = None
        return base, 0
    base["label_end_ts"] = int(f"{int(h1_date)}150000")

    # ---- S = first ARMED enter_ts ----
    tr_json = row.get("state_transitions_json")
    transitions = json.loads(tr_json) if isinstance(tr_json, str) else []
    S = _first_armed(transitions)
    if S is None:
        base["timing_ineligible_reason"] = "NEVER_ARMED"
        return base, 0

    # ---- 分钟 (带缓存, 上限 400 防内存膨胀) ----
    key = (code, d_date)
    m = minute_cache.get(key) if minute_cache is not None else None
    if m is None:
        m = L.load_minute(code, d_date)
        if minute_cache is not None:
            if len(minute_cache) > 400:
                minute_cache.clear()
            minute_cache[key] = m
    if m is None or m.empty:
        base["timing_ineligible_reason"] = "minute_empty"
        return base, 0
    open_ts = int(m["ts"].min())

    # ---- 重构 primary Zone (frozen P_CLOSE) ----
    try:
        zone = Zone(
            zone_id=str(row.get("primary_zone_id")),
            zone_source=str(row.get("primary_zone_source", "")),
            zone_lower=float(row["primary_zone_lower"]),
            zone_upper=float(row["primary_zone_upper"]),
            zone_mid=float(row["primary_zone_mid"]),
            zone_timestamp=int(plan_date),
            zone_width_atr=float(row.get("zone_band_atr", 0.0) or 0.0),
            zone_strength=int(row.get("zone_strength", 1) or 1),
            zone_source_count=1,
            zone_distance_atr=float(row.get("primary_zone_distance_atr", 0.0) or 0.0),
        )
    except Exception as e:  # noqa
        base["timing_ineligible_reason"] = f"zone_build_err:{type(e).__name__}"
        return base, 0

    chain = chains.get(code)
    if chain is None:
        base["timing_ineligible_reason"] = "no_adjustment_chain"
        return base, 0
    z_scale = (chain.factor_at(int(d_date)) or 1.0) / (chain.factor_at(int(plan_date)) or 1.0)

    # ---- A3 at S (严格因果: 只用 mS=ts<=S, 不碰 S 之后的分钟) ----
    mS = m[m["ts"] <= S].copy()
    if mS.empty:
        base["timing_ineligible_reason"] = "no_minute_before_S"
        return base, 0
    open_adj_S = float(mS.iloc[0]["open"])  # 当日复权开盘价 (S 必已知)
    # prev_close_adj 由 P_CLOSE-safe 的 gap_atr 反推: 两者在 S 均已可知, 无前视.
    gap_raw = row.get("gap_atr")
    if gap_raw is not None and not (isinstance(gap_raw, float) and np.isnan(gap_raw)):
        gap_atr = float(gap_raw)
        prev_close_adj = open_adj_S - gap_atr * atr_ref
    else:
        gap_atr = np.nan
        prev_close_adj = np.nan
    low_15_S = float(mS[mS["ts"] <= open_ts + 1500]["low"].min()) \
        if (mS["ts"] <= open_ts + 1500).any() else np.nan
    open_dip_atr_S = (open_adj_S - low_15_S) / atr_ref if not np.isnan(low_15_S) else np.nan
    day_low_S = float(mS["low"].min())
    total_dip_atr_S = (prev_close_adj - day_low_S) / atr_ref if not np.isnan(prev_close_adj) else np.nan

    def _depth(span):
        w = mS[(mS["ts"] > open_ts) & (mS["ts"] <= open_ts + span)]
        return None if w.empty else (open_adj_S - w["low"].min()) / atr_ref

    d5 = _depth(500)
    d30 = _depth(3000)
    dip_speed_5m = (d5 / 5) if d5 is not None else None
    dip_speed_15m = open_dip_atr_S / 15.0 if not np.isnan(open_dip_atr_S) else None
    dip_speed_30m = (d30 / 30) if d30 is not None else None

    # ---- A4 at S (frozen engine, snapshot=S) ----
    z_out = run_state_engine(
        zone, m, z_scale, setup, atr_ref, S,
        float(row["open_dip_atr"]), float(row.get("primary_zone_distance_atr", 0.0) or 0.0),
        dip_stage=dip_stage)
    a4 = {k: z_out.get(k) for k in A4_KEYS}
    final_state = z_out.get("final_state")

    # ---- exec at S ----
    ex = execute(S, m, "buy")
    actual_fill_new = ex.get("actual_fill")
    fill_ts = ex.get("fill_ts")
    if actual_fill_new is None or fill_ts is None:
        base["timing_ineligible_reason"] = f"NO_FILL:{ex.get('exec_status','')}"
        return base, 0

    # ---- label ratio rescale (无 proxy) ----
    # 注意: 行内 timing_market_alpha_h1 存的是「个股 alpha = 个股收益 - 市场收益」, 不是市场收益本身.
    timing_ret_old = float(row["timing_return_h1"])
    actual_fill_old = float(row["actual_fill"])
    timing_ret_new = (1.0 + timing_ret_old) * actual_fill_old / actual_fill_new - 1.0
    if row.get("proxy_timing_return") is not None:
        market_ret = float(row["proxy_timing_return"])
    else:
        market_ret = float(row["timing_market_alpha_h1"]) - timing_ret_old
    timing_alpha_new = timing_ret_new - market_ret

    base.update({
        "open_dip_atr": open_dip_atr_S,
        "total_dip_atr": total_dip_atr_S,
        "gap_atr": gap_atr,
        "dip_depth_5m_atr": d5,
        "dip_speed_5m": dip_speed_5m,
        "dip_depth_30m_atr": d30,
        "dip_speed_30m": dip_speed_30m,
        "dip_speed_15m": dip_speed_15m,
        **a4,
        "final_state": final_state,
        "signal_ts": S,
        "signal_time": f"{S:014d}",
        "fill_ts": int(fill_ts),
        "actual_fill": float(actual_fill_new),
        "feature_asof_max_ts": S,
        "first_armed_ts": S,
        "timing_return_h1": timing_ret_new,
        "timing_alpha_h1": timing_alpha_new,
        "timing_eligible": 1,
        "timing_ineligible_reason": None,
        "exec_status": ex.get("exec_status"),
    })
    return base, 1


def run_gates(tdf):
    """仅对 timing_eligible==1 子集校验 (infeasible 行无 timing 特征, 不计入)."""
    out = {}
    el = tdf[tdf["timing_eligible"] == 1] if "timing_eligible" in tdf else tdf
    if el.empty:
        out["n_eligible"] = 0
        return out
    feat_cols = [c for c in el.columns if c not in NON_FEATURE]
    s0_only = [c for c in feat_cols
               if C.LINEAGE_REGISTRY.get(c, {}).get("role_allowed") == ["S0"]]
    out["MODEL_ROLE_FEATURE_PASS"] = (len(s0_only) == 0)
    out["MODEL_ROLE_offending"] = s0_only
    out["TRAINING_TIME_CAUSALITY_PASS"] = bool(
        (el["feature_asof_max_ts"] <= el["signal_ts"]).all()
        and (el["signal_ts"] < el["fill_ts"]).all()
        and (el["fill_ts"] < el["label_end_ts"]).all())
    out["TIMING_TRIGGER_SEMANTICS_PASS"] = bool(
        (el["signal_ts"] == el["first_armed_ts"]).all())
    out["n_eligible"] = len(el)
    return out


def _monitor(n, t0):
    try:
        c = shutil.disk_usage("C:\\")
        d = shutil.disk_usage("D:\\")
        rss = (psutil.Process().memory_info().rss / 1e9) if psutil else float("nan")
        print(f"  [monitor] n={n} elapsed={time.time()-t0:.0f}s "
              f"C_free={c.free/1e9:.1f}GB D_free={d.free/1e9:.1f}GB RSS={rss:.1f}GB",
              flush=True)
        if c.free / 1e9 < 15:
            print("!! C free < 15GB -> STOP", flush=True); sys.exit(7)
        if d.free / 1e9 < 30:
            print("!! D free < 30GB -> STOP", flush=True); sys.exit(7)
    except Exception as e:  # noqa
        print(f"  [monitor] skip: {e}", flush=True)


def _coverage(df):
    """触发覆盖率 + 首触时刻分布 (用户要求的第 1/4 组口径)."""
    n_total = len(df)
    el = df[df["timing_eligible"] == 1]
    n_elig = len(el)
    rep = {"n_total": n_total, "n_eligible": n_elig,
           "overall_arm_rate": round(n_elig / n_total, 4) if n_total else 0.0}
    # by setup
    if "primary_setup" in df:
        g = df.groupby("primary_setup")["timing_eligible"].agg(["mean", "count"])
        rep["arm_rate_by_setup"] = {k: {"rate": round(v["mean"], 4), "n": int(v["count"])}
                                    for k, v in g.iterrows()}
    if "dip_stage" in df:
        g = df.groupby("dip_stage")["timing_eligible"].agg(["mean", "count"])
        rep["arm_rate_by_dip_stage"] = {k: {"rate": round(v["mean"], 4), "n": int(v["count"])}
                                        for k, v in g.iterrows()}
    # first trigger time distribution (eligible)
    if n_elig:
        hr = (el["signal_ts"] // 10000) % 100
        rep["first_trigger_hour_dist"] = {int(k): int(v) for k, v in hr.value_counts().sort_index().items()}
        later = int((el["signal_ts"] > (el["effective_trade_date"] * 1000000 + 93000)).sum())
        rep["later_triggers(S>open)"] = later
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="最多前 N 个候选 (smoke test)")
    ap.add_argument("--stride", type=int, default=None, help="跨全表等距抽样 (场景审计)")
    ap.add_argument("--candidates", default=str(C.DATASETS / "a14_universe.parquet"))
    args = ap.parse_args()

    t0 = time.time()
    print(f"[timing_builder] start n={args.n} stride={args.stride} src={args.candidates}", flush=True)
    df = pd.read_parquet(args.candidates)
    if args.stride:
        df = df.iloc[::args.stride].copy()
    if args.n:
        df = df.head(args.n)

    # ---- 预扫: 仅对会 ARMED 的 code 加载复权链 (避免 NEVER_ARMED 也拉链) ----
    armed_codes = set()
    for _, r in df.iterrows():
        tr_json = r.get("state_transitions_json")
        tr = json.loads(tr_json) if isinstance(tr_json, str) else []
        if _first_armed(tr) is not None:
            armed_codes.add(str(r.get("symbol") or r.get("code")).zfill(6))
    print(f"[timing_builder] {len(df)} rows, {len(armed_codes)} armed-codes; loading chains...", flush=True)
    chains = {}
    _ac = sorted(armed_codes)
    for _i in range(0, len(_ac), 300):
        _batch = _ac[_i:_i + 300]
        _part = L.load_adjustment_chain(_batch)
        chains.update(_part)
        print(f"  [chains] {min(_i+300, len(_ac))}/{len(_ac)} loaded", flush=True)
    print(f"[timing_builder] chains loaded ({len(chains)})", flush=True)

    minute_cache = {}
    rows = []
    reasons = Counter()
    n_elig = 0
    for ci, (_, r) in enumerate(df.iterrows()):
        sid = r.get("sample_id")
        try:
            new_row, elig = build_timing_row(r, chains, minute_cache)
        except Exception as e:  # noqa
            import traceback
            print(f"[err] {sid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            new_row = {"sample_id": sid, "timing_eligible": 0,
                       "timing_ineligible_reason": f"exception:{type(e).__name__}"}
            elig = 0
        new_row["sample_id"] = sid
        rows.append(new_row)
        if elig:
            n_elig += 1
            if ci < 10 or ci % 500 == 0:
                print(f"[{ci+1}/{len(df)}] {r.get('symbol')} {r['effective_trade_date']} "
                      f"ARMED S={new_row['signal_ts']} fill={new_row['fill_ts']} "
                      f"ret1={new_row['timing_return_h1']:.4f}", flush=True)
        else:
            reasons[new_row.get("timing_ineligible_reason")] += 1
        if (ci + 1) % 100 == 0:
            _monitor(ci + 1, t0)

    tdf = pd.DataFrame(rows)
    out = C.DATASETS / OUT_NAME
    tdf.to_parquet(out, index=False)
    print(f"[timing_builder] wrote {out} rows={len(tdf)} eligible={n_elig} "
          f"({time.time()-t0:.0f}s)", flush=True)

    gates = run_gates(tdf)
    cov = _coverage(tdf)
    print("[GATES]", json.dumps(gates, ensure_ascii=False), flush=True)
    print("[COVERAGE]", json.dumps(cov, ensure_ascii=False), flush=True)
    (C.DATASETS / "timing_causal_v1_manifest.json").write_text(
        json.dumps({"out": OUT_NAME, "n_total": len(tdf), "n_eligible": n_elig,
                    "reasons": dict(reasons), "gates": gates, "coverage": cov,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "note": "PATH A: frozen engine, FIRST_ARMED as S, as-of-S A3/A4, proxy alpha reused (S-invariant), NEVER_ARMED retained timing_eligible=0"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if reasons:
        print("[reasons]", dict(reasons), flush=True)


if __name__ == "__main__":
    main()
