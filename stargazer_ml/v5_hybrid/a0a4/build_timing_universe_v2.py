"""build_timing_universe_v2 — Causal Timing Dataset v2 构建器 (TIMING-TRIGGER-v2).

设计 (用户 2026-09-19 授权):
  - 复用 a14_universe.parquet 作为 P_CLOSE-safe base (白名单拷贝, 绝不读入旧 D 日动态字段作重建输入).
  - 对每个候选全天空跑 run_state_engine_v2 -> S=score_ts 处冻结的 as-of-S 特征 (compute_features_asof 产出).
  - 以 execute(score_ts, m) 得 as-of-S 真实 actual_fill (⑥ Real Execution).
  - 标签 MUST 重算 (不可继承 v1 fill):
        new_ret  = (1+ret_old) * old_fill / new_fill - 1   (exit 价 S-不变量, 等价重算)
        market   = ret_old - alpha_old                      (S-不变量市场段)
        new_alpha= new_ret - market
    即个股 Timing 收益严格锚定新 v2 actual_fill, 不混入 v1 fill (避免标签错位).
  - 窗口字段严格门控: window_complete_Xm==0 -> 对应窗口字段 = NaN (不是 0).
  - second_test_flag / second_test_higher_low: 从模型特征集剔除 (MODEL_ELIGIBLE=False),
    仅留作 master 审计列, 供 v3 启用.
  - reclaim_zone_* : 保留 NaN (合法缺失, 不补 0).

产物:
  datasets/timing_universe_v2_all.parquet    — 全体 16,192 审计母表
  datasets/timing_universe_v2_train.parquet  — timing_eligible_v2==1 AND label_status==OK
  datasets/timing_universe_v2_manifest.json  — 审计/特征契约/5 Gate 结果

不动 zone_state_engine_v2 / K=5 / Zone / Trigger 路径 (Step 3 已冻结).
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
from zone_state_engine import Zone  # noqa (冻结: 仅类型)
from zone_state_engine_v2 import run_state_engine_v2, V2_FEATURE_REGISTRY  # noqa (v2)
from zone_state_engine import volume_baseline, _hhmmss_to_min  # noqa (v1, 仅借 A4 合同量能基准)
from execution_engine import execute  # ⑥ (冻结)
import latr_builder as L  # noqa (load_minute / load_adjustment_chain)
import trading_calendar as TC  # noqa
import v2_gates as G  # noqa (5 个契约检查)

# ---------- 继承的 P_CLOSE-safe 特征 (来自 a14, 与 a0a4 A1/A2 同口径) ----------
INHERITED = [
    "primary_setup", "dip_stage",
    "prior_strength_pass", "structure_intact_flag", "prior_impulse_return",
    "ma20_slope_5", "ret_60d", "ret_20d", "dist_from_60d_high", "range_pos60",
    "range_pos20", "rsi14", "dist_low60", "distribution_score_raw", "support_break_depth_atr",
    "atr_ref", "zone_strength", "zone_distance_atr", "penetration_atr",
    "primary_zone_distance_atr", "zone_band_atr", "n_zones", "zone_n_sources",
]

# ---------- v2 as-of-S 特征 (compute_features_asof 产出, 门控后入模型) ----------
V2_ASOF = [
    # A3 v2 新 (无窗口)
    "realized_dip_atr", "current_price_dip_atr", "zone_penetration_low_atr",
    "current_zone_distance_atr", "current_break_depth_atr",
    # A3 价格 (窗口语义, 受 WINDOWED 门控)
    "open_dip_atr", "total_dip_atr", "dip_depth_5m_atr", "dip_depth_30m_atr",
    "dip_speed_5m", "dip_speed_15m", "dip_speed_30m",
    # A4 状态 (v2 语义)
    "recovery_from_zone_low_atr", "recovery_atr", "dip_recovery_ratio",
    "reclaim_zone_lower", "reclaim_zone_mid", "reclaim_zone_upper", "reclaim_failure_count",
    "time_since_zone_touch", "time_below_zone",
    "lower_wick_ratio", "sell_volume_ratio", "rebound_volume_ratio", "vwap_reclaim",
]

# A4 合同中原本存在、v2 compute_features_asof 缺失、现按 S causal 补全的 5 个字段
# (B = A4 causal contract completion, 不是新特征设计). 严格按 S 重算, 不照搬 a14 的 v1 值.
V2_A4_CONTRACT = [
    "close_position_15m",
    "zone_break_depth_atr",
    "zone_break_duration",
    "cum_volume_vs_hist_same_time",
    "cum_turnover_vs_hist_same_time",
]

# 窗口字段 -> 其完整性 flag (门控: flag==0 -> 字段 NaN)
WINDOWED = {
    "open_dip_atr": "window_complete_15m",
    "dip_depth_5m_atr": "window_complete_5m",
    "dip_depth_30m_atr": "window_complete_30m",
    "dip_speed_5m": "window_complete_5m",
    "dip_speed_15m": "window_complete_15m",
    "dip_speed_30m": "window_complete_30m",
}

# 仅作 CONTEXT/METADATA (第一轮不进模型)
METADATA = ["window_complete_5m", "window_complete_15m", "window_complete_30m",
            "zone_width_atr", "open_to_zone_upper_atr", "open_to_zone_mid_atr",
            "open_to_zone_lower_atr"]

# 明确剔除出 v2-primary 模型特征集 (保留作 master 审计列)
EXCLUDED = ["second_test_flag", "second_test_higher_low"]

# 模型特征 = 继承 + v2-asof (second_test 已被排除, window/geometry 仅 metadata)
MODEL_FEATURES = list(INHERITED) + list(V2_ASOF) + list(V2_A4_CONTRACT)

# ---- A4 合同 causal-at-S 特征计算 (B) ----
_CALENDAR = None  # 由 main 填充
_PAST_CACHE = {}  # (code, date) -> minute df, 跨样本去重 (过去 60 日量能基准)


def _min_to_hhmmss(mins: int) -> str:
    hh = mins // 60
    mm = mins % 60
    return f"{hh:02d}{mm:02d}00"


def _completed_15m_closepos(m: pd.DataFrame, S: int):
    """close_position_15m: 仅用最近一根【已完成】15min bar 的 (close-low)/(high-low).

    S=09:36 -> 第一根 15m bar(09:30-09:45) 未完成 -> NaN.
    S=09:47 -> 可用 09:30-09:45 bar.
    """
    s_min = int(_hhmmss_to_min(int(S)))
    base = 570  # 09:30 in minutes
    if s_min < base + 15:
        return float("nan")
    idx = (s_min - base) // 15 - 1
    if idx < 0:
        return float("nan")
    win_end = base + (idx + 1) * 15
    win_start = base + idx * 15
    date = int(S) // 1000000
    lo = int(f"{date}{_min_to_hhmmss(win_start)}")
    hi = int(f"{date}{_min_to_hhmmss(win_end)}")
    bar = m[(m["ts"] >= lo) & (m["ts"] <= hi)]
    if bar.empty:
        return float("nan")
    low_15 = float(bar["low"].min())
    high_15 = float(bar["high"].max())
    close_15 = float(bar["close"].iloc[-1])
    if (high_15 - low_15) <= 0:
        return float("nan")
    return (close_15 - low_15) / (high_15 - low_15)


def _past_minutes(code: str, d_date: int):
    """过去 60 完整交易日同 code 的分钟 (用于 cum_volume 基准), 跨样本缓存去重."""
    past_dates = [int(x) for x in _CALENDAR if int(x) < int(d_date)][-60:]
    out = []
    for pd_ in past_dates:
        key = (code, pd_)
        df = _PAST_CACHE.get(key)
        if df is None:
            try:
                df = _retry_call(L.load_minute, code, pd_)
            except Exception:  # noqa
                df = None
            if len(_PAST_CACHE) > 20000:
                _PAST_CACHE.clear()
            _PAST_CACHE[key] = df
        if df is not None and not (isinstance(df, pd.DataFrame) and df.empty):
            out.append(df)
    return out


def compute_a4_contract(m: pd.DataFrame, zone, atr_ref: float, S: int, code: str, d_date: int):
    """原 A4 合同 5 字段, 严格 causal 重算于 S (score_ts)."""
    feats = {}
    lower = float(zone.zone_lower)
    # 1) close_position_15m (最近已完成 15m bar)
    feats["close_position_15m"] = _completed_15m_closepos(m, S)
    # 2) zone_break_depth/duration as-of-S
    sub = m[m["ts"] <= int(S)]
    if len(sub):
        below = sub[sub["low"] < lower]
        if len(below):
            break_low = float(below["low"].min())
            feats["zone_break_depth_atr"] = (lower - break_low) / atr_ref
            feats["zone_break_duration"] = int(len(below))
        else:
            feats["zone_break_depth_atr"] = 0.0
            feats["zone_break_duration"] = 0
    else:
        feats["zone_break_depth_atr"] = 0.0
        feats["zone_break_duration"] = 0
    # 3) cum_volume / cum_turnover vs 过去 60 日同 minute-of-day 中位累计
    past = _past_minutes(code, int(d_date))
    try:
        vb = volume_baseline(int(d_date), m, past, int(S))
    except Exception:  # noqa
        vb = {}
    feats["cum_volume_vs_hist_same_time"] = vb.get("cum_volume_vs_hist_same_time")
    feats["cum_turnover_vs_hist_same_time"] = vb.get("cum_turnover_vs_hist_same_time")
    return feats

OUT_ALL = "timing_universe_v2_all.parquet"
OUT_TRAIN = "timing_universe_v2_train.parquet"
OUT_MANIFEST = "timing_universe_v2_manifest.json"


def _is_nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _retry_call(fn, *args, n=4, sleep=1.5):
    """网络 (get_minute_http / get_adjustment_http) 偶发超时 -> 重试, 避免整轮崩溃."""
    last = None
    for _ in range(n):
        try:
            return fn(*args)
        except Exception as e:  # noqa
            last = e
            time.sleep(sleep)
    print(f"[warn] retry exhausted {fn.__name__}{args[:1]}: {type(last).__name__}: {last}",
          flush=True)
    return None


def build_row(r, chains, minute_cache):
    """返回 (row_dict, eligible, label_status, reason)。始终返回完整母表行。"""
    code = str(r.get("symbol") or r.get("code")).zfill(6)
    d_date = int(r["effective_trade_date"])
    plan_date = int(r["plan_date"])
    atr_ref = float(r["atr_ref"])
    setup = str(r.get("primary_setup", ""))
    dip_stage = r.get("dip_stage")
    gap_raw = r.get("gap_atr")
    gap_atr = float(gap_raw) if (gap_raw is not None and not _is_nan(gap_raw)) else None

    # base = a14 整行 (traceability); 之后覆盖 v2 字段
    base = dict(r) if hasattr(r, "to_dict") else dict(r)
    base["timing_eligible_v2"] = 0
    base["trigger_path_v2"] = None
    base["trigger_reject_reason_v2"] = None
    base["score_ts"] = None
    base["actual_fill_ts"] = None
    base["actual_fill"] = None
    base["feature_asof_max_ts"] = None
    base["label_end_ts"] = None
    base["proxy_start_ts"] = None
    base["proxy_end_ts"] = None
    base["label_status"] = "NA_TRIGGER"
    base["timing_return_h1"] = np.nan
    base["timing_market_alpha_h1"] = np.nan
    # A4 合同 5 字段: 非触发行置 NaN (触发行在 score_ts 处 causal 重算; 不继承 a14 的 v1 值)
    for _k in V2_A4_CONTRACT:
        base[_k] = np.nan

    # 标签源 (v1, 用于 S-不变量重算) — 必须在覆盖前读出
    old_ret = r.get("timing_return_h1")
    old_fill = r.get("actual_fill")
    old_alpha = r.get("timing_market_alpha_h1")

    # ---- 右删失: 算不出 H1 -> 标签不可用 ----
    h1_date = TC.next_trading_day(d_date)
    if h1_date is None:
        base["trigger_reject_reason_v2"] = "TAIL_CENSORED"
        base["label_status"] = "TAIL_CENSORED"
        return base, 0, "TAIL_CENSORED", "TAIL_CENSORED"
    base["label_end_ts"] = int(f"{int(h1_date)}150000")

    # ---- Zone (冻结 P_CLOSE 几何) ----
    try:
        zone = Zone(
            zone_id=str(r.get("primary_zone_id")),
            zone_source=str(r.get("primary_zone_source", "")),
            zone_lower=float(r["primary_zone_lower"]),
            zone_upper=float(r["primary_zone_upper"]),
            zone_mid=float(r["primary_zone_mid"]),
            zone_timestamp=int(plan_date),
            zone_width_atr=float(r.get("zone_band_atr", 0.0) or 0.0),
            zone_strength=int(r.get("zone_strength", 1) or 1),
            zone_source_count=1,
            zone_distance_atr=float(r.get("primary_zone_distance_atr", 0.0) or 0.0),
        )
    except Exception as e:  # noqa
        base["trigger_reject_reason_v2"] = f"zone_err:{type(e).__name__}"
        return base, 0, "NA_TRIGGER", base["trigger_reject_reason_v2"]

    chain = chains.get(code)
    if chain is None:
        base["trigger_reject_reason_v2"] = "no_adjustment_chain"
        return base, 0, "NA_TRIGGER", "no_adjustment_chain"
    z_scale = (chain.factor_at(int(d_date)) or 1.0) / (chain.factor_at(int(plan_date)) or 1.0)

    m = minute_cache.get((code, d_date))
    if m is None:
        m = _retry_call(L.load_minute, code, d_date)
        if m is None:
            m = pd.DataFrame()
        if minute_cache is not None:
            if len(minute_cache) > 400:
                minute_cache.clear()
            minute_cache[(code, d_date)] = m
    if m is None or m.empty:
        base["trigger_reject_reason_v2"] = "minute_empty"
        return base, 0, "NA_TRIGGER", "minute_empty"

    snap = int(m["ts"].max())
    out = run_state_engine_v2(
        zone, m, z_scale, setup, atr_ref, snap,
        float(r.get("open_dip_atr", 0.0) or 0.0),
        float(r.get("primary_zone_distance_atr", 0.0) or 0.0),
        dip_stage=dip_stage, gap_atr=gap_atr)

    if out.get("score_ts") is None:
        base["trigger_reject_reason_v2"] = out.get("rejection")
        return base, 0, "NA_TRIGGER", out.get("rejection")

    score_ts = int(out["score_ts"])
    path = out["score_path"]
    feats = dict(out.get("features_asof_S", {}))

    # ---- 窗口门控: flag==0 -> 字段 NaN ----
    for f, flag in WINDOWED.items():
        if feats.get(flag, 0) != 1:
            feats[f] = np.nan

    # ---- A4 合同补全 (causal at S, 不改 Trigger/K/Zone) ----
    if score_ts is not None:
        a4c = compute_a4_contract(m, zone, atr_ref, score_ts, code, d_date)
        feats.update(a4c)

    # ---- 执行 actual_fill ----
    ex = execute(score_ts, m, "buy")
    actual_fill = ex.get("actual_fill")
    fill_ts = ex.get("fill_ts")
    if actual_fill is None or fill_ts is None:
        base["trigger_reject_reason_v2"] = f"NO_FILL:{ex.get('exec_status','')}"
        base["label_status"] = "NO_FILL"
        # 仍记录触发路径与特征 (供审计), 但 eligible=0
        base["trigger_path_v2"] = path
        base["score_ts"] = score_ts
        for k, v in feats.items():
            base[k] = v
        base["second_test_flag"] = feats.get("second_test_flag")
        base["second_test_higher_low"] = feats.get("second_test_higher_low")
        return base, 0, "NO_FILL", base["trigger_reject_reason_v2"]

    # ---- 标签重算 (S-不变量 rescale, 锚定新 v2 fill) ----
    if _is_nan(old_ret) or _is_nan(old_fill) or _is_nan(actual_fill) or old_fill <= 0:
        label_status = "LABEL_SRC_MISSING"
        new_ret = np.nan
        new_alpha = np.nan
    else:
        new_ret = (1.0 + float(old_ret)) * float(old_fill) / float(actual_fill) - 1.0
        if _is_nan(old_alpha):
            new_alpha = np.nan
        else:
            market_ret = float(old_ret) - float(old_alpha)
            new_alpha = new_ret - market_ret
        label_status = "OK"

    # ---- 写回 v2 字段 ----
    base["trigger_path_v2"] = path
    base["trigger_reject_reason_v2"] = None
    base["score_ts"] = score_ts
    base["actual_fill_ts"] = int(fill_ts)
    base["actual_fill"] = float(actual_fill)
    base["feature_asof_max_ts"] = score_ts
    base["proxy_start_ts"] = int(fill_ts)
    base["proxy_end_ts"] = int(base["label_end_ts"])
    base["timing_return_h1"] = new_ret
    base["timing_market_alpha_h1"] = new_alpha
    base["label_status"] = label_status
    for k, v in feats.items():
        base[k] = v
    # second_test 仅作 master 审计列 (不进 MODEL_FEATURES)
    base["second_test_flag"] = feats.get("second_test_flag")
    base["second_test_higher_low"] = feats.get("second_test_higher_low")
    # metadata
    for k in METADATA:
        base[k] = feats.get(k)

    eligible = 1 if label_status == "OK" else 0
    base["timing_eligible_v2"] = eligible
    return base, eligible, label_status, None


def _monitor(n, t0):
    try:
        c = shutil.disk_usage("C:\\")
        d = shutil.disk_usage("D:\\")
        rss = (psutil.Process().memory_info().rss / 1e9) if psutil else float("nan")
        print(f"  [monitor] n={n} elapsed={time.time()-t0:.0f}s "
              f"C_free={c.free/1e9:.1f}GB D_free={d.free/1e9:.1f}GB RSS={rss:.1f}GB", flush=True)
        if c.free / 1e9 < 15:
            print("!! C free < 15GB -> STOP", flush=True); sys.exit(7)
        if d.free / 1e9 < 30:
            print("!! D free < 30GB -> STOP", flush=True); sys.exit(7)
    except Exception as e:  # noqa
        print(f"  [monitor] skip: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--candidates", default=str(C.DATASETS / "a14_universe.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    print(f"[v2_builder] start n={args.n} stride={args.stride} src={args.candidates}", flush=True)
    df = pd.read_parquet(args.candidates)
    if args.stride:
        df = df.iloc[::args.stride].copy()
    if args.n:
        df = df.head(args.n)
    print(f"[v2_builder] rows={len(df)}", flush=True)
    global _CALENDAR
    _CALENDAR = sorted(int(x) for x in TC.load_calendar())

    codes = sorted({str(r.get("symbol") or r.get("code")).zfill(6)
                    for _, r in df.iterrows()})
    print(f"[v2_builder] loading chains for {len(codes)} codes...", flush=True)
    chains = {}
    for i in range(0, len(codes), 300):
        part = _retry_call(L.load_adjustment_chain, codes[i:i + 300])
        if part:
            chains.update(part)
        print(f"  [chains] {min(i + 300, len(codes))}/{len(codes)} "
              f"(loaded={len(chains)})", flush=True)

    minute_cache: dict = {}
    rows = []
    elig = 0
    label_ok = 0
    reasons = Counter()

    for ci, (_, r) in enumerate(df.iterrows()):
        sid = r.get("sample_id")
        try:
            new_row, e, ls, reason = build_row(r, chains, minute_cache)
        except Exception as e2:  # noqa
            import traceback
            print(f"[err] {sid}: {type(e2).__name__}: {e2}", flush=True)
            traceback.print_exc()
            new_row = {"sample_id": sid, "timing_eligible_v2": 0,
                       "label_status": "EXCEPTION", "trigger_reject_reason_v2": f"exception:{type(e2).__name__}"}
            e, ls, reason = 0, "EXCEPTION", f"exception:{type(e2).__name__}"
        new_row["sample_id"] = sid
        rows.append(new_row)
        if e:
            elig += 1
            if ls == "OK":
                label_ok += 1
            else:
                reasons[f"elig_no_label:{ls}"] += 1
        else:
            reasons[new_row.get("trigger_reject_reason_v2") or ls or "NA"] += 1
        if (ci + 1) % 5 == 0:
            _monitor(ci + 1, t0)

    tdf = pd.DataFrame(rows)
    all_path = C.DATASETS / OUT_ALL
    tdf.to_parquet(all_path, index=False)
    print(f"[v2_builder] wrote {all_path} rows={len(tdf)} eligible={elig} label_ok={label_ok} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- train 表 (eligible==1 AND label OK) ----
    train_df = tdf[(tdf["timing_eligible_v2"] == 1) & (tdf["label_status"] == "OK")].copy()
    train_cols = (["sample_id", "symbol", "plan_date", "effective_trade_date",
                   "trigger_path_v2", "score_ts", "actual_fill_ts", "actual_fill",
                   "feature_asof_max_ts", "label_end_ts", "proxy_start_ts", "proxy_end_ts",
                   "timing_return_h1", "timing_market_alpha_h1", "label_status"]
                  + MODEL_FEATURES + METADATA)
    train_cols = [c for c in train_cols if c in train_df.columns]
    train_out = train_df[train_cols].copy()
    train_path = C.DATASETS / OUT_TRAIN
    train_out.to_parquet(train_path, index=False)
    print(f"[v2_builder] wrote {train_path} rows={len(train_out)}", flush=True)

    # ---- 5 个契约 Gate ----
    gates = G.run_all_gates(tdf, MODEL_FEATURES)
    # ---- 窗口完整率审计 (报告用) ----
    win_audit = {}
    scored = tdf[tdf["score_ts"].notna()]
    n_scored = len(scored)
    for f, flag in WINDOWED.items():
        wc = int((scored[flag] == 1).sum()) if flag in scored else 0
        win_audit[f] = {"complete_rate": round(wc / n_scored, 4) if n_scored else 0.0,
                        "n_complete": wc}
    path_dist = scored["trigger_path_v2"].value_counts().to_dict() if n_scored else {}

    manifest = {
        "out_all": OUT_ALL, "out_train": OUT_TRAIN,
        "n_total": len(tdf), "n_scored_v2": n_scored,
        "n_eligible": elig, "n_label_ok": label_ok,
        "reasons": dict(reasons),
        "gates": gates,
        "window_completeness_audit": win_audit,
        "path_distribution": path_dist,
        "feature_manifest": {
            "model_features": MODEL_FEATURES,
            "metadata_features": METADATA,
            "excluded_features": EXCLUDED,
            "inherited_features": INHERITED,
            "v2_asof_features": V2_ASOF,
        },
        "v2_feature_registry_ref": "zone_state_engine_v2.V2_FEATURE_REGISTRY",
        "elapsed_sec": round(time.time() - t0, 1),
        "note": ("v2 causal dataset: as-of-S features (compute_features_asof) + execute() actual_fill "
                 "at score_ts; timing labels recomputed via S-invariant rescale (new fill), "
                 "window gating (NaN not 0), second_test excluded from model, reclaim_* NaN preserved."),
    }
    (C.DATASETS / OUT_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[GATES]", json.dumps(gates, ensure_ascii=False, default=str), flush=True)
    print(f"[v2_builder] wrote {C.DATASETS / OUT_MANIFEST}", flush=True)
    print(f"[v2_builder] scored_v2={n_scored} eligible={elig} label_ok={label_ok}", flush=True)


if __name__ == "__main__":
    main()
