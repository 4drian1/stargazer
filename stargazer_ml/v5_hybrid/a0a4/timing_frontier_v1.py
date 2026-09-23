"""TIMING_FRONTIER_V1 — TIMING-TRIGGER-v2 入场时机对照实验 (5 因果臂, 不训练).

严格按用户 2026-09-22 规范:
  - 5 个因果入场臂, 只研究"确认程度 vs 入场时点"; 不改 Selection / 特征 / 风控 / 模型.
  - 每臂独立 eligibility (不能用最终 trigger_path 反推更早买点 -> 无 hindsight bias).
  - BOTTOM / DIP 分开统计.
  - 每个 Arm 的 Market Proxy 对齐自己的 actual_fill (复用 a14 的 market_ret = ret - alpha,
    与 v2 标签同窗口; proxy 不依赖 fill, 故各臂 alpha = 自臂 stock_ret - market_ret).
  - H1 exit 用 v1 exit 的 S-不变量 exit_adj_v1 = (1+old_ret)*old_fill, 与现有 v2 标签同 horizon,
    便于直接对照 (T4 臂应重现 v2 标签 timing_return_h1).
  - MFE/MAE 用 [fill_ts, h1 150000] 原始分钟路径 (D 日 + h1 日, h1 价格按复权因子对齐到 D 日尺度).

臂定义 (预注册, 只读观测):
  T0_TOUCH        = 第一次真实 ZONE_TOUCH 后的下一合法分钟
  T1_STABLE_1     = TOUCH 后第 1 根完整 bar 未创新低 (bars_since_last_low==1)
  T2_STABLE_3     = 连续 3 根 bar 未创新低 (bars_since_last_low==3)
  T3_EARLY_RECOVERY_25 = dip_recovery_ratio(S) >= 0.25
      dip_recovery_ratio(S) = (close_S - running_pullback_low_S)
                              / max(EPS, touch_high(S) - running_pullback_low_S)
  T4_V2_PRIMARY   = 当前 v2 正式 MODEL_SCORE_REQUEST (控制组)
"""
from __future__ import annotations
import sys
import time
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))
from zone_state_engine import Zone  # noqa
from zone_state_engine_v2 import run_state_engine_v2  # noqa
import execution_engine as EX  # noqa
import latr_builder as L  # noqa
import trading_calendar as TC  # noqa

ARMS = ["T0_TOUCH", "T1_STABLE_1", "T2_STABLE_3", "T3_EARLY_RECOVERY_25", "T4_V2_PRIMARY"]
RECOVERY_RATIO = 0.25  # T3 预注册阈值 (dip_recovery_ratio 口径, 分母 = touch_high - running_pullback_low)


def _is_nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _retry(fn, *a, n=4, sleep=1.5):
    last = None
    for _ in range(n):
        try:
            return fn(*a)
        except Exception as e:  # noqa
            last = e
            time.sleep(sleep)
    return None


def _mkrec(sid, code, setup, arm, eligible, filled, reason=None, signal_ts=None,
           fill_ts=None, stock_ret=np.nan, alpha=np.nan, mfe=np.nan, mae=np.nan, loss=np.nan,
           dip_stage=None, plan_date=None):
    return dict(sample_id=sid, code=code, setup=setup, arm=arm, eligible=bool(eligible),
                filled=bool(filled), reason=reason, signal_ts=signal_ts, fill_ts=fill_ts,
                stock_ret=stock_ret, alpha=alpha, mfe=mfe, mae=mae, loss=loss,
                dip_stage=dip_stage, plan_date=plan_date)


def _hhmm(hhmm):
    h = int(hhmm)
    return f"{h // 100:02d}:{h % 100:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    df = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    if args.stride:
        df = df.iloc[::args.stride].copy()
    if args.n:
        df = df.head(args.n)
    N = len(df)
    print(f"[frontier] start n={N}", flush=True)

    cal = sorted(int(x) for x in TC.load_calendar())
    codes = sorted({str(r.get("symbol") or r.get("code")).zfill(6) for _, r in df.iterrows()})
    print(f"[frontier] loading chains for {len(codes)} codes...", flush=True)
    chains = {}
    for i in range(0, len(codes), 300):
        part = _retry(L.load_adjustment_chain, codes[i:i + 300])
        if part:
            chains.update(part)
        print(f"  [chains] {min(i + 300, len(codes))}/{len(codes)} loaded={len(chains)}", flush=True)

    minute_cache: dict = {}
    h1_cache: dict = {}
    recs = []

    for ci, (_, r) in enumerate(df.iterrows()):
        sid = r.get("sample_id")
        code = str(r.get("symbol") or r.get("code")).zfill(6)
        d_date = int(r["effective_trade_date"])
        plan_date = int(r["plan_date"])
        atr_ref = float(r["atr_ref"])
        setup = str(r.get("primary_setup", ""))
        dip_stage = r.get("dip_stage")
        gap_raw = r.get("gap_atr")
        gap_atr = float(gap_raw) if (gap_raw is not None and not _is_nan(gap_raw)) else None
        old_ret = r.get("timing_return_h1")
        old_fill = r.get("actual_fill")
        old_alpha = r.get("timing_market_alpha_h1")

        # ---- 标签源 (定义 exit + market, S-不变量) ----
        exit_adj_v1 = None
        market_ret = None
        if not (_is_nan(old_ret) or _is_nan(old_fill) or old_fill <= 0 or _is_nan(old_alpha)):
            exit_adj_v1 = (1.0 + float(old_ret)) * float(old_fill)
            market_ret = float(old_ret) - float(old_alpha)
        h1_date = TC.next_trading_day(d_date)
        if h1_date is None:
            exit_adj_v1 = None
            market_ret = None

        chain = chains.get(code)
        if chain is None:
            for arm in ARMS:
                recs.append(_mkrec(sid, code, setup, arm, False, False, "no_chain",
                                   dip_stage=dip_stage, plan_date=plan_date))
            continue
        z_scale = (chain.factor_at(int(d_date)) or 1.0) / (chain.factor_at(int(plan_date)) or 1.0)

        m = minute_cache.get((code, d_date))
        if m is None:
            m = _retry(L.load_minute, code, d_date)
            if m is None:
                m = pd.DataFrame()
            if minute_cache is not None:
                if len(minute_cache) > 400:
                    minute_cache.clear()
                minute_cache[(code, d_date)] = m
        if m is None or m.empty:
            for arm in ARMS:
                recs.append(_mkrec(sid, code, setup, arm, False, False, "minute_empty",
                               dip_stage=dip_stage, plan_date=plan_date))
            continue
        snap = int(m["ts"].max())

        try:
            zone = Zone(zone_id=str(r.get("primary_zone_id")),
                        zone_source=str(r.get("primary_zone_source", "")),
                        zone_lower=float(r["primary_zone_lower"]),
                        zone_upper=float(r["primary_zone_upper"]),
                        zone_mid=float(r["primary_zone_mid"]),
                        zone_timestamp=int(plan_date),
                        zone_width_atr=float(r.get("zone_band_atr", 0.0) or 0.0),
                        zone_strength=int(r.get("zone_strength", 1) or 1),
                        zone_source_count=1,
                        zone_distance_atr=float(r.get("primary_zone_distance_atr", 0.0) or 0.0))
        except Exception:
            for arm in ARMS:
                recs.append(_mkrec(sid, code, setup, arm, False, False, "zone_err",
                               dip_stage=dip_stage, plan_date=plan_date))
            continue

        out = run_state_engine_v2(zone, m, z_scale, setup, atr_ref, snap,
                                  float(r.get("open_dip_atr", 0.0) or 0.0),
                                  float(r.get("primary_zone_distance_atr", 0.0) or 0.0),
                                  dip_stage=dip_stage, gap_atr=gap_atr)
        arm_ts = {"T0_TOUCH": out.get("arm_t0"), "T1_STABLE_1": out.get("arm_t1"),
                  "T2_STABLE_3": out.get("arm_t2"), "T3_EARLY_RECOVERY_25": out.get("arm_t3"),
                  "T4_V2_PRIMARY": out.get("score_ts")}

        h1m = None
        if exit_adj_v1 is not None and h1_date is not None:
            h1m = h1_cache.get((code, h1_date))
            if h1m is None:
                h1m = _retry(L.load_minute, code, int(h1_date))
                if h1m is None:
                    h1m = pd.DataFrame()
                if h1_cache is not None:
                    if len(h1_cache) > 400:
                        h1_cache.clear()
                    h1_cache[(code, h1_date)] = h1m
        adj_h1 = (chain.factor_at(int(h1_date)) / chain.factor_at(int(d_date))) if (chain and h1_date) else 1.0

        for arm, sts in arm_ts.items():
            if sts is None:
                recs.append(_mkrec(sid, code, setup, arm, False, False, "not_reached",
                                   dip_stage=dip_stage, plan_date=plan_date))
                continue
            ex = EX.execute(int(sts), m, "buy")
            if ex.get("exec_status") != "FILLED" or ex.get("actual_fill") is None or ex.get("fill_ts") is None:
                recs.append(_mkrec(sid, code, setup, arm, True, False,
                                   reason=ex.get("exec_status", "NO_FILL"), signal_ts=int(sts),
                                   dip_stage=dip_stage, plan_date=plan_date))
                continue
            fill = float(ex["actual_fill"])
            fill_ts = int(ex["fill_ts"])
            stock_ret = np.nan
            alpha = np.nan
            mfe = np.nan
            mae = np.nan
            loss = np.nan
            if exit_adj_v1 is not None and fill > 0:
                stock_ret = exit_adj_v1 / fill - 1.0
                if market_ret is not None and not _is_nan(market_ret):
                    alpha = stock_ret - market_ret
                loss = 1 if stock_ret < 0 else 0
                # MFE/MAE over [fill_ts, h1 150000] raw(d-scale)
                path = []
                dm = m[m["ts"] >= fill_ts]
                for _, row in dm.iterrows():
                    path.append((int(row["ts"]), float(row["low"]), float(row["high"])))
                if h1m is not None and not h1m.empty:
                    h1end = int(f"{int(h1_date)}150000")
                    hm = h1m[h1m["ts"] <= h1end]
                    for _, row in hm.iterrows():
                        path.append((int(row["ts"]), float(row["low"]) * adj_h1, float(row["high"]) * adj_h1))
                if path:
                    lows = [p[1] for p in path]
                    highs = [p[2] for p in path]
                    mx = max(highs)
                    mn = min(lows)
                    if fill > 0:
                        mfe = mx / fill - 1.0
                        mae = 1.0 - mn / fill
            recs.append(_mkrec(sid, code, setup, arm, True, True, signal_ts=int(sts),
                               fill_ts=fill_ts, stock_ret=stock_ret, alpha=alpha,
                               mfe=mfe, mae=mae, loss=loss,
                               dip_stage=dip_stage, plan_date=plan_date))

        if (ci + 1) % 500 == 0:
            print(f"  [frontier] {ci + 1}/{N} elapsed={time.time() - t0:.0f}s recs={len(recs)}", flush=True)

    rdf = pd.DataFrame(recs)
    out_path = C.DATASETS / "timing_frontier_v1.parquet"
    rdf.to_parquet(out_path, index=False)
    print(f"[frontier] wrote {out_path} rows={len(rdf)} ({time.time() - t0:.0f}s)", flush=True)
    _report(rdf, N)


def _fmt(x):
    return f"{x * 100:+.2f}" if (x is not None and not _is_nan(x)) else "  --  "


def _arm_table(sub, N):
    rows = []
    for arm in ARMS:
        a = sub[sub["arm"] == arm]
        elig = a[a["eligible"] == True]  # noqa
        fill = a[a["filled"] == True]  # noqa
        n_e = len(elig)
        n_f = len(fill)
        rate = 100.0 * n_e / N if N else 0.0
        entry = "--"
        if n_f:
            hhmm = [int(x) % 1_000_000 // 100 for x in fill["fill_ts"].dropna()]
            if hhmm:
                entry = _hhmm(int(np.median(hhmm)))
        sr = fill["stock_ret"].mean() if n_f else np.nan
        al = fill["alpha"].mean() if n_f else np.nan
        lr = fill["loss"].mean() * 100 if n_f else np.nan
        mfe = fill["mfe"].mean() if n_f else np.nan
        mae = fill["mae"].mean() if n_f else np.nan
        rows.append(dict(arm=arm, n_elig=n_e, rate=round(rate, 2), n_fill=n_f, entry=entry,
                         h1_raw=(None if _is_nan(sr) else round(float(sr), 4)),
                         alpha=(None if _is_nan(al) else round(float(al), 4)),
                         loss_rate=(None if _is_nan(lr) else round(float(lr), 2)),
                         mfe=(None if _is_nan(mfe) else round(float(mfe), 4)),
                         mae=(None if _is_nan(mae) else round(float(mae), 4))))
    return rows


def _print_arms(rows):
    print(f"{'arm':22s} {'n_elig':>7s} {'rate%':>6s} {'n_fill':>7s} {'entry':>6s} {'H1raw%':>8s} {'alpha%':>8s} {'loss%':>6s} {'MFE%':>7s} {'MAE%':>7s}")
    for r in rows:
        lr = "-" if r["loss_rate"] is None else f"{r['loss_rate']:6.1f}"
        print(f"{r['arm']:22s} {r['n_elig']:7d} {r['rate']:6.1f} {r['n_fill']:7d} {r['entry']:>6s} "
              f"{_fmt(r['h1_raw']):>8s} {_fmt(r['alpha']):>8s} {lr:>6s} "
              f"{_fmt(r['mfe']):>7s} {_fmt(r['mae']):>7s}")


def _report(rdf, N):
    summary = {}
    print("\n==================== TIMING FRONTIER V1 ====================")
    for setup in ["ALL", "BOTTOM_SETUP", "INTRADAY_DIP_SETUP"]:
        sub = rdf if setup == "ALL" else rdf[rdf["setup"] == setup]
        print(f"\n=== setup = {setup}  (N_total = {N}) ===")
        rows = _arm_table(sub, N)
        _print_arms(rows)
        summary[setup] = rows
        if setup != "ALL":
            # pre/post 20260701 数据 regime 分层 (FLOAT_MV regime change)
            for tag, mask in [("pre_20260701", sub["plan_date"] < 20260701),
                              ("post_20260701", sub["plan_date"] >= 20260701)]:
                s2 = sub[mask]
                if len(s2) == 0:
                    continue
                n_samp = len(s2) // len(ARMS)
                print(f"  -- {tag} (n_sample={n_samp}) --")
                _print_arms(_arm_table(s2, N))
                summary[f"{setup}/{tag}"] = _arm_table(s2, N)
    # DIP 按 dip_stage 细分
    dip = rdf[rdf["setup"] == "INTRADAY_DIP_SETUP"]
    print("\n=== DIP by dip_stage ===")
    for stage in ["PRE_PULLBACK", "PULLBACK_ACTIVE", "NEAR_ZONE"]:
        s3 = dip[dip["dip_stage"] == stage]
        if len(s3) == 0:
            continue
        n_samp = len(s3) // len(ARMS)
        print(f"  -- {stage} (n_sample={n_samp}) --")
        _print_arms(_arm_table(s3, N))
        summary[f"DIP/{stage}"] = _arm_table(s3, N)
    (C.DATASETS / "timing_frontier_v1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n[frontier] wrote timing_frontier_v1_summary.json", flush=True)


if __name__ == "__main__":
    main()
