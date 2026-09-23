"""coverage_02_full — ② MARKET_PROXY-V1 全量真实 coverage (含按日期分布).

设计 (与 user ② 要求一致):
  - 基于生产候选池 pool_daily_v11.parquet (v11 最终门控).
  - 仅计算 ② 需要的覆盖率字段, 跳过 ④ zone 引擎与 60 历史分钟 volume_baseline.
  - 性能关键 (已修复): 
      * 预建 code->daily DataFrame 内存字典, 避免 get_daily 每次重扫全 parquet;
      * proxy 收益按 P(plan_date) 只算一次: 同一 plan_date 的所有候选共享相同的
        D/H1/H3 交易日, 故 leave-one-out 的"可用性"对所有候选一致, 一次性预计算即可;
      * 逐候选只取 1 次 D 日分钟算 timing 标签 (S0 标签仅需日线).
  - 逐 plan_date 输出 7 指标 + 计数; 总体 + 骤降告警写入 JSON/CSV.

指标 (user 指定):
  s0_return_h3 / s0_market_alpha_h3 / timing_return_h1 / timing_market_alpha_h1 /
  proxy_weight_coverage / proxy_constituent_count / alpha_label_available
  —— 重点按日期分布看, 不能被均值掩盖.

运行:
  python coverage_02_full.py --cap 30
  python coverage_02_full.py --cap 30 --max-days 5   # 冒烟
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_source_provenance import get_universe, get_adjustment_http, get_daily_parquet  # noqa
from market_proxy_v1 import MarketProxyV1  # noqa
from execution_engine import execute  # noqa
from adjustment_chain import build_chains  # noqa
from green_watch_candidate_loader import CandidateRow, REQUIRED_FIELDS, OPTIONAL_FIELDS  # noqa
import trading_calendar as TC  # noqa


def _nth_trading_day(d: int, n: int):
    """canonical 日历上 d 之后的第 n 个交易日; 超出数据范围返回 None。"""
    cur = int(d)
    for _ in range(n):
        nxt = TC.next_trading_day(cur)
        if nxt is None:
            return None
        cur = int(nxt)
    return cur

_HERE = Path(__file__).resolve().parent
V11_POOL = _HERE.parent / "green_watch_v0" / "pool_daily_v11.parquet"
MINUTE_START = 20250102
ATR_LOOK = 14
ATR_REF_DAYS = 5


def lookback_start(d_date: int, years: int = 2) -> int:
    y = d_date // 10000
    return (y - years) * 10000 + 101


# ---------------- 快速日线字典 (避免每次重扫全 parquet) ----------------
def build_daily_dict():
    print("[02] building in-memory daily dict ...", flush=True)
    p = get_daily_parquet()
    keep = [c for c in ("code", "date", "open", "high", "low", "close", "pre_close",
                       "volume", "amount", "float_mv") if c in p.columns]
    p = p[keep].copy()
    for c in keep:
        if c in ("code", "date"):
            continue
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["code"] = p["code"].astype(str).str.zfill(6)
    p["date"] = p["date"].astype(int)
    out = {}
    for code, sub in p.groupby("code"):
        sub = sub.sort_values("date").reset_index(drop=True)
        out[code] = (sub["date"].to_numpy(np.int64), sub)
    print(f"[02]   daily dict: {len(out)} codes", flush=True)
    return out


_EMPTY_DF = pd.DataFrame()


def fast_daily(daily_dict, code, start, end):
    """按 code 预索引日期数组 + searchsorted 取切片 (避免每次布尔掩码)。"""
    e = daily_dict.get(code.zfill(6))
    if e is None:
        return _EMPTY_DF
    dates, df = e
    i0 = int(np.searchsorted(dates, int(start), side="left"))
    i1 = int(np.searchsorted(dates, int(end), side="right"))
    if i1 <= i0:
        return _EMPTY_DF
    return df.iloc[i0:i1]


# ---------------- 候选加载 (向量化) ----------------
def _build_row(r):
    row = CandidateRow(
        symbol=str(r["code"]).zfill(6),
        plan_date=int(r["plan_date"]),
        effective_trade_date=int(r["effective_trade_date"]),
        primary_setup=str(r["primary_setup"]),
        bottom_score=float(r["bottom_score"]),
        intraday_dip_score=float(r["intraday_dip_score"]),
        atr_ref=float(r["atr_ref"]),
        bottom_tier=str(r.get("bottom_tier", "")),
        dip_tier=str(r.get("dip_tier", "")),
        primary_tier=str(r.get("primary_tier", "")),
        bottom_subtype=str(r.get("bottom_subtype", "")) if "bottom_subtype" in r else "",
        raw=dict(r) if hasattr(r, "to_dict") else r,
    )
    row.validate()
    return row


def load_v11_candidates(cap_per_day: int, max_days: int = None, seed: int = 20260918,
                        min_plan: int = None, max_plan: int = None):
    import pyarrow.parquet as pq
    avail = set(pq.read_schema(V11_POOL).names)
    need = ["code", "plan_date", "effective_trade_date", "primary_setup",
            "bottom_score", "intraday_dip_score", "atr_ref",
            "bottom_tier", "dip_tier", "primary_tier", "bottom_subtype"]
    use_cols = [c for c in need if c in avail]
    print(f"[02] reading v11 pool (cols={len(use_cols)}) ...", flush=True)
    df = pd.read_parquet(V11_POOL, columns=use_cols)
    print(f"[02]   pool rows={len(df)}", flush=True)
    df = df[df["effective_trade_date"].notna()]
    df["_plan"] = df["plan_date"].astype(int)
    df["_eff"] = df["effective_trade_date"].astype(int)
    df = df[(df["_eff"] > df["_plan"]) & (df["_eff"] >= MINUTE_START)]
    if min_plan is not None:
        df = df[df["_plan"] >= int(min_plan)]
    if max_plan is not None:
        df = df[df["_plan"] <= int(max_plan)]
    print(f"[02]   valid candidates (eff>plan, in minute window)={len(df)}", flush=True)

    rng = np.random.default_rng(seed)
    parts = []
    for p, g in df.groupby("_plan"):
        n = min(cap_per_day, len(g))
        parts.append(g.sample(n=n, random_state=int(rng.integers(0, 1_000_000))))
    sampled_df = pd.concat(parts) if parts else df.iloc[0:0]
    print(f"[02]   sampled rows={len(sampled_df)} across {sampled_df['_plan'].nunique()} plan_dates", flush=True)

    groups = defaultdict(list)
    for _, r in sampled_df.iterrows():
        try:
            groups[int(r["_plan"])].append(_build_row(r))
        except Exception:
            continue
    plan_dates = sorted(groups.keys())
    if max_days:
        plan_dates = plan_dates[:max_days]
        groups = {p: groups[p] for p in plan_dates}
    return groups, plan_dates


# ---------------- 单票 label (S0 + Timing) ----------------
def stock_labels(cand, chains, daily_dict):
    """复用 build_one_sample 的 label 数学 (S0: D-open->H3-close; Timing: actual_fill->H1-close)."""
    code = cand.symbol
    d_date = cand.effective_trade_date
    atr_ref = cand.atr_ref
    if not atr_ref or atr_ref <= 0:
        return None
    start = lookback_start(d_date)
    end_label = int((pd.Timestamp(str(d_date)) + pd.Timedelta(days=20)).strftime("%Y%m%d"))
    daily = fast_daily(daily_dict, code, start, end_label)
    if daily is None or daily.empty or len(daily) < ATR_LOOK + ATR_REF_DAYS:
        return None
    chain = chains.get(code.zfill(6))
    if chain is None:
        return None
    dates_arr = daily["date"].to_numpy(int)
    ref_fac = chain.factor_at(int(d_date)) or 1.0

    def _adj(col):
        raw = daily[col].to_numpy(float)
        fac = np.array([chain.factor_at(int(dt)) for dt in dates_arr], dtype=float)
        return raw * (fac / ref_fac)

    adj_close = _adj("close")
    adj_open = _adj("open")
    raw_open = daily["open"].to_numpy(float)
    dates = daily["date"].to_numpy(int)
    d_idx = int(np.where(dates == d_date)[0][0]) if d_date in dates else len(dates) - 1
    if d_idx < ATR_LOOK + ATR_REF_DAYS:
        return None

    # 分钟 (仅 1 次 HTTP)
    from data_source_provenance import get_minute_http
    m = get_minute_http(code, str(d_date))
    if not isinstance(m, pd.DataFrame) or m.empty:
        return None
    if "date" in m.columns and "ts" not in m.columns:
        m = m.rename(columns={"date": "ts"})
    for c in ("open", "high", "low", "close", "volume"):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    open_ts = int(m["ts"].min())
    exec = execute(open_ts, m, side="buy")
    if exec["exec_status"] != "FILLED":
        return None
    actual_fill = exec["actual_fill"]

    h3_idx = d_idx + 3
    h1_idx = d_idx + 1
    s0_open = adj_open[d_idx]
    s0_ret = adj_close[h3_idx] / s0_open - 1 if h3_idx < len(adj_close) else np.nan
    raw_open_d = raw_open[d_idx] if d_idx < len(raw_open) else np.nan
    if raw_open_d:
        timing_fill_analytic = actual_fill * (adj_open[d_idx] / raw_open_d)
    else:
        timing_fill_analytic = actual_fill
    timing_ret = adj_close[h1_idx] / timing_fill_analytic - 1 if h1_idx < len(adj_close) else np.nan

    return {
        "symbol": code, "d_date": int(d_date),
        "s0_return_h3": s0_ret, "timing_return_h1": timing_ret,
        "h1_date": int(dates[h1_idx]) if h1_idx < len(dates) else None,
        "h3_date": int(dates[h3_idx]) if h3_idx < len(dates) else None,
        "d_date_ref": int(d_date),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--min-plan", type=int, default=None)
    ap.add_argument("--max-plan", type=int, default=None)
    ap.add_argument("--out-dir", default=str(_HERE))
    ap.add_argument("--only-dates", default=None,
                    help="逗号分隔 plan_date 白名单 (复测特定日期; 不影响全量语义)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    log = open(out_dir / "coverage_02.log", "w", encoding="utf-8", buffering=1)

    def logp(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")

    t0 = time.time()
    daily_dict = build_daily_dict()
    groups, plan_dates = load_v11_candidates(args.cap, args.max_days,
                                             min_plan=args.min_plan, max_plan=args.max_plan)
    if args.only_dates:
        keep = {int(x) for x in str(args.only_dates).split(",") if x.strip()}
        plan_dates = [p for p in plan_dates if int(p) in keep]
        groups = {p: groups[p] for p in plan_dates}
    total_cands = sum(len(v) for v in groups.values())
    logp(f"[02] plan_dates={len(plan_dates)} sampled_candidates={total_cands} "
         f"date_range={plan_dates[0]}..{plan_dates[-1]}  ({time.time()-t0:.1f}s)")

    # proxy (快速 daily + 快速 chains provider)
    # 预建全 universe 复权链一次 (避免 compute_proxy_return 内逐 constituent HTTP)
    univ = get_universe()
    logp(f"[02] building universe adjustment chains ({len(univ)} codes) ...", )
    by_u = defaultdict(list)
    for code in univ:
        try:
            for d, rec in get_adjustment_http(code):
                by_u[code.zfill(6)].append((str(d), rec))
        except Exception:
            continue
    chains_full = build_chains(by_u)
    logp(f"[02] universe chains built: {len(chains_full)}  ({time.time()-t0:.1f}s)")

    def chains_provider(codes):
        return {c.zfill(6): chains_full.get(c.zfill(6)) for c in codes}

    proxy = MarketProxyV1(weight="float", leave_one_out=False,
                          universe_provider=lambda: univ,
                          daily_provider=lambda code, s, e: fast_daily(daily_dict, code, s, e),
                          chains_provider=chains_provider)
    # 抽样 symbol 的 chains (用于逐票 label)
    all_codes = sorted({c.symbol for g in groups.values() for c in g})
    by = defaultdict(list)
    for code in all_codes:
        try:
            for d, rec in get_adjustment_http(code):
                by[code.zfill(6)].append((str(d), rec))
        except Exception:
            continue
    chains = build_chains(by)
    logp(f"[02] chains built for {len(chains)} sampled symbols  ({time.time()-t0:.1f}s)")

    rows = []
    csv_path = out_dir / "coverage_02_by_date.csv"
    for p in plan_dates:
        g = groups[p]
        # D/H1/H3 锚点来自 canonical 交易日历 (用户 ③.1-1).
        # 修复 (2026-09-18): 旧实现用 g[0] 推导锚点 -> 若当日首个抽样候选建 label 失败
        # 则 H1=H3=None, 当日所有候选 proxy/alpha 全为 NaN (alpha_label_available_cov=0),
        # 造成 raw cov 1.0 而 alpha cov 0.0 的假性缺口 (见 verify_02_ref_single_point.py).
        # 生产 attach_alpha_labels 按每候选自身 h3_date 计算, 无此单点依赖。
        D = int(g[0].effective_trade_date)
        H1 = _nth_trading_day(D, 1)
        H3 = _nth_trading_day(D, 3)
        # proxy universe 冻结 (按 P 一次)
        try:
            _, elig_count, cov = proxy.freeze_proxy_weights(int(p))
        except Exception:
            elig_count, cov = 0, 0.0
        # proxy 收益 (按 plan_date 一次, exclude=None; leave-one-out 可用性对同 P 一致)
        try:
            proxy_s0, _ = proxy.compute_proxy_return(int(D), "open", int(H3), "close", weight_date=int(p)) if H3 else (np.nan, {})
        except Exception:
            proxy_s0 = np.nan
        try:
            proxy_t1, _ = proxy.compute_proxy_return(int(D), "open", int(H1), "close", weight_date=int(p)) if H1 else (np.nan, {})
        except Exception:
            proxy_t1 = np.nan
        proxy_s0_ok = not (proxy_s0 is None or (isinstance(proxy_s0, float) and np.isnan(proxy_s0)))
        proxy_t1_ok = not (proxy_t1 is None or (isinstance(proxy_t1, float) and np.isnan(proxy_t1)))

        recs = []
        for cand in g:
            sl = stock_labels(cand, chains, daily_dict)
            if sl is None:
                continue
            s0_ok = not (sl["s0_return_h3"] is None or (isinstance(sl["s0_return_h3"], float) and np.isnan(sl["s0_return_h3"])))
            t_ok = not (sl["timing_return_h1"] is None or (isinstance(sl["timing_return_h1"], float) and np.isnan(sl["timing_return_h1"])))
            s0_alpha_ok = s0_ok and proxy_s0_ok
            t_alpha_ok = t_ok and proxy_t1_ok
            alpha_avail = 1 if (s0_alpha_ok and t_alpha_ok) else 0
            recs.append({
                "symbol": cand.symbol, "plan_date": int(p), "d_date": int(cand.effective_trade_date),
                "s0_return_h3": sl["s0_return_h3"],
                "timing_return_h1": sl["timing_return_h1"],
                "s0_market_alpha_h3": (sl["s0_return_h3"] - proxy_s0) if s0_alpha_ok else None,
                "timing_market_alpha_h1": (sl["timing_return_h1"] - proxy_t1) if t_alpha_ok else None,
                "alpha_label_available": alpha_avail,
                "proxy_weight_coverage": cov,
                "proxy_constituent_count": int(elig_count),
            })
        n = len(recs)

        def _ok(key):
            return sum(1 for x in recs if x.get(key) is not None and not
                       (isinstance(x.get(key), float) and np.isnan(x.get(key)))) if n else 0

        rows.append({
            "plan_date": int(p),
            "d_date": int(g[0].effective_trade_date),
            "n_sampled": len(g),
            "n_built": n,
            "s0_return_h3_cov": round(_ok("s0_return_h3") / n, 4) if n else 0.0,
            "s0_market_alpha_h3_cov": round(_ok("s0_market_alpha_h3") / n, 4) if n else 0.0,
            "timing_return_h1_cov": round(_ok("timing_return_h1") / n, 4) if n else 0.0,
            "timing_market_alpha_h1_cov": round(_ok("timing_market_alpha_h1") / n, 4) if n else 0.0,
            "alpha_label_available_cov": round(sum(x["alpha_label_available"] for x in recs) / n, 4) if n else 0.0,
            "proxy_weight_coverage": round(float(cov), 4),
            "proxy_constituent_count": int(elig_count),
        })
        if len(rows) % 25 == 0:
            logp(f"[02] progress {len(rows)}/{len(plan_dates)}  ({time.time()-t0:.1f}s)")
        if len(rows) % 50 == 0 and rows:
            try:
                pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
            except Exception:
                pass
        # 关键: 每 P 清空 proxy 逐 constituent 日线缓存, 否则 416 天累积 ~7M DataFrame 爆内存
        try:
            proxy._daily_cache.clear()
        except Exception:
            pass

    df = pd.DataFrame(rows).sort_values("plan_date").reset_index(drop=True)
    csv_path = out_dir / "coverage_02_by_date.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logp(f"[02] wrote {csv_path} ({len(df)} dates)")

    overall = {
        "n_plan_dates": len(df),
        "date_range": [int(df["plan_date"].min()), int(df["plan_date"].max())] if len(df) else None,
        "total_sampled": int(df["n_sampled"].sum()),
        "total_built": int(df["n_built"].sum()),
        "s0_return_h3_cov_mean": round(float(df["s0_return_h3_cov"].mean()), 4),
        "s0_return_h3_cov_min": round(float(df["s0_return_h3_cov"].min()), 4),
        "s0_market_alpha_h3_cov_mean": round(float(df["s0_market_alpha_h3_cov"].mean()), 4),
        "s0_market_alpha_h3_cov_min": round(float(df["s0_market_alpha_h3_cov"].min()), 4),
        "timing_return_h1_cov_mean": round(float(df["timing_return_h1_cov"].mean()), 4),
        "timing_return_h1_cov_min": round(float(df["timing_return_h1_cov"].min()), 4),
        "timing_market_alpha_h1_cov_mean": round(float(df["timing_market_alpha_h1_cov"].mean()), 4),
        "timing_market_alpha_h1_cov_min": round(float(df["timing_market_alpha_h1_cov"].min()), 4),
        "alpha_label_available_cov_mean": round(float(df["alpha_label_available_cov"].mean()), 4),
        "alpha_label_available_cov_min": round(float(df["alpha_label_available_cov"].min()), 4),
        "proxy_weight_coverage_mean": round(float(df["proxy_weight_coverage"].mean()), 4),
        "proxy_weight_coverage_min": round(float(df["proxy_weight_coverage"].min()), 4),
        "proxy_constituent_count_mean": int(df["proxy_constituent_count"].mean()),
        "proxy_constituent_count_min": int(df["proxy_constituent_count"].min()),
        "proxy_constituent_count_max": int(df["proxy_constituent_count"].max()),
    }

    median_constit = float(df["proxy_constituent_count"].median()) if len(df) else 0
    alerts = []
    for _, r in df.iterrows():
        reasons = []
        if r["proxy_constituent_count"] < 0.6 * median_constit:
            reasons.append(f"proxy_constituent_count={int(r['proxy_constituent_count'])} < 0.6*median({int(median_constit)})")
        if r["proxy_weight_coverage"] < 0.5:
            reasons.append(f"proxy_weight_coverage={r['proxy_weight_coverage']} < 0.5")
        if r["s0_return_h3_cov"] < 0.9:
            reasons.append(f"s0_return_h3_cov={r['s0_return_h3_cov']} < 0.9")
        if r["timing_return_h1_cov"] < 0.9:
            reasons.append(f"timing_return_h1_cov={r['timing_return_h1_cov']} < 0.9")
        if r["alpha_label_available_cov"] < 0.9:
            reasons.append(f"alpha_label_available_cov={r['alpha_label_available_cov']} < 0.9")
        if reasons:
            alerts.append({"plan_date": int(r["plan_date"]), "reasons": reasons})

    summary = {"overall": overall, "alerts": alerts, "n_alerts": len(alerts)}
    json_path = out_dir / "coverage_02_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logp(f"[02] wrote {json_path}")
    logp(f"[02] OVERALL: {json.dumps(overall, ensure_ascii=False)}")
    logp(f"[02] ALERTS: {len(alerts)} dates flagged")
    for a in alerts[:40]:
        logp(f"   ALERT {a['plan_date']}: {a['reasons']}")
    logp(f"[02] DONE total_time={time.time()-t0:.1f}s")


def _chains_for(codes, daily_dict):
    # 简化: compute_proxy_return 内部按 constituent 调用 chains_provider 时我们已全量预建,
    # 这里直接返回已建 chains 的 subset (MarketProxyV1 期望 code->SymbolChain).
    by = defaultdict(list)
    for code in codes:
        try:
            for d, rec in get_adjustment_http(code):
                by[code.zfill(6)].append((str(d), rec))
        except Exception:
            continue
    return build_chains(by)


if __name__ == "__main__":
    main()
