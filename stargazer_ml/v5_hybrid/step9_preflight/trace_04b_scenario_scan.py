"""trace_04b_scenario_scan — ④ 定向场景 Trace 覆盖扫描 (为 Gate4 生成场景证书).

背景 (用户 2026-09-18 裁定):
  随机抽样 155 个样本里 DIP_TO_BREAKDOWN / SECOND_TEST_HIGHER_LOW / LATE_SESSION_TOUCH
  命中数为 0。但 Gate4 旧版只检查「6 个末态是否出现」, 并不覆盖预注册的「场景路径」。
  「抽样没抽到」≠「数据里没有」——必须定向扩大搜索窗口来证明。

做法:
  - 直接从 v11 候选池均匀抽样 (默认优先 INTRADAY_DIP_SETUP), 覆盖整个 plan_date 范围;
  - 每个候选只做「Zone 构建 + D 日分钟状态机」这条最小必要路径
    (不算 label / proxy / volume_baseline) -> 快, 且与被检验的状态机完全同源;
  - 用 gate4_harness 的同一套场景判定函数 (保证证书与 Gate4 语义一致);
  - 输出 scenario_cert.json: 每场景 status / scan_scope / n_scanned / found_n + 样例。

status:
  FOUND_IN_SCAN   扫描窗口内确实存在 (样例已导出)
  SCENARIO_RARE   扫描了 n_scanned 个候选仍未命中 -> 证书化「数据里确实罕见/不存在」
  （gate4 的 TRACE_SCENARIO_COVERAGE_PASS 只在样本内 0 命中时消费该证书）

运行:
  python trace_04b_scenario_scan.py --budget 1500
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

import coverage_02_full as C02  # noqa
import gate4_harness as G  # noqa
import latr_builder as LB  # noqa
from fast_infra import FastContext, mk_chains  # noqa
from zone_state_engine import build_zones, select_primary_zone, run_state_engine  # noqa

_HERE = Path(__file__).resolve().parent
SNAPSHOT_MINUTE = 150000


def _sample_candidates(budget: int, setup_first: str, min_plan, max_plan, seed):
    """从 v11 池读候选 (全列), 均匀抽样覆盖整个 plan_date 范围。"""
    import pyarrow.parquet as pq
    avail = set(pq.read_schema(C02.V11_POOL).names)
    need = ["code", "plan_date", "effective_trade_date", "primary_setup",
            "bottom_score", "intraday_dip_score", "atr_ref",
            "bottom_tier", "dip_tier", "primary_tier", "bottom_subtype",
            "dip_stage"]
    use = [c for c in need if c in avail]
    df = pd.read_parquet(C02.V11_POOL, columns=use)
    df = df[df["effective_trade_date"].notna()]
    df["_plan"] = df["plan_date"].astype(int)
    df["_eff"] = df["effective_trade_date"].astype(int)
    df = df[(df["_eff"] > df["_plan"]) & (df["_eff"] >= C02.MINUTE_START)]
    if min_plan is not None:
        df = df[df["_plan"] >= int(min_plan)]
    if max_plan is not None:
        df = df[df["_plan"] <= int(max_plan)]
    total_pool = len(df)
    total_dip = int((df["primary_setup"] == "INTRADAY_DIP_SETUP").sum())

    rng = np.random.default_rng(seed)
    # 按 plan_date 均匀抽样; DIP 优先 (先抽 DIP, 不足再补 BOTTOM)
    order = []
    for st in (setup_first, None):
        sub = df if st is None else df[df["primary_setup"] == st]
        parts = [g.sample(n=min(3, len(g)), random_state=int(rng.integers(0, 10 ** 6)))
                 for _, g in sub.groupby("_plan")]
        if parts:
            s = pd.concat(parts)
            order.append(s)
    pooled = pd.concat(order) if order else df.iloc[0:0]
    pooled = pooled.drop_duplicates(subset=["code", "_plan"])
    if len(pooled) > budget:
        pooled = pooled.sample(n=budget, random_state=seed)
    return pooled, total_pool, total_dip


def _scan_one(r, chains, daily_dict):
    """最小必要路径: Zone + D 日分钟状态机 -> 场景判定所需字段。"""
    code = str(r["code"]).zfill(6)
    plan_date = int(r["plan_date"])
    d_date = int(r["effective_trade_date"])
    atr_ref = float(r["atr_ref"])
    if not atr_ref or atr_ref <= 0:
        return None, "atr_ref_invalid"
    chain = chains.get(code)
    if chain is None:
        return None, "no_adjustment_chain"
    daily = C02.fast_daily(daily_dict, code, LB.lookback_start(d_date),
                           int((pd.Timestamp(str(d_date)) + pd.Timedelta(days=20)).strftime("%Y%m%d")))
    if daily is None or daily.empty or len(daily) < LB.ATR_LOOK + LB.ATR_REF_DAYS:
        return None, "daily_empty_or_short"
    dates_arr = daily["date"].to_numpy(int)
    if d_date not in dates_arr:
        return None, "d_date_not_in_daily"
    d_idx = int(np.where(dates_arr == d_date)[0][0])
    if d_idx < LB.ATR_LOOK + LB.ATR_REF_DAYS:
        return None, "history_too_short"
    m = LB.load_minute(code, d_date)
    if m is None or m.empty:
        return None, "minute_empty"

    # ANALYTIC at D (ref_fac 锚 D => adj_open[D] == raw_open[D]), 口径与 latr_builder 完全一致
    ref_fac = chain.factor_at(int(d_date)) or 1.0
    fac = np.array([chain.factor_at(int(dt)) for dt in dates_arr], dtype=float)
    adj_open = daily["open"].to_numpy(float) * (fac / ref_fac)
    open_ts = int(m["ts"].min())
    m15 = m[m["ts"] <= open_ts + 1500]
    if m15.empty:
        m15 = m.iloc[: max(1, len(m) // 10)]
    low_15 = float(m15["low"].min())
    open_dip_atr = (float(adj_open[d_idx]) - low_15) / atr_ref

    zones = build_zones(daily, plan_date, atr_ref, chain)
    if not zones:
        return None, "no_zone"
    # P 日收盘价 (analytic at P) 选主 zone
    d = daily[daily["date"] <= plan_date]
    if d.empty:
        return None, "no_p_close"
    p_close = float(d["close"].iloc[-1])
    primary = select_primary_zone(zones, p_close)
    if primary is None:
        return None, "no_primary_zone"
    scale = (chain.factor_at(int(d_date)) or 1.0) / (chain.factor_at(int(plan_date)) or 1.0)
    snap = int(f"{d_date}{SNAPSHOT_MINUTE}")
    out = run_state_engine(primary, m, scale, str(r["primary_setup"]), atr_ref, snap,
                           float(open_dip_atr), float(primary.zone_distance_atr),
                           dip_stage=(str(r.get("dip_stage")) if r.get("dip_stage") is not None else None))
    trans = out.get("state_transitions") or out.get("transitions") or []
    row = {
        "primary_setup": str(r["primary_setup"]),
        "final_state": out.get("final_state"),
        "state_transitions_json": json.dumps(trans, ensure_ascii=False),
        "second_test_flag": out.get("second_test_flag"),
        "second_test_higher_low": out.get("second_test_higher_low"),
    }
    top = trans[0]["state"] if trans else None
    return row, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--setup-first", default="INTRADAY_DIP_SETUP")
    ap.add_argument("--min-plan", type=int, default=20250102)
    ap.add_argument("--max-plan", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default=str(_HERE / "scenario_cert.json"))
    args = ap.parse_args()
    t0 = time.time()

    def log(*a):
        print(" ".join(str(x) for x in a), flush=True)

    pooled, total_pool, total_dip = _sample_candidates(
        args.budget, args.setup_first, args.min_plan, args.max_plan, args.seed)
    log(f"[04b] pool={total_pool} (DIP={total_dip}); scan sample={len(pooled)}")

    daily_dict = C02.build_daily_dict()
    ctx = FastContext(daily_dict, {}, [])   # 只需 daily 快取 + load_minute 缓存, 不建全市场 chain
    ctx.patch()
    codes = sorted({str(c).zfill(6) for c in pooled["code"].unique()})
    log(f"[04b] chains for {len(codes)} codes ...")
    chains = mk_chains(codes)

    found = defaultdict(list)
    rej = defaultdict(int)
    n_ok = 0
    for i, (_, r) in enumerate(pooled.iterrows()):
        try:
            row, why = _scan_one(r, chains, daily_dict)
        except Exception as e:  # noqa
            rej[f"exc:{type(e).__name__}"] += 1
            continue
        if row is None:
            rej[why] += 1
            continue
        n_ok += 1
        hit = G._scenario_present(pd.DataFrame([row]))
        for k, v in hit.items():
            if v > 0 and len(found[k]) < 8:
                found[k].append({"symbol": str(r["code"]).zfill(6),
                                 "plan_date": int(r["plan_date"]),
                                 "d_date": int(r["effective_trade_date"]),
                                 "setup": str(r["primary_setup"]),
                                 "final_state": row["final_state"],
                                 "first_state": (json.loads(row["state_transitions_json"])[0]["state"]
                                                 if row["state_transitions_json"] != "[]" else None)})
        if (i + 1) % 100 == 0:
            log(f"[04b] scanned {i+1}/{len(pooled)} ok={n_ok} found="
                f"{ {k: len(v) for k, v in found.items()} } ({time.time()-t0:.0f}s)")
        if (i + 1) % 500 == 0:
            ctx._mc.clear()

    scope = (f"v11 pool uniform scan: plan_date {args.min_plan}..{args.max_plan or 'latest'}, "
             f"pool={total_pool} (DIP={total_dip}), sampled={len(pooled)}, "
             f"state_engine_ok={n_ok}, seed={args.seed}, "
             f"path=build_zones+run_state_engine (与 gate4 同源)")

    cert = {}
    for scen in G.SCENARIO_DEFS:
        n = len(found[scen])
        cert[scen] = {
            "status": "FOUND_IN_SCAN" if n > 0 else "SCENARIO_RARE",
            "scan_scope": scope,
            "n_scanned": n_ok,
            "n_pool": total_pool,
            "found_n": n,
            "examples": found[scen],
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
        }
    Path(args.out).write_text(json.dumps(cert, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[04b] DONE ok={n_ok} rejects={dict(rej)} ({time.time()-t0:.0f}s)")
    for k, v in cert.items():
        log(f"[04b]   {k:24s} status={v['status']} found_n={v['found_n']}")
    log(f"[04b] wrote {args.out}")


if __name__ == "__main__":
    main()
