"""diag_02_alpha_gap — 诊断 ② 中 alpha coverage=0 但 raw coverage=1 的日期.

现象 (coverage_02_by_date.csv): 11 个中段 plan_date s0_return_h3_cov=1.0
但 s0_market_alpha_h3_cov=0.0 / alpha_label_available_cov=0.0; proxy 覆盖正常(~0.71).

本脚本对每个 alert 日期 + 邻近对照日期, 复现 MarketProxyV1.compute_proxy_return
(D.open -> H3.close) 并输出详细 meta (used/total/weight_sum), 定位 NaN 根因。

只读诊断, 不改任何冻结文件。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_02_full as C02  # noqa
import trading_calendar as TC  # noqa
from fast_infra import mk_chains  # noqa
from data_source_provenance import get_universe  # noqa
from market_proxy_v1 import MarketProxyV1  # noqa

ALERTS = [20250515, 20250616, 20250819, 20260203, 20260409, 20260413,
          20260512, 20260624, 20260703, 20260717, 20260720]
CONTROLS = [20250514, 20250516, 20250613, 20250617, 20250818,
            20260202, 20260408, 20260410, 20260623, 20260702, 20260716,
            20260911]

PLAN_DATES = sorted(set(ALERTS + CONTROLS))


def nth_trading_day(d, n):
    cur = int(d)
    for _ in range(n):
        nxt = TC.next_trading_day(cur)
        if nxt is None:
            return None
        cur = int(nxt)
    return cur


def main():
    print("[diag] building daily dict ...", flush=True)
    daily_dict = C02.build_daily_dict()
    univ = get_universe()
    print(f"[diag] universe={len(univ)}; building chains ...", flush=True)
    chains_full = mk_chains(univ)
    print(f"[diag] chains={len(chains_full)}", flush=True)

    proxy = MarketProxyV1(
        weight="float", leave_one_out=False,
        universe_provider=lambda: univ,
        daily_provider=lambda code, s, e: C02.fast_daily(daily_dict, code, s, e),
        chains_provider=lambda codes: {c.zfill(6): chains_full.get(c.zfill(6)) for c in codes},
    )

    print(f"\n{'plan_date':>10} {'D':>9} {'H3':>9} {'cov':>7} {'n_elig':>7} "
          f"{'proxy_s0':>12} {'used':>6} {'total':>6} {'wsum':>8}  tag", flush=True)
    for P in PLAN_DATES:
        D = int(TC.next_trading_day(P))
        H3 = nth_trading_day(D, 3)
        tag = "ALERT" if P in ALERTS else "ctrl"
        try:
            _, elig, cov = proxy.freeze_proxy_weights(int(P))
        except Exception as e:  # noqa
            print(f"{P:>10} freeze EXC: {type(e).__name__}: {e}", flush=True)
            continue
        try:
            r, meta = proxy.compute_proxy_return(int(D), "open", int(H3), "close",
                                                 weight_date=int(P))
        except Exception as e:  # noqa
            print(f"{P:>10} {D:>9} {H3:>9} {cov:>7.4f} {elig:>7} "
                  f"{'EXC':>12} {'-':>6} {'-':>6} {'-':>8}  {tag} {type(e).__name__}: {e}",
                  flush=True)
            continue
        print(f"{P:>10} {D:>9} {H3:>9} {cov:>7.4f} {elig:>7} "
              f"{r:>12.6f} {meta.get('constituent_count'):>6} {meta.get('total'):>6} "
              f"{meta.get('weight_sum', float('nan')):>8.4f}  {tag}", flush=True)
        proxy._daily_cache.clear()

    # 深入: 对第一个 alert 日期, 抽样统计 constituent 的 _daily 可用性
    print("\n[diag] deep-dive on first ALERT ...", flush=True)
    P = ALERTS[0]
    D = int(TC.next_trading_day(P))
    H3 = nth_trading_day(D, 3)
    w, elig, cov = proxy.freeze_proxy_weights(int(P))
    print(f"[diag] P={P} D={D} H3={H3} n_weights={len(w)} elig={elig} cov={cov}", flush=True)
    miss_daily = miss_price = ok = 0
    reasons = {}
    sample_codes = list(w.keys())[:400]
    for c in sample_codes:
        df = C02.fast_daily(daily_dict, c, D, H3)
        if df is None or df.empty:
            miss_daily += 1
            reasons["empty_daily"] = reasons.get("empty_daily", 0) + 1
            continue
        dts = set(df["date"].tolist())
        has_d = int(D) in dts
        has_h3 = int(H3) in dts
        if not (has_d and has_h3):
            miss_price += 1
            k = "missing_" + ("D" if not has_d else "") + ("H3" if not has_h3 else "")
            reasons[k] = reasons.get(k, 0) + 1
            continue
        ok += 1
    print(f"[diag] sample {len(sample_codes)} constituents: ok={ok} "
          f"empty_daily={miss_daily} missing_date={miss_price}", flush=True)
    print(f"[diag] reasons={reasons}", flush=True)

    # 全 universe: D / H3 当天有多少 code 有行
    n_present_d = n_present_h3 = 0
    for code, (dates_arr, _df) in daily_dict.items():
        if np.any(dates_arr == int(D)):
            n_present_d += 1
        if np.any(dates_arr == int(H3)):
            n_present_h3 += 1
    print(f"[diag] universe daily_dict={len(daily_dict)}; codes with row on D={n_present_d}, "
          f"on H3={n_present_h3}", flush=True)


if __name__ == "__main__":
    main()
