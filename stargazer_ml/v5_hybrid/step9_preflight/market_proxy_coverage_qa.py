"""market_proxy_coverage_qa — ② 覆盖率 QA (真实候选 + 有界真实 universe).

验证: s0/timing 的 alpha coverage 应非常接近对应 raw coverage。
若 raw~99% 而 alpha~70%, 说明 Proxy pipeline 有缺口, 不能带缺口训练。

为控制开销, universe 限定为 get_universe() 前 N 支 (真实 float_mv/daily/adjustment),
候选取少量真实 GREEN-WATCH 候选。这是管线健康检查, 非全市场训练。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from market_proxy_v1 import MarketProxyV1, coverage_qa
from latr_builder import (load_green_watch_candidates, load_adjustment_chain,
                          build_one_sample, load_daily_window)


def main(n_cands=12, universe_topn=120, plan_date=20260105):
    cands = load_green_watch_candidates(plan_date=plan_date)
    if not cands:
        print(f"[coverage_qa] 无候选 @ {plan_date}, 试全量前 {n_cands} 个")
        cands = load_green_watch_candidates()[:n_cands]
    cands = cands[:n_cands]
    codes = sorted({c.symbol for c in cands})
    try:
        chains = load_adjustment_chain(codes)
        adj_mode = "corporate-action-safe (real adjustment chain)"
    except Exception as e:  # noqa  stockdb 不可达时降级为 raw-only
        chains = {}
        adj_mode = f"RAW-ONLY (adjustment HTTP 不可用: {type(e).__name__})"
    print(f"[coverage_qa] adjustment mode: {adj_mode}")

    # 有界真实 universe (前 N 支, 真实数据)
    from data_source_provenance import get_universe
    univ = get_universe()[:universe_topn]

    proxy = MarketProxyV1(weight="float", leave_one_out=True,
                          universe_provider=lambda: univ,
                          daily_provider=load_daily_window,
                          chains_provider=load_adjustment_chain)

    out = []
    for cand in cands:
        try:
            s = build_one_sample(cand, chains, proxy)
            if s:
                out.append(s)
        except Exception as e:  # noqa
            continue
    print(f"[coverage_qa] 成功构建样本: {len(out)} / {len(cands)}")

    cov = coverage_qa(out)
    print("\n===== MARKET_PROXY-V1 覆盖率 QA =====")
    for k, v in cov.items():
        print(f"  {k:32s} = {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    gap_ok = cov["alpha_vs_raw_gap_s0"] <= 0.20 and cov["alpha_vs_raw_gap_timing"] <= 0.20
    print(f"\n  alpha~raw 缺口合格 (tol=0.20): {gap_ok}")
    print(f"  MARKET_PROXY_V1_PASS(覆盖率维度): {gap_ok}")
    return cov


if __name__ == "__main__":
    main()
