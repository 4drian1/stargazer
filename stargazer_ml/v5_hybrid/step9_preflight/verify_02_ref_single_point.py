"""verify_02_ref_single_point — 验证 ② coverage 脚本的单点脆弱设计.

假设: coverage_02_full.main 中
    ref = stock_labels(g[0], chains, daily_dict)
    if ref is None: H1 = H3 = None
    proxy_s0 = compute_proxy_return(D.open -> H3.close) if H3 else np.nan
=> 只要当日"首个抽样候选 g[0]" 建 label 失败, 该 plan_date 所有候选的
   proxy 收益与 alpha 全为 NaN (alpha_label_available_cov=0), 而逐票 raw
   s0_return_h3 仍可正常 -> raw cov 1.0 但 alpha cov 0.0.

本脚本用与 ② 完全相同的采样 (cap=30, seed=20260918) 复现 g[0], 并检验:
  (a) alert 日期 g[0] 是否 stock_labels -> None (预期 True);
  (b) 对照日期 (n_built<n_sampled 但非 alert) g[0] 是否成功 (预期 True)。

只读, 不改冻结文件。
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_02_full as C02  # noqa
from data_source_provenance import get_adjustment_http  # noqa
from adjustment_chain import build_chains  # noqa

ALERTS = [20250515, 20250616, 20250819, 20260203, 20260409, 20260413,
          20260512, 20260624, 20260703, 20260717, 20260720]
# 对照: n_built<n_sampled 但非 alert (从上一步输出中取若干)
CONTROLS = [20250110, 20250516, 20250619, 20250703, 20250818, 20250902,
            20251009, 20251208]


def chains_for(codes):
    by = defaultdict(list)
    for code in codes:
        try:
            for d, rec in get_adjustment_http(code):
                by[code.zfill(6)].append((str(d), rec))
        except Exception:
            continue
    return build_chains(by)


def main():
    print("[verify] building daily dict ...", flush=True)
    daily_dict = C02.build_daily_dict()
    print("[verify] loading candidates (same seed/cap as ②) ...", flush=True)
    groups, plan_dates = C02.load_v11_candidates(30, None, seed=20260918)
    print(f"[verify] plan_dates={len(plan_dates)}", flush=True)

    need = set()
    for p in ALERTS + CONTROLS:
        if p in groups and groups[p]:
            need.add(groups[p][0].symbol)
    # 另取一个 alert 日的全部候选, 复现 n_built 计数
    full_date = ALERTS[0]
    if full_date in groups:
        need.update(c.symbol for c in groups[full_date])
    codes = sorted(need)
    print(f"[verify] building chains for {len(codes)} codes ...", flush=True)
    chains = chains_for(codes)
    print(f"[verify] chains={len(chains)}", flush=True)

    print(f"\n{'plan_date':>10} {'g0_code':>8} {'ref_ok':>7} {'n_in_group':>10}  tag", flush=True)
    for p in ALERTS + CONTROLS:
        g = groups.get(p)
        if not g:
            print(f"{p:>10}  <no group>", flush=True)
            continue
        ref = C02.stock_labels(g[0], chains, daily_dict)
        tag = "ALERT" if p in ALERTS else "ctrl"
        print(f"{p:>10} {g[0].symbol:>8} {str(ref is not None):>7} {len(g):>10}  {tag}", flush=True)

    # 复现 alert 日的 n_built
    g = groups.get(full_date)
    if g:
        built = sum(1 for c in g if C02.stock_labels(c, chains, daily_dict) is not None)
        print(f"\n[verify] date={full_date}: n_sampled={len(g)} n_built={built} "
              f"(g0 ref_ok={C02.stock_labels(g[0], chains, daily_dict) is not None})", flush=True)


if __name__ == "__main__":
    main()
