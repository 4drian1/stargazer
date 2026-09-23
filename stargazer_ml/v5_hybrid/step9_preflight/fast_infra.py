"""fast_infra — ④/⑤ 真实构建的加速取数层 (纯运行时 monkeypatch, 不改动 latr_builder 文件).

问题: build_one_sample 每次调用 load_daily_window (get_daily, 重扫全 parquet) 与
load_minute (D 日 + 60 历史日), 并让 MarketProxyV1 用默认 daily/chains provider
(逐构成股 get_daily + HTTP) -> 单样本 ~69s, 全量不可行。

方案 (不改动被冻结的 latr_builder / market_proxy_v1 文件, 仅运行时替换):
  1. code->daily 内存字典 (预建一次) 替换 latr_builder.load_daily_window;
  2. (code,date) 分钟缓存替换 latr_builder.load_minute;
  3. 全 universe 复权链预建 -> 注入 MarketProxyV1 的 fast chains_provider;
  4. MarketProxyV1 用 fast daily_provider。
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_02_full as C02  # noqa
import latr_builder as LB  # noqa
from market_proxy_v1 import MarketProxyV1  # noqa
from adjustment_chain import build_chains  # noqa
from data_source_provenance import get_universe, get_adjustment_http  # noqa


def mk_chains(codes):
    by = defaultdict(list)
    for code in codes:
        try:
            for d, rec in get_adjustment_http(code):
                by[code.zfill(6)].append((str(d), rec))
        except Exception:
            continue
    return build_chains(by)


class FastContext:
    def __init__(self, daily_dict, chains_full, univ):
        self.daily_dict = daily_dict
        self.chains_full = chains_full
        self.univ = univ
        self._mc = {}
        self._orig_minute = LB.load_minute

    def patch(self):
        dd = self.daily_dict
        LB.load_daily_window = lambda code, start, end: C02.fast_daily(dd, code, start, end)
        LB.load_minute = self.cached_minute

    def cached_minute(self, code, date):
        k = (str(code).zfill(6), int(date))
        if k not in self._mc:
            self._mc[k] = self._orig_minute(code, date)
        return self._mc[k]

    def chains_provider(self, codes):
        return {c.zfill(6): self.chains_full.get(c.zfill(6)) for c in codes}

    def make_proxy(self, leave_one_out: bool = True):
        return MarketProxyV1(
            weight="float", leave_one_out=leave_one_out,
            universe_provider=lambda: self.univ,
            daily_provider=lambda code, s, e: C02.fast_daily(self.daily_dict, code, s, e),
            chains_provider=self.chains_provider,
        )


def build_context(cap_per_day=25, min_plan=20250102, max_plan=None, seed=20260918):
    """返回 (ctx, cands, plan_dates). 已 patch latr_builder 取数."""
    daily_dict = C02.build_daily_dict()
    univ = get_universe()
    chains_full = mk_chains(univ)
    ctx = FastContext(daily_dict, chains_full, univ)
    ctx.patch()
    groups, plan_dates = C02.load_v11_candidates(cap_per_day, None,
                                                 min_plan=min_plan, max_plan=max_plan, seed=seed)
    cands = [c for p in plan_dates for c in groups[p]]
    return ctx, cands, plan_dates
