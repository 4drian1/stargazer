"""a0a4_fast_proxy — MarketProxyV1.compute_proxy_return 的等价加速层.

不修改冻结的 market_proxy_v1.py 任何逻辑; 仅在运行时包装 compute_proxy_return:

事实 (由 attach_alpha_labels 的调用结构 + effective_trade_date=canonical 下一交易日推出):
  同一 plan_date 的全部候选共享完全相同的 (weight_date, d_date, h1_date, h3_date) 窗口,
  唯一差异是 leave-one-out 的 exclude_symbol.

加速原理:
  1) 每个窗口 key=(weight_date, start, sf, end, ef) 只跑一次冻结原版 exclude=None 的
     全量计算, 缓存 (ret_full, meta_full)。
  2) leave-one-out 由恒等代数给出 (与冻结逐码求和完全等价):
       若 X 在冻结权重中且其 [start,end] 价格有效 (即 X ∈ used_full):
         num_excl = ret_full*den_full - w_X*adj_X ;  den_excl = den_full - w_X
       否则 (X 不在权重 / 价格无效): 结果与全量完全一致。
     w_X 取自冻结 freeze_proxy_weights; adj_X 用冻结 _daily/_price/_chain 按冻结公式
     ep/sp*(f_e/f_s)-1 计算。

等价性验证 (install 时执行):
  对调用方提供的真实探针窗口 (weight_date,start,sf,end,ef,exclude) 原版 vs 加速版逐一对拍,
  要求 |diff| < 1e-9; 任一不过则不安装 (调用方必须 abort, 保证绝不静默漂移)。
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _diff(a, b) -> float:
    an, bn = _is_nan(a), _is_nan(b)
    if an and bn:
        return 0.0
    if an or bn:
        return 9e9
    return abs(float(a) - float(b))


def install_fast_proxy(proxy, probe_windows, tol: float = 1e-9):
    """安装加速 compute_proxy_return; 返回 (installed, verify_report)。

    probe_windows: [(weight_date, start, sf, end, ef, exclude), ...] 真实调用探针。
    """
    orig = proxy.compute_proxy_return
    cache = {}
    last_calls = deque(maxlen=32)

    def fast(start_date, start_field, end_date, end_field,
             exclude_symbol=None, weight_date=None):
        last_calls.append((start_date, start_field, end_date, end_field,
                           exclude_symbol, weight_date))
        if weight_date is None:
            weight_date = int(start_date)
        key = (int(weight_date), int(start_date), str(start_field),
               int(end_date), str(end_field))
        if key not in cache:
            cache[key] = orig(start_date, start_field, end_date, end_field,
                              exclude_symbol=None, weight_date=weight_date)
        ret_full, meta_full = cache[key]
        if exclude_symbol is None or not getattr(proxy, "leave_one_out", True):
            return ret_full, meta_full
        ex = str(exclude_symbol)
        weights = proxy.freeze_proxy_weights(int(weight_date))[0]
        if ex not in weights:
            return ret_full, meta_full
        df = proxy._daily(ex, int(start_date), int(end_date))
        sp = proxy._price(df, int(start_date), start_field)
        ep = proxy._price(df, int(end_date), end_field)
        if sp is None or sp <= 0 or ep is None or ep <= 0:
            return ret_full, meta_full          # X 价格无效 -> 未进 used_full
        ch = proxy._chain(ex)
        f_s = ch.factor_at(int(start_date)) if ch else 1.0
        f_e = ch.factor_at(int(end_date)) if ch else 1.0
        adj = ep / sp * (f_e / f_s) - 1.0
        w = weights[ex]
        den_full = float(meta_full.get("weight_sum", 0.0))
        num_full = float(ret_full) * den_full
        den = den_full - w
        ret = (num_full - w * adj) / den if den > 0 else np.nan
        meta = dict(meta_full)
        meta["constituent_count"] = meta_full.get("constituent_count", 0) - 1
        meta["weight_sum"] = den
        meta["proxy_excluded_symbol"] = exclude_symbol
        return ret, meta

    report = {"n_tests": 0, "max_abs_diff": 0.0, "passed": False,
              "n_probe_windows": len(probe_windows), "tol": tol}
    for (wd, s, sf, e, ef, ex) in probe_windows:
        r_o, _ = orig(s, sf, e, ef, exclude_symbol=ex, weight_date=wd)
        r_f, _ = fast(s, sf, e, ef, exclude_symbol=ex, weight_date=wd)
        d = _diff(r_o, r_f)
        report["n_tests"] += 1
        report["max_abs_diff"] = max(report["max_abs_diff"], d)
    report["passed"] = (report["n_tests"] > 0
                        and report["max_abs_diff"] < tol)
    if report["passed"]:
        proxy.compute_proxy_return = fast
        fast.last_calls = last_calls
        proxy._a0a4_window_cache = cache
        proxy._a0a4_orig_cpr = orig
        return True, report
    proxy.compute_proxy_return = orig
    return False, report


def recheck_equivalence(proxy, n_recent: int = 4):
    """对最近 n 个真实调用重放原版对拍; 返回 (n_checks, max_abs_diff, ok)。"""
    fast = proxy.compute_proxy_return
    orig = getattr(proxy, "_a0a4_orig_cpr", None)
    if orig is None or not hasattr(fast, "last_calls"):
        return 0, None, False
    calls = list(fast.last_calls)[-n_recent:]
    dmax = 0.0
    for (s0, sf, e0, ef, ex, wd) in calls:
        r_o = orig(s0, sf, e0, ef, exclude_symbol=ex, weight_date=wd)
        r_f = fast(s0, sf, e0, ef, exclude_symbol=ex, weight_date=wd)
        dmax = max(dmax, _diff(r_o[0], r_f[0]))
    return len(calls), dmax, bool(calls) and dmax < 1e-9
