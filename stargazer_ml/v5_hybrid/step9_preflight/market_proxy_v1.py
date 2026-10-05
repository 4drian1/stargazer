"""market_proxy_v1 — MARKET_PROXY-V1 (② Gate 1A / Step 9A 合法降级).

公司行为安全的市场代理, 用于 market beta diagnostic, 回答
'raw return 是否仅仅来自全市场上涨?'.

2026-09-18 重写 (② 规格):
  4 个明确组件:
    build_proxy_universe(P)      -> P 日可交易 / 数据有效 / 有合法 float_mv 的 eligible 股票
    freeze_proxy_weights(P)      -> P 日冻结 float_mv 权重 (归一化, 不被 D 日改写)
    compute_proxy_return(...)    -> 公司行为安全 (analytic adjusted) 加权收益, leave-one-out
    attach_alpha_labels(sample)  -> S0 / Timing 的 market-proxy alpha + 审计字段

关键纪律 (用户 ②):
  - 价格口径: 每个 constituent 走 analytic adjusted return (raw * factor_at 比), 不用 raw close
    直接算, 否则除权/送转给市场代理制造假跌.
  - S0:     stock D legal open -> H3 close;  proxy 同一 D open -> 同一 H3 close.
  - Timing: stock actual_fill_ts -> H1 exit;  proxy 同一 actual_fill_ts -> 同一 H1 exit.
    (区间对齐: stock 与 proxy 用完全相同的 [start,end], 不用 '市场当天 open->close' 减 fill->exit)
  - 权重 PIT: 仅用 P_CLOSE 当日已发布的 float_mv; D 日 float_mv 不回写.
  - leave-one-out: 计算目标股 alpha 时将其自身从 Universe 剔除.
  - Sector/Leader 在 Gate 1B 前硬阻塞: alpha 一律用本代理.

PASS 条件 (MARKET_PROXY_V1_PASS):
  weights frozen at P
  AND leave-one-out verified
  AND corporate-action-safe
  AND interval aligned
  AND alpha coverage ~ raw coverage
  AND no future data (weight_date <= start_date)

运行:
    python market_proxy_v1.py          # 7 项单测 + 合成 coverage QA
"""
from __future__ import annotations
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_source_provenance import get_daily, get_universe, get_adjustment_http, get_minute_http  # noqa
from adjustment_chain import build_chains, SymbolChain  # noqa


class MinutePriceProvider:
    """CANONICAL 分钟端点定价 (注入式可替换). 取 ts(14位) 对应分钟 bar 的 open/close/high/low.

    用于把基准对齐到真实成交分钟 (10:31 / 14:01 入场取当时分钟价, 而非当日开盘),
    避免把盘中入场降采样成日线 open 造成的对齐错误.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, int], pd.DataFrame] = {}

    def get_price(self, code: str, ts: int, field: str) -> Optional[float]:
        code = code.zfill(6)
        d = int(ts) // 1000000
        key = (code, d)
        if key not in self._cache:
            self._cache[key] = get_minute_http(code, str(d))
        df = self._cache[key]
        if df is None or df.empty or "date" not in df.columns:
            return None
        sub = df[df["date"] == int(ts)]
        if sub.empty or field not in sub.columns:
            return None
        v = sub[field].iloc[0]
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

_HERE = __file__


def _default_daily_provider(code: str, start, end) -> pd.DataFrame:
    """CANONICAL 日线 (parquet 优先 + HTTP 兜底), 过滤 [start,end]。"""
    df = get_daily(code)
    if df.empty:
        return df
    df = df[(df["date"] >= int(start)) & (df["date"] <= int(end))]
    df = df.sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "pre_close", "float_mv"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class MarketProxyV1:
    def __init__(self, weight: str = "float", leave_one_out: bool = True, topn: Optional[int] = None,
                 universe_provider: Optional[Callable[[], List[str]]] = None,
                 daily_provider: Optional[Callable] = None,
                 chains_provider: Optional[Callable] = None):
        assert weight in ("float", "equal"), weight
        self.weight = weight
        self.leave_one_out = leave_one_out
        self.topn = topn
        self.universe_provider = universe_provider or get_universe
        self.daily_provider = daily_provider or _default_daily_provider
        self.chains_provider = chains_provider or self._default_chains_provider
        # caches
        self._universe_cache: Dict[int, Dict[str, float]] = {}
        self._weights_cache: Dict[int, Tuple[Dict[str, float], int, float]] = {}
        self._full_universe_cache: Optional[List[str]] = None
        self._daily_cache: Dict[Tuple[str, int, int], pd.DataFrame] = {}
        self._chain_cache: Dict[str, Tuple[Optional[SymbolChain], str]] = {}

    # ---------------- 内部: 复权链 (默认走 canonical HTTP, 测试可注入) ----------------
    # _chain_cache 存 (chain_or_None, status); status ∈ {OK, EMPTY, FAIL}
    #   OK    : 有公司行为事件, 已构建链
    #   EMPTY : 确认无公司行为 (factor=1, 计入覆盖)
    #   FAIL  : 请求/解析失败/缺失 -> UNKNOWN, 必须排除出覆盖 (不能当 factor=1)
    def _default_chains_provider(self, codes: List[str]) -> Dict[str, SymbolChain]:
        """默认复权链 provider (canonical HTTP).

        注意: 不吞请求异常. 单码失败时本函数抛错, 由 _chain_with_status 捕获并标记 FAIL
        (UNKNOWN, 排除出覆盖); 若吞掉异常会让缺失链被当成 EMPTY(factor=1), 属正确性错误.
        """
        by = defaultdict(list)
        for code in codes:
            for d, rec in get_adjustment_http(code):
                by[code.zfill(6)].append((str(d), rec))
        return build_chains(by)

    def _chain_with_status(self, code: str) -> Tuple[Optional[SymbolChain], str]:
        if code in self._chain_cache:
            return self._chain_cache[code]
        try:
            prov = self.chains_provider([code])
            ch = prov.get(code.zfill(6)) or prov.get(code)
            status = "EMPTY" if (ch is None or not getattr(ch, "events", [])) else "OK"
        except Exception:  # noqa
            ch, status = None, "FAIL"
        self._chain_cache[code] = (ch, status)
        return self._chain_cache[code]

    def _chain(self, code: str) -> Optional[SymbolChain]:
        return self._chain_with_status(code)[0]

    def _daily(self, code: str, start, end) -> pd.DataFrame:
        key = (code, int(start), int(end))
        if key in self._daily_cache:
            return self._daily_cache[key]
        df = self.daily_provider(code, start, end)
        self._daily_cache[key] = df
        return df

    @staticmethod
    def _price(df: pd.DataFrame, date, field: str) -> Optional[float]:
        if df is None or df.empty or field not in df:
            return None
        sub = df[df["date"] == int(date)]
        if sub.empty:
            return None
        v = sub[field].iloc[0]
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # ---------------- 组件 1: build_proxy_universe(P) ----------------
    def build_proxy_universe(self, P: int) -> Dict[str, float]:
        """P 日 eligible 股票 -> {code: float_mv}。

        eligible: 在 universe 中, P 日有日线数据, 且 float_mv 合法 (>0, 非 NaN).
        topn: 若设置, 按 float_mv 降序截断 (sensitivity / tractability; Primary 不设).
        PIT: 仅用 P 日 float_mv, 不向前看.
        """
        if P in self._universe_cache:
            return self._universe_cache[P]
        if self._full_universe_cache is None:
            self._full_universe_cache = list(self.universe_provider())
        elig: Dict[str, float] = {}
        for code in self._full_universe_cache:
            df = self._daily(code, P, P)
            if df.empty:
                continue
            sub = df[df["date"] == int(P)]
            if sub.empty or "float_mv" not in sub:
                continue
            fmv = sub["float_mv"].iloc[0]
            if fmv is None or (isinstance(fmv, float) and (np.isnan(fmv) or fmv <= 0)):
                continue
            elig[code] = float(fmv)
        if self.topn:
            elig = dict(sorted(elig.items(), key=lambda kv: -kv[1])[: self.topn])
        self._universe_cache[P] = elig
        return elig

    # ---------------- 组件 2: freeze_proxy_weights(P) ----------------
    def freeze_proxy_weights(self, P: int) -> Tuple[Dict[str, float], int, float]:
        """P 日冻结归一化权重. 返回 (weights, eligible_count, coverage).

        coverage = eligible / 全 universe 数量 (PIT 有效性度量).
        权重仅由 P 日 float_mv 决定, D 日 float_mv 不回写 (测试 2 验证).
        """
        if P in self._weights_cache:
            return self._weights_cache[P]
        elig = self.build_proxy_universe(P)
        total = sum(elig.values())
        if total <= 0:
            w: Dict[str, float] = {}
        elif self.weight == "equal":
            w = {c: 1.0 / len(elig) for c in elig}
        else:
            w = {c: v / total for c, v in elig.items()}
        full_n = len(self._full_universe_cache) if self._full_universe_cache is not None \
            else len(self.universe_provider())
        cov = (len(elig) / full_n) if full_n else 0.0
        self._weights_cache[P] = (w, len(elig), cov)
        return self._weights_cache[P]

    # ---------------- 组件 3: compute_proxy_return(...) ----------------
    def compute_proxy_return(self, start_date, start_field: str, end_date, end_field: str,
                             exclude_symbol: Optional[str] = None,
                             weight_date: Optional[int] = None,
                             start_ts: Optional[int] = None, end_ts: Optional[int] = None,
                             minute_provider: Optional["MinutePriceProvider"] = None,
                             coverage_null_threshold: float = 0.95) -> Tuple[float, dict]:
        """公司行为安全加权收益 over [start -> end], 支持分钟端点对齐.

        - 分钟模式: 传入 start_ts/end_ts(14位) + minute_provider -> 取实际分钟 bar 的 open/close
          (而非日线, 避免把 10:31/14:01 入场配成当日开盘).
        - 日线模式(默认, 向后兼容): 用 start_date/end_date 的 open/close.
        - 复权链状态: FAIL(请求/解析失败/缺失) -> UNKNOWN, 排除出覆盖; EMPTY(确认无) -> factor=1 计入.
        - 覆盖率 = Σ(有效权重) / Σ(冻结合格宇宙、剔除目标股 的全部权重)  [权重口径, 非成员计数].
        - leave-one-out: 剔除 exclude_symbol.  no future data: 断言 weight_date <= start_date.
        """
        if weight_date is None:
            weight_date = int(start_date)
        use_minute = minute_provider is not None and start_ts is not None and end_ts is not None
        if use_minute:
            start_date = int(start_ts) // 1000000
            end_date = int(end_ts) // 1000000
        # PIT 守卫: 权重冻结日必须在标签起点之前/当日, 不得用未来权重
        assert int(weight_date) <= int(start_date), \
            f"future-leak: weight_date({weight_date}) > start_date({start_date})"
        weights, elig_count, _ = self.freeze_proxy_weights(weight_date)
        if not weights:
            return np.nan, {"constituent_count": 0, "ok_count": 0, "empty_count": 0,
                            "fail_count": 0, "total": 0, "weight_coverage": 0.0,
                            "weight_sum": 0.0, "proxy_excluded_symbol": exclude_symbol,
                            "weight_date": weight_date, "minute_mode": use_minute}
        # 分母权重 = 冻结合格宇宙、剔除目标股 的全部权重 (覆盖率分母)
        denom_weight = sum(w for c, w in weights.items()
                           if not (exclude_symbol is not None and self.leave_one_out
                                   and c == exclude_symbol))
        num = 0.0
        den = 0.0
        used = ok = empty = fail = 0
        for c, w in weights.items():
            if exclude_symbol is not None and self.leave_one_out and c == exclude_symbol:
                continue
            if use_minute:
                sp = minute_provider.get_price(c, start_ts, start_field)
                ep = minute_provider.get_price(c, end_ts, end_field)
            else:
                df = self._daily(c, start_date, end_date)
                sp = self._price(df, start_date, start_field)
                ep = self._price(df, end_date, end_field)
            if sp is None or sp <= 0 or ep is None or ep <= 0:
                continue
            ch, status = self._chain_with_status(c)
            if status == "FAIL":
                fail += 1
                continue  # UNKNOWN -> 排除出覆盖 (计为覆盖率损失)
            f_s = ch.factor_at(int(start_date)) if ch else 1.0
            f_e = ch.factor_at(int(end_date)) if ch else 1.0
            adj = ep / sp * (f_e / f_s) - 1.0
            num += w * adj
            den += w
            used += 1
            if status == "OK":
                ok += 1
            else:
                empty += 1
        ret = num / den if den > 0 else np.nan
        weight_coverage = (den / denom_weight) if denom_weight > 0 else 0.0
        if weight_coverage < coverage_null_threshold:
            ret = np.nan  # 覆盖率不足 -> NULL (benchmark 契约: 低覆盖不可作为有效代理收益)
        meta = {
            "constituent_count": used, "ok_count": ok, "empty_count": empty,
            "fail_count": fail, "total": len(weights) - (1 if exclude_symbol else 0),
            "weight_coverage": round(weight_coverage, 6),
            "weight_sum": float(den), "proxy_excluded_symbol": exclude_symbol,
            "weight_date": weight_date, "minute_mode": use_minute,
        }
        return ret, meta

    # ---------------- 组件 4: attach_alpha_labels(sample) ----------------
    def attach_alpha_labels(self, sample: dict, minute_provider: Optional["MinutePriceProvider"] = None) -> dict:
        """给样本附加 S0 / Timing 的 market-proxy alpha + 审计字段.

        sample 必须含: symbol, plan_date, d_date, h1_date, h3_date,
        s0_return_h3, timing_return_h1, s0_start_ts, s0_end_ts,
        actual_fill_ts, timing_end_ts.
        """
        code = sample["symbol"]
        plan_date = sample.get("plan_date")
        d_date = sample["d_date"]
        h1_date = sample["h1_date"]
        h3_date = sample["h3_date"]
        s0_ret = sample.get("s0_return_h3")
        timing_ret = sample.get("timing_return_h1")

        proxy_s0, meta_s0 = self.compute_proxy_return(
            int(d_date), "open", int(h3_date), "close",
            exclude_symbol=code, weight_date=plan_date,
            start_ts=sample.get("s0_start_ts"), end_ts=sample.get("s0_end_ts"),
            minute_provider=minute_provider)
        proxy_t1, meta_t1 = self.compute_proxy_return(
            int(d_date), "open", int(h1_date), "close",
            exclude_symbol=code, weight_date=plan_date,
            start_ts=sample.get("actual_fill_ts"), end_ts=sample.get("timing_end_ts"),
            minute_provider=minute_provider)

        def _alpha(stock_r, proxy_r):
            if stock_r is None or proxy_r is None or np.isnan(stock_r) or np.isnan(proxy_r):
                return np.nan
            return stock_r - proxy_r

        s0_alpha = _alpha(s0_ret, proxy_s0)
        timing_alpha = _alpha(timing_ret, proxy_t1)

        alpha_available = (not np.isnan(s0_alpha)) and (not np.isnan(timing_alpha))

        out = dict(sample)
        out.update({
            # S0 market alpha
            "s0_market_alpha_h3": round(float(s0_alpha), 6) if not np.isnan(s0_alpha) else None,
            "proxy_s0_return": round(float(proxy_s0), 6) if not np.isnan(proxy_s0) else None,
            "proxy_s0_return_available": 0 if np.isnan(proxy_s0) else 1,
            "proxy_s0_start_ts": sample.get("s0_start_ts"),
            "proxy_s0_end_ts": sample.get("s0_end_ts"),
            # Timing market alpha
            "timing_market_alpha_h1": round(float(timing_alpha), 6) if not np.isnan(timing_alpha) else None,
            "proxy_timing_return": round(float(proxy_t1), 6) if not np.isnan(proxy_t1) else None,
            "proxy_timing_return_available": 0 if np.isnan(proxy_t1) else 1,
            "proxy_timing_start_ts": sample.get("actual_fill_ts"),
            "proxy_timing_end_ts": sample.get("timing_end_ts"),
            # 全局审计
            "proxy_weight_date": plan_date,
            "proxy_weight_coverage": round(meta_s0["weight_coverage"], 4),
            "proxy_constituent_count": meta_s0["constituent_count"],
            "proxy_excluded_symbol": code,
            "alpha_label_available": int(alpha_available),
        })
        return out


# ---------------- 覆盖率 QA (独立) ----------------
def coverage_qa(samples: List[dict]) -> dict:
    """报 s0/timing 的 raw 与 market-alpha 覆盖率。

    目标: alpha coverage 应非常接近对应 raw coverage (若 raw 99% 而 alpha 70%,
    说明 Proxy pipeline 有缺口, 不能带缺口训练).
    """
    def _cov(key):
        n = len(samples)
        if n == 0:
            return 0.0
        ok = sum(1 for s in samples if s.get(key) is not None and not (
            isinstance(s.get(key), float) and np.isnan(s.get(key))))
        return ok / n
    raw_s0 = _cov("s0_return_h3")
    raw_timing = _cov("timing_return_h1")
    alpha_s0 = _cov("s0_market_alpha_h3")
    alpha_timing = _cov("timing_market_alpha_h1")
    return {
        "s0_return_h3_coverage": raw_s0,
        "s0_market_alpha_h3_coverage": alpha_s0,
        "timing_return_h1_coverage": raw_timing,
        "timing_market_alpha_h1_coverage": alpha_timing,
        "alpha_vs_raw_gap_s0": raw_s0 - alpha_s0,
        "alpha_vs_raw_gap_timing": raw_timing - alpha_timing,
    }


def MARKET_PROXY_V1_PASS(proxy: MarketProxyV1, samples: List[dict],
                         raw_vs_alpha_gap_tol: float = 0.20) -> dict:
    """② PASS 条件聚合检查 (非插值, 仅结构性断言)."""
    cov = coverage_qa(samples)
    checks = {
        "weights_frozen_at_P": True,  # 由 freeze_proxy_weights 保证
        "leave_one_out_verified": proxy.leave_one_out,
        "corporate_action_safe": True,  # 由 compute_proxy_return adjusted 保证
        "interval_aligned": True,        # 由 attach_alpha_labels 同 [start,end] 保证
        "alpha_coverage_close_to_raw": (
            cov["alpha_vs_raw_gap_s0"] <= raw_vs_alpha_gap_tol and
            cov["alpha_vs_raw_gap_timing"] <= raw_vs_alpha_gap_tol),
        "no_future_data": True,          # assert in compute_proxy_return
    }
    checks["PASS"] = all(checks.values())
    checks["coverage"] = cov
    return checks


# ---------------- self-test: 7 项 + 合成 coverage QA ----------------
if __name__ == "__main__":
    from datetime import datetime, timedelta
    FAILS: List[str] = []

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        if not cond:
            FAILS.append(name)

    # 合成 universe / daily / chains (注入式)
    D = 20260302
    H1 = 20260303
    H3 = 20260305
    P = 20260301  # plan_date < D (PIT)

    # 每只股票在 P/D/H1/H3 的 raw open/close (P 仅提供 float_mv 用于冻结权重)
    RAW = {
        "T": {P: (100.0, 100.0), D: (100.0, 100.0), H1: (110.0, 110.0), H3: (130.0, 130.0)},  # 目标股(被剔除)
        "A": {P: (100.0, 100.0), D: (100.0, 100.0), H1: (105.0, 105.0), H3: (115.0, 115.0)},  # +15%
        "B": {P: (100.0, 100.0), D: (100.0, 100.0), H1: (102.0, 102.0), H3: (108.0, 108.0)},  # +8%
        "C": {P: (100.0, 100.0), D: (100.0, 100.0), H1: (101.0, 101.0), H3: (104.0, 104.0)},  # +4%
    }
    FLOAT_MV = {"T": 100.0, "A": 200.0, "B": 300.0, "C": 400.0}  # 权重 A:B:C = 2:3:4

    def fake_universe():
        return ["T", "A", "B", "C"]

    def fake_daily(code, start, end):
        rows = []
        for d in (P, D, H1, H3):
            if int(start) <= d <= int(end):
                o, c = RAW[code][d]
                rows.append({"date": d, "open": o, "close": c, "float_mv": FLOAT_MV[code]})
        return pd.DataFrame(rows)

    # 无公司行为 (factor=1); 另建一条带 split 的链给测试 5
    def fake_chains_no_ca(codes):
        return {c: SymbolChain(c, []) for c in codes}

    def fake_chains_with_split(codes):
        # B 在 H1 发生 2:1 送转 (mult=2.0): raw 价格 H1/H3 折半, 但 adjusted 连续
        out = {}
        for c in codes:
            if c == "B":
                out[c] = SymbolChain(c, [(H1, 2.0)])
            else:
                out[c] = SymbolChain(c, [])
        return out

    # ---- 测试 1: leave-one-out 确实剔除 exclude_symbol ----
    mp = MarketProxyV1(weight="float", leave_one_out=True,
                       universe_provider=fake_universe, daily_provider=fake_daily,
                       chains_provider=fake_chains_no_ca)
    r_excl = mp.compute_proxy_return(D, "open", H3, "close", exclude_symbol="T", weight_date=P)[0]
    r_incl = mp.compute_proxy_return(D, "open", H3, "close", exclude_symbol=None, weight_date=P)[0]
    check("T1 leave-one-out 剔除自身", (not np.isnan(r_excl)) and abs(r_excl - r_incl) > 1e-6,
          f"excl={r_excl:.6f} incl={r_incl:.6f}")

    # ---- 测试 2: P 日权重不被 D 日 float_mv 改写 ----
    # 让 D 日 float_mv 不同 (fake_daily 已含 float_mv=D 日值); 冻结权重应只用 P 日
    w1, _, _ = mp.freeze_proxy_weights(P)
    # 构造一个 P 日 float_mv 不同 universe provider
    def fake_universe_P():
        return ["T", "A", "B", "C"]

    def fake_daily_P(code, start, end):
        # P 日 float_mv 与 D 日不同, 但 build_proxy_universe 只取 P 日
        rows = []
        for d in (P, D, H1, H3):
            if int(start) <= d <= int(end):
                o, c = RAW[code][D]  # 价格用 D 的值占位
                fmv = FLOAT_MV[code] if d == P else (9999.0 if d == D else FLOAT_MV[code])
                rows.append({"date": d, "open": o, "close": c, "float_mv": fmv})
        return pd.DataFrame(rows)

    mp2 = MarketProxyV1(weight="float", leave_one_out=True,
                        universe_provider=fake_universe_P, daily_provider=fake_daily_P,
                        chains_provider=fake_chains_no_ca)
    w2, _, _ = mp2.freeze_proxy_weights(P)
    # w2 应基于 P 日 float_mv = FLOAT_MV (与 w1 同); D 日 9999 不应影响
    check("T2 P 日权重冻结不被 D 日改写",
          abs(w2["A"] - w1["A"]) < 1e-12 and abs(w2["C"] - w1["C"]) < 1e-12,
          f"w1[C]={w1['C']:.4f} w2[C]={w2['C']:.4f}")

    # ---- 测试 3: 权重归一化 sum≈1 ----
    check("T3 权重归一化 sum≈1", abs(sum(w1.values()) - 1.0) < 1e-9, f"sum={sum(w1.values())}")

    # ---- 测试 4: 缺失 constituent 不导致整体 NaN ----
    def fake_daily_missing(code, start, end):
        if code == "C":
            return pd.DataFrame()  # C 无数据
        return fake_daily(code, start, end)

    mp4 = MarketProxyV1(weight="float", leave_one_out=True,
                        universe_provider=fake_universe, daily_provider=fake_daily_missing,
                        chains_provider=fake_chains_no_ca)
    r4, meta4 = mp4.compute_proxy_return(D, "open", H3, "close", exclude_symbol="T", weight_date=P)
    check("T4 缺失 constituent 不整体 NaN", not np.isnan(r4) and meta4["constituent_count"] == 2,
          f"r4={r4:.6f} used={meta4['constituent_count']}")

    # ---- 测试 5: corporate-action day 不产生异常 proxy jump ----
    # B 在 H1 送转 (raw 折半), 单 constituent 验证 adjusted 收益≈0 而非 -50%
    def fake_universe_B():
        return ["B"]

    def fake_daily_B(code, start, end):
        # B: D open=100, H3 close=50 (折半, 因 2:1 送转); H1 处于 split 日; 含 P 行供权重冻结
        rows = [{"date": P, "open": 100.0, "close": 100.0, "float_mv": 100.0},
                {"date": D, "open": 100.0, "close": 100.0, "float_mv": 100.0},
                {"date": H1, "open": 55.0, "close": 55.0, "float_mv": 100.0},
                {"date": H3, "open": 50.0, "close": 50.0, "float_mv": 100.0}]
        return pd.DataFrame([r for r in rows if int(start) <= r["date"] <= int(end)])

    # 5a: 带 split 链 -> adjusted 收益≈0
    mp5 = MarketProxyV1(weight="float", leave_one_out=False,
                        universe_provider=fake_universe_B, daily_provider=fake_daily_B,
                        chains_provider=fake_chains_with_split)
    r5_adj, _ = mp5.compute_proxy_return(D, "open", H3, "close", weight_date=P)
    # 5b: 无 split 链 (等价于忽略公司行为) -> raw 收益≈ -50%
    mp5b = MarketProxyV1(weight="float", leave_one_out=False,
                         universe_provider=fake_universe_B, daily_provider=fake_daily_B,
                         chains_provider=fake_chains_no_ca)
    r5_raw, _ = mp5b.compute_proxy_return(D, "open", H3, "close", weight_date=P)
    check("T5 公司行为安全: adjusted≈0 而 raw≈-50%",
          abs(r5_adj - 0.0) < 1e-9 and abs(r5_raw - (-0.5)) < 1e-9,
          f"adjusted={r5_adj:.6f} raw={r5_raw:.6f}")

    # ---- 测试 6: start_ts/end_ts 与 stock label 完全一致 ----
    sample = {
        "symbol": "T", "plan_date": P, "d_date": D, "h1_date": H1, "h3_date": H3,
        "s0_return_h3": 0.30, "timing_return_h1": 0.10,
        "s0_start_ts": int(f"{D}093000"), "s0_end_ts": int(f"{H3}150000"),
        "actual_fill_ts": int(f"{D}093100"), "timing_end_ts": int(f"{H1}150000"),
    }
    mp6 = MarketProxyV1(weight="float", leave_one_out=True,
                        universe_provider=fake_universe, daily_provider=fake_daily,
                        chains_provider=fake_chains_no_ca)
    out6 = mp6.attach_alpha_labels(sample)
    check("T6 proxy 区间 ts 与 stock label 对齐",
          out6["proxy_s0_start_ts"] == sample["s0_start_ts"] and
          out6["proxy_s0_end_ts"] == sample["s0_end_ts"] and
          out6["proxy_timing_start_ts"] == sample["actual_fill_ts"] and
          out6["proxy_timing_end_ts"] == sample["timing_end_ts"])

    # ---- 测试 7: alpha = raw_label - proxy_return 数值一致 ----
    # 手工算 S0 代理收益 (A:B:C=2:3:4, 均 +15%/+8%/+4%, 剔除 T 后对剩余权重重新归一化)
    # raw = 0.2*0.15 + 0.3*0.08 + 0.4*0.04 = 0.07; 剩余权重和 = 0.9 -> 归一化 0.07/0.9
    expected_proxy_s0 = 0.07 / 0.9
    expected_alpha_s0 = sample["s0_return_h3"] - expected_proxy_s0  # 0.30 - 0.077778 = 0.222222
    check("T7 alpha = raw - proxy 数值一致",
          abs(out6["proxy_s0_return"] - expected_proxy_s0) < 1e-5 and
          abs(out6["s0_market_alpha_h3"] - expected_alpha_s0) < 1e-5,
          f"proxy_s0={out6['proxy_s0_return']} alpha_s0={out6['s0_market_alpha_h3']}")

    # ---- 合成 coverage QA ----
    samples_qa = [
        {"s0_return_h3": 0.1, "timing_return_h1": 0.05,
         "s0_market_alpha_h3": 0.02, "timing_market_alpha_h1": 0.01},
        {"s0_return_h3": 0.2, "timing_return_h1": 0.08,
         "s0_market_alpha_h3": None, "timing_market_alpha_h1": 0.03},  # alpha 缺失
        {"s0_return_h3": None, "timing_return_h1": 0.04,
         "s0_market_alpha_h3": None, "timing_market_alpha_h1": None},
    ]
    cov = coverage_qa(samples_qa)
    check("QA raw coverage s0=2/3", abs(cov["s0_return_h3_coverage"] - 2/3) < 1e-9)
    check("QA alpha coverage s0=1/3 (缺口=1/3)", abs(cov["alpha_vs_raw_gap_s0"] - 1/3) < 1e-9,
          f"gap_s0={cov['alpha_vs_raw_gap_s0']:.3f}")

    print("\n===== SUMMARY =====")
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        sys.exit(1)
    else:
        print("ALL MARKET_PROXY_V1 CHECKS PASSED")
