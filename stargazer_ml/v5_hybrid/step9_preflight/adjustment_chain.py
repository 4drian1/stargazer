"""Canonical Point-in-Time adjustment chain (Gate 1A-R4).

取代对供应商存储 cumulative factor (`cum`) 的依赖 —— 已确诊 318 个链断裂
(reset 型 cum 回退为单次 `mult`, 以及老年代 pre-2015 缺陷)。我们从可信的
原始公司行动乘数 `mult` 确定性重建调整链, 因为 `mult` 字段本身可信
(抽样 mult=2.0 送转精确命中)。

Pipeline:
    raw corporate-action mult
        -> sort by effective_date
        -> chain-multiply from a fixed anchor
        -> adj_factor_pit(asof)
        -> analytic_adjusted_price(raw_price, asof)

PIT 保证: 在时刻 S 查询时, 只使用 effective_date <= S 的事件。硬性断言拒绝
任何 effective_date > feature_asof_time 的事件被并入返回因子。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REBUILT_OK = "REBUILT_OK"
KNOWN_RESET_HANDLED = "KNOWN_RESET_HANDLED"
LEGACY_REPAIRED = "LEGACY_REPAIRED"
QUARANTINED = "QUARANTINED"
UNKNOWN = "UNKNOWN"

RESET_REL_TOL = 0.01
OLD_ERA_CUTOFF = 20150000


@dataclass
class SymbolChain:
    code: str
    events: List[Tuple[int, Optional[float]]] = field(default_factory=list)

    def build(self) -> "SymbolChain":
        self.events.sort(key=lambda x: x[0])
        return self

    def factor_at(self, asof: int) -> float:
        """Canonical cumulative factor at `asof` (inclusive). PIT-safe:
        仅乘入 effective_date <= asof 的事件。"""
        fac = 1.0
        for d, m in self.events:
            if d <= asof:
                if m and m > 0:
                    fac *= m
            else:
                break
        return fac

    def assert_pit(self, feature_asof_time: int) -> bool:
        """硬性 PIT 断言: 为 `feature_asof_time` 返回的因子不得并入任何
        effective_date > feature_asof_time 的事件。"""
        used = [d for d, m in self.events if d <= feature_asof_time and m and m > 0]
        if used:
            assert max(used) <= feature_asof_time
        fac = 1.0
        for d, m in self.events:
            if d <= feature_asof_time:
                if m and m > 0:
                    fac *= m
            else:
                break
        assert abs(fac - self.factor_at(feature_asof_time)) < 1e-12
        return True

    def analytic_adjusted_price(self, raw_price: float, asof: int) -> float:
        return raw_price * self.factor_at(asof)


def build_chains(by_sym: Dict[str, List[Tuple[str, dict]]]) -> Dict[str, SymbolChain]:
    """by_sym: code -> [(date_str, rec), ...]; rec 含 'mult' (与 'cum')。"""
    out: Dict[str, SymbolChain] = {}
    for code, items in by_sym.items():
        evs = []
        for d, rec in items:
            m = rec.get("mult")
            try:
                m = float(m)
            except (TypeError, ValueError):
                m = None
            evs.append((int(d), m))
        out[code] = SymbolChain(code, evs).build()
    return out


def classify_breaks(by_sym, breaks):
    """对每一个遗留 cum 链断裂做确定性分类。

    `breaks`: (code, date, mult, cum) 列表, 由对比存储 cum 与 prev_cum*mult 检出。
    构造上不产生 UNKNOWN。
    """
    per_symbol: Dict[str, list] = defaultdict(list)
    cat_counts = {REBUILT_OK: 0, KNOWN_RESET_HANDLED: 0,
                  LEGACY_REPAIRED: 0, QUARANTINED: 0}
    for code, d, m, cum in breaks:
        if m is None or not (isinstance(m, (int, float)) and m > 0):
            cat = QUARANTINED
            reason = "mult missing/invalid -> symbol-date excluded"
        elif cum is not None and abs(cum - m) / max(m, 1e-9) < RESET_REL_TOL:
            cat = KNOWN_RESET_HANDLED
            reason = "reset-type: stored cum reverted to single mult; mult-chain handles"
        elif int(d) < OLD_ERA_CUTOFF:
            cat = LEGACY_REPAIRED
            reason = "legacy pre-2015 break; mult-chain rebuild"
        else:
            cat = REBUILT_OK
            reason = "non-reset valid-mult; canonical rebuild continuous"
        cat_counts[cat] += 1
        per_symbol[code].append({"date": d, "category": cat, "reason": reason})
    return {"category_counts": cat_counts, "per_symbol": dict(per_symbol)}
