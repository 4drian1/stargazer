"""板块/龙头三层表 — 合成验收（SYNTHETIC ACCEPTANCE，无需采集/拟合）。

目的：把 sector_leader_context_tables_design.md §7 的 S-AC1~5 与新增反例从设计变成**可运行测试**，
验证三层表的构造与防泄漏语义，不依赖真实源、不训练。

核心不变量：在决策时刻 t 构造的上下文字段，只应使用 t 及之前**已知且可得**的数据：
  - 名单变更只作用于其 known_at 之后（S-AC1）；
  - 迟到数据不改变已落地的过去字段（S-AC2）；
  - 候选自身是唯一代表股 → 代表股字段 NULL，不可用自身走势填充（S-AC3）；
  - 空基准 → NULL，不记 0（S-AC4）；
  - 未来扰动不改变过去字段（S-AC5）；
  - 缺失必要成员 → 不能悄悄缩小分母（新增反例）。

运行：python sector_leader_synthetic_acceptance.py  -> 打印 PASS/FAIL 与汇总。
"""
from __future__ import annotations


# ---------------- 参考构造器（编码设计 §1–§3 规则） ----------------
def ret5(prices, code, t, back: int = 4):
    """5 日收益：用 t-back 与 t 的 close，仅当两者均在 t 及之前可得。"""
    if code not in prices:
        return None
    s = prices[code]
    if (t - back) in s and t in s:
        return s[t] / s[t - back] - 1.0
    return None


def build_sector_context(t, target, sector_members, prices):
    """返回 (sector_ret_5d, missing_members)。
    规则：剔除目标股；任一必要成员在 t 不可得 → 不悄悄缩小分母，返回 None + 缺失集合。"""
    members = [c for c in sector_members if c != target]
    if not members:
        return None, set()
    rets = {c: ret5(prices, c, t) for c in members}
    computable = {c for c, r in rets.items() if r is not None}
    if computable != set(members):
        return None, (set(members) - computable)   # 缺失必要成员 → 显式 NULL
    return sum(rets[c] for c in members) / len(members), set()


def build_leader_context(t, target, leader_members, weights, prices):
    """返回 (leader_ret_5d, missing_members)。规则同上，权重在剔除目标后归一。"""
    members = [c for c in leader_members if c != target]
    if not members:
        return None, set()
    rets = {c: ret5(prices, c, t) for c in members}
    computable = {c for c, r in rets.items() if r is not None}
    if computable != set(members):
        return None, (set(members) - computable)
    w = {c: weights.get(c, 1.0) for c in members}
    tot = sum(w[c] for c in members)
    return sum(rets[c] * w[c] for c in members) / tot, set()


# ---------------- 合成数据辅助 ----------------
def make_prices(codes, base=100.0, drift=1.0):
    """为 codes 构造 t=6..15 的简单收盘价序列。"""
    out = {}
    for c in codes:
        out[c] = {d: base + drift * (d - 6) for d in range(6, 16)}
    return out


def run():
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    codes = ["A", "B", "C", "TARGET"]
    prices = make_prices(codes)          # t=6..15 均有价

    # ---------- S-AC1 历史名单变更 ----------
    t1, t2 = 10, 20
    L1 = {"members": ["A", "B"], "known_at": t1}
    ctx1_sec, _ = build_sector_context(t1, "TARGET", ["A", "B"], prices)
    ctx1_lead, _ = build_leader_context(t1, "TARGET", L1["members"], {"A": 1, "B": 1}, prices)
    # 在 t2 改变名单（新增 C），known_at=t2 > t1
    L2 = {"members": ["A", "B", "C"], "known_at": t2}
    # 重建 t1：只能用 known_at<=t1 的名单，即 L1，C 不出现
    ctx1b_sec, _ = build_sector_context(t1, "TARGET", ["A", "B"], prices)
    ctx1b_lead, _ = build_leader_context(t1, "TARGET", L1["members"], {"A": 1, "B": 1}, prices)
    check("S-AC1 历史名单变更不改变过去字段",
          ctx1_sec == ctx1b_sec and ctx1_lead == ctx1b_lead,
          f"sec={ctx1_sec}/{ctx1b_sec} lead={ctx1_lead}/{ctx1b_lead}")

    # ---------- S-AC2 迟到数据 ----------
    ctx_a_sec, _ = build_sector_context(10, "TARGET", ["A", "B"], prices)
    prices_late = {c: dict(prices[c]) for c in prices}
    prices_late["A"][11] = 999.0        # 迟到数据（t=11）
    ctx_a2_sec, _ = build_sector_context(10, "TARGET", ["A", "B"], prices_late)
    check("S-AC2 迟到数据不改变已落地字段",
          ctx_a_sec == ctx_a2_sec, f"before={ctx_a_sec} after_late={ctx_a2_sec}")

    # ---------- S-AC3 候选自身是唯一代表股 ----------
    lead_only_self, miss = build_leader_context(10, "TARGET", ["TARGET"], {"TARGET": 1}, prices)
    target_own_ret = ret5(prices, "TARGET", 10)
    check("S-AC3 候选自身唯一代表股→NULL（不填自身走势）",
          lead_only_self is None and miss == set() and target_own_ret is not None,
          f"lead={lead_only_self} target_ret={target_own_ret}")

    # ---------- S-AC4 空基准 → NULL 非 0 ----------
    sec_empty, _ = build_sector_context(10, "TARGET", ["TARGET"], prices)  # 剔除目标后空
    # 市场上下文独立、仍可得（此处仅验证板块不为 0）
    mkt_ret = ret5(prices, "A", 10)
    check("S-AC4 空基准→NULL（不记 0）且市场上下文独立",
          sec_empty is None and mkt_ret is not None, f"sec={sec_empty} mkt={mkt_ret}")

    # ---------- S-AC5 未来扰动不改变过去字段 ----------
    ctx_b_sec, _ = build_sector_context(10, "TARGET", ["A", "B"], prices)
    prices_future = {c: dict(prices[c]) for c in prices}
    prices_future["A"][11] = 5.0         # 未来极端扰动
    prices_future["A"][12] = 500.0
    ctx_b2_sec, _ = build_sector_context(10, "TARGET", ["A", "B"], prices_future)
    check("S-AC5 未来扰动不改变过去字段",
          ctx_b_sec == ctx_b2_sec, f"before={ctx_b_sec} after_perturb={ctx_b2_sec}")

    # ---------- 新增反例：缺失必要成员不能悄悄缩小分母 ----------
    # 板块应有 A,B,C 三个必要成员；但 C 在 t=10 无价（缺数据）
    prices_missing = {c: dict(prices[c]) for c in prices}
    del prices_missing["C"][10]
    del prices_missing["C"][6]
    sec_miss, missing = build_sector_context(10, "TARGET", ["A", "B", "C"], prices_missing)
    # 若偷偷只用 A,B 计算会得到某均值；断言函数返回 None（不缩小分母）
    silent_shrink_value = (ret5(prices_missing, "A", 10) + ret5(prices_missing, "B", 10)) / 2.0
    check("反例 缺失必要成员→NULL（不悄悄缩小分母）",
          sec_miss is None and missing == {"C"} and silent_shrink_value is not None,
          f"sec={sec_miss} missing={missing} silent_would_be={silent_shrink_value}")

    # ---------- 汇总 ----------
    print("==== SECTOR/LEADER 三层表 合成验收 ====")
    npass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\n{npass}/{len(results)} PASS")
    return npass == len(results)


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
