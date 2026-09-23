"""阶段2.1 审计专用复放：直接指定入场日，绕过 G0 候选筛选，仅验证退出引擎。

用例（来自冻结规格要求）：
  600313  2026-09-04
  600479  2026-09-07
  600479  2026-09-08

运行：python audit_replay.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from build_exit_labels import replay_case, VARIANTS

CASES = [("600313", "20260904"), ("600479", "20260907"), ("600479", "20260908")]


def main():
    for code, ed in CASES:
        print(f"\n==== {code} 入场日 {ed} ====")
        for v in VARIANTS:
            r = replay_case(code, ed, v)
            if r is None:
                print(f"  {v:12s}: None (无数据 / 窗口内不可买入)")
                continue
            print(f"  {v:12s}: entry={r['entry_date']} exit={r['exit_reason']:22s} "
                  f"price={r['exit_price']:.4f} net={r['net_return']:+.4f} "
                  f"hold={r['holding_days']} blocked={r['exit_blocked']} prec={r['label_precision']}")


if __name__ == "__main__":
    main()
