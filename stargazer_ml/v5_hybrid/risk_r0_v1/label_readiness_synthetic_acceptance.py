"""label_readiness_synthetic_acceptance.py — 就绪盘点合成验收（无需真实数据/取数）。

验证 label_readiness_inventory 的价格加载与冲突阻断逻辑：
- S-AC-R1 重复键数值冲突 → 登记 value_mismatch 且 data_ok 阻断（不静默放行）；
- S-AC-R2 同值不同来源重复 → 登记来源但 value_mismatch=False，data_ok 不阻断；
- S-AC-R3 价格 NaN/≤0 → valid_price 判无效；
- S-AC-R4 正常窗口 → data_ok=True。
仅验证构造与防泄漏逻辑，不依赖真实 parquet/训练。
"""
from label_readiness_inventory import (
    aggregate_prices, evaluate_window_prices, valid_price,
)


def run():
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # ---- S-AC-R1 重复键数值冲突 ----
    rows = [
        ('000001', 20260105, 10.0, 9.0, 'part_00000.parquet'),
        ('000001', 20260105, 11.0, 9.0, 'part_incr_01.parquet'),  # 不同 close → 数值冲突
        ('000001', 20260106, 10.0, 9.0, 'part_00000.parquet'),
    ]
    out, conflicts = aggregate_prices(rows)
    key = ('000001', 20260105)
    check('S-AC-R1 数值冲突被登记',
          key in conflicts and conflicts[key]['value_mismatch'])
    ck = {k for k, c in conflicts.items() if c['value_mismatch']}
    p_close, lows, data_ok, any_c = evaluate_window_prices(
        out, ck, '000001', 20260105, ['2026-01-05', '2026-01-06'])
    check('S-AC-R1 冲突键阻断 data_ok',
          data_ok is False and any_c is True,
          f'data_ok={data_ok} any_conflict={any_c}')

    # ---- S-AC-R2 同值不同来源（良性重复）----
    rows2 = [
        ('000002', 20260105, 10.0, 9.0, 'part_00000.parquet'),
        ('000002', 20260105, 10.0, 9.0, 'part_incr_01.parquet'),  # 同值
    ]
    out2, conflicts2 = aggregate_prices(rows2)
    c2 = conflicts2.get(('000002', 20260105))
    check('S-AC-R2 同值重复不记数值冲突',
          c2 is not None and c2['value_mismatch'] is False)
    ck2 = {k for k, c in conflicts2.items() if c['value_mismatch']}
    _, _, data_ok2, any_c2 = evaluate_window_prices(
        out2, ck2, '000002', 20260105, ['2026-01-05'])
    check('S-AC-R2 良性重复不阻断 data_ok',
          data_ok2 is True and any_c2 is False, f'data_ok={data_ok2}')

    # ---- S-AC-R3 NaN/≤0 无效 ----
    check('S-AC-R3 NaN 无效', not valid_price(float('nan')))
    check('S-AC-R3 ≤0 无效', not valid_price(0.0) and not valid_price(-1.0))
    check('S-AC-R3 正数有效', valid_price(9.5))

    # ---- S-AC-R4 正常窗口 ----
    rows3 = [('000003', 20260105, 10.0, 9.0, 'p0'),
             ('000003', 20260106, 10.5, 9.2, 'p0')]
    out3, conflicts3 = aggregate_prices(rows3)
    ck3 = {k for k, c in conflicts3.items() if c['value_mismatch']}
    pc, lows3, dok, ac = evaluate_window_prices(
        out3, ck3, '000003', 20260105, ['2026-01-05', '2026-01-06'])
    check('S-AC-R4 正常窗口 data_ok',
          dok is True and ac is False and pc == 10.0,
          f'data_ok={dok} p_close={pc}')

    # ---- 汇总 ----
    print('==== 就绪盘点 合成验收 ====')
    npass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\n{npass}/{len(results)} PASS")
    return npass == len(results)


if __name__ == '__main__':
    raise SystemExit(0 if run() else 1)
