# -*- coding: utf-8 -*-
"""688656 Event A 只读触发诊断 v3.1 (用户 2026-10-04 批准只读补证后).

依赖 gcp_readonly_supplement_evidence.reconstruct_event_a(): 用 E 日分钟 + P 日及以前日线,
按已冻结入场规则重建【模拟 fill】与 atr_ref, 再按 §6.3 合同止损公式对缺口前日(20250723, 遵守 T+1)做只读扫描.

重要: 模拟 fill 非已验证实际成交; 止损后能否成交仍须 §6.5 及执行对账, 不凭触价写任何收益. H3/H5 净收益保持 NULL.
本脚本仅把重建结果落盘为 Event A 诊断 JSON, 不产生/不写入任何收益标签.

产物: gcp_round_b_small_sample_v1/gcp_event_a_stopout_diagnosis.json
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gcp_readonly_supplement_evidence import reconstruct_event_a

OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_event_a_stopout_diagnosis.json"


def main():
    res = reconstruct_event_a()
    res = {**res,
           "note": ("只读诊断 v3.1: 模拟 fill (规则重建, 非已验证实际成交) 与 atr_ref 已由只读补证重建 "
                    "(gcp_readonly_supplement_evidence). 缺口前日网格完整->以模拟 fill 为条件的价格触发诊断可判; "
                    "止损扫描遵守 T+1; 但止损成交可行性待 §6.5, 不写收益. H3/H5 净收益保持 NULL."),
           "writes_return": False}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Event A v3.1] {res.get('code')} diag_day={res.get('diag_day')} -> verdict={res.get('verdict')}")
    cs = res.get("contract_stop", {})
    print(f"  atr_ref={res.get('atr_ref')} sim_fill={res.get('sim_fill',{}).get('sim_fill_price')} "
          f"StopPrice={cs.get('StopPrice')} 首触={res.get('first_stop_breach')}")
    print(f"  收益写入=False (待 §6.5). 明细: {OUT.name}")


if __name__ == "__main__":
    main()
