# -*- coding: utf-8 -*-
"""顺延层待定范围盘点 (Task 5, 2026-10-04, 零抓取, 不冻结/不批准).

§6.4 成交规则与上限尚未签署, 本脚本只做**待定范围盘点**:
  - 对 247 事件, 以 H5(=E+5 交易日) 为起点, 向后顺延 k 个"交易日"(按工作日近似, 为保守上界)
    枚举顺延日 (code,date), 去重得增量, 并统计超出数据终点 20260928 的数量.
  - 不冻结请求量, 不批准采集; 仅给出候选视界下的范围, 供 §6.4 签署后另算冻结量.
产物: gcp_round_b_small_sample_v1/gcp_rollover_scope.json
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
VERIFY = HERE / "gcp_round_b_small_sample_v1" / "gcp_247_coverage_verify.json"
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_rollover_scope.json"
DATA_END = 20260928
MAX_K = 10  # 候选顺延视界上界 (交易日)


def to_date(d: int) -> date:
    s = str(int(d))
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def to_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def next_weekdays(start: date, k: int):
    """从 start 的次日起, 取接下来 k 个工作日 (Mon-Fri), 保守上界."""
    out, cur = [], start
    while len(out) < k:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:  # 0=Mon .. 4=Fri
            out.append(cur)
    return out


def main():
    verify = json.loads(VERIFY.read_text(encoding="utf-8"))
    rows = verify["rows"]

    # 每个事件 H5 日期
    events = []
    for r in rows:
        code = str(r["code"]).zfill(6)
        h5 = None
        for s in r["scan"]:
            if s.get("offset") == 5:
                h5 = int(s["date"])
        events.append((code, h5))

    by_k = {}
    for k in range(1, MAX_K + 1):
        pairs = set()
        exceed = 0
        per_seg = defaultdict(lambda: {"increment": 0, "exceed": 0})
        for code, h5 in events:
            if h5 is None:
                continue
            seg = None
            for r in rows:
                if str(r["code"]).zfill(6) == code and int(r["E"]) == int(r["E"]):
                    seg = r.get("segment")
                    break
            for d in next_weekdays(to_date(h5), k):
                di = to_int(d)
                pairs.add((code, di))
                if di > DATA_END:
                    exceed += 1
                    if seg:
                        per_seg[seg]["exceed"] += 1
        by_k[k] = {
            "rollover_trading_days": k,
            "dedup_increment_pairs": len(pairs),
            "exceed_data_end": exceed,
            "by_segment": {s: dict(v) for s, v in per_seg.items()},
        }

    report = {
        "version": "gcp_rollover_scope_v1",
        "network_requests": 0,
        "NOT_APPROVED_REQUEST": True,
        "note": ("待定范围盘点(非核准请求量): 仅按工作日近似枚举 H5 之后 k 个交易日的去重增量与超出数据终点数; "
                 "所列 247/673/1095/2143 等数字【不是已核准请求量】, 不得据此发起采集. "
                 "§6.4 退出成交与限价规则签署后, 须按实际触发的顺延事件 + 交易所日历 + 现有缓存重算增量, "
                 "再冻结请求量并另行批准采集. 非交易日(法定假日)会使实际增量更小, 故本表仅为保守上界."),
        "h5_basis": "E+5 交易日 (基础层已采集窗口终点)",
        "data_end": DATA_END,
        "max_k": MAX_K,
        "by_k": by_k,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[顺延层待定范围盘点] 事件={len(events)} 数据终点={DATA_END} 候选视界 1..{MAX_K} 交易日")
    for k in (1, 3, 5, 10):
        b = by_k[k]
        print(f"  k={k:>2}: 去重增量={b['dedup_increment_pairs']:>4}  超出数据终点={b['exceed_data_end']}")
    print(f"明细: {OUT.name}")


if __name__ == "__main__":
    main()
