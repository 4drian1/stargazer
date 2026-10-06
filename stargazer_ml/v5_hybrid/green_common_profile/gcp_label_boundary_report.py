# -*- coding: utf-8 -*-
"""首轮 247 事件 label_end_ts 跨段报告 (任务单 §4, 2026-10-04).

本地只读. 口径:
  - label_end_ts = int(f"{H5}150000"), H5 = E+5 交易日 (合同最大标签视界).
    注: §6.4 顺延会延伸 label_end, 本报告为"基础层"视界, 不移动切分点.
  - 段映射复用冻结边界 segment_of (与既有研究一致):
      train: E<=20260305 ; val: 20260309..20260609 ; test: E>=20260610 ; 其余 gap.
  - 跨段: segment_of(E) != segment_of(H5_date). 这些事件标签窗口 spill 到后续段,
    供以后闭环隔离 (训练/买入建议继续锁定, 本报告不移动现有切分点).
产物: gcp_round_b_small_sample_v1/gcp_label_boundary_report.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIFY_JSON = HERE / "gcp_round_b_small_sample_v1" / "gcp_247_coverage_verify.json"
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_label_boundary_report.json"

SEG_TRAIN_END = 20260305
SEG_VAL_START = 20260309
SEG_VAL_END = 20260609
SEG_TEST_START = 20260610


def segment_of(e: int) -> str:
    if e <= SEG_TRAIN_END:
        return "train"
    if SEG_VAL_START <= e <= SEG_VAL_END:
        return "val"
    if e >= SEG_TEST_START:
        return "test"
    return "gap"


def main():
    verify = json.loads(VERIFY_JSON.read_text(encoding="utf-8"))
    rows = verify["rows"]

    per = []
    cross_counter = Counter()           # (seg_E -> seg_H5)
    by_seg = defaultdict(lambda: {"n": 0, "cross": 0})
    crossing = []
    for r in rows:
        code = r["code"]
        E = int(r["E"])
        H5 = None
        for s in r["scan"]:
            if s.get("offset") == 5:
                H5 = s.get("date")
        if H5 is None:
            H5 = E
        seg_E = segment_of(E)
        seg_H5 = segment_of(H5)
        label_end_ts = int(f"{int(H5)}150000")
        is_cross = seg_E != seg_H5
        by_seg[seg_E]["n"] += 1
        rec = {"code": code, "E": E, "H5": H5, "seg_E": seg_E,
               "seg_H5": seg_H5, "label_end_ts": label_end_ts, "cross": is_cross}
        per.append(rec)
        if is_cross:
            cross_counter[(seg_E, seg_H5)] += 1
            by_seg[seg_E]["cross"] += 1
            crossing.append(rec)

    report = {
        "version": "gcp_label_boundary_report_v1",
        "network_requests": 0,
        "horizon": "H5 = E+5 交易日 (基础层; §6.4 顺延会延伸, 本报告不计)",
        "label_end_ts_def": "int(f'{H5}150000')",
        "frozen_segment_boundaries": {
            "train": "E<=20260305", "val": "20260309..20260609",
            "test": "E>=20260610", "gap": "其余"},
        "n_events": len(rows),
        "n_crossing": len(crossing),
        "cross_by_transition": {"->".join(k): v for k, v in sorted(cross_counter.items())},
        "by_seg_E": {k: dict(v) for k, v in sorted(by_seg.items())},
        "split_points_unchanged": True,
        "crossing_events": crossing,
        "per_event": per,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[label_end_ts 跨段报告] 事件={len(rows)} 跨段={len(crossing)}")
    print(f"  跨段转移: {report['cross_by_transition']}")
    for k, v in sorted(by_seg.items()):
        print(f"    {k}: n={v['n']} 跨段={v['cross']}")
    print(f"  (切分点未移动: {report['split_points_unchanged']})")
    print(f"  明细: {OUT.name}")


if __name__ == "__main__":
    main()
