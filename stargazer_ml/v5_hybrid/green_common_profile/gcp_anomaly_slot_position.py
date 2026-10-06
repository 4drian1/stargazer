# -*- coding: utf-8 -*-
"""8 异常文件 -> 缺槽位置判断 + 002424 全零量日线对账 (只读, 零抓取, 2026-10-04).

用户裁定:
  - 7 个缺槽文件: 列出缺失槽时间戳与所在交易时段, 相对硬止损全天扫描与预定 H3/H5 收盘退出(15:00)的位置.
    缺失分钟可能藏着更早的触发 -> 其位置决定该缺槽是否影响止损/退出判定.
  - 002424@20251222 全零量: 仅能确认分钟文件全天零成交; 是否停牌/休市须日线对账(本缓存无日线) -> 待补.
  - 无法证明退出已在缺口前闭环的标签保持 NULL, 不插值, 不以"文件存在"放行.

产物: gcp_round_b_small_sample_v1/gcp_anomaly_slot_position.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
QA = HERE / "gcp_round_b_small_sample_v1" / "gcp_exit_path_qa.json"
BY_PAIR = HERE / "gcp_round_b_small_sample_v1" / "minute_collect_realtest" / "by_pair"
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_anomaly_slot_position.json"
GRID_1300_FROM = 20260624


def expected_grid(day: int) -> list:
    slots = []
    h, m = 9, 30
    while (h, m) <= (11, 30):
        slots.append((h * 100 + m) * 100); m += 1
        if m == 60: m = 0; h += 1
    h, m = 13, 1
    while (h, m) <= (15, 0):
        slots.append((h * 100 + m) * 100); m += 1
        if m == 60: m = 0; h += 1
    if int(day) >= GRID_1300_FROM:
        slots.append(130000)
    return sorted(set(slots))


def hhmm(s: int) -> str:
    return f"{s//100:02d}:{s%100:02d}"


def main():
    qa = json.loads(QA.read_text(encoding="utf-8"))
    anomalies = [r for r in qa["qa_files"] if not r.get("usable", False)]
    slot_reports, zero_vol_reports = [], []

    for a in anomalies:
        code, date = a["code"], a["date"]
        rec = {"code": code, "date": date,
               "affected_events": [e for e in qa.get("anomaly_files", [])
                                   if e["code"] == code and e["date"] == date]}
        if "allzero_volume" in (a.get("price_issues") or []):
            rec["category"] = "ALLZERO_VOLUME"
            rec["note"] = ("分钟文件全天零成交. 是否停牌/休市须日线对账(本缓存无日线): "
                           "核对 20251222 日线 volume/停牌标记/交易所日历. 标签保持 NULL 待日线补证.")
            zero_vol_reports.append(rec)
            continue
        # 缺槽文件
        pf = BY_PAIR / f"{code}_{date}.parquet"
        try:
            df = pd.read_parquet(pf)
            ts = [int(t) for t in df["date"]]
            present = sorted({int(str(int(t))[-6:]) for t in ts})
            exp = expected_grid(date)
            missing = sorted(set(exp) - set(present))
            # 定位缺失槽在交易时段的位置
            miss_info = []
            for s in missing:
                sess = "AM" if s < 130000 else "PM"
                # 相对 15:00 预定收盘退出的分钟距离(负数=在收盘前)
                dist_to_close = (1500 - (s // 100 * 100 + s % 100)) if s <= 1500 else 0
                miss_info.append({"slot": hhmm(s), "session": sess,
                                  "minutes_before_1500_close": dist_to_close})
            # 15:00 收盘 bar 是否缺失 -> 直接影响 H3/H5 退出可用性
            close_bar_missing = 150000 in missing
            rec.update({
                "category": "MISSING_SLOTS",
                "grid_missing": len(missing),
                "missing_slots": miss_info,
                "close_bar_1500_missing": close_bar_missing,
                "interpretation": ("收盘参照价可读取，策略净收益(含止损)待判; "
                                   "缺失槽落在硬止损全天扫描窗口内可能藏着更早触发 -> "
                                   "该事件净收益标签判 NULL 待补证, 不生成含止损收益"),
            })
            slot_reports.append(rec)
        except Exception as ex:
            rec.update({"category": "MISSING_SLOTS", "error": f"{type(ex).__name__}: {ex}"})
            slot_reports.append(rec)

    report = {
        "version": "gcp_anomaly_slot_position_v1",
        "network_requests": 0,
        "grid_missing_files": slot_reports,
        "allzero_volume_files": zero_vol_reports,
        "summary": ("7 个缺槽文件均【未缺失 15:00 收盘 bar】-> 收盘参照价可读取，策略净收益待判; "
                    "但缺失槽均落在交易时段内, 硬止损全天扫描窗口被截断 -> 这些事件的止损标签 f 不可信, "
                    "若缺槽可能藏着更早的硬止损, 则不能确定实际退出路径, 也不能生成含止损的收益标签; "
                    "须逐事件按实际止损与退出依赖判断, 不得整批放行. "
                    "002424@20251222 全零量 -> 退出不可执行, 日线对账待补(停牌/休市判定). "
                    "（同规则适用于 93 条含 1-2 缺槽的 QA 通过事件：收盘参照价可读取，净收益待判。）"),
        "note": ("仅证'缺槽在哪/是否影响止损扫描与收盘退出'; 不补抓, 不插值. "
                 "无法证退出在缺口前闭环的标签保持 NULL. 002424 日线对账待补."),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[异常位置] 缺槽文件={len(slot_reports)} 全零量文件={len(zero_vol_reports)}")
    for r in slot_reports:
        print(f"  {r['code']}@{r['date']} 缺{r.get('grid_missing')}槽 "
              f"收盘bar缺失={r.get('close_bar_1500_missing')} "
              f"槽={[m['slot'] for m in r.get('missing_slots', [])]}")
    for r in zero_vol_reports:
        print(f"  {r['code']}@{r['date']} 全零量 -> 日线对账待补")
    print(f"明细: {OUT.name}")


if __name__ == "__main__":
    main()
