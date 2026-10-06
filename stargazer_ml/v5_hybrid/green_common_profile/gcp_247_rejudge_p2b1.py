# -*- coding: utf-8 -*-
"""Step: p2b1 采集后 247 人口工程状态重判 (零网络; 相同冻结事件/S/行动分支/时间切分, 只变缓存).

- 读取路径追加 p2b1_collect/by_pair (位于 deferral 之后, realtest 之前; 本批日均不存在于其他目录).
- 复用 gcp_247_phase1.main(); 输出重定向 *_p2b1 新文件, 不覆盖 v0_1_1 / 旧历史.
- 对照基线 = gcp_247_phase1_status_v0_1_1.json (采集前重判);
  报告: 新增闭环 (INCOMPLETE->COMPLETE) 逐条及原因, 剩余阻断, 跨段隔离变化,
  H3/H5 有效数, 分段完整率前后对比 (不保证改善; 不做收益汇总).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_247_phase1 as P
import dual_path_execution_v0_1_1 as D

BASE = P.BASE_DIR
P2B1_CACHE = BASE / "p2b1_collect" / "by_pair"
BEFORE = BASE / "gcp_247_phase1_status_v0_1_1.json"


def seg_rates(evs):
    per = defaultdict(lambda: [0, 0])
    for e in evs:
        per[e["segment"]][1] += 1
        if e["class"] == "COMPLETE":
            per[e["segment"]][0] += 1
    return {k: {"complete": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(per.items())}


def main():
    t0 = time.time()
    assert P2B1_CACHE.exists(), "p2b1 缓存目录缺失"
    D.MIN_DIRS.insert(1, P2B1_CACHE)  # deferral(0) 之后, realtest 之前
    P.OUT_STATUS = BASE / "gcp_247_phase1_status_p2b1.json"
    P.OUT_NEEDS = BASE / "gcp_247_phase1_collection_needs_p2b1.json"
    P.OUT_REPORT = BASE / "gcp_247_phase1_report_p2b1.md"
    skip_names = {P.OUT_FROZEN.name, P.OUT_INV.name}
    orig_check = P.check_and_write
    def _check(path, text):
        if Path(path).name in skip_names:
            print(f"  skip rewrite (inputs unchanged): {Path(path).name}")
            return
        orig_check(path, text)
    P.check_and_write = _check
    P.build_inventory = lambda: json.loads(
        (BASE / "gcp_247_phase1_inventory.json").read_text(encoding="utf-8"))
    P.main()

    new = json.loads(P.OUT_STATUS.read_text(encoding="utf-8"))
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    bef_by = {(e["code"], e["E"]): e for e in before["events"]}

    closures, new_incomplete, other = [], [], []
    for ev in new["events"]:
        b = bef_by.get((ev["code"], ev["E"]))
        if not b:
            continue
        if b["class"] != "COMPLETE" and ev["class"] == "COMPLETE":
            why = []
            for tr in ("H3", "H5"):
                bt, nt = b.get(tr) or {}, ev.get(tr) or {}
                if (bt.get("status"), bt.get("fully_exited")) != (nt.get("status"), nt.get("fully_exited")):
                    why.append(f"{tr}: {bt.get('status')} -> {nt.get('status')} "
                               f"(label_end={nt.get('label_end_ts')})")
            closures.append({"code": ev["code"], "E": ev["E"], "segment": ev["segment"],
                             "setup": ev.get("setup"), "board": ev.get("board"),
                             "old_class": b["class"], "why": why})
        elif b["class"] == "COMPLETE" and ev["class"] != "COMPLETE":
            new_incomplete.append({"code": ev["code"], "E": ev["E"], "new_class": ev["class"]})
        elif b["class"] != ev["class"]:
            other.append({"code": ev["code"], "E": ev["E"], "old": b["class"], "new": ev["class"]})

    trk = {"H3": Counter(), "H5": Counter()}
    cross = {"H3": 0, "H5": 0}
    for ev in new["events"]:
        for tr in ("H3", "H5"):
            t = ev.get(tr) or {}
            trk[tr][t.get("status") or "NONE"] += 1
            if (ev.get("cross_segment_label") or {}).get(tr):
                cross[tr] += 1
    cross_before = {"H3": 0, "H5": 0}
    for ev in before["events"]:
        for tr in ("H3", "H5"):
            if (ev.get("cross_segment_label") or {}).get(tr):
                cross_before[tr] += 1

    rates_b, rates_a = seg_rates(before["events"]), seg_rates(new["events"])
    cls_counts = Counter(e["class"] for e in new["events"])
    # 剩余阻断: 新 status 的 track_missing 首阻对
    needs_p2b1 = json.loads(P.OUT_NEEDS.read_text(encoding="utf-8"))

    L = ["# GCP 247 人口重判报告 p2b1 (39 对采集后, 零网络, 只变缓存)", "",
         f"- 生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | 重判用时 {time.time()-t0:.1f}s",
         f"- 新缓存: p2b1_collect/by_pair (39 文件, 429,017B; 2 文件带登记源洞内洞: "
         f"600888@20260211 缺141300, 688038@20260313 缺135300; QA/哈希见 p2b1_collect/collection_report.json)",
         "- 冻结不变: 事件/S/行动分支/时间切分/阈值; 不补算信号; 无盈利汇总/左尾/训练.", "",
         "## 总体对照 (基线 = v0_1_1 采集前重判)", "",
         f"- 采集前 class: {before['summary']['class_counts']}",
         f"- 采集后 class: {dict(cls_counts)}",
         f"- H3: {dict(trk['H3'])}", f"- H5: {dict(trk['H5'])}", "",
         "## 新增闭环 (INCOMPLETE->COMPLETE) 逐条", ""]
    if not closures:
        L.append("- 无")
    for c in closures:
        L.append(f"- {c['code']}@{c['E']} ({c['segment']}/{c['setup']}/{c['board']}): " + " ; ".join(c["why"]))
    L += ["", "## 反向变化 (COMPLETE->其他) 逐条", ""]
    if not new_incomplete:
        L.append("- 无")
    for c in new_incomplete:
        L.append(f"- {c['code']}@{c['E']} -> {c['new_class']}")
    for c in other:
        L.append(f"- {c['code']}@{c['E']}: {c['old']} -> {c['new']}")
    L += ["", "## 跨段 CROSS_SEGMENT_LABEL 变化", "",
          f"- H3: {cross_before['H3']} -> {cross['H3']} | H5: {cross_before['H5']} -> {cross['H5']}", "",
          "## 分段完整率对比 (COMPLETE 占比; 如实报告, 不保证改善)", "",
          "| 段 | 采集前 | 采集后 | 变化 |", "|---|---|---|---|"]
    for seg in sorted(set(rates_b) | set(rates_a)):
        b_, a_ = rates_b.get(seg), rates_a.get(seg)
        L.append(f"| {seg} | {b_['complete']}/{b_['n']} = {b_['rate']*100:.1f}% | "
                 f"{a_['complete']}/{a_['n']} = {a_['rate']*100:.1f}% | {a_['complete']-b_['complete']:+d} |")
    L += ["", "## 剩余阻断与后续需求", "",
          f"- 采集后剩余首阻去重对: {needs_p2b1['counts']['dedup_pairs_total']} "
          f"分类: {needs_p2b1['counts']['by_class']}",
          "- 后续未核查窗口日与四类需求更新见 gcp_247_phase2_needs_v0_1_2.json (只登记, 不追抓)",
          "- 注: 完整样本只代表可复放子集, 不得当作无偏表现; 本轮不做收益汇总.", ""]
    out_md = BASE / "gcp_247_phase1_status_p2b1.md"
    out_md.write_text("\n".join(L), encoding="utf-8")

    new["diff_from_v0_1_1_p2b1"] = {
        "baseline": "gcp_247_phase1_status_v0_1_1.json (采集前)",
        "new_closures": closures, "new_incomplete": new_incomplete, "other_changes": other,
        "segment_complete_rate": {"before": rates_b, "after": rates_a},
        "cross_segment": {"before": cross_before, "after": cross},
        "track_status_after": {k: dict(v) for k, v in trk.items()},
    }
    P.OUT_STATUS.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"closures={len(closures)} new_incomplete={len(new_incomplete)} other={len(other)}")
    print(f"rates before={rates_b} after={rates_a}")
    print(f"cross H3 {cross_before['H3']}->{cross['H3']} H5 {cross_before['H5']}->{cross['H5']}")


if __name__ == "__main__":
    main()
