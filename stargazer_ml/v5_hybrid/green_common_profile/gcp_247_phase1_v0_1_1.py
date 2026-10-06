# -*- coding: utf-8 -*-
"""Step4 零网络 247 人口重判 (v0.1.1 缺槽修复后).

复用 gcp_247_phase1.main() (不重写引擎核对逻辑), 仅:
  - 输出重定向到 *_v0_1_1 新文件, 不覆盖历史报告;
  - frozen/inventory 输入未变 -> 跳过重写 (运行前已复验, 见 step1_frozen_inputs.json);
  - 运行后生成人读 gcp_247_phase1_status_v0_1_1.md:
    状态分布 (段 x setup(A/B池) x 板块), H3/H5 有效/缺失/跨段隔离,
    原先完整现不完整逐条原因, 未成交/入场未知/残仓/源缺统计 (remaining_shares 单独列示, 不折现金),
    逐笔对账字段差异 (对比修复前 gcp_247_phase1_status.json).
零网络; 不汇总收益排名/左尾/策略有效性.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_247_phase1 as P

BASE = P.BASE_DIR
OLD_STATUS = BASE / "gcp_247_phase1_status.json"


def main():
    t0 = time.time()
    # ---- 输出重定向 / 冻结件跳写 ----
    P.OUT_STATUS = BASE / "gcp_247_phase1_status_v0_1_1.json"
    P.OUT_NEEDS = BASE / "gcp_247_phase1_collection_needs_v0_1_1.json"
    P.OUT_REPORT = BASE / "gcp_247_phase1_report_v0_1_1.md"
    skip_names = {P.OUT_FROZEN.name, P.OUT_INV.name}
    orig_check = P.check_and_write
    def _check(path, text):
        if Path(path).name in skip_names:
            print(f"  skip rewrite (inputs unchanged, verified in step1): {Path(path).name}")
            return
        orig_check(path, text)
    P.check_and_write = _check
    P.build_inventory = lambda: json.loads(
        (BASE / "gcp_247_phase1_inventory.json").read_text(encoding="utf-8"))
    P.main()

    # ---- 运行后处理: 人读 md + 前后对照 ----
    new = json.loads(P.OUT_STATUS.read_text(encoding="utf-8"))
    old = json.loads(OLD_STATUS.read_text(encoding="utf-8"))
    old_by = {(e["code"], e["E"]): e for e in old["events"]}

    def st_tr(ev, tr):
        t = ev.get(tr) or {}
        return (t.get("status"), t.get("fully_exited"))

    changes, residuals, src_missing = [], [], Counter()
    for ev in new["events"]:
        k = (ev["code"], ev["E"])
        oe = old_by.get(k)
        if oe and oe.get("class") != ev.get("class"):
            why = []
            ent = ev.get("entry") or {}
            if (oe.get("entry") or {}).get("status") != ent.get("status"):
                why.append(f"entry: {(oe.get('entry') or {}).get('status')} -> {ent.get('status')} ({ent.get('reason')})")
            for tr in ("H3", "H5"):
                o_t, n_t = oe.get(tr) or {}, ev.get(tr) or {}
                if (o_t.get("status"), o_t.get("fully_exited")) != (n_t.get("status"), n_t.get("fully_exited")):
                    deps = "; ".join(f"{m.get('type')}@{m.get('date')}" for m in (n_t.get("missing_dependencies") or []))
                    why.append(f"{tr}: {o_t.get('status')} -> {n_t.get('status')} [{deps or n_t.get('reason')}]")
            changes.append({"code": ev["code"], "E": ev["E"], "segment": ev["segment"],
                            "setup": ev.get("setup"), "board": ev.get("board"),
                            "old_class": oe.get("class"), "new_class": ev.get("class"),
                            "why": why})
        for tr in ("H3", "H5"):
            t = ev.get(tr) or {}
            if t.get("remaining_shares"):
                residuals.append({"code": ev["code"], "E": ev["E"], "track": tr,
                                  "segment": ev["segment"], "status": t.get("status"),
                                  "remaining_shares": t.get("remaining_shares"),
                                  "reason": t.get("reason"),
                                  "last_fill_ts": t.get("last_fill_ts")})
        # 源缺统计口径
        if ev.get("class") == "NO_FILL_EOD":
            src_missing["confirmed_no_fill"] += 1
        if ev.get("class") == "ENTRY_METADATA_UNKNOWN" or (ev.get("entry") or {}).get("status") == "ENTRY_METADATA_UNKNOWN":
            src_missing["entry_unknown"] += 1
        if ev.get("exec_status") == "ENTRY_DATA_MISSING":
            src_missing["entry_data_missing"] += 1

    # 汇总表
    cls_seg = defaultdict(Counter); cls_setup = defaultdict(Counter); cls_board = defaultdict(Counter)
    trk = {"H3": Counter(), "H5": Counter()}
    trk_seg_valid = defaultdict(lambda: {"H3": 0, "H5": 0})
    cross = {"H3": [], "H5": []}
    for ev in new["events"]:
        cls_seg[ev["segment"]][ev["class"]] += 1
        cls_setup[ev.get("setup") or "UNKNOWN"][ev["class"]] += 1
        cls_board[ev.get("board") or "UNKNOWN"][ev["class"]] += 1
        for tr in ("H3", "H5"):
            t = ev.get(tr) or {}
            st = t.get("status") or "NONE"
            trk[tr][st] += 1
            if t.get("fully_exited"):
                trk_seg_valid[ev["segment"]][tr] += 1
            if (ev.get("cross_segment_label") or {}).get(tr):
                cross[tr].append({"code": ev["code"], "E": ev["E"], "segment": ev["segment"],
                                  "label_end_ts": t.get("label_end_ts")})
    n_events = len(new["events"])

    L = ["# GCP 247 人口重判报告 v0.1.1 (缺槽修复后, 零网络)",
         "",
         f"- 生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | 网络请求: 0 | 重判用时 {time.time()-t0:.1f}s",
         f"- 数据源: 冻结 247 人口 / 冻结 S / 登记缓存 (哈希见 step1_frozen_inputs.json; manifest 5 文件复验全匹配)",
         "- 口径: 缺槽依赖判断按修复后新执行路径 (窗内缺 1 槽即拒; 完成 bar 后缺槽含 14:57-15:00 不否定已闭环; "
         "入场等待窗缺槽 -> ENTRY_DATA_MISSING); 旧 validate_minute_day 历史口径未改动.",
         "- 本报告只做工程状态与逐笔对账字段: 无盈利排名/平均收益/左尾/策略有效性.", "",
         "## 完整 247 人口状态分布", ""]
    L.append(f"- 总事件: {n_events} | class: {dict(Counter(e['class'] for e in new['events']))}")
    L.append("")
    L.append("### 按段 (E 切分: train E<=20260305 / val 20260309-20260609 / test E>=20260610)")
    L += ["", "| 段 | " + " | ".join(sorted({c for v in cls_seg.values() for c in v})) + " |",
          "|" + "---|" * (1 + len({c for v in cls_seg.values() for c in v}))]
    for seg in sorted(cls_seg):
        cols = cls_seg[seg]
        allc = sorted({c for v in cls_seg.values() for c in v})
        L.append(f"| {seg} | " + " | ".join(str(cols.get(c, 0)) for c in allc) + " |")
    L.append("")
    L.append("### 按 A/B 池 (setup)")
    L += ["", "| setup | " + " | ".join(sorted({c for v in cls_setup.values() for c in v})) + " |",
          "|" + "---|" * (1 + len({c for v in cls_setup.values() for c in v}))]
    for s in sorted(cls_setup):
        allc = sorted({c for v in cls_setup.values() for c in v})
        L.append(f"| {s} | " + " | ".join(str(cls_setup[s].get(c, 0)) for c in allc) + " |")
    L.append("")
    L.append("### 按板块")
    L += ["", "| board | " + " | ".join(sorted({c for v in cls_board.values() for c in v})) + " |",
          "|" + "---|" * (1 + len({c for v in cls_board.values() for c in v}))]
    for b in sorted(cls_board):
        allc = sorted({c for v in cls_board.values() for c in v})
        L.append(f"| {b} | " + " | ".join(str(cls_board[b].get(c, 0)) for c in allc) + " |")
    L += ["", "## H3/H5 逐轨道统计", "",
          f"- H3 状态: {dict(trk['H3'])}", f"- H5 状态: {dict(trk['H5'])}",
          f"- 完全退出数: H3={sum(1 for e in new['events'] if (e.get('H3') or {}).get('fully_exited'))} "
          f"H5={sum(1 for e in new['events'] if (e.get('H5') or {}).get('fully_exited'))}",
          f"- 跨段隔离 (label_end_ts 落在下一段, CROSS_SEGMENT_LABEL): H3={len(cross['H3'])} H5={len(cross['H5'])} "
          "(test 无下一段; 明细见 status json isolation_list)", "",
          "## 修复前后对照 (原状态 = gcp_247_phase1_status.json 旧口径)", "",
          f"- 原 class: {old['summary']['class_counts']}",
          f"- 新 class: {dict(Counter(e['class'] for e in new['events']))}",
          f"- 原轨道: H3={old['summary']['track_status']['H3']} H5={old['summary']['track_status']['H5']}",
          f"- 新轨道: H3={dict(trk['H3'])} H5={dict(trk['H5'])}",
          f"- class 变化事件数: {len(changes)} (逐条见 status json diff_from_v0_1_0)",
          f"- 原先标完整、现改不完整: {sum(1 for c in changes if c['old_class'] == 'COMPLETE')} 条", "",
          "### 原先完整 -> 现不完整逐条原因"]
    for c in changes:
        if c["old_class"] == "COMPLETE":
            L.append(f"- {c['code']}@{c['E']} ({c['segment']}/{c['setup']}/{c['board']}): "
                     + " ; ".join(c["why"]))
    L += ["", "### 修复后转完整 (原不完整 -> 现完整) 逐条"]
    for c in changes:
        if c["new_class"] == "COMPLETE" and c["old_class"] != "COMPLETE":
            L.append(f"- {c['code']}@{c['E']} ({c['segment']}/{c['setup']}/{c['board']}): "
                     f"{c['old_class']} -> {c['new_class']} ; " + " ; ".join(c["why"]))
    L += ["", "## 未成交 / 入场未知 / 残仓 / 源缺统计", "",
          f"- 确认未成交 (NO_FILL_EOD, 留现金): {src_missing['confirmed_no_fill']}",
          f"- 入场未知/元数据未知 (ENTRY_METADATA_UNKNOWN): {src_missing['entry_unknown']}",
          f"- 入场数据缺失 (ENTRY_DATA_MISSING, 含等待窗缺槽): {src_missing['entry_data_missing']}",
          f"- 残仓轨道 (remaining_shares>0, 股数单列不折现金): {len(residuals)} 条",
          f"- 源缺 (EXIT_DATA_MISSING 轨道): H3={trk['H3'].get('EXIT_DATA_MISSING', 0)} "
          f"H5={trk['H5'].get('EXIT_DATA_MISSING', 0)}", "",
          "### 残仓逐条 (remaining_shares)"]
    for r in residuals:
        L.append(f"- {r['code']}@{r['E']} {r['track']} ({r['segment']}): remaining_shares={r['remaining_shares']} "
                 f"status={r['status']} reason={r['reason']} last_fill={r['last_fill_ts']}")
    L += ["", "## 证据与依赖缺口声明", "",
          "- 必要依赖缺口 (分钟缺槽/缺日/日级元数据 is_st 或 pre_close 未知) 已逐轨道登记于 "
          "status json track_missing_index / events[].H3|H5.missing_dependencies; 不凭网格完整宣布合同全面通过.",
          "- 量纲/价格尺度/公司行为证据: 本阶段未重算信号, 复权沿用 244 OK + 3 EMPTY 验收口径, 未做新增裁决.",
          "- 行为标签 (缺口后 MAE/MFE/箱体) 不在本报告范围, 另判.", ""]
    out_md = BASE / "gcp_247_phase1_status_v0_1_1.md"
    out_md.write_text("\n".join(L), encoding="utf-8")

    # diff 嵌入 status json (工程字段)
    new["diff_from_v0_1_0"] = {
        "old_status_file": "gcp_247_phase1_status.json (缺槽修复前旧口径)",
        "old_class_counts": old["summary"]["class_counts"],
        "old_track_status": old["summary"]["track_status"],
        "class_changes": changes,
        "n_complete_to_incomplete": sum(1 for c in changes if c["old_class"] == "COMPLETE"),
    }
    P.OUT_STATUS.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"status md written: {out_md.name} ({out_md.stat().st_size} bytes)")
    print(f"class changes: {len(changes)} (complete->incomplete "
          f"{new['diff_from_v0_1_0']['n_complete_to_incomplete']})")


if __name__ == "__main__":
    main()
