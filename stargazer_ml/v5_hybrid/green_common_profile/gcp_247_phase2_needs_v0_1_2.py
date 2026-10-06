# -*- coding: utf-8 -*-
"""Step5 第二阶段准备: 缺失需求四类登记 + 拟议请求清单 (v0.1.1).

输入: gcp_247_phase1_status_p2b1.json (重判后) + step1_frozen_inputs.json 缓存登记 + 现存 inventory.
分类:
  ① 首次阻断点: 逐轨道第一个缺失依赖 (dedup (code,date))
  ② 后续未检查路径: 阻断点至该轨道窗口上界 (d0+5 或数据截止) 之间未核查的交易日计数
     (未证明完整 != 零需求; 补齐①后须继续核查)
  ③ 已有无效文件: 阻断日已有缓存文件但缺槽/无效 -> 不自动重抓, 单列
  ④ 已知源洞/元数据未知/超数据截止: 保持隔离, 不混入普通采集请求
拟议请求清单 = ①中无已有文件覆盖的 (code,date) (minute_day_missing 类), 登记相关现存文件 SHA256.
预算口径: 明确区分"本批 (首次阻断可请求)"与"完整路径"两个口径, 不把首次阻断清单包装成全部补齐需求.
零网络. 产物: gcp_round_b_small_sample_v1/gcp_247_phase2_needs_v0_1_2.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D

BASE = HERE / "gcp_round_b_small_sample_v1"
STATUS = json.loads((BASE / "gcp_247_phase1_status_p2b1.json").read_text(encoding="utf-8"))
STEP1 = json.loads((BASE / "step1_frozen_inputs.json").read_text(encoding="utf-8"))
INV = json.loads((BASE / "gcp_247_phase1_inventory.json").read_text(encoding="utf-8"))
OUT = BASE / "gcp_247_phase2_needs_v0_1_2.json"

cal = D.load_calendar()
ci = {d: i for i, d in enumerate(cal)}
DATA_CUTOFF = D.DATA_CUTOFF
KNOWN_SOURCE_HOLES = {("688656", 20250724)}

# ---- 缓存覆盖索引: (code,day) -> 登记文件 ----
cover = defaultdict(list)   # 有文件覆盖 (可能无效)
code_files = defaultdict(list)
for r in STEP1["cache_registry"]["files"]:
    code = Path(r["file"]).stem.split("_")[0]
    code_files[code].append(r)
for r in INV["files"]:
    stem = Path(r["file"]).stem
    code = stem.split("_")[0]
    if r.get("naming") == "by_pair":
        day = int(stem.split("_")[-1])
        cover[(code, day)].append(r)
    elif r.get("error"):
        continue
    else:
        days = r.get("days")
        if isinstance(days, list) and days and all(isinstance(x, int) for x in days):
            for d in days:
                cover[(code, d)].append(r)
        # 整码切片且天数过多未枚举: 以 code_files 兜底 (请求判定只依赖"是否存在覆盖文件")

def has_cover(code, day):
    if (code, day) in cover:
        return True
    return bool(code_files.get(code))  # 保守: 整码切片可能含该日, 引擎报 minute_day_missing 时才判无覆盖

# ---- ① 首次阻断点 + ② 未检查路径 ----
first_block, unchecked = [], []
trk_records = STATUS["track_missing_index"]
by_pair = defaultdict(lambda: {"kinds": set(), "needed_by": [], "has_file_cover": None})
for rec in trk_records:
    code, day, typ = rec["code"], int(rec["date"]), rec["type"]
    k = (code, day)
    e = by_pair[k]
    e["kinds"].add(typ)
    nb = {"E": rec["E"], "track": rec.get("track")}
    if nb not in e["needed_by"]:
        e["needed_by"].append(nb)

# 逐轨道窗口上界: 用事件轨道字段估 d0+5 或数据截止
ev_map = {(e["code"], e["E"]): e for e in STATUS["events"]}
unchecked_days_total = 0
pair_rows = []
for (code, day), e in sorted(by_pair.items()):
    if day > DATA_CUTOFF:
        cls4 = "BEYOND_DATA_CUTOFF"
    elif (code, day) in KNOWN_SOURCE_HOLES:
        cls4 = "KNOWN_SOURCE_HOLE"
    else:
        cls4 = "COLLECTABLE"
    cover_recs = cover.get((code, day), [])
    # 引擎报 minute_day_missing 即无任何文件含该日; day_gap 则已有文件但带内洞
    no_file = ("minute_day_missing" in e["kinds"]) and not cover_recs
    pair_rows.append({"code": code, "day": day, "class": cls4, "kinds": sorted(e["kinds"]),
                      "needed_by": e["needed_by"],
                      "existing_file_cover": [{"file": r["file"], "sha256": r["sha256"], "trusted": r.get("trusted")}
                                              for r in cover_recs],
                      "no_existing_file": no_file})

# ② 未检查路径: 对每条受阻轨道, 阻断日之后至窗口上界的交易日
for rec in trk_records:
    code, E_, day = rec["code"], rec["E"], int(rec["date"])
    ev = ev_map.get((code, E_))
    tr = (ev.get(rec["track"]) or {}) if rec.get("track") and ev else {}
    d0 = tr.get("deferral_d0")
    plan = tr.get("planned_exit_date")
    if day not in ci:
        continue
    if d0 and d0 in ci:
        end_i = min(ci[d0] + 5, len(cal) - 1)
    elif plan and plan in ci:
        end_i = min(ci[plan] + 5, len(cal) - 1)  # 保守: 若无 d0, 计划日+顺延上限
    else:
        end_i = min(ci[day] + 5, len(cal) - 1)
    # 窗口上界同时受数据截止约束
    while end_i > ci[day] and cal[end_i] > DATA_CUTOFF:
        end_i -= 1
    n_unchecked = max(0, end_i - ci[day])
    unchecked_days_total += n_unchecked
    if n_unchecked:
        unchecked.append({"code": code, "E": E_, "track": rec.get("track"), "blocked_at": day,
                          "window_end": cal[end_i], "unchecked_trading_days": n_unchecked,
                          "note": "补齐首阻后须逐日核查; 其中可能含完整日/缺槽/涨跌停锁死, 未知"})

# ---- ③ 已有无效文件 (247 人口相关 code 的异常文件) ----
codes247 = {e["code"] for e in STATUS["events"]}
invalid_files = [r for r in INV["files"]
                 if not r.get("trusted") and Path(r["file"]).stem.split("_")[0] in codes247]
inv_by_kind = Counter()
for r in invalid_files:
    for it in r.get("issues") or []:
        inv_by_kind[it.get("kind") or "other"] += 1

# ---- ④ 已知源洞 / 元数据未知 / 超截止 ----
daily_meta = STATUS.get("needs_daily_meta_gaps")
# status json 不含 daily_meta_gaps (在 needs 文件); 读 v0_1_1 needs
needs_v = json.loads((BASE / "gcp_247_phase1_collection_needs_p2b1.json").read_text(encoding="utf-8"))
cat4 = {
    "known_source_holes": [{"code": c, "day": d, "note": "GAP-2026-10-04-01 源洞; 保持隔离, 不混入采集请求"}
                           for (c, d) in sorted(KNOWN_SOURCE_HOLES)],
    "beyond_data_cutoff": [p for p in pair_rows if p["class"] == "BEYOND_DATA_CUTOFF"],
    "daily_metadata_unknown": needs_v.get("daily_meta_gaps", []),
    "isolation_controls": [i for i in STATUS["isolation_list"] if i.get("track") is None],
}

# ---- 拟议请求清单: 无已有文件覆盖的 COLLECTABLE 首阻对 ----
proposed = [p for p in pair_rows if p["class"] == "COLLECTABLE" and p["no_existing_file"]]
covered_but_invalid = [p for p in pair_rows if p["class"] == "COLLECTABLE" and not p["no_existing_file"]]
avg_bytes = needs_v["budget"]["avg_by_pair_bytes"]
est_batch = len(proposed) * avg_bytes

out = {
    "phase": "GCP_247_PHASE2_NEEDS_V0_1_2",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "network_requests": 0,
    "no_auto_collection": True,
    "category_1_first_blockers": {
        "description": "逐轨道当前执行的第一个缺失依赖; 引擎在首阻即停, 故每轨至多一条",
        "dedup_pairs": pair_rows,
        "counts": {"pairs_total": len(pair_rows),
                   "by_class": dict(Counter(p["class"] for p in pair_rows)),
                   "track_records": len(trk_records),
                   "by_type": dict(Counter(r["type"] for r in trk_records))}},
    "category_2_unchecked_paths": {
        "description": "首阻之后至该轨道窗口上界 (d0+5 交易日 或 数据截止) 未核查的交易日; "
                       "未证明完整 != 零需求; 补齐①后这些天数须继续逐日核查",
        "records": unchecked,
        "unchecked_trading_days_total": unchecked_days_total},
    "category_3_existing_invalid_files": {
        "description": "阻断日已有缓存文件但存在内洞/无效 (同源分钟内洞); 按规则不自动重抓, "
                       "是否补采须第二阶段另行裁决",
        "n_files_in_247_scope": len(invalid_files),
        "issue_kinds": dict(inv_by_kind),
        "files_head": [{"file": r["file"], "sha256": r["sha256"],
                        "issues": [{"day": i.get("day"), "kind": i.get("kind"),
                                    "missing_head": (i.get("missing") or [])[:6]}
                                   for i in (r.get("issues") or [])[:4]]}
                       for r in invalid_files[:200]]},
    "category_4_isolated": {
        "description": "已知源洞 / 日级元数据未知 / 超数据截止: 保持隔离, 不混入普通采集请求",
        **cat4},
    "proposed_requests": {
        "description": "拟议第二阶段采集请求: COLLECTABLE 首阻对 且 无任何已有缓存文件覆盖 "
                       "(minute_day_missing); 已按 (code,date) 去重; 相关现存文件 SHA256 已在 "
                       "category_1.existing_file_cover / category_3 登记",
        "list": proposed,
        "n_requests": len(proposed),
        "excluded_covered_but_invalid": [{"code": p["code"], "day": p["day"],
                                          "reason": "已有文件但无效 (category_3), 不自动重抓"}
                                         for p in covered_but_invalid]},
    "budget_statement": {
        "basis_avg_by_pair_bytes": avg_bytes,
        "first_batch_estimate_bytes": est_batch,
        "first_batch_scope": "仅本批可请求清单 (无覆盖的 COLLECTABLE 首阻对); "
                             "不含 category_2 未核查天数, 不含 category_3 无效文件替换, 不含 category_4 隔离项",
        "full_path_budget": "无法在本阶段定界: 首阻补齐后 category_2 的未核查天数才暴露, "
                            "其缺槽/无效形态未知; 本批清单不得包装成全部补齐需求",
        "dir_bytes_now": needs_v["budget"]["dir_bytes_now"],
        "dir_budget": needs_v["budget"]["dir_budget"],
        "dir_headroom_bytes": needs_v["budget"]["dir_budget"] - needs_v["budget"]["dir_bytes_now"],
        "first_batch_fits": (needs_v["budget"]["dir_bytes_now"] + est_batch) <= needs_v["budget"]["dir_budget"]},
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"step5 written: {OUT.name} ({OUT.stat().st_size} bytes)")
print("pairs:", len(pair_rows), dict(Counter(p['class'] for p in pair_rows)))
print("proposed requests:", len(proposed), "covered_but_invalid:", len(covered_but_invalid))
print("unchecked days total:", unchecked_days_total, "| invalid files in scope:", len(invalid_files))
print("est batch bytes:", est_batch)
