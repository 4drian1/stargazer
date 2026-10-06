# -*- coding: utf-8 -*-
"""任务 A: 1,046 股日 (139 轨道后续窗口累计) 只读覆盖盘点 — 零网络, 零采集.

口径: 1,046 = 139 条轨道的后续窗口累计, 不是 1,046 个独立请求; 盘点先按 (code,date) 去重,
报告 "窗口数 vs 去重后对数" 两个数字.
每个去重 (code,date) 分四类:
  ①已有完整数据 (含本批 p2b1 新采)  ②已有缺槽或元数据问题 (指到已有文件/内洞登记)
  ③真正无文件  ④超出截止范围 (>20260928)
覆盖盘点不修首阻; 未知路径不得据此推算真实退出或收益 (只登记覆盖状态).
产物: gcp_round_b_small_sample_v1/gcp_247_coverage_scope_v0_1_2.json + 同名 md
交叉: 首阻 69 对 / 无效 350 文件 / 已知源洞 / p2b1 新采 39 文件.
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D
import gcp_round_b_engine as E

BASE = HERE / "gcp_round_b_small_sample_v1"
NEEDS = BASE / "gcp_247_phase2_needs_v0_1_2.json"
OUT_JSON = BASE / "gcp_247_coverage_scope_v0_1_2.json"
OUT_MD = BASE / "gcp_247_coverage_scope_v0_1_2.md"
DATA_CUTOFF = D.DATA_CUTOFF

BY_PAIR_DIRS = [BASE / "minute_collect_realtest" / "by_pair",
                BASE / "deferral_collect_v2" / "by_pair",
                BASE / "p2b1_collect" / "by_pair"]
CODE_SLICE_DIRS = [BASE / "minute_collect_realtest", BASE / "minute_ext",
                   BASE / "minute", BASE / "minute_car"]
P2B1_MANIFEST = json.loads((BASE / "p2b1_collect" / "collection_manifest.json").read_text(encoding="utf-8"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def write_checked(path: Path, text: str):
    if dir_size(BASE) + len(text.encode("utf-8")) > 50 * 1024 * 1024:
        raise RuntimeError("budget exceeded before writing (stop)")
    path.write_text(text, encoding="utf-8")


def main():
    t0 = time.time()
    needs = json.loads(NEEDS.read_text(encoding="utf-8"))
    c2 = needs["category_2_unchecked_paths"]["records"]
    cal = D.load_calendar()
    ci = {d: i for i, d in enumerate(cal)}
    n_windows = sum(r["unchecked_trading_days"] for r in c2)

    # ---- 展开窗口 -> 去重 (code,day) ----
    pair_tracks = defaultdict(list)   # (code,day) -> [records]
    for r in c2:
        code = str(r["code"]).zfill(6)
        d0, dend = int(r["blocked_at"]), int(r["window_end"])
        if d0 not in ci:
            continue
        i = ci[d0] + 1
        while i < len(cal) and cal[i] <= dend:
            pair_tracks[(code, cal[i])].append({"E": r["E"], "track": r["track"],
                                                "segment": None, "blocked_at": d0})
            i += 1
    pairs = sorted(pair_tracks)
    n_pairs = len(pairs)

    # 段标注 (按 status p2b1 事件段)
    status = json.loads((BASE / "gcp_247_phase1_status_p2b1.json").read_text(encoding="utf-8"))
    seg_of = {(e["code"], e["E"]): e["segment"] for e in status["events"]}
    for k, lst in pair_tracks.items():
        for r in lst:
            r["segment"] = seg_of.get((k[0], r["E"]), "UNKNOWN")

    # ---- 覆盖索引 ----
    by_pair_index = {}
    for d in BY_PAIR_DIRS:
        for f in d.glob("*.parquet"):
            stem = f.stem
            if "_" in stem and stem.split("_")[-1].isdigit():
                by_pair_index.setdefault((stem.split("_")[0], int(stem.split("_")[-1])), f)
    slice_cache = {}
    codes_needed = sorted({k[0] for k in pairs})
    for d in CODE_SLICE_DIRS:
        for code in codes_needed:
            f = d / f"{code}.parquet"
            if f.exists() and (d, code) not in slice_cache:
                try:
                    df = pd.read_parquet(f, columns=["date"])
                    days = set(int(x) // 1000000 for x in df["date"].astype(int))
                    slice_cache[(d, code)] = (f, days)
                except Exception:
                    slice_cache[(d, code)] = (f, set())

    # 日线 (元数据判定)
    tbl = ds.dataset(sorted(__import__("glob").glob(D.DAILY_PARTS + "/part_*.parquet")), format="parquet").to_table(
        columns=["date", "code", "pre_close", "is_st"],
        filter=ds.field("code").isin(codes_needed)).to_pandas()
    tbl["date"] = tbl["date"].astype(int)
    daily_idx = {(r["code"], int(r["date"])): r for _, r in tbl.iterrows()}

    # 交叉清单
    cat1_pairs = {(str(q["code"]).zfill(6), int(q["day"]))
                  for q in needs["category_1_first_blockers"]["dedup_pairs"]}
    invalid_files = {(r["file"], (i.get("day") or 0)) for r in needs["category_3_existing_invalid_files"]["files_head"]
                     for i in r.get("issues", [])}
    invalid_files |= {(r["file"], None) for r in needs["category_3_existing_invalid_files"]["files_head"]}
    known_holes = {("688656", 20250724)}
    p2b1_files = {f.stem: sha256_file(f) for f in (BASE / "p2b1_collect" / "by_pair").glob("*.parquet")}

    # ---- 逐对分类 ----
    rows = []
    cls_counts = Counter()
    seg_dist = defaultdict(Counter)
    for (code, day) in pairs:
        rec = {"code": code, "day": day,
               "needed_by": pair_tracks[(code, day)],
               "segments": sorted({r["segment"] for r in pair_tracks[(code, day)]})}
        cross = []
        if (code, day) in cat1_pairs:
            cross.append("first_blocker_69")
        if (code, day) in known_holes:
            cross.append("known_source_hole")
        fkey = f"{code}_{day}"
        if fkey in p2b1_files:
            cross.append("p2b1_newly_collected")
        rec["cross_refs"] = cross
        if day > DATA_CUTOFF:
            rec["class"] = "BEYOND_CUTOFF"
        else:
            src = by_pair_index.get((code, day))
            if src is None:
                for (d, c), (f, days) in slice_cache.items():
                    if c == code and day in days:
                        src = f
                        break
            if src is None:
                rec["class"] = "NO_FILE"
            else:
                rec["cover_file"] = str(src.relative_to(BASE))
                rec["cover_sha256"] = sha256_file(src)
                if any(rec["cover_file"] == f and (day == d or d is None) for (f, d) in invalid_files):
                    rec["cross_refs"].append("invalid_350_listed")
                try:
                    df = pd.read_parquet(src)
                    df = df[df["date"].astype(int) // 1000000 == day]
                    v = E.validate_minute_day(df, D.day_grid(day), day)
                    price_bad = bool((df[["open", "high", "low", "close"]].astype(float) <= 0).any().any()) \
                        if len(df) else True
                    zero_all = bool((df["volume"].astype(float) > 0).sum() == 0) if len(df) else True
                    dr = daily_idx.get((code, day))
                    meta_bad = (dr is None) or (pd.isna(dr.get("pre_close"))) or (pd.isna(dr.get("is_st")))
                    if v["missing"] or v["dups"] or v["unexpected"] or price_bad or zero_all or meta_bad:
                        rec["class"] = "HAS_GAP_OR_METADATA"
                        rec["detail"] = {"missing": v["missing"][:10], "dups": v["dups"][:6],
                                         "unexpected": v["unexpected"][:6],
                                         "price_bad": price_bad, "all_zero_volume": zero_all,
                                         "daily_metadata_missing": bool(meta_bad)}
                    else:
                        rec["class"] = "COMPLETE_DATA"
                except Exception as ex:
                    rec["class"] = "HAS_GAP_OR_METADATA"
                    rec["detail"] = {"unreadable": repr(ex)[:120]}
        cls_counts[rec["class"]] += 1
        for s in rec["segments"]:
            seg_dist[s][rec["class"]] += 1
        rows.append(rec)

    out = {"phase": "GCP_247_COVERAGE_SCOPE_V0_1_2",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0, "no_collection": True,
           "scope_note": "1,046 = 139 轨道后续窗口累计 (窗口数), 非 1,046 个独立请求; "
                         "按 (code,date) 去重后为 %d 对. 只登记覆盖状态, 不修首阻, 不推算退出/收益." % n_pairs,
           "window_days_total": n_windows,
           "n_tracks": len(c2),
           "dedup_pairs_total": n_pairs,
           "class_counts": dict(cls_counts),
           "class_by_segment": {k: dict(v) for k, v in seg_dist.items()},
           "cross_check": {
               "first_blocker_pairs_69": len(cat1_pairs),
               "overlap_with_unchecked": sum(1 for k in pairs if k in cat1_pairs),
               "invalid_350_files": needs["category_3_existing_invalid_files"]["n_files_in_247_scope"],
               "pairs_covered_by_invalid_listed": sum(1 for r in rows
                                                      if "invalid_350_listed" in r["cross_refs"]),
               "invalid_cross_note": "category_3 files_head 仅含 350 个无效文件中的前 200 个, 该交叉为下界; "
                                     "逐对分类不依赖此清单 (直接按网格/量价/元数据实测)",
               "known_source_holes": sorted(list(known_holes)),
               "p2b1_new_files": len(p2b1_files),
               "pairs_covered_by_p2b1": sum(1 for r in rows if "p2b1_newly_collected" in r["cross_refs"]),
           },
           "data_cutoff": DATA_CUTOFF,
           "cache_dirs_scanned": [str(d.relative_to(BASE)) for d in BY_PAIR_DIRS + CODE_SLICE_DIRS],
           "pairs": rows}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 1,046 股日只读覆盖盘点 (v0.1.2, 零网络零采集)", "",
         f"- 生成: {out['generated_at']} | 窗口数 **{n_windows}** (139 轨道后续窗口累计) | "
         f"去重后对数 **{n_pairs}** | 用时 {time.time()-t0:.1f}s",
         "- 口径: 1,046 是窗口累计不是独立请求; 每对只登记覆盖状态, 不修首阻, 不推算退出或收益.", "",
         "## 四分类计数 (去重后对)", ""]
    for k in ("COMPLETE_DATA", "HAS_GAP_OR_METADATA", "NO_FILE", "BEYOND_CUTOFF"):
        L.append(f"- {k}: {cls_counts.get(k, 0)}")
    L += ["", "## 按段分布 (对-段归属; 一对服务多段时各计一次)", "",
          "| 段 | COMPLETE | GAP/META | NO_FILE | BEYOND |", "|---|---|---|---|---|"]
    for s in sorted(seg_dist):
        v = seg_dist[s]
        L.append(f"| {s} | {v.get('COMPLETE_DATA',0)} | {v.get('HAS_GAP_OR_METADATA',0)} | "
                 f"{v.get('NO_FILE',0)} | {v.get('BEYOND_CUTOFF',0)} |")
    cc = out["cross_check"]
    L += ["", "## 与既有清单交叉核对", "",
          f"- 首阻 69 对中与未核查窗口重叠: {cc['overlap_with_unchecked']} (首阻本身不在本盘点范围, 重叠=对其它轨道是首阻)",
          f"- 被 350 无效文件清单覆盖的对: {cc['pairs_covered_by_invalid_listed']}",
          f"- 被本批 p2b1 新采覆盖的对: {cc['pairs_covered_by_p2b1']}",
          f"- 已知源洞 688656@20250724 落入窗口: {('688656', 20250724) in set(pairs)}",
          "", "## 说明", "",
          "- 类别②含内洞/重复/错位/量价非法/日线元数据缺失, 均指向具体文件 (cover_file+cover_sha256).",
          "- 类别③才是真正可考虑采集的对象, 但本任务零采集; 是否申请下一批另议.",
          "- 类别④超出 data_cutoff 20260928 的不在任何补齐范围.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"windows={n_windows} dedup_pairs={n_pairs} class={dict(cls_counts)} ({time.time()-t0:.1f}s)")
    print(f"cross: {out['cross_check']}")


if __name__ == "__main__":
    main()
