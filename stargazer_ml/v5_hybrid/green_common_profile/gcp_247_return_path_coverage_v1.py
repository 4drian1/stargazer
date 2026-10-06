# -*- coding: utf-8 -*-
"""收益依赖覆盖补齐 v1 — 行动轨道必查窗完整核对 (零网络, 零收益计算, 零采集).

口径更正 (用户): v3 收益路径检查漏 E+4、未覆盖实际顺延日. 本件对 1,013 研究人口第 1 类行动轨道
(187 条 × H3/H5 两轨) 重新核对分钟覆盖:
  必查窗 = 入场窗(E日) + T+1 起至实际(247线, 取 p2b1 复放最后成交/首阻日)或计划(非247线, H3=E+3 /
  H5=E+5)最后退出完成 bar 的止损扫描窗; 247 线含顺延日(复放已扫); 非 247 线顺延未复放 ->
  标注"顺延未检查", 不猜.
字段口径: "全路径分钟" -> "已检查日期覆盖" (checked_dates 列表 + 未检查原因); 不是收益就绪证明.
E+4 单独计数; 与 328 集合交集实测并登记 (328 挂起不采).
产出: gcp_247_return_path_coverage_v1.json/.md (RUN 新文件; v3 不动, 本件开头注明与 v3 检查窗差异).
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D
import gcp_round_b_engine as E

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_return_path_coverage_v1.json"
OUT_MD = BASE / "gcp_247_return_path_coverage_v1.md"
DATA_CUTOFF = D.DATA_CUTOFF


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
        raise RuntimeError("budget exceeded (stop)")
    path.write_text(text, encoding="utf-8")


def main():
    t0 = time.time()
    v3 = json.loads(open(BASE / "gcp_247_full_population_inventory_v3.json", encoding="utf-8").read())
    status = json.loads(open(BASE / "gcp_247_phase1_status_p2b1.json", encoding="utf-8").read())
    scope = json.loads(open(BASE / "gcp_247_coverage_scope_v0_1_2.json", encoding="utf-8").read())
    set328 = {(r["code"], int(r["day"])) for r in scope["pairs"] if r.get("class") == "NO_FILE"}

    cal = D.load_calendar()
    ci = {d: i for i, d in enumerate(cal)}
    actions = [r for r in v3["records"] if r["mutual"] == "1_有行动"]
    end247 = {}
    for e in status["events"]:
        for tr in ("H3", "H5"):
            t = e.get(tr) or {}
            blk = None
            for m in (t.get("missing_dependencies") or []):
                if m.get("date"):
                    blk = int(m["date"])
                    break
            end247[(e["code"], int(e["E"]), tr)] = {
                "complete": bool(t.get("fully_exited")),
                "last_fill_day": (int(t["last_fill_ts"]) // 1000000) if t.get("last_fill_ts") else None,
                "blocker_day": blk, "status": t.get("status")}

    # 分钟索引 (同 v3 修正版: 多 slice 全保留)
    by_pair = {}
    for d in [BASE / "minute_collect_realtest" / "by_pair",
              BASE / "deferral_collect_v2" / "by_pair",
              BASE / "p2b1_collect" / "by_pair"]:
        for f in d.glob("*.parquet"):
            stem = f.stem
            if "_" in stem and stem.split("_")[-1].isdigit():
                by_pair.setdefault((stem.split("_")[0], int(stem.split("_")[-1])), str(f))
    slice_days = defaultdict(list)
    codes = sorted({a["code"] for a in actions})
    for d in [BASE / "minute_collect_realtest", BASE / "minute_ext", BASE / "minute", BASE / "minute_car",
              HERE.parent / "pullback_train" / "minute_cache_clean"]:
        for code in codes:
            f = d / f"{code}.parquet"
            if f.exists():
                try:
                    df = pd.read_parquet(f, columns=["date"])
                    slice_days[code].append((str(f), set(int(x) // 1000000 for x in df["date"].astype(int))))
                except Exception:
                    pass

    def day_class(code, day):
        if day is None:
            return "BEYOND_CALENDAR"
        if day > DATA_CUTOFF:
            return "BEYOND_CUTOFF"
        f = by_pair.get((code, day))
        if f is None:
            for sf, days in slice_days.get(code, []):
                if day in days:
                    f = sf
                    break
        if f is None:
            return "NO_FILE"
        try:
            df = pd.read_parquet(f)
            df = df[df["date"].astype(int) // 1000000 == day]
            if not len(df):
                return "GAP"
            v = E.validate_minute_day(df, D.day_grid(day), day)
            bad = bool((df[["open", "high", "low", "close"]].astype(float) <= 0).any().any())
            zero = bool((df["volume"].astype(float) > 0).sum() == 0)
            return "COMPLETE" if not (v["missing"] or v["dups"] or v["unexpected"] or bad or zero) else "GAP"
        except Exception:
            return "GAP"

    tracks = []
    e4_records = []
    for a in actions:
        code, Ee = a["code"], a["E"]
        i_e = ci.get(Ee)
        for tr, plan_off in (("H3", 3), ("H5", 5)):
            key = (code, Ee, tr)
            ep = end247.get(key)
            checked = []      # [(date, class)]
            unchecked = []    # [{window, reason}]
            if ep is not None:
                line = "247线(复放终点)"
                end_day = ep["last_fill_day"] or ep["blocker_day"]
                if end_day is None:
                    end_day = cal[i_e + plan_off] if i_e is not None and i_e + plan_off < len(cal) else None
                checked.append((Ee, day_class(code, Ee)))
                if i_e is not None and end_day is not None:
                    i = i_e + 1
                    while i < len(cal) and cal[i] <= end_day:
                        checked.append((cal[i], day_class(code, cal[i])))
                        i += 1
                    # 顺延余日未检查 (首阻后至 d0+5; 复放已终止处之后的顺延日)
                    if not ep["complete"]:
                        unchecked.append({"window": f"{cal[i]}..d0+5顺延余日", "reason": "复放首阻后顺延未检查(同category_2语义)"})
            else:
                line = "非247线(计划窗+顺延未检查)"
                checked.append((Ee, day_class(code, Ee)))
                for off in range(1, plan_off + 1):
                    d = cal[i_e + off] if (i_e is not None and i_e + off < len(cal)) else None
                    checked.append((d, day_class(code, d)) if d else (None, "BEYOND_CALENDAR"))
                unchecked.append({"window": f"计划E+{plan_off}之后顺延日",
                                  "reason": "顺延未复放, 不猜 (顺延未检查)"})
            # E+4 单独登记 (无论是否在窗内)
            d4 = cal[i_e + 4] if (i_e is not None and i_e + 4 < len(cal)) else None
            c4 = day_class(code, d4) if d4 else "BEYOND_CALENDAR"
            e4_records.append({"code": code, "E": Ee, "track": tr, "E+4": d4, "class": c4,
                               "in_window": any(d == d4 for d, _ in checked)})
            gaps = [d for d, c in checked if c in ("GAP", "NO_FILE")]
            tracks.append({"code": code, "E": Ee, "track": tr, "line": line,
                           "segment": a["segment"], "setup": a["setup"],
                           "checked_dates": [{"date": d, "class": c} for d, c in checked],
                           "n_checked": len(checked),
                           "gaps": gaps, "n_unchecked_windows": len(unchecked),
                           "unchecked": unchecked,
                           "all_checked_complete": (not gaps) and all(c == "COMPLETE" for _, c in checked)})

    # E+4 与 328 交集
    e4_gap = [(r["code"], r["E+4"]) for r in e4_records if r["class"] in ("GAP", "NO_FILE") and r["E+4"]]
    inter_e4_328 = sorted(set(e4_gap) & set328)
    all_checked_gap_pairs = set()
    for t in tracks:
        for d in t["gaps"]:
            if d:
                all_checked_gap_pairs.add((t["code"], d))
    inter_328_checked = sorted(all_checked_gap_pairs & set328)

    agg = defaultdict(lambda: Counter())
    for t in tracks:
        k = f"{t['line']}/{t['segment']}/{t['setup']}/{t['track']}"
        agg[k]["n"] += 1
        if t["all_checked_complete"]:
            agg[k]["all_checked_complete"] += 1
        if t["gaps"]:
            agg[k]["has_gap_in_window"] += 1
        if t["n_unchecked_windows"]:
            agg[k]["deferral_unchecked"] += 1
    e4_agg = Counter(r["class"] for r in e4_records)

    out = {"phase": "GCP_247_RETURN_PATH_COVERAGE_V1",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0, "no_collection": True, "no_return_computation": True, "set_328_suspended": True,
           "note_vs_v3": "v3 收益路径检查仅 E+1/E+2/E+3/E+5 四定点且无实际顺延; 本件按必查窗(入场窗+T+1至实际"
                         "(247线复放终点)/计划(非247线)最后退出完成bar, 含顺延)逐日核对, 字段为"
                         "'已检查日期覆盖'(checked_dates+未检查原因), 不是收益就绪证明",
           "n_action_candidates": len(actions), "n_tracks": len(tracks),
           "by_cell": {k: dict(v) for k, v in sorted(agg.items())},
           "e4_summary": {"by_class": dict(e4_agg),
                          "e4_gap_or_nofile_n": len(e4_gap),
                          "intersection_with_328": {"n": len(inter_e4_328), "pairs": inter_e4_328},
                          "note": "E+4 对 H3 计划窗是顺延候选日、对 H5 窗是窗内日; 单独计数, 缺口与 328 交集登记"},
           "checked_window_gap_intersection_with_328": {"n": len(inter_328_checked), "pairs": inter_328_checked[:40],
                                                         "note": "已检查窗内 GAP/NO_FILE 日期与 328 的交集 (328 挂起不采)"},
           "deferral_unchecked_tracks": sum(1 for t in tracks if t["line"].startswith("非247")),
           "tracks": tracks,
           "elapsed_s": round(time.time() - t0, 1)}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 收益依赖覆盖补齐 v1（已检查日期覆盖口径）", "",
         f"- 生成: {out['generated_at']} | 零网络零采集零收益计算 | 用时 {out['elapsed_s']}s",
         f"- {out['note_vs_v3']}", "",
         "## 与 v3 检查窗的差异", "",
         "- v3: E+1/E+2/E+3/E+5 四定点, 无顺延, 字段'全路径分钟'",
         "- v1(本件): 必查窗逐日(E, E+1..实际/计划终点), 247线含顺延(复放终点), 非247线顺延未检查不猜; "
         "字段'已检查日期覆盖'", "",
         "## 汇总 (line/segment/setup/track)", "",
         "| 格 | 轨数 | 已检查全COMPLETE | 窗内有GAP/NO_FILE | 顺延未检查 |", "|---|---|---|---|---|"]
    for k, v in sorted(agg.items()):
        L.append(f"| {k} | {v['n']} | {v.get('all_checked_complete',0)} | {v.get('has_gap_in_window',0)} | {v.get('deferral_unchecked',0)} |")
    L += ["", "## E+4 单独计数", "",
          f"- E+4 覆盖分布: {dict(e4_agg)}",
          f"- E+4 GAP/NO_FILE 日期数: {len(e4_gap)}",
          f"- E+4 缺口 ∩ 328 = {len(inter_e4_328)} 对" + (f": {inter_e4_328}" if inter_e4_328 else ""),
          "", "## 已检查窗缺口 ∩ 328", "",
          f"- 交集 {len(inter_328_checked)} 对" + (f" (前40): {inter_328_checked[:40]}" if inter_328_checked else ""),
          "- 328 维持挂起不采; 各集合独立预算口径", "",
          "## 声明", "",
          "- '已检查日期覆盖'不是收益就绪证明; 非 247 线顺延未检查轨道数: "
          f"{out['deferral_unchecked_tracks']} (顺延未复放, 不猜)", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"return-path coverage done ({time.time()-t0:.1f}s)")
    print("tracks:", len(tracks), "| non247 deferral unchecked:", out["deferral_unchecked_tracks"])
    print("e4:", dict(e4_agg), "| e4∩328:", len(inter_e4_328), "| checked∩328:", len(inter_328_checked))


if __name__ == "__main__":
    main()
