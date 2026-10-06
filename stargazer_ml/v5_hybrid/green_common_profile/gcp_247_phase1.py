# -*- coding: utf-8 -*-
"""GCP 247 第一阶段: 纯本地盘点 + 执行质量规模化核对 (任务单 v0.2 第一阶段, 已批准 2026-10-05).

零网络请求; 不采集 / 不训练 / 不做盈利裁决.
用户三点执行澄清:
  1) 日期 gap 与数据缺口分开: E∈[20260306,20260308] -> UNASSIGNED, 不事后归段.
  2) 超预算停机不砍范围: 247 人口保持完整; 无法处理事件记待处理+原因;
     新增报告受 5MiB 与整目录 50MiB 双上限约束, 超限即 RuntimeError 停机 (不截断).
  3) 不做盈利裁决: 只核对成交/现金/残仓/标签依赖; 不汇总收益排名/左尾/策略有效性;
     缺冻结 S 的事件列输入缺口, 不现场重算.
盘点覆盖全部已登记缓存 (含 E 日旧缓存): minute_collect_realtest/by_pair(1005),
deferral_collect_v2/by_pair(30), minute_ext(664), minute(16), minute_car(539).
运行前冻结: 执行器 / 日线(逐 part) / 日历 / 网格 / 全部缓存文件 SHA256 + manifest 5 文件复验.
247 事件全部使用冻结 S (= coverage verify rows entry.action_ts; 已核与 manifest 12 条一致).

产物: gcp_247_phase1_{frozen_inputs,inventory,status,collection_needs}.json + report.md
"""
from __future__ import annotations
import sys, json, hashlib, glob, os, time
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D
import gcp_round_b_engine as EG

BASE_DIR = HERE / "gcp_round_b_small_sample_v1"
MANIFEST = HERE / "contracts" / "dual_path_execution_v0_1_1_manifest.json"
EXECUTOR = HERE / "dual_path_execution_v0_1_1.py"
GRID_PATH = BASE_DIR / "minute_grid.json"
CV_JSON = BASE_DIR / "gcp_247_coverage_verify.json"
DAILY_PARTS = Path(os.environ.get("STARGAZER_DATA_ROOT", r"D:/A股数据") + r"\data_daily_parts")
DEFERRAL_DIR = BASE_DIR / "deferral_collect_v2" / "by_pair"

OUT_FROZEN = BASE_DIR / "gcp_247_phase1_frozen_inputs.json"
OUT_INV = BASE_DIR / "gcp_247_phase1_inventory.json"
OUT_STATUS = BASE_DIR / "gcp_247_phase1_status.json"
OUT_NEEDS = BASE_DIR / "gcp_247_phase1_collection_needs.json"
OUT_REPORT = BASE_DIR / "gcp_247_phase1_report.md"

DATA_CUTOFF = D.DATA_CUTOFF  # 20260928
TRAIN_E_MAX, VAL_E_MIN, VAL_E_MAX = 20260305, 20260309, 20260609
NEXT_SEG_START = {"train": VAL_E_MIN, "val": 20260610}   # test 无下一段
REPORT_BUDGET = 5 * 1024 * 1024          # manifest new_report_budget_bytes
DIR_BUDGET = 50 * 1024 * 1024            # 整个研究目录
KNOWN_SOURCE_HOLES = {("688656", 20250724)}   # GAP-2026-10-04-01
REGISTERED_DIRS = {                       # 已登记缓存 (引擎 MIN_DIRS + 顺延缓存)
    "minute_collect_realtest": "engine MIN_DIRS[0] (by_pair)",
    "minute_ext": "engine MIN_DIRS",
    "minute": "engine MIN_DIRS",
    "minute_car": "engine MIN_DIRS",
    "deferral_collect_v2": "reconcile deferral cache (by_pair)",
}
COLS = ["date", "code", "open", "high", "low", "close", "volume", "pre_close", "amount", "is_st"]

# 顺延缓存插到读取路径首位 (与已批准对账驱动一致, 仅追加不覆盖)
D.MIN_DIRS.insert(0, DEFERRAL_DIR)

# ---- memoized sha256 (引擎内部依赖哈希同走此缓存: 同值, 仅免重复读) ----
_H = {}
def sha256_file(p) -> str:
    k = str(p)
    if k not in _H:
        h = hashlib.sha256()
        with open(k, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _H[k] = h.hexdigest()
    return _H[k]
D._sha256_file = sha256_file


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def seg_of(e: int) -> str:
    if e <= TRAIN_E_MAX:
        return "train"
    if e <= 20260308:          # 20260306-20260308: 日期 gap -> UNASSIGNED
        return "UNASSIGNED"
    if e <= VAL_E_MAX:
        return "val"
    return "test"


_new_report_bytes = 0
def check_and_write(path: Path, text: str):
    global _new_report_bytes
    b = len(text.encode("utf-8"))
    if _new_report_bytes + b > REPORT_BUDGET:
        raise RuntimeError(f"report budget exceeded: {_new_report_bytes}+{b} > {REPORT_BUDGET} "
                           f"(stop, no truncation)")
    if dir_size(BASE_DIR) + b > DIR_BUDGET:
        raise RuntimeError(f"research-dir budget exceeded before writing {path.name} (stop, no truncation)")
    path.write_text(text, encoding="utf-8")
    _new_report_bytes += b
    print(f"  written {path.name} ({b/1024:.1f} KiB)")


def verify_manifest(man: dict) -> dict:
    res = {"manifest_version": man.get("manifest_version"), "files_checked": [], "all_match": True}
    for f in man["files"]:
        p = Path("D:/stargazer") / f["path"]
        if not p.exists():
            res["files_checked"].append({"role": f["role"], "present": False})
            res["all_match"] = False
            continue
        actual = sha256_file(p)
        m = (actual == f["sha256"])
        res["files_checked"].append({"role": f["role"], "path": f["path"], "match": m})
        if not m:
            res["all_match"] = False
    return res


# ================= 1) 冻结输入 =================
def build_frozen(man: dict, mv: dict) -> dict:
    parts = sorted(DAILY_PARTS.glob("part_*.parquet"))
    cal = D.load_calendar()
    g = D.load_minute_grid()
    exec_sha = sha256_file(EXECUTOR)
    return {
        "phase": "GCP_247_PHASE1", "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0,
        "manifest_verification": mv,
        "executor": {"path": str(EXECUTOR.relative_to(HERE)), "sha256": exec_sha,
                     "bytes": EXECUTOR.stat().st_size},
        "minute_grid": {"path": str(GRID_PATH), "sha256": sha256_file(GRID_PATH),
                        "grid_1300_from": g.get("grid_1300_from"),
                        "grid_241_n": len(g.get("grid_241", [])), "grid_n": len(g.get("grid", []))},
        "daily_parts": {
            "dir": str(DAILY_PARTS), "n_files": len(parts),
            "total_bytes": sum(p.stat().st_size for p in parts),
            "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                      for p in parts]},
        "calendar": {"n_days": len(cal), "first": cal[0], "last": cal[-1],
                     "data_cutoff": DATA_CUTOFF,
                     "sha256": hashlib.sha256(json.dumps(cal).encode()).hexdigest()},
        "budget": {"report_bytes": REPORT_BUDGET, "dir_bytes": DIR_BUDGET,
                   "dir_bytes_at_start": dir_size(BASE_DIR)},
    }


# ================= 2) 缓存盘点 (全部已登记缓存, 含 E 日旧缓存) =================
def check_cache_file(fp: Path, rel_dir: str) -> dict:
    rec = {"file": str(fp.relative_to(BASE_DIR)), "bytes": fp.stat().st_size,
           "sha256": sha256_file(fp)}
    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        rec["error"] = f"unreadable:{repr(e)[:120]}"
        return rec
    stem = fp.stem
    if "_" in stem and stem.split("_")[-1].isdigit():
        naming, code, expect_day = "by_pair", stem.split("_")[0], int(stem.split("_")[-1])
        rec.update({"naming": "by_pair", "code": code})
    else:
        naming, code, expect_day = "code", stem, None
        rec.update({"naming": "code", "code": code})
    if "date" not in df.columns or not len(df):
        rec["error"] = "no_date_or_empty"
        return rec
    ds = df["date"].astype(int)
    days = sorted({int(x) // 1000000 for x in ds})
    issues = []
    if naming == "by_pair" and expect_day not in days:
        issues.append({"day": expect_day, "kind": "day_mismatch", "detail": f"actual days={days}"})
    dup_n = int(ds.duplicated().sum())
    price_bad = int(((df[["open", "high", "low", "close"]].astype(float) <= 0).any(axis=1)).sum()) \
        if all(c in df.columns for c in ("open", "high", "low", "close")) else None
    vol_bad = int((df["volume"].astype(float) < 0).sum()) if "volume" in df.columns else None
    for day in days:
        sub = df[ds // 1000000 == day]
        v = EG.validate_minute_day(sub, D.day_grid(day), day)
        if (not v["ok"]) or v["missing"] or v["dups"] or v["unexpected"]:
            issues.append({"day": day, "kind": "grid" if not v["ok"] else "grid_partial",
                           "engine_ok": v["ok"], "reason": v["reason"],
                           "missing": v["missing"][:12], "dups": v["dups"][:6],
                           "unexpected": v["unexpected"][:6],
                           "expected_slots": v.get("expected_slots")})
    rec.update({"n_days": len(days), "days": days if len(days) <= 16 else [days[0], days[-1]],
                "dup_rows": dup_n, "price_bad_rows": price_bad, "vol_bad_rows": vol_bad,
                "issues": issues})
    rec["trusted"] = not (issues or dup_n or (price_bad or 0) or (vol_bad or 0) or rec.get("error"))
    return rec


def build_inventory() -> dict:
    dirs = {}
    for name in REGISTERED_DIRS:
        p = BASE_DIR / name
        files = sorted(p.rglob("*.parquet"))
        dirs[name] = {"registered": True, "role": REGISTERED_DIRS[name],
                      "n_files": len(files), "total_bytes": sum(f.stat().st_size for f in files)}
    # 未登记 parquet 目录 (只报告, 不使用)
    known = set(REGISTERED_DIRS)
    for p in BASE_DIR.rglob("*.parquet"):
        top = p.relative_to(BASE_DIR).parts[0]
        if top not in known:
            e = dirs.setdefault(top, {"registered": False, "n_files": 0, "total_bytes": 0,
                                      "files": []})
            e["n_files"] += 1
            e["total_bytes"] += p.stat().st_size
    recs = []
    for name in sorted(REGISTERED_DIRS):
        for fp in sorted((BASE_DIR / name).rglob("*.parquet")):
            recs.append(check_cache_file(fp, name))
    summary = {}
    for name in sorted(REGISTERED_DIRS):
        sub = [r for r in recs if r["file"].startswith(name + os.sep) or r["file"].startswith(name + "/")]
        summary[name] = {"n_files": len(sub),
                         "trusted": sum(1 for r in sub if r.get("trusted")),
                         "anomaly": sum(1 for r in sub if not r.get("trusted")),
                         "bytes": sum(r["bytes"] for r in sub)}
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "network_requests": 0,
            "grid_rule": "day<grid_1300_from(20260624)->241 slots; else 242 slots; "
                         "engine tolerance missing<=2 recorded separately (issues carry exact slots)",
            "dirs": dirs, "summary": summary, "files": recs}


# ================= 3) 引擎规模化核对 =================
def norm_track(t):
    if not t:
        return None
    out = {k: t.get(k) for k in ("status", "reason", "planned_exit_date", "deferral_d0",
                                 "fully_exited", "remaining_shares", "C_buy", "C_sell", "pnl",
                                 "net", "label_end_ts", "last_fill_ts", "observed_endpoint_ts",
                                 "missing_dependencies", "sells")}
    out["sim_fill"] = t.get("sim_fill")
    return out


def classify(ev: dict) -> str:
    st = ev.get("exec_status")
    if st != "EXECUTED":
        return st
    h3, h5 = ev.get("H3") or {}, ev.get("H5") or {}
    sts = {h3.get("status"), h5.get("status")}
    if sts <= {"SIM_COMPLETE"}:
        return "COMPLETE" if (h3.get("fully_exited") and h5.get("fully_exited")) else "UNEXITED_AT_CUTOFF"
    if "EXIT_DATA_MISSING" in sts:
        return "INCOMPLETE"
    if sts <= {"EXIT_DEFERRED_CAP"}:
        return "DEFERRED_CAP"
    return "MIXED"


def cross_flag(seg: str, t) -> bool:
    if seg not in NEXT_SEG_START or not t:
        return False
    le = t.get("label_end_ts")
    return bool(le is not None and (le // 1000000) >= NEXT_SEG_START[seg])


def main():
    t0 = time.time()
    print("=== GCP 247 Phase 1: 纯本地盘点 + 执行质量规模化核对 (零网络) ===")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mv = verify_manifest(man)
    print(f"manifest 5 文件复验: all_match={mv['all_match']}")

    # ---- 1) 冻结输入 (引擎运行前) ----
    frozen = build_frozen(man, mv)
    check_and_write(OUT_FROZEN, json.dumps(frozen, ensure_ascii=False, indent=1))

    # ---- 2) 缓存盘点 (引擎运行前) ----
    inv = build_inventory()
    check_and_write(OUT_INV, json.dumps(inv, ensure_ascii=False, indent=1))
    for k, v in inv["summary"].items():
        print(f"  cache {k}: files={v['n_files']} trusted={v['trusted']} anomaly={v['anomaly']}")

    # ---- 3) 共享数据 ----
    cal = D.load_calendar()
    rows = json.loads(CV_JSON.read_text(encoding="utf-8"))["rows"]
    control_set = {(c["code"], c["E"]) for c in man["isolation_controls"]}
    codes = sorted({r["code"] for r in rows})
    print(f"247 事件 / {len(codes)} codes; controls in scope: "
          f"{sum(1 for c in control_set if c in {(r['code'], r['E']) for r in rows})}/4")

    import pyarrow.dataset as ds
    files = sorted(glob.glob(str(DAILY_PARTS / "part_*.parquet")))
    tbl = ds.dataset(files, format="parquet").to_table(
        columns=COLS, filter=ds.field("code").isin(codes))
    pdf = tbl.to_pandas()
    pdf["date"] = pdf["date"].astype(int)
    daily_by_code = {c: g.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                     for c, g in pdf.groupby("code")}
    print(f"日线 panel (filtered): {len(pdf)} rows, {len(daily_by_code)} codes")

    mbd_cache, mprov_cache = {}, {}
    def get_minute(code: str):
        if code not in mbd_cache:
            mbd_cache[code], mprov_cache[code] = D.load_minute_code(code)
        return mbd_cache[code]

    # ---- 4) 逐事件引擎核对 (全部用冻结 S; 缺冻结 S -> 输入缺口, 不重算) ----
    events = []
    n_fallback_daily = 0
    for row in rows:
        code, E_ = row["code"], int(row["E"])
        seg = seg_of(E_)
        S = row.get("entry", {}).get("action_ts")
        ev = {"code": code, "P": row.get("P"), "E": E_, "setup": row.get("setup"),
              "board": row.get("board"), "segment_frozen": row.get("segment"), "segment": seg,
              "segment_mismatch": (seg != row.get("segment")),
              "control": (code, E_) in control_set,
              "frozen_S": S, "signal_source": ("frozen_coverage_action_ts" if S else None)}
        if S is None:
            ev.update({"exec_status": "INPUT_GAP_NO_FROZEN_S",
                       "input_gap": "no frozen S/action branch; not recomputed"})
            ev["class"] = classify(ev)
            events.append(ev)
            continue
        try:
            mbd = get_minute(code)
            ev["minute_cache_files"] = sorted(Path(p["path"]).name for p in mprov_cache.get(code, []))
            dd = daily_by_code.get(code)
            if dd is None:
                n_fallback_daily += 1
            r = D.execute_event(code, E_, row["board"], None, row.get("setup"),
                                signal_ts=int(S), minute_by_day=mbd,
                                daily_df=dd, cal=cal)
        except Exception as ex:
            r = {"status": "EXECUTE_ERROR", "error": repr(ex)[:300]}
        ev["exec_status"] = r.get("status", "EXECUTE_ERROR")
        ev["entry_reason"] = r.get("reason")
        ev["signal_type"] = r.get("signal_type")
        ent = r.get("entry")
        if isinstance(ent, dict):
            slim_e = {k: ent.get(k) for k in ("status", "sim_fill_ts", "sim_fill_price",
                                              "shares", "amount", "D_stop", "stop_price")}
            if isinstance(ent.get("reasons"), list):
                slim_e["reasons_head"] = ent["reasons"][:5]
                slim_e["reasons_n"] = len(ent["reasons"])
            ev["entry"] = slim_e
        for tr in ("H3", "H5"):
            ev[tr] = norm_track(r.get(tr))
        ev["error"] = r.get("error")
        ev["class"] = classify(ev)
        ev["cross_segment_label"] = {
            "next_segment_start": NEXT_SEG_START.get(seg),
            "H3": cross_flag(seg, ev.get("H3")), "H5": cross_flag(seg, ev.get("H5"))}
        events.append(ev)
    print(f"引擎核对完成: {len(events)} 事件 ({time.time()-t0:.1f}s; daily fallback={n_fallback_daily})")

    # ---- 5) 逐轨道缺失与隔离清单 / 去重补采需求 ----
    track_missing, isolation, daily_meta_gaps = [], [], []
    pairs = {}
    def add_pair(code, day, E_, tr, kind):
        k = (code, int(day))
        e = pairs.setdefault(k, {"needed_by": [], "kinds": []})
        if {"E": E_, "track": tr} not in e["needed_by"]:
            e["needed_by"].append({"E": E_, "track": tr})
        if kind not in e["kinds"]:
            e["kinds"].append(kind)
    for ev in events:
        code, E_ = ev["code"], ev["E"]
        if ev.get("exec_status") == "ENTRY_DATA_MISSING" and ev.get("entry_reason") == "no E-day minute":
            add_pair(code, E_, E_, None, "E_day_minute_missing")
            track_missing.append({"code": code, "E": E_, "track": None,
                                  "type": "E_day_minute_missing", "date": E_})
        for tr in ("H3", "H5"):
            t = ev.get(tr)
            if not t:
                continue
            for md in (t.get("missing_dependencies") or []):
                typ = md.get("type")
                if typ in ("minute_day_missing", "day_invalid"):
                    add_pair(code, md["date"], E_, tr, typ)
                    track_missing.append({"code": code, "E": E_, "track": tr, "type": typ,
                                          "date": md.get("date"), "detail": md.get("detail")})
                elif typ in ("is_st_unknown", "pre_close_missing"):
                    daily_meta_gaps.append({"code": code, "E": E_, "track": tr, "type": typ,
                                            "date": md.get("date")})
                elif typ not in ("deferral_cap",):
                    track_missing.append({"code": code, "E": E_, "track": tr, "type": typ,
                                          "date": md.get("date")})
            if (ev.get("cross_segment_label") or {}).get(tr):
                isolation.append({"code": code, "E": E_, "track": tr,
                                  "segment": ev["segment"],
                                  "label_end_ts": t.get("label_end_ts"),
                                  "next_segment_start": NEXT_SEG_START.get(ev["segment"]),
                                  "rule": "label_end_ts >= next segment start"})
        if ev.get("control"):
            isolation.append({"code": code, "E": E_, "track": None, "segment": ev["segment"],
                              "rule": "isolation_control (known issue; validation only)"})
    # 补采分类与预算
    pair_list = []
    by_cls = Counter()
    for (code, day), e in sorted(pairs.items()):
        if day > DATA_CUTOFF:
            cls = "BEYOND_DATA_CUTOFF"
        elif (code, day) in KNOWN_SOURCE_HOLES:
            cls = "KNOWN_SOURCE_HOLE"
        else:
            cls = "COLLECTABLE"
        by_cls[cls] += 1
        pair_list.append({"code": code, "day": day, "class": cls, "kinds": e["kinds"],
                          "needed_by": e["needed_by"]})
    by_pair_files = [f for f in inv["files"]
                     if f.get("naming") == "by_pair" and not f.get("error")]
    avg_bytes = int(sum(f["bytes"] for f in by_pair_files) / max(1, len(by_pair_files)))
    est_collect = by_cls["COLLECTABLE"] * avg_bytes
    dir_now = dir_size(BASE_DIR)
    needs = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "network_requests": 0,
        "note": "仅盘点与预算; 不采集. 第二阶段须另行批准.",
        "dedup_pairs": pair_list,
        "counts": {"dedup_pairs_total": len(pair_list), "by_class": dict(by_cls),
                   "track_missing_records": len(track_missing),
                   "daily_meta_gaps": len(daily_meta_gaps)},
        "daily_meta_gaps": daily_meta_gaps,
        "budget": {"avg_by_pair_bytes": avg_bytes,
                   "estimated_collect_bytes": est_collect,
                   "dir_bytes_now": dir_now, "dir_budget": DIR_BUDGET,
                   "projected_dir_after_collect": dir_now + est_collect,
                   "collect_budget_ok": (dir_now + est_collect) <= DIR_BUDGET,
                   "report_bytes_this_run": _new_report_bytes,
                   "report_budget": REPORT_BUDGET},
    }

    # ---- 6) 状态表 ----
    cls_counts = Counter(e["class"] for e in events)
    seg_counts = defaultdict(Counter)
    for e in events:
        seg_counts[e["segment"]][e["class"]] += 1
    tr_status = {"H3": Counter(), "H5": Counter()}
    tr_full = {"H3": 0, "H5": 0}
    for e in events:
        for tr in ("H3", "H5"):
            t = e.get(tr)
            if t:
                tr_status[tr][t.get("status")] += 1
                if t.get("fully_exited"):
                    tr_full[tr] += 1
    status = {
        "phase": "GCP_247_PHASE1", "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0,
        "no_profitability_aggregation": True,
        "note": "仅状态/成交/现金/残仓/标签依赖; 不汇总收益排名/左尾/策略有效性; "
                "net/pnl 为逐事件字段, 未做任何聚合统计.",
        "population": {"n_events": len(events), "segments_rederived": dict(Counter(e["segment"] for e in events)),
                       "segment_mismatches": sum(1 for e in events if e["segment_mismatch"]),
                       "unassigned_rule": "E in [20260306,20260308] -> UNASSIGNED (date gap, not data gap)",
                       "input_gaps_no_frozen_S": sum(1 for e in events if e["class"] == "INPUT_GAP_NO_FROZEN_S")},
        "summary": {"class_counts": dict(cls_counts),
                    "class_by_segment": {k: dict(v) for k, v in seg_counts.items()},
                    "track_status": {k: dict(v) for k, v in tr_status.items()},
                    "track_fully_exited": tr_full},
        "track_missing_index": track_missing,
        "isolation_list": isolation,
        "events": events,
    }
    check_and_write(OUT_STATUS, json.dumps(status, ensure_ascii=False, indent=1))
    check_and_write(OUT_NEEDS, json.dumps(needs, ensure_ascii=False, indent=1))

    # ---- 7) report.md ----
    L = ["# GCP 247 第一阶段报告 (纯本地盘点 + 执行质量规模化核对)", "",
         f"- 生成: {status['generated_at']}  | 网络请求: 0 | 用时 {time.time()-t0:.1f}s",
         f"- manifest 5 文件复验: {'全部匹配' if mv['all_match'] else '不一致'}",
         f"- 运行前冻结: 执行器/日线({frozen['daily_parts']['n_files']} parts)/日历/网格/全部缓存 SHA256 "
         f"(见 gcp_247_phase1_frozen_inputs.json)", "",
         "## 缓存盘点 (全部已登记缓存, 含 E 日旧缓存)", "",
         "| 目录 | 文件 | 可信 | 异常 | 字节 |", "|---|---|---|---|---|"]
    for k, v in inv["summary"].items():
        L.append(f"| {k} | {v['n_files']} | {v['trusted']} | {v['anomaly']} | {v['bytes']} |")
    L += ["", "## 247 状态分布", "",
          f"- class: {dict(cls_counts)}",
          f"- 分段(重推导, UNASSIGNED 规则): {dict(Counter(e['segment'] for e in events))}"
          f" | 段标注不一致: {status['population']['segment_mismatches']}",
          f"- 输入缺口(无冻结 S): {status['population']['input_gaps_no_frozen_S']}",
          f"- 轨道状态: H3={dict(tr_status['H3'])} | H5={dict(tr_status['H5'])}",
          f"- 完全退出: H3={tr_full['H3']} H5={tr_full['H5']}", "",
          "## 逐轨道缺失与隔离", "",
          f"- 缺失记录: {len(track_missing)} (明细在 status.track_missing_index)",
          f"- 日线元数据缺口(ST/pre_close, 非分钟采集): {len(daily_meta_gaps)}",
          f"- 隔离清单: {len(isolation)} 条 (CROSS_SEGMENT_LABEL + 对照)", "",
          "## 去重补采需求与预算", "",
          f"- 去重 (code,day) 对: {len(pair_list)} | 分类: {dict(by_cls)}",
          f"- 预估补采字节: {est_collect} (均值 {avg_bytes}B/文件)",
          f"- 研究目录现 {dir_now/1048576:.2f}MiB -> 预计 {needs['budget']['projected_dir_after_collect']/1048576:.2f}MiB "
          f"(上限 50MiB: {'OK' if needs['budget']['collect_budget_ok'] else '超限'})", "",
          "## 边界声明", "",
          "- 本阶段未做盈利裁决: 无收益排名/左尾/策略有效性汇总; 逐事件 net/pnl 仅核对字段.",
          "- 未成交/未闭环不记零收益; 残仓仅记录股数, 不折现金.",
          "- 网络采集、第二阶段经济复放、训练与买入建议均未批准, 到此停止."]
    check_and_write(OUT_REPORT, "\n".join(L))

    print("class_counts:", dict(cls_counts))
    print("track_status H3:", dict(tr_status["H3"]), "H5:", dict(tr_status["H5"]))
    print(f"needs: dedup={len(pair_list)} by_class={dict(by_cls)} est={est_collect}B "
          f"budget_ok={needs['budget']['collect_budget_ok']}")
    print("Phase 1 完成, 停止 (无第二阶段).")


if __name__ == "__main__":
    main()
