# -*- coding: utf-8 -*-
"""全量离线 QA + 逐事件退出路径重报 (Task A, 2026-10-04 修订版).

修复两处盲点 (用户 2026-10-04 复核):
  (A) 重复分钟检测先前先对时间戳去重, 永远发现不了重复 -> 改为频率计数, 报告重复时间戳.
  (B) 文件日期只核对首行 -> 改为核对 ALL 行是否与文件名日期一致.

关键区分 (用户裁定):
  - "退出路径完整 (exit_complete)" 仅表示 5 个所需日期都有文件 (本次采集 or 原缓存).
  - "可复放 (replay_valid)" 还要求每个所需日的文件 网格OK/价格OK/量OK/日期全行一致/无重复时间戳.
    8 个异常文件 (跨 7 条事件) 使 replay_valid 低于 exit_complete.

产物: gcp_round_b_small_sample_v1/gcp_exit_path_qa.json
网络请求: 0 (仅读取已落盘文件).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERIFY = HERE / "gcp_round_b_small_sample_v1" / "gcp_247_coverage_verify.json"
BY_PAIR = HERE / "gcp_round_b_small_sample_v1" / "minute_collect_realtest" / "by_pair"
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_exit_path_qa.json"

MIN_DIRS = [HERE / "gcp_round_b_small_sample_v1" / "minute",
            HERE / "gcp_round_b_small_sample_v1" / "minute_ext",
            HERE / "gcp_round_b_small_sample_v1" / "minute_car"]
FALLBACK_DIR = ROOT / "pullback_train" / "minute_cache_clean"

GRID_1300_FROM = 20260624
GAP_CODE = "688656"
GAP_DAY = 20250724
MISSING_SLOT_TOLERANCE = 2  # 引擎 validate_minute_day 允许的缺失槽上限


def expected_grid(day: int) -> list:
    """复刻引擎: 09:30-11:30(121) + 13:01-15:00(120) = 241; day>=20260624 含 13:00 -> 242."""
    slots = []
    h, m = 9, 30
    while (h, m) <= (11, 30):
        slots.append((h * 100 + m) * 100)
        m += 1
        if m == 60:
            m = 0
            h += 1
    h, m = 13, 1
    while (h, m) <= (15, 0):
        slots.append((h * 100 + m) * 100)
        m += 1
        if m == 60:
            m = 0
            h += 1
    if day is not None and int(day) >= GRID_1300_FROM:
        slots.append(130000)
    return sorted(set(slots))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def qa_one_file(df: pd.DataFrame, code: str, date: int) -> dict:
    """单文件 QA. 修复重复检测(频率)与全行日期核对."""
    rec = {"code": code, "date": date, "n_bars": int(len(df)), "cols": list(df.columns)}
    ts = [int(t) for t in df["date"]] if "date" in df.columns and len(df) else []
    ts_counts = Counter(ts)
    dup_ts = {t: c for t, c in ts_counts.items() if c > 1}
    rec["dup_timestamps"] = sorted(dup_ts.keys())
    rec["dup_timestamp_n"] = sum(c - 1 for c in dup_ts.values())
    # 日期全行核对 (盲点 B 修复)
    if ts:
        day_ints = [int(str(t)[:8]) for t in ts]
        rec["data_day_all_match"] = all(d == date for d in day_ints)
        rec["date_mismatch_rows"] = sum(1 for d in day_ints if d != date)
    else:
        rec["data_day_all_match"] = False
        rec["date_mismatch_rows"] = 0
    # 网格 (盲点 A 修复: 用频率而非去重集合)
    slot_counts = Counter(int(str(t)[-6:]) for t in ts)
    dups_slots = sorted(s for s, c in slot_counts.items() if c > 1)
    present_set = set(slot_counts)
    exp = expected_grid(date)
    missing = sorted(set(exp) - present_set)
    unexpected = sorted(present_set - set(exp))
    rec["grid_missing"] = len(missing)
    rec["grid_dups"] = len(dups_slots)
    rec["grid_unexpected"] = len(unexpected)
    rec["grid_expected_n"] = len(exp)
    rec["grid_ok"] = (len(missing) <= MISSING_SLOT_TOLERANCE) and not dups_slots and not unexpected
    # 量价尺度
    issues = []
    cols = list(df.columns)
    for c in ("open", "high", "low", "close"):
        if c not in cols:
            issues.append(f"missing_{c}")
    if "close" in cols:
        cl = pd.to_numeric(df["close"], errors="coerce")
        if cl.isna().any():
            issues.append("null_close")
        if (cl <= 0).any():
            issues.append("nonpositive_close")
    if {"high", "low"}.issubset(cols):
        hi = pd.to_numeric(df["high"], errors="coerce")
        lo = pd.to_numeric(df["low"], errors="coerce")
        if (hi < lo).any():
            issues.append("high_below_low")
    if "volume" in cols:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        if (vol < 0).any():
            issues.append("negative_volume")
        if vol.sum() == 0:
            issues.append("allzero_volume")
    rec["price_issues"] = issues
    rec["price_ok"] = (len(issues) == 0)
    rec["usable"] = bool(rec["grid_ok"] and rec["price_ok"]
                         and rec["data_day_all_match"] and rec["dup_timestamp_n"] == 0)
    return rec


def load_original_index():
    idx = set()
    for d in MIN_DIRS + ([FALLBACK_DIR] if FALLBACK_DIR.exists() else []):
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            try:
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    for dd in pd.unique(m["date"].astype(str).str[:8].astype(int)):
                        idx.add((pf.stem, int(dd)))
            except Exception:
                pass
    return idx


def load_original_day(code: str, date: int) -> pd.DataFrame | None:
    for d in MIN_DIRS + ([FALLBACK_DIR] if FALLBACK_DIR.exists() else []):
        if not d.exists():
            continue
        f = d / f"{code}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f)
            sub = df[df["date"].astype(int).astype(str).str[:8].astype(int) == int(date)]
            if len(sub):
                return sub
        except Exception:
            pass
    return None


def qa_by_pair_files():
    rows, seen = [], Counter()
    for pf in sorted(BY_PAIR.glob("*.parquet")):
        raw = pf.read_bytes()
        code, date_s = pf.stem.split("_")
        date = int(date_s)
        seen[(code, date)] += 1
        rec = {"sha256": sha256_bytes(raw), "bytes": len(raw)}
        try:
            df = pd.read_parquet(pf)
            rec.update(qa_one_file(df, code, date))
        except Exception as ex:
            rec.update({"error": f"{type(ex).__name__}: {ex}", "usable": False,
                        "grid_ok": False, "price_ok": False, "data_day_all_match": False})
        rows.append(rec)
    dup = [{"code": k[0], "date": k[1], "count": v} for k, v in seen.items() if v > 1]
    return rows, seen, dup


def main():
    verify = json.loads(VERIFY.read_text(encoding="utf-8"))
    rows_events = verify["rows"]

    qa, seen, dup_keys = qa_by_pair_files()
    by_pair_qa = {(r["code"], r["date"]): r for r in qa}
    original_idx = load_original_index()
    bypair_keys = set(seen.keys())

    # 事件级: 退出路径完整性 与 可复放性
    day_to_events = defaultdict(list)
    exit_reports = []
    orig_qa_cache = {}
    for r in rows_events:
        code = str(r["code"]).zfill(6)
        offsets = []
        all_present = True
        all_usable = True
        for s in r["scan"]:
            d = int(s["date"])
            day_to_events[(code, d)].append({"E": int(r["E"]), "segment": r.get("segment")})
            present = (code, d) in bypair_keys or (code, d) in original_idx
            if not present:
                all_present = False
                all_usable = False
                offsets.append({"offset": s.get("offset"), "date": d,
                                "present": False, "usable": False})
                continue
            if (code, d) in by_pair_qa:
                rec = by_pair_qa[(code, d)]
            else:
                key = (code, d)
                if key not in orig_qa_cache:
                    df = load_original_day(code, d)
                    if df is None:
                        orig_qa_cache[key] = {"n_bars": 0, "usable": False,
                                              "grid_ok": False, "price_ok": False,
                                              "data_day_all_match": False,
                                              "dup_timestamp_n": 0,
                                              "price_issues": ["original_load_failed"],
                                              "grid_missing": -1}
                    else:
                        orig_qa_cache[key] = qa_one_file(df, code, d)
                rec = orig_qa_cache[key]
            offsets.append({"offset": s.get("offset"), "date": d, "present": True,
                            "usable": bool(rec.get("usable", False)),
                            "grid_ok": rec.get("grid_ok"),
                            "price_ok": rec.get("price_ok"),
                            "grid_missing": rec.get("grid_missing"),
                            "price_issues": rec.get("price_issues")})
            all_usable = all_usable and bool(rec.get("usable", False))
        has_partial_gap = any((o.get("grid_missing") or 0) in (1, 2)
                              for o in offsets if o.get("present"))
        exit_reports.append({
            "code": code, "E": int(r["E"]), "segment": r.get("segment"),
            "board": r.get("board"), "offsets": offsets,
            "exit_complete": all_present, "qa_passed": all_usable,
            "has_partial_gap": has_partial_gap,
        })

    # 异常文件 -> 受影响事件 (关键: 8 文件跨 7 事件, 不能直接用 245 当可复放数)
    anomalies = [r for r in qa if not r.get("usable", False)]
    affected_events = set()
    anomaly_event_map = []
    for a in anomalies:
        evs = day_to_events.get((a["code"], a["date"]), [])
        for ev in evs:
            affected_events.add((ev["E"], ev["segment"]))
        anomaly_event_map.append({
            "code": a["code"], "date": a["date"],
            "issues": {"grid_missing": a.get("grid_missing"),
                       "grid_dups": a.get("grid_dups"),
                       "grid_unexpected": a.get("grid_unexpected"),
                       "price_issues": a.get("price_issues"),
                       "date_mismatch_rows": a.get("date_mismatch_rows"),
                       "dup_timestamp_n": a.get("dup_timestamp_n")},
            "affected_events": evs,
        })

    # 688656@20250724 缺口对两条事件的隔离判定
    gap_events = []
    for r in rows_events:
        if str(r["code"]).zfill(6) != GAP_CODE:
            continue
        scan = r["scan"]
        dates = {int(s["date"]): s.get("offset") for s in scan}
        if GAP_DAY not in dates:
            continue
        gap_off = dates[GAP_DAY]
        pre_gap = [int(s["date"]) for s in scan
                   if int(s["date"]) < GAP_DAY
                   and ((str(r["code"]).zfill(6), int(s["date"])) in bypair_keys
                        or (str(r["code"]).zfill(6), int(s["date"])) in original_idx)]
        affected = [int(s["date"]) for s in scan if int(s["date"]) >= GAP_DAY]
        if not pre_gap:
            decision = "EXIT_DATA_MISSING_NULL"
            note = "缺口为首个退出日, 无缺口前退出日可证闭环 -> H3/H5 收益 NULL"
        else:
            decision = "PRE_GAP_DATA_PRESENT_PENDING_ENGINE"
            note = (f"缺口前退出日 {pre_gap} 分钟齐备, 须跑引擎判定是否合法止损; "
                    "若成立保留闭环结果, 否则 H3/H5 = NULL")
        gap_events.append({
            "code": GAP_CODE, "E": int(r["E"]), "gap_day": GAP_DAY, "gap_offset": gap_off,
            "pre_gap_exit_days_present": pre_gap, "affected_refs": affected,
            "policy_decision": decision, "note": note,
        })

    report = {
        "version": "gcp_exit_path_qa_v2",
        "network_requests": 0,
        "fixes": ["dup_timestamp detection now frequency-based (was set-dedup, blind)",
                  "date check now ALL rows (was first row only)",
                  "added event-level replay_valid (grid+price+volume+date+no-dup)"],
        "by_pair_file_count": len(qa),
        "by_pair_total_bytes": sum(r.get("bytes", 0) for r in qa),
        "by_pair_anomalies": len(anomalies),
        "by_pair_dup_keys": dup_keys,
        "exit_complete_n": sum(1 for e in exit_reports if e["exit_complete"]),
        "exit_incomplete_n": sum(1 for e in exit_reports if not e["exit_complete"]),
        "qa_passed_n": sum(1 for e in exit_reports if e["qa_passed"]),
        "qa_passed_with_partial_gap_n": sum(1 for e in exit_reports
                                            if e["qa_passed"] and e["has_partial_gap"]),
        "partial_gap_events": [{"code": e["code"], "E": e["E"], "segment": e.get("segment"),
                                 "gap_dates": [o["date"] for o in e["offsets"]
                                               if (o.get("grid_missing") or 0) in (1, 2)]}
                                for e in exit_reports if e["qa_passed"] and e["has_partial_gap"]],
        "events_with_anomaly_n": len(affected_events),
        "partial_gap_rule": ("含 1-2 缺槽的 QA 通过事件(同上 93 条): 收盘参照价可读取, "
                             "但缺槽可能藏更早硬止损 -> 实际退出路径/含止损收益待判, "
                             "标签 NULL 待补证, 不得整批放行"),
        "anomaly_files": anomaly_event_map,
        "gap_code": GAP_CODE, "gap_day": GAP_DAY,
        "gap_affected_events": gap_events,
        "qa_files": qa,
        "exit_reports": exit_reports,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[离线 QA v2] 文件={len(qa)} 字节={round(report['by_pair_total_bytes']/1024/1024,2)}MB "
          f"异常={len(anomalies)} 重复键={len(dup_keys)}")
    print(f"[退出路径] 完整(有文件)={report['exit_complete_n']}  "
          f"当前QA通过(网格/价格/量/日期/无重复全OK)={report['qa_passed_n']}  "
          f"其中含缺槽(1-2槽)={report['qa_passed_with_partial_gap_n']}")
    print(f"[异常->事件] {len(anomalies)} 异常文件跨 {len(affected_events)} 条事件 "
          f"(245 不可直接当可复放数)")
    for g in gap_events:
        print(f"[缺口隔离] {g['code']} E={g['E']} 缺口offset={g['gap_offset']} "
              f"缺口前齐备={g['pre_gap_exit_days_present']} -> {g['policy_decision']}")
    print(f"明细: {OUT.name}")


if __name__ == "__main__":
    main()
