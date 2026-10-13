"""R0-V1 外部证据核验（只读、零网络、可复跑）。

产出机器可核验的证据，不强行通过任何一项；缺证据即登记 UNVERIFIED(未核销)。
- 日历：从 canonical daily parts 实算（镜像 trading_calendar.load_calendar），核对与冻结快照指纹的差异（证明旧指纹不能绑定当前输入）。
- 复权：本地缓存 adj_cache.json + union_codes.json + 失败日志，逐码状态与事件覆盖。
- 候选：dual_path_e_day_registry.json 汇总；逐行绑定字段标记 UNVERIFIED（实际存储表待定位）。
- 价格尺度：日线 parquet schema + 重算 SHA256 与库存清单逐文件比对（内容哈希，非指纹）。
"""
from __future__ import annotations
import glob, os, sys, json, hashlib
from collections import Counter
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
DAILY_PARTS = r"D:\a股数据\data_daily_parts"
ADJ_CACHE = HERE.parent / "opportunity_v2" / "gate1c" / "feb2026_adjustment_cache_v1" / "adj_cache.json"
UNION = HERE.parent / "opportunity_v2" / "gate1c" / "feb2026_universe_freeze_v1" / "union_codes.json"
FREEZE_MANIFEST = HERE.parent / "opportunity_v2" / "gate1c" / "feb2026_universe_freeze_v1" / "manifest.json"
REGISTRY = HERE.parent / "green_common_profile" / "contracts" / "dual_path_e_day_registry.json"
INVENTORY = HERE / "risk_population_inventory_v2.json"

import pyarrow.dataset as ds  # noqa: E402


def compute_calendar():
    files = sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
    if not files:
        return {"error": "NO_DAILY_PARTS", "n_files": 0}
    dates = ds.dataset(files, format="parquet").to_table(columns=["date"]).column("date").to_pylist()
    cal = sorted({int(d) for d in dates})
    sizes = [os.path.getsize(f) for f in files]
    mtimes = [os.path.getmtime(f) for f in files]
    epoch = max(cal) if cal else 0
    h = hashlib.sha256()
    for f, s in zip(files, sizes):
        h.update(f.encode("utf-8")); h.update(str(s).encode("utf-8"))
    digest = h.hexdigest()[:16]
    fp = (f"daily_parts@{len(files)}files_s{sum(sizes)}_"
          f"m{max(mtimes):.0f}_epoch{epoch}_h{digest}")
    # 自洽检查（非权威；权威完整性需独立交易所日历）
    wk_bad = [d for d in cal if datetime.strptime(str(d), "%Y%m%d").weekday() >= 5]
    gaps = []
    for a, b in zip(cal, cal[1:]):
        if b - a > 4:  # 跨周末外的长缺口（仅提示）
            gaps.append((a, b, b - a))
    return {"n_files": len(files), "n_trading_days": len(cal),
            "min_date": cal[0], "max_date": cal[-1],
            "data_epoch_fingerprint": fp,
            "weekday_contaminated_days": wk_bad[:20], "weekday_contaminated_n": len(wk_bad),
            "large_gaps_gt4d": gaps[:20], "large_gaps_n": len(gaps)}


def load_freeze_fingerprint():
    if not FREEZE_MANIFEST.exists():
        return None
    m = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    return m.get("data_epoch_fingerprint"), m.get("union_n"), m.get("union_sha256")


def verify_adjustment():
    if not ADJ_CACHE.exists():
        return {"error": "NO_ADJ_CACHE"}
    cache = json.loads(ADJ_CACHE.read_text(encoding="utf-8"))
    union = json.loads(UNION.read_text(encoding="utf-8")) if UNION.exists() else []
    # 失败日志
    fail_path = ADJ_CACHE.with_suffix(ADJ_CACHE.suffix + ".failures.jsonl")
    fails = []
    if fail_path.exists():
        for line in fail_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try: fails.append(json.loads(line))
                except Exception: pass
    fail_codes = {r.get("code") for r in fails if r.get("code")}
    status = Counter(v.get("status") for v in cache.values())
    missing = [c for c in union if c not in cache]
    # 逐码覆盖（OK 事件的日期跨度）
    cov = []
    for c, rec in cache.items():
        if rec.get("status") == "OK":
            evs = [int(d) for d, _ in rec.get("events", []) if d is not None]
            if evs:
                cov.append((c, min(evs), max(evs), len(evs)))
    max_event = max((mx for _, _, mx, _ in cov), default=None)
    min_event = min((mn for _, mn, _, _ in cov), default=None)
    return {
        "cache_total": len(cache),
        "ok": status.get("OK", 0), "empty": status.get("EMPTY", 0),
        "union_n": len(union),
        "missing_vs_union": len(missing), "missing_sample": missing[:20],
        "fail_distinct": len(fail_codes), "fail_sample": sorted(fail_codes)[:20],
        "coverage_ok_n": len(cov),
        "coverage_min_event_date": min_event, "coverage_max_event_date": max_event,
        "coverage_sample": cov[:10],
        "note": "逐候选 [P, W末日] 覆盖交叉核验 UNVERIFIED：1,013 候选实际存储表待定位（见候选表）。",
    }


def verify_price_scale():
    files = sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
    schema = None
    if files:
        schema = ds.dataset(files[0:1], format="parquet").schema.names
    # 重算【仅日线 parquet】逐文件 SHA256 并比对库存清单（内容哈希，非指纹）
    cross = []
    match = mismatch = unrecorded = 0
    inv_map = {}
    if INVENTORY.exists():
        inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
        for i in inv.get("input_manifest", []):
            if str(i.get("path", "")).lower().endswith(".parquet"):
                inv_map[i["path"]] = i.get("sha256")
    inventory_mixed = 0
    if INVENTORY.exists():
        inventory_mixed = sum(1 for i in json.loads(INVENTORY.read_text(encoding="utf-8"))
                              .get("input_manifest", []) if not str(i.get("path", "")).lower().endswith(".parquet"))
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        got = h.hexdigest(); exp = inv_map.get(p)
        if exp is None:
            unrecorded += 1; st = "UNRECORDED"
        elif got == exp:
            match += 1; st = "MATCH"
        else:
            mismatch += 1; st = "MISMATCH"
        cross.append({"file": os.path.basename(p), "status": st})
    return {"daily_part_schema_sample": schema,
            "actual_daily_part_files": len(files),
            "inventory_parquet_entries": len(inv_map),
            "inventory_non_data_entries_mixed": inventory_mixed,
            "sha256_match": match, "sha256_mismatch": mismatch, "sha256_unrecorded": unrecorded,
            "inventory_mixing_note": "库存清单 input_manifest 混入了 .py/.md/.json 等非数据文件；"
                                    "其 parquet 条目与实际文件数也可能不一致（陈旧）。应以本脚本重算的 39 文件为准。",
            "raw_claim": "UNVERIFIED_HYPOTHESIS：get_daily 为 parquet 优先+HTTP 兜底，未做价格尺度转换/验证；"
                         "原样返回记录≠返回未复权价格。参见 R0_V1_DEVELOPMENT.md §2（待验证假设）。",
            "crosscheck_sample": cross[:10]}


def verify_candidates():
    if not REGISTRY.exists():
        return {"error": "NO_REGISTRY"}
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    # 逐行绑定字段：registry 仅为汇总，无 P / decision_available_ts / 原始资格引用 / 规则版本
    return {
        "registry_version": reg.get("version"),
        "registry_date": reg.get("date"),
        "summary_counts": reg.get("counts"),
        "by_month": reg.get("by_month"),
        "daily_snapshot": reg.get("daily_snapshot"),
        "per_row_binding": {
            "original_eligibility_ref": "UNVERIFIED（实际存储表待定位，risk_candidate_source_reconciliation_v1.md §4.1 TODO）",
            "P_date": "UNVERIFIED（registry 无逐事件 P）",
            "candidate_rule_version": "UNVERIFIED（规则 SHA256 已记录于 reconciliation 文档，未绑定到逐行）",
            "decision_available_ts_status": "UNVERIFIED（无逐行到达时刻）",
        },
        "note": "DUAL_PATH=1,013 已识别但未物化逐行存储；不得用后续行动反推资格。",
    }


def main():
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "network_requests": 0,
        "calendar": compute_calendar(),
        "freeze_fingerprint": load_freeze_fingerprint(),
        "adjustment": verify_adjustment(),
        "price_scale": verify_price_scale(),
        "candidates": verify_candidates(),
    }
    # 打印摘要
    print("==== R0-V1 EXTERNAL EVIDENCE VERIFICATION ====")
    print(f"calendar: n_files={out['calendar'].get('n_files')} n_days={out['calendar'].get('n_trading_days')} "
          f"range={out['calendar'].get('min_date')}..{out['calendar'].get('max_date')}")
    print(f"current fingerprint: {out['calendar'].get('data_epoch_fingerprint')}")
    print(f"freeze  fingerprint: {out['freeze_fingerprint']}")
    adj = out["adjustment"]
    print(f"adjustment: ok={adj.get('ok')} empty={adj.get('empty')} missing={adj.get('missing_vs_union')} "
          f"fail={adj.get('fail_distinct')} cov_max_event={adj.get('coverage_max_event_date')}")
    ps = out["price_scale"]
    print(f"price_scale: schema={ps.get('daily_part_schema_sample')}")
    print(f"price_scale: daily_parts={ps.get('actual_daily_part_files')} "
          f"sha256 match={ps.get('sha256_match')} mismatch={ps.get('sha256_mismatch')} "
          f"unrecorded={ps.get('sha256_unrecorded')} "
          f"(inventory non-data entries mixed={ps.get('inventory_non_data_entries_mixed')})")
    cand = out["candidates"]
    print(f"candidates: registry_counts={cand.get('summary_counts')}")
    print(f"candidates per_row_binding: {cand.get('per_row_binding')}")
    out_path = HERE / "verify_external_evidence_output.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
