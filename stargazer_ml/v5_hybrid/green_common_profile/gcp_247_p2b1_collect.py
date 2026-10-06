# -*- coding: utf-8 -*-
"""第二阶段本批 (p2b1) 39 对定向采集: 登记校验 -> 有限重试采集 -> 严格 QA.

边界 (用户批准 2026-10-05):
  - 唯一请求来源 = gcp_247_phase2_needs_v0_1_1.json proposed_requests.list (39 对),
    非 108 对首阻全集; 执行前逐对校验 + 输入版本哈希复验, 落冻结清单.
  - 网络仅限本脚本内 39 对分钟请求 (stockdb HTTP), 除此之外零网络.
  - 独立缓存目录 p2b1_collect/by_pair (不覆盖任何已有目录);
    已有文件/无效文件清单覆盖的 (code,day) 不采 (批前复核).
  - 有限重试 (3); 源缺/空/结构非法/全零量 -> 留因不落盘; 不插值/不替补/不扩围/不重抓无效文件.
  - 严格 QA 全过后才允许 247 重判 (本脚本产出 QA 门, 重判另跑):
    逐文件 哈希 / 网格按日期版本 241-242 / 量价尺度 (分钟量汇总 vs 日线量, 价在日线高低内) /
    no-trade 语义 (全零量=源缺, 不落盘) / 与已有缓存不冲突.
  - 预算: 整个 gcp_round_b_small_sample_v1 <=50MiB, 含采集+快照+报告; 每写前检查.
"""
from __future__ import annotations
import sys, os, json, hashlib, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "step9_preflight"))
import data_source_provenance as PROV
import dual_path_execution_v0_1_1 as D
import pyarrow.dataset as ds

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_DIR = BASE / "p2b1_collect"
CACHE = OUT_DIR / "by_pair"
FROZEN_REQ = OUT_DIR / "frozen_request_list.json"
SNAPSHOT = OUT_DIR / "collection_snapshot.json"
REPORT = OUT_DIR / "collection_report.json"
MANIFEST = OUT_DIR / "collection_manifest.json"
NEEDS = BASE / "gcp_247_phase2_needs_v0_1_1.json"
STEP1 = BASE / "step1_frozen_inputs.json"
MANIFEST_CONTRACT = HERE / "contracts" / "dual_path_execution_v0_1_1_manifest.json"

BUDGET = 50 * 1024 * 1024
RETRIES = 3
REGISTERED_DIRS = ["minute_collect_realtest", "minute_ext", "minute", "minute_car", "deferral_collect_v2"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def check_budget(nbytes: int) -> bool:
    return dir_size(BASE) + nbytes <= BUDGET


def write_checked(path: Path, text: str):
    b = len(text.encode("utf-8"))
    if not check_budget(b):
        raise RuntimeError(f"budget exceeded before writing {path.name} (stop, no truncation)")
    path.write_text(text, encoding="utf-8")


# ---------- 1) 清单登记与校验 (零网络) ----------
def register_and_verify() -> list:
    needs = json.loads(NEEDS.read_text(encoding="utf-8"))
    step1 = json.loads(STEP1.read_text(encoding="utf-8"))
    # 输入版本哈希复验
    ver = []
    for f in json.loads(MANIFEST_CONTRACT.read_text(encoding="utf-8"))["files"]:
        p = Path("D:/stargazer") / f["path"]
        actual = sha256_file(p)
        ver.append({"role": f["role"], "match": actual == f["sha256"]})
        if actual != f["sha256"]:
            raise SystemExit(f"输入哈希变化: {f['role']}, 中止")
    for name, p, expect in [
            ("needs_v0_1_1", NEEDS, None),
            ("step1_frozen_inputs", STEP1, None),
            ("executor", HERE / "dual_path_execution_v0_1_1.py", None)]:
        ver.append({"role": name, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    ver.append({"role": "population_247", "match": sha256_file(BASE / "gcp_247_coverage_verify.json")
                == step1["population_247"]["sha256"]})
    if not all(v.get("match", True) for v in ver):
        raise SystemExit("输入复验存在不匹配, 中止")

    lst = needs["proposed_requests"]["list"]
    assert needs["proposed_requests"]["n_requests"] == 39 and len(lst) == 39, "清单不是 39 对"
    pairs, seen = [], set()
    for p in lst:
        code, day = str(p["code"]).zfill(6), int(p["day"])
        assert p["kinds"] == ["minute_day_missing"] and p.get("no_existing_file") is True, f"清单项口径异常: {p}"
        assert (code, day) not in seen, f"清单内重复: {code}@{day}"
        seen.add((code, day))
        pairs.append({"code": code, "day": day, "needed_by": p["needed_by"]})
    # 与 category_1 首阻集合一致性: 每对必须是 COLLECTABLE 首阻
    cat1 = {(q["code"].zfill(6) if isinstance(q["code"], str) else str(q["code"]).zfill(6), int(q["day"]))
            for q in needs["category_1_first_blockers"]["dedup_pairs"] if q["class"] == "COLLECTABLE"}
    for p in pairs:
        assert (p["code"], p["day"]) in cat1, f"{p['code']}@{p['day']} 不在 COLLECTABLE 首阻集合"
    # 批前复核: 任何已登记缓存目录均无该 (code,day) 覆盖文件 (含无效文件清单, 不重抓)
    conflicts = []
    for p in pairs:
        code, day = p["code"], p["day"]
        for dname in REGISTERED_DIRS:
            d = BASE / dname
            if (d / f"{code}_{day}.parquet").exists():
                conflicts.append({"code": code, "day": day, "file": str(d / f'{code}_{day}.parquet')})
            fcode = d / f"{code}.parquet"
            if fcode.exists():
                try:
                    df = pd.read_parquet(fcode, columns=["date"])
                    if int(day) in set(df["date"].astype(int) // 1000000) or int(day) in set(df["date"].astype(int)):
                        conflicts.append({"code": code, "day": day, "file": str(fcode), "note": "code-slice covers day"})
                except Exception:
                    pass
    frozen = {"phase": "GCP_247_P2B1_FROZEN_REQUESTS",
              "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "network_requests_budget": 39, "source": "gcp_247_phase2_needs_v0_1_1.json proposed_requests.list",
              "needs_sha256": sha256_file(NEEDS),
              "input_verification": ver,
              "n_pairs": len(pairs), "pairs": pairs,
              "pre_collect_conflicts": conflicts}
    if conflicts:
        # 与已有文件冲突的对 -> 不采 (注册留因); 若全部冲突则无事可做
        frozen["excluded_due_to_conflict"] = conflicts
        keep = [p for p in pairs if not any(c["code"] == p["code"] and c["day"] == p["day"] for c in conflicts)]
        frozen["pairs_to_collect"] = keep
    else:
        frozen["pairs_to_collect"] = pairs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_checked(FROZEN_REQ, json.dumps(frozen, ensure_ascii=False, indent=1))
    print(f"frozen list: {len(pairs)} pairs, conflicts={len(conflicts)}, "
          f"to_collect={len(frozen['pairs_to_collect'])}")
    return frozen["pairs_to_collect"]


# ---------- 2) 采集 ----------
def fetch_with_retry(code: str, day: int):
    last = None
    for a in range(1, RETRIES + 1):
        try:
            df = PROV.get_minute_http(code, str(day))
            return df, None, a
        except Exception as e:
            last = f"attempt{a}: {repr(e)[:160]}"
            time.sleep(0.3 * a)
    return None, last, RETRIES


def struct_validate(df, day: int):
    if df is None or len(df) == 0:
        return False, "empty_or_none"
    for c in ("date", "open", "high", "low", "close", "volume"):
        if c not in df.columns:
            return False, f"missing_col:{c}"
    try:
        ds_ = df["date"].astype(int)
    except Exception as e:
        return False, f"date_not_int:{repr(e)[:80]}"
    if not ((ds_ // 1000000) == int(day)).all():
        return False, "date_not_belong_to_day"
    if (df[["open", "high", "low", "close"]].astype(float) <= 0).any().any():
        return False, "nonpositive_price"
    if (df["volume"].astype(float) < 0).any():
        return False, "negative_volume"
    if (df["volume"].astype(float) > 0).sum() == 0:
        return False, "all_zero_volume_source_hole"
    return True, None


def collect(pairs_to_collect: list) -> list:
    CACHE.mkdir(parents=True, exist_ok=True)
    statuses = []
    for p in pairs_to_collect:
        code, day = p["code"], p["day"]
        target = CACHE / f"{code}_{day}.parquet"
        if target.exists():
            statuses.append({"code": code, "day": day, "status": "skipped_existing",
                             "bytes": target.stat().st_size, "sha256": sha256_file(target)})
            continue
        df, err, n_att = fetch_with_retry(code, day)
        if err is not None:
            statuses.append({"code": code, "day": day, "status": "fetch_failed",
                             "cause": err, "attempts": n_att})
            continue
        ok, why = struct_validate(df, day)
        if not ok:
            statuses.append({"code": code, "day": day, "status": "source_gap_or_invalid",
                             "cause": why, "attempts": n_att, "rows": int(len(df))})
            continue
        est = len(df) * 90 + 2048
        if not check_budget(est):
            statuses.append({"code": code, "day": day, "status": "budget_exceeded",
                             "cause": "would exceed 50MiB dir cap"})
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(target, index=False)
        statuses.append({"code": code, "day": day, "status": "collected",
                         "rows": int(len(df)), "attempts": n_att,
                         "bytes": target.stat().st_size, "sha256": sha256_file(target)})
    return statuses


# ---------- 3) 严格 QA (门) ----------
def qa(statuses: list) -> dict:
    # 日线对照 (量价尺度)
    codes = sorted({s["code"] for s in statuses if s["status"] in ("collected", "skipped_existing")})
    daily = {}
    if codes:
        tbl = ds.dataset(sorted(__import__("glob").glob(D.DAILY_PARTS + "/part_*.parquet")),
                         format="parquet").to_table(
            columns=["date", "code", "open", "high", "low", "close", "volume"],
            filter=ds.field("code").isin(codes)).to_pandas()
        tbl["date"] = tbl["date"].astype(int)
        daily = {(r["code"], int(r["date"])): r for _, r in tbl.iterrows()}
    qa_rows, grid_gaps = [], []
    for s in statuses:
        if s["status"] not in ("collected", "skipped_existing"):
            qa_rows.append({**s, "qa": "not_collected"})
            continue
        code, day = s["code"], s["day"]
        target = CACHE / f"{code}_{day}.parquet"
        try:
            df = pd.read_parquet(target)
        except Exception as e:
            qa_rows.append({**s, "qa": "unreadable", "reason": repr(e)[:160]})
            continue
        rec = {"code": code, "day": day, "bytes": target.stat().st_size,
               "sha256": sha256_file(target), "rows": int(len(df))}
        ok, why = struct_validate(df, day)
        rec["struct_ok"] = ok
        if not ok:
            rec["qa"] = "invalid"
            rec["reason"] = why
            qa_rows.append(rec)
            continue
        # 网格按日期版本
        exp = D.day_grid(day)
        slots = sorted({int(x) % 1000000 for x in df["date"].astype(int)})
        missing = sorted(set(exp) - set(slots))
        dup_n = len(slots) - len(set(slots))
        rec["expected_slots"] = len(exp)
        rec["grid_ok"] = (not missing) and dup_n == 0
        rec["grid_missing"] = missing
        rec["dup_slots"] = dup_n
        if missing:
            grid_gaps.append({"code": code, "day": day, "missing": missing})
        # 量价尺度
        dr = daily.get((code, day))
        if dr is None:
            rec["daily_ref"] = "no_daily_row"
            rec["vol_ratio"] = None
            rec["price_ok"] = None
        else:
            vol_ratio = float(df["volume"].astype(float).sum() / float(dr["volume"])) if float(dr["volume"]) > 0 else None
            rec["vol_ratio"] = round(vol_ratio, 4) if vol_ratio is not None else None
            rec["price_ok"] = bool((df[["open", "high", "low", "close"]].min().min() >= float(dr["low"]) * 0.999)
                                   and (df[["open", "high", "low", "close"]].max().max() <= float(dr["high"]) * 1.001))
        rec["zero_vol_rows"] = int((df["volume"] == 0).sum())
        # 冲突: 与已登记目录不得出现同 (code,day) 文件
        conflict = []
        for dname in REGISTERED_DIRS:
            if (BASE / dname / f"{code}_{day}.parquet").exists():
                conflict.append(dname)
        rec["conflict_with_existing"] = conflict
        rec["qa"] = "valid" if (rec["grid_ok"] and not conflict
                                and (rec["vol_ratio"] is None or 0.9 <= rec["vol_ratio"] <= 1.1)
                                and (rec["price_ok"] in (True, None))) else "invalid"
        if rec["qa"] == "invalid":
            rec["reason"] = "; ".join(x for x in [
                None if rec["grid_ok"] else f"grid_missing={missing[:8]}",
                "conflict" if conflict else None,
                None if (rec["vol_ratio"] is None or 0.9 <= rec["vol_ratio"] <= 1.1) else f"vol_ratio={rec['vol_ratio']}",
                None if (rec["price_ok"] in (True, None)) else "price_out_of_daily_range"] if x)
        qa_rows.append(rec)
    n_collected = sum(1 for s in statuses if s["status"] in ("collected", "skipped_existing"))
    qa_valid = sum(1 for q in qa_rows if q.get("qa") == "valid")
    gate = (qa_valid == n_collected) and n_collected > 0 or (n_collected == 0)
    return {"qa_rows": qa_rows, "n_collected": n_collected, "qa_valid": qa_valid,
            "grid_gaps": grid_gaps, "qa_gate_pass": (qa_valid == n_collected)}


def main():
    t0 = time.time()
    pairs = register_and_verify()
    statuses = collect(pairs)
    new_bytes = sum(s.get("bytes", 0) for s in statuses if s["status"] == "collected")
    snap = {"phase": "GCP_247_P2B1_COLLECTION",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "network_requests": sum(s.get("attempts", 0) for s in statuses),
            "network_scope": "stockdb HTTP 分钟k vals, 仅限本批 39 对",
            "cache_dir": str(CACHE), "canonical_source": PROV.CANONICAL_BASE,
            "retries_max": RETRIES, "dir_bytes_after": dir_size(BASE),
            "budget_bytes": BUDGET, "budget_ok": dir_size(BASE) <= BUDGET,
            "this_batch_new_bytes": new_bytes,
            "statuses": statuses,
            "by_status": {k: sum(1 for s in statuses if s["status"] == k)
                          for k in sorted({s["status"] for s in statuses})},
            "elapsed_s": round(time.time() - t0, 1)}
    write_checked(SNAPSHOT, json.dumps(snap, ensure_ascii=False, indent=1))
    res_qa = qa(statuses)
    report = {"summary": snap, "qa": res_qa}
    write_checked(REPORT, json.dumps(report, ensure_ascii=False, indent=1))
    write_checked(MANIFEST, json.dumps({
        "frozen_list": str(FROZEN_REQ), "frozen_list_sha256": sha256_file(FROZEN_REQ),
        "snapshot_sha256": sha256_file(SNAPSHOT), "report_sha256": sha256_file(REPORT),
        "cache_dir": str(CACHE), "canonical_source": PROV.CANONICAL_BASE,
        "qa_gate_pass": res_qa["qa_gate_pass"],
        "status": "QA_PASS_READY_FOR_REJUDGE" if res_qa["qa_gate_pass"] else "QA_FAIL_STOP"}, ensure_ascii=False, indent=1))
    print(f"collect: {snap['by_status']} new_bytes={new_bytes} dir={snap['dir_bytes_after']} "
          f"budget_ok={snap['budget_ok']} net_requests={snap['network_requests']} ({time.time()-t0:.1f}s)")
    print(f"QA: collected={res_qa['n_collected']} valid={res_qa['qa_valid']} "
          f"gate={'PASS' if res_qa['qa_gate_pass'] else 'FAIL'} grid_gaps={len(res_qa['grid_gaps'])}")
    if not res_qa["qa_gate_pass"]:
        raise SystemExit("QA 门未过 -> 停止, 不做 247 重判")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
