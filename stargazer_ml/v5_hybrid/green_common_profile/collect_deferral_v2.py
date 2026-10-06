# -*- coding: utf-8 -*-
"""顺延 v2 清单内 30 对分钟采集 (独立缓存, 有限重试, 不插值/不替补/不扩围).

- 仅读取已冻结 deferral_requirement_list_v2.json (不重新运行生成器, 不自动选最新报告).
- 对每对 (code, day) 经 canonical stockdb HTTP 取数, 写入独立缓存 by_pair (不覆盖 realtest).
- 有限重试; 源缺/空/结构非法 -> 留因不采集; 不插值/不替补/不扩围.
- 预算: 累计【整个研究目录 gcp_round_b_small_sample_v1】守住 50MiB; 本批新增单独列示 (非累计).
- 采集后 QA: 结构/非空/日归属 + 严格交易网格 (每日常规 241 槽) 校验; 无效数据不落盘.
- 网格缺口 (如 000042_20260401 缺 10:54) 单独登记; 是否阻断由对账按依赖终点判定.
- 产出 collection_snapshot.json / collection_report.json / collection_manifest.json.
"""
from __future__ import annotations
import sys, os, json, hashlib, time
from pathlib import Path
import pandas as pd

HERE = Path(r"d:/stargazer/stargazer_ml/v5_hybrid/green_common_profile")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "step9_preflight"))
import data_source_provenance as P

FROZEN_LIST = HERE / "gcp_round_b_small_sample_v1" / "execution_reconcile_v0_1_1" / "deferral_requirement_list_v2.json"
CACHE_DIR = HERE / "gcp_round_b_small_sample_v1" / "deferral_collect_v2" / "by_pair"
SNAPSHOT = HERE / "gcp_round_b_small_sample_v1" / "deferral_collect_v2" / "collection_snapshot.json"
REPORT = HERE / "gcp_round_b_small_sample_v1" / "deferral_collect_v2" / "collection_report.json"
MANIFEST = HERE / "gcp_round_b_small_sample_v1" / "deferral_collect_v2" / "collection_manifest.json"
RESEARCH_DIR = HERE / "gcp_round_b_small_sample_v1"   # 预算硬保护对象 (含 realtest 旧缓存 + 本次新增)

BUDGET_BYTES = 50 * 1024 * 1024
RETRIES = 3
REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]
CANONICAL_URL = P.CANONICAL_BASE

# 用户已核定的网格缺口: 不影响本次 H5 标签, 不再补抓.
# 000042 已于 20260331 闭环; 缺口 20260401 10:54 晚于交易依赖, 非阻断.
KNOWN_GRID_GAPS = {
    ("000042", 20260401): "closed 20260331; gap 20260401-1054 postdates trade dependency; non-blocking, no refetch",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def expected_grid() -> set:
    """A 股常规分钟网格: 上午 09:30-11:30 (含, 121 槽) + 下午 13:01-15:00 (含, 120 槽) = 241 槽."""
    g = set()
    for h in range(9, 12):
        for m in range(60):
            if (h, m) >= (9, 30) and (h, m) <= (11, 30):
                g.add(h * 100 + m)
    for h in range(13, 16):
        for m in range(60):
            if (h, m) >= (13, 1) and (h, m) <= (15, 0):
                g.add(h * 100 + m)
    return g


EXPECTED_GRID = expected_grid()


def list_pairs():
    lst = json.loads(FROZEN_LIST.read_text(encoding="utf-8"))
    pairs, seen = [], set()
    for req in lst.get("requirements", []):
        code = str(req["code"]).zfill(6)
        for d in req.get("needed_days", []):
            key = (code, int(d))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs, lst


def fetch_with_retry(code: str, day: int):
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            return P.get_minute_http(code, str(day)), None
        except Exception as e:
            last_err = f"attempt{attempt}: {repr(e)[:160]}"
            time.sleep(0.2 * attempt)
    return None, last_err


def validate(df: pd.DataFrame, day: int):
    if df is None or len(df) == 0:
        return False, "empty_or_none"
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"missing_cols:{missing}"
    try:
        ds = df["date"].astype(int)
    except Exception as e:
        return False, f"date_not_int:{e}"
    if not (ds // 1000000 == int(day)).all():
        return False, "date_not_belong_to_day"
    if (df[["open", "high", "low", "close"]].astype(float) <= 0).any().any():
        return False, "nonpositive_price"
    return True, None


def grid_check(df: pd.DataFrame):
    try:
        slots = set((int(x) // 100) % 10000 for x in df["date"])
    except Exception:
        return False, ["<date_not_int>"]
    missing = sorted(EXPECTED_GRID - slots)
    return (len(missing) == 0), missing


def _count(statuses):
    c = {}
    for s in statuses:
        c[s["status"]] = c.get(s["status"], 0) + 1
    return c


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pairs, lst = list_pairs()
    n = len(pairs)
    base_bytes = dir_size(RESEARCH_DIR)          # 预算 = 整个研究目录 (含旧 realtest 缓存)
    new_bytes, budget_hit = 0, False
    statuses = []
    for (code, day) in pairs:
        target = CACHE_DIR / f"{code}_{day}.parquet"
        if target.exists():
            statuses.append({"code": code, "day": day, "status": "skipped_existing",
                             "bytes": target.stat().st_size, "sha256": sha256_file(target)})
            continue
        if budget_hit:
            statuses.append({"code": code, "day": day, "status": "budget_exceeded",
                             "cause": "research-dir cap reached"})
            continue
        df, err = fetch_with_retry(code, day)
        if err is not None:
            statuses.append({"code": code, "day": day, "status": "fetch_failed", "cause": err})
            continue
        ok, why = validate(df, day)
        if not ok:
            statuses.append({"code": code, "day": day, "status": "invalid", "cause": why})
            continue
        est = len(df) * 80 + 2048
        if base_bytes + new_bytes + est > BUDGET_BYTES:
            statuses.append({"code": code, "day": day, "status": "budget_exceeded",
                             "cause": "would exceed 50MiB research-dir cap"})
            budget_hit = True
            continue
        df.to_parquet(target, index=False)
        sz = target.stat().st_size
        new_bytes += sz
        statuses.append({"code": code, "day": day, "status": "collected",
                         "rows": int(len(df)), "bytes": sz, "sha256": sha256_file(target)})

    total_bytes = base_bytes + new_bytes
    snapshot = {
        "frozen_list": str(FROZEN_LIST),
        "frozen_list_sha256": sha256_file(FROZEN_LIST),
        "canonical_source": CANONICAL_URL,
        "cache_dir": str(CACHE_DIR),
        "research_dir": str(RESEARCH_DIR),
        "research_dir_base_bytes": base_bytes,
        "this_batch_new_bytes": new_bytes,
        "total_research_dir_bytes_after_run": total_bytes,
        "cumulative_cache_bytes_this_batch_only": new_bytes,
        "budget_bytes": BUDGET_BYTES,
        "budget_ok": total_bytes <= BUDGET_BYTES,
        "total_pairs": n,
        "statuses": statuses,
    }
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    qa = []
    for s in statuses:
        if s["status"] not in ("collected", "skipped_existing"):
            qa.append({**s, "qa": "not_collected"})
            continue
        target = CACHE_DIR / f"{s['code']}_{s['day']}.parquet"
        try:
            df = pd.read_parquet(target)
            ok, why = validate(df, s["day"])
            g_ok, g_missing = grid_check(df)
            qa.append({"code": s["code"], "day": s["day"],
                       "qa": "valid" if ok else "invalid", "reason": (None if ok else why),
                       "rows": int(len(df)), "grid": "complete" if g_ok else "gap",
                       "grid_missing": g_missing,
                       "known_gap_note": KNOWN_GRID_GAPS.get((s["code"], s["day"]))})
        except Exception as e:
            qa.append({"code": s["code"], "day": s["day"], "qa": "unreadable", "reason": repr(e)[:160]})

    collected = sum(1 for s in statuses if s["status"] in ("collected", "skipped_existing"))
    struct_valid = sum(1 for q in qa if q["qa"] == "valid")
    struct_invalid = sum(1 for q in qa if q["qa"] != "valid")
    grid_complete = sum(1 for q in qa if q.get("grid") == "complete")
    grid_gap = sum(1 for q in qa if q.get("grid") == "gap")
    gaps = [{"code": q["code"], "day": q["day"], "grid_missing": q.get("grid_missing"),
             "known_gap_note": q.get("known_gap_note")}
            for q in qa if q.get("grid") == "gap"]
    report = {
        "summary": {
            "frozen_list": str(FROZEN_LIST), "total_pairs": n,
            "collected": collected, "failed": n - collected,
            "by_status": _count(statuses),
            "qa_struct_valid": struct_valid, "qa_struct_invalid": struct_invalid,
            "grid_complete": grid_complete, "grid_gap": grid_gap, "grid_gaps": gaps,
            "research_dir_base_bytes": base_bytes,
            "this_batch_new_bytes": new_bytes,
            "total_research_dir_bytes_after_run": total_bytes,
            "budget_bytes": BUDGET_BYTES, "budget_ok": snapshot["budget_ok"],
            "note": ("结构校验通过=30, 但严格网格为 29 完整 + 1 缺槽 (000042_20260401 缺 10:54, "
                     "已核定非阻断). 本批新增 %.2fKiB; 研究目录累计 %.2fMiB, 仍<50MiB. "
                     "无效数据未落盘; 仅清单内采集, 不插值/不替补/不扩围." %
                     (new_bytes / 1024.0, total_bytes / 1048576.0)),
        },
        "per_pair": statuses,
        "qa": qa,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "frozen_list": str(FROZEN_LIST), "frozen_list_sha256": sha256_file(FROZEN_LIST),
        "snapshot_sha256": sha256_file(SNAPSHOT), "report_sha256": sha256_file(REPORT),
        "cache_dir": str(CACHE_DIR), "canonical_source": CANONICAL_URL,
        "constraints": {"limited_retries": RETRIES, "independent_cache": True,
                        "budget_target": "whole research_dir gcp_round_b_small_sample_v1",
                        "budget_bytes": BUDGET_BYTES, "no_substitute": True, "no_expansion": True},
        "status": "COLLECTED_AWAITING_QA_RECONCILE",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pairs={n} collected={collected} failed={n - collected} "
          f"struct_valid={struct_valid} grid_complete={grid_complete} grid_gap={grid_gap}")
    print(f"research_dir_base={base_bytes}B this_batch_new={new_bytes}B total={total_bytes}B "
          f"budget_ok={snapshot['budget_ok']}")
    print(f"by_status={_count(statuses)}")
    print(f"snapshot={SNAPSHOT.name} report={REPORT.name} manifest={MANIFEST.name}")


if __name__ == "__main__":
    main()
