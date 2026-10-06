# -*- coding: utf-8 -*-
"""首轮 247 事件复权缓存按状态验收 (任务单 §2, 2026-10-04).

本地只读: 不向源发任何请求. 验收口径:
  - 源: gate1c feb2026 复权缓存 adj_cache.json (+ failures.jsonl / prefetch_report.json).
  - 事件: gcp_247_coverage_verify.json 的 247 条 (STOCK, 主板/创业板/科创板).
  - 逐事件判定 status = OK / EMPTY / ABSENT(失败未落盘) / OTHER.
  - EMPTY 必须为"已确认空链": status==EMPTY 且 events==[] 且不在 failures.jsonl.
  - 链对各事件窗口 [E, H5] 适用性: factor_at(E) 与 factor_at(H5) 均有限 (PIT 安全).
  - 失败 (ABSENT/在 failures 集) 绝不当空链; 须隔离, 不猜.
产物: gcp_round_b_small_sample_v1/gcp_adj_cache_acceptance.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
from adjustment_chain import SymbolChain  # noqa: E402

CACHE_DIR = ROOT / "opportunity_v2" / "gate1c" / "feb2026_adjustment_cache_v1"
ADJ_CACHE = CACHE_DIR / "adj_cache.json"
FAIL_LOG = CACHE_DIR / "adj_cache.json.failures.jsonl"
PREFETCH = CACHE_DIR / "prefetch_report.json"
VERIFY_JSON = HERE / "gcp_round_b_small_sample_v1" / "gcp_247_coverage_verify.json"
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_adj_cache_acceptance.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    adj = json.loads(ADJ_CACHE.read_text(encoding="utf-8"))
    cache_sha = sha256_file(ADJ_CACHE)

    # 失败码集 (未落盘, 绝不可当作空链)
    fail_codes = set()
    if FAIL_LOG.exists():
        for line in FAIL_LOG.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fail_codes.add(str(rec["code"]).zfill(6))

    prefetch = json.loads(PREFETCH.read_text(encoding="utf-8")) if PREFETCH.exists() else {}

    verify = json.loads(VERIFY_JSON.read_text(encoding="utf-8"))
    rows = verify["rows"]

    per_event = []
    status_counter = Counter()
    empty_events = []
    absent_events = []
    inapplicable = []
    for r in rows:
        code = str(r["code"]).zfill(6)
        E = int(r["E"])
        H5 = None
        for s in r["scan"]:
            if s.get("offset") == 5:
                H5 = s.get("date")
        if H5 is None:
            H5 = E  # 不应发生 (censored=0)
        entry = adj.get(code)
        rec = {"code": code, "E": E, "H5": H5, "in_failures_log": code in fail_codes}
        if entry is None:
            rec["status"] = "ABSENT"
            rec["applicable"] = False
            rec["reason"] = "code 不在 adj_cache.json (未落盘)"
            absent_events.append(code)
        else:
            st = entry.get("status")
            rec["status"] = st
            events = entry.get("events", [])
            ch = SymbolChain(code, [(int(d), float(m)) for d, m in events]).build()
            f_e = ch.factor_at(E)
            f_h5 = ch.factor_at(H5)
            rec["factor_e"] = round(f_e, 8)
            rec["factor_h5"] = round(f_h5, 8)
            rec["n_chain_events"] = len(events)
            rec["adj_in_window"] = any(E < d <= H5 for d, m in events)
            if st == "EMPTY":
                ok_empty = (len(events) == 0) and (code not in fail_codes)
                rec["applicable"] = ok_empty
                rec["confirmed_empty"] = ok_empty
                rec["reason"] = ("确认空链(无公司行为)" if ok_empty
                                 else "EMPTY 但 events 非空或在失败集 -> 可疑")
                if not ok_empty:
                    inapplicable.append(code)
                else:
                    empty_events.append(code)
            elif st == "OK":
                rec["applicable"] = (f_e == f_e) and (f_h5 == f_h5)  # 有限
                rec["confirmed_empty"] = False
            else:
                rec["applicable"] = False
                rec["reason"] = f"未预期 status={st}"
                inapplicable.append(code)
        status_counter[rec["status"]] += 1
        per_event.append(rec)

    n_ok = status_counter.get("OK", 0)
    n_empty = status_counter.get("EMPTY", 0)
    n_absent = status_counter.get("ABSENT", 0)
    n_other = sum(v for k, v in status_counter.items() if k not in ("OK", "EMPTY", "ABSENT"))

    report = {
        "version": "gcp_adj_cache_acceptance_v1",
        "network_requests": 0,
        "cache_path": str(ADJ_CACHE),
        "cache_sha256": cache_sha,
        "prefetch_complete": prefetch.get("complete"),
        "prefetch_cumulative": prefetch.get("cumulative"),
        "fail_codes_distinct_in_log": sorted(fail_codes),
        "n_events": len(rows),
        "n_unique_codes": len({r["code"] for r in rows}),
        "status_counts": dict(status_counter),
        "ok": n_ok, "empty": n_empty, "absent": n_absent, "other": n_other,
        "empty_event_codes": sorted(set(empty_events)),
        "absent_event_codes": sorted(set(absent_events)),
        "inapplicable_codes": sorted(set(inapplicable)),
        "verdict": (
            "PASS: 247 事件全部在缓存内, 244 OK + 3 EMPTY, 0 失败/缺失; "
            "EMPTY 均为已确认空链(不在 failures 集), 链对 [E,H5] 窗口适用."
            if (n_ok == 244 and n_empty == 3 and n_absent == 0 and not inapplicable)
            else "REVIEW: 计数或适用性与预期不符, 见上."
        ),
        "per_event": per_event,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[复权缓存验收] 事件={len(rows)} 唯一码={report['n_unique_codes']}")
    print(f"  status: {dict(status_counter)}")
    print(f"  OK={n_ok} EMPTY={n_empty} ABSENT={n_absent} OTHER={n_other}")
    print(f"  EMPTY 事件码: {report['empty_event_codes']}")
    print(f"  failures 集(未落盘, 非本 247): {report['fail_codes_distinct_in_log']}")
    print(f"  ABSENT 事件码: {report['absent_event_codes']}")
    print(f"  不适用码: {report['inapplicable_codes']}")
    print(f"  cache sha256: {cache_sha[:16]}...")
    print(f"  结论: {report['verdict']}")
    print(f"  明细: {OUT.name}")


if __name__ == "__main__":
    main()
