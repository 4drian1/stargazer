# -*- coding: utf-8 -*-
"""双路径执行 v0.1.1 有限事件离线对账 (§9): 12 普通 + 4 隔离对照.

步骤:
  1. 校验 5 个文件 SHA256 与清单一致 (§11 交付校验).
  2. 对 16 个事件调用 execute_event (新执行路径, 不做旧 y/f).
  3. 分类状态并停止 (不展开 247 收益复放 / 训练).
产物: gcp_round_b_small_sample_v1/execution_reconcile_v0_1_1/<run_id>/{manifest_verification,limited_reconciliation,summary}.json + report.md
"""
from __future__ import annotations
import sys, json, hashlib, glob, os
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D

# 可选: 顺延补采的独立缓存目录 (清单内 30 对分钟). 仅追加为读取路径, 不覆盖 realtest 旧文件.
import argparse
_argv = argparse.ArgumentParser(add_help=False)
_argv.add_argument("--deferral-cache-dir", default=None)
_args, _ = _argv.parse_known_args()
if _args.deferral_cache_dir and Path(_args.deferral_cache_dir).exists():
    D.MIN_DIRS.insert(0, Path(_args.deferral_cache_dir))

MANIFEST = HERE / "contracts" / "dual_path_execution_v0_1_1_manifest.json"
BASE = Path("D:/stargazer")
OUT_ROOT = HERE / "gcp_round_b_small_sample_v1" / "execution_reconcile_v0_1_1"

CONTROL_BOARD = {"688656": "STAR", "002424": "SZ_MAIN", "603395": "SH_MAIN"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest() -> dict:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    res = {"manifest_version": man.get("manifest_version"), "files_checked": [], "all_match": True}
    for f in man["files"]:
        p = BASE / f["path"]
        if not p.exists():
            res["files_checked"].append({"role": f["role"], "path": f["path"], "present": False})
            res["all_match"] = False
            continue
        actual = sha256_of(p)
        match = (actual == f["sha256"])
        res["files_checked"].append({"role": f["role"], "path": f["path"],
                                     "expected": f["sha256"], "actual": actual, "match": match})
        if not match:
            res["all_match"] = False
    return res


def classify(res: dict) -> str:
    if res.get("status") != "EXECUTED":
        return res.get("status", "UNKNOWN")
    h3, h5 = res.get("H3", {}), res.get("H5", {})
    st = {h3.get("status"), h5.get("status")}
    if st <= {"SIM_COMPLETE"}:
        return "COMPLETE"
    if "EXIT_DATA_MISSING" in st or "CENSORED_TAIL" in st:
        return "INCOMPLETE"
    if st <= {"EXIT_DEFERRED_CAP"}:
        return "DEFERRED_CAP"
    return "MIXED"


def main():
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mv = verify_manifest()
    (out_dir / "manifest_verification.json").write_text(json.dumps(mv, indent=2), encoding="utf-8")

    events = []
    for e in man["ordinary_candidates"]:
        events.append({"code": e["code"], "E": e["E"], "board": e["board"], "setup": e["setup"],
                       "segment": e["segment"], "role": "ORDINARY", "S": e.get("S"),
                       "known_issue": None})
    for c in man["isolation_controls"]:
        events.append({"code": c["code"], "E": c["E"], "board": CONTROL_BOARD.get(c["code"], "SH_MAIN"),
                       "setup": None, "segment": None, "role": "CONTROL",
                       "known_issue": c.get("known_issue"), "S": None})

    results = []
    for ev in events:
        try:
            # Bug2: 普通事件使用冻结双路径信号 (manifest 的 S); 隔离对照无冻结 S 时回退 detect_signal
            sig = ev["S"] if ev["role"] == "ORDINARY" else None
            # Bug4: is_st=None -> 从日线 is_st 列按日派生, 不硬编码 False
            r = D.execute_event(ev["code"], ev["E"], ev["board"], None, ev["setup"], signal_ts=sig)
        except Exception as ex:
            r = {"code": ev["code"], "E": ev["E"], "status": "EXECUTE_ERROR", "error": repr(ex)[:300]}
        r["_role"] = ev["role"]
        r["_known_issue"] = ev["known_issue"]
        r["_class"] = classify(r)
        results.append(r)

    summary = {
        "run_id": run_id,
        "manifest_all_match": mv["all_match"],
        "n_events": len(results),
        "n_ordinary": sum(1 for r in results if r["_role"] == "ORDINARY"),
        "n_control": sum(1 for r in results if r["_role"] == "CONTROL"),
        "class_counts": {},
        "ordinary_classes": {},
        "control_classes": {},
    }
    for r in results:
        summary["class_counts"][r["_class"]] = summary["class_counts"].get(r["_class"], 0) + 1
        bucket = summary["ordinary_classes"] if r["_role"] == "ORDINARY" else summary["control_classes"]
        bucket[r["_class"]] = bucket.get(r["_class"], 0) + 1

    # 精简结果 (去除巨型原始依赖, 仅留关键字段)
    slim = []
    for r in results:
        dep = r.get("dependencies", {})
        slim.append({
            "code": r.get("code"), "E": r.get("E"), "role": r["_role"], "board": r.get("board"),
            "setup": r.get("setup"), "status": r.get("status"), "class": r["_class"],
            "known_issue": r["_known_issue"],
            "signal_ts": r.get("signal_ts"), "signal_source": r.get("signal_source"),
            "signal_type": r.get("signal_type"), "is_st_source": r.get("is_st_source"),
            "atr_ref": r.get("atr_ref"),
            "entry": r.get("entry"),
            "n_daily_parts": len(dep.get("daily_parts", [])) if isinstance(dep, dict) else None,
            "n_minute_files": len(dep.get("minute_files", [])) if isinstance(dep, dict) else None,
            "H3": {k: r.get("H3", {}).get(k) for k in ("status", "reason", "net", "fully_exited",
                     "planned_exit_date", "deferral_d0", "remaining_shares", "close_ref_Hn",
                     "label_end_ts", "last_fill_ts", "observed_endpoint_ts", "missing_dependencies",
                     "sells", "C_buy", "C_sell", "pnl")},
            "H5": {k: r.get("H5", {}).get(k) for k in ("status", "reason", "net", "fully_exited",
                     "planned_exit_date", "deferral_d0", "remaining_shares", "close_ref_Hn",
                     "label_end_ts", "last_fill_ts", "observed_endpoint_ts", "missing_dependencies",
                     "sells", "C_buy", "C_sell", "pnl")},
            "error": r.get("error"),
        })

    (out_dir / "limited_reconciliation.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # report.md
    lines = [f"# 双路径执行 v0.1.1 有限事件对账报告 ({run_id})", "",
             f"- 清单文件一致性: {'全部匹配' if mv['all_match'] else '存在不一致(见 manifest_verification.json)'}",
             f"- 事件总数: {summary['n_events']} (普通 {summary['n_ordinary']} / 隔离对照 {summary['n_control']})",
             f"- 状态分类: {summary['class_counts']}", "",
             "## 普通事件", "", "| code | E | board | setup | status | class | H3 | H5 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in slim:
        if r["role"] != "ORDINARY":
            continue
        h3 = r["H3"]; h5 = r["H5"]
        lines.append(f"| {r['code']} | {r['E']} | {r['board']} | {r['setup']} | {r['status']} | {r['class']} | "
                     f"{h3.get('status')} | {h5.get('status')} |")
    lines += ["", "## 隔离对照", "", "| code | E | board | known_issue | status | class |",
              "|---|---|---|---|---|---|"]
    for r in slim:
        if r["role"] != "CONTROL":
            continue
        lines.append(f"| {r['code']} | {r['E']} | {r['board']} | {r['known_issue']} | {r['status']} | {r['class']} |")
    lines += ["", "## 结论", "",
              "- 本对账仅覆盖 16 事件; 全 247 收益复放与训练仍待对账结果裁决 (未执行).",
              "- 普通事件若 class=INCOMPLETE/DEFERRED_CAP, 因持仓窗口内分钟缺失或 5 日顺延上限, 净收益保持 NULL.",
              "- 隔离对照预期为 ISOLATED/NULL, 用于校验隔离逻辑而非产出收益."]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"对账完成 run_id={run_id}")
    print(f"清单一致: {mv['all_match']}")
    print(f"分类: {summary['class_counts']}")
    print(f"产物目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
