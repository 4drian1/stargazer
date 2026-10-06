# -*- coding: utf-8 -*-
"""规则 A (S1/S2) 逐候选触发记录冻结件导出 — 已有数据的格式化, 零网络零新计算.

从 v3 盘点 (gcp_247_full_population_inventory_v3.json) 导出 1,013 研究人口逐候选触发记录:
含 S1/S2 独立标志 (s1_ts/s2_ts/watch) 与互斥归属 (mutual/first_action/double_trigger).
绑定哈希: 本件 / v3 盘点 json / v3 脚本 / scan 引擎模块 (gcp_dual_path_coverage_probe.py).
产物: gcp_round_b_small_sample_v1/gcp_247_ruleA_trigger_records_v1.json
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "gcp_round_b_small_sample_v1"
OUT = BASE / "gcp_247_ruleA_trigger_records_v1.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    v3_p = BASE / "gcp_247_full_population_inventory_v3.json"
    v3 = json.loads(open(v3_p, encoding="utf-8").read())
    recs = [{
        "code": r["code"], "E": r["E"], "P": r["P"], "setup": r["setup"], "segment": r["segment"],
        "watch_flag": r.get("watch"), "s1_ts": r.get("s1_ts"), "s2_ts": r.get("s2_ts"),
        "mutual_status": r["mutual"],
        "first_action_ts": r.get("first_action"), "double_trigger": r.get("double_trigger", False),
        "action_kind": (r.get("reason") if r["mutual"] == "1_有行动" else None),
        "in_251": r.get("in_251"), "in_247": r.get("in_247"),
    } for r in v3["records"]]
    n_action = sum(1 for r in recs if r["mutual_status"] == "1_有行动")
    n_double = sum(1 for r in recs if r.get("double_trigger"))
    out = {"phase": "GCP_247_RULEA_TRIGGER_RECORDS_V1",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0, "formatting_only": True,
           "source": {"v3_inventory": {"file": str(v3_p), "sha256": sha256_file(v3_p)},
                      "v3_script": {"file": str(HERE / "gcp_247_full_population_inventory_v3.py"),
                                    "sha256": sha256_file(HERE / "gcp_247_full_population_inventory_v3.py")},
                      "scan_engine_module": {"file": str(HERE / "gcp_dual_path_coverage_probe.py"),
                                             "sha256": sha256_file(HERE / "gcp_dual_path_coverage_probe.py")},
                      "note": "scan_views/evaluate_calibers 冻结规则同码复用; 触发记录= v3 既有结果格式化导出"},
           "population_n": len(recs), "n_action": n_action, "n_double_trigger": n_double,
           "fields": ["code", "E", "P", "setup", "segment", "watch_flag", "s1_ts", "s2_ts",
                      "mutual_status", "first_action_ts", "double_trigger", "action_kind",
                      "in_251", "in_247"],
           "records": recs}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if dir_size(BASE) + len(text.encode("utf-8")) > 50 * 1024 * 1024:
        raise RuntimeError("budget exceeded (stop)")
    OUT.write_text(text, encoding="utf-8")
    print(f"trigger records: {len(recs)} (action {n_action}, double {n_double})")
    print("sha256:", sha256_file(OUT))


if __name__ == "__main__":
    main()
