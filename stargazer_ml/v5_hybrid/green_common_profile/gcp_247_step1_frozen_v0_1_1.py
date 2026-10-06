# -*- coding: utf-8 -*-
"""Step1 冻结输入记录 (v0.1.1 缺槽修复工程): 合同/执行器/测试/247人口/日历/网格/日线/缓存 全量 SHA256.

零网络. 缓存清单复用 gcp_247_phase1_inventory.json (运行前校验其目录级计数与字节与当前文件系统一致).
产物: gcp_round_b_small_sample_v1/step1_frozen_inputs.json
"""
from __future__ import annotations
import sys, json, hashlib, glob, os, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D

BASE = HERE / "gcp_round_b_small_sample_v1"
RUN = BASE
MANIFEST = HERE / "contracts" / "dual_path_execution_v0_1_1_manifest.json"
OUT = RUN / "step1_frozen_inputs.json"
DAILY_PARTS = Path(os.environ.get("STARGAZER_DATA_ROOT", r"D:/A股数据") + r"\data_daily_parts")
EXECUTOR_PRE_FIX = "b835ac0cc4e3f6f2b77c554e69c72b791c4dfe5cbc6facfe6b6d50695cbc23ee"
REGISTERED_DIRS = ["minute_collect_realtest", "minute_ext", "minute", "minute_car", "deferral_collect_v2"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files_checked = []
    all_match = True
    for f in man["files"]:
        p = Path("D:/stargazer") / f["path"]
        actual = sha256_file(p) if p.exists() else None
        m = (actual == f["sha256"])
        all_match &= m
        files_checked.append({"role": f["role"], "path": f["path"], "expected": f["sha256"],
                              "actual": actual, "match": m, "bytes": p.stat().st_size if p.exists() else None})
    if not all_match:
        raise SystemExit("FROZEN INPUT MISMATCH: manifest 5 文件复验失败, 中止")

    # 缓存清单复验 (目录级计数/字节 与现存 inventory 一致)
    inv = json.loads((BASE / "gcp_247_phase1_inventory.json").read_text(encoding="utf-8"))
    cache_ok = True
    for name, d in inv["dirs"].items():
        if not d.get("registered"):
            continue
        p = BASE / name
        fps = sorted(p.rglob("*.parquet"))
        nbytes = sum(f.stat().st_size for f in fps)
        ok = (len(fps) == d["n_files"] and nbytes == d["total_bytes"])
        cache_ok &= ok
        d["recheck_at_step1"] = {"n_files_now": len(fps), "bytes_now": nbytes, "match": ok}
    if not cache_ok:
        raise SystemExit("FROZEN INPUT MISMATCH: 缓存目录与既有 inventory 不一致, 中止")

    # 日线/日历/网格
    parts = sorted(DAILY_PARTS.glob("part_*.parquet"))
    cal = D.load_calendar()
    grid = D.load_minute_grid()

    scripts = ["gcp_247_phase1.py", "gcp_247_event_coverage_verify.py",
               "dual_path_execution_v0_1_1.py", "dual_path_limited_reconcile.py",
               "dual_path_synthetic_acceptance.py", "dual_path_gap_regress_v0_1_1.py",
               "gcp_round_b_engine.py"]
    rec = {
        "phase": "GCP_247_STEP1_FROZEN_INPUTS_V0_1_1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0,
        "note": "缺槽修复工程执行前冻结输入; 事件级实际使用缓存文件名为 status_v0_1_1 逐事件 minute_cache_files, "
                "其 SHA256 在本文件 cache_registry 中按相对路径索引. 历史 inventory 未改动.",
        "manifest_verification": {"manifest_version": man["manifest_version"], "all_match": all_match,
                                  "files_checked": files_checked},
        "executor": {"path": "dual_path_execution_v0_1_1.py",
                     "sha256": sha256_file(HERE / "dual_path_execution_v0_1_1.py"),
                     "sha256_pre_fix": EXECUTOR_PRE_FIX,
                     "bytes": (HERE / "dual_path_execution_v0_1_1.py").stat().st_size},
        "related_scripts": [{"path": s, "sha256": sha256_file(HERE / s),
                             "bytes": (HERE / s).stat().st_size} for s in scripts],
        "population_247": {"path": "gcp_round_b_small_sample_v1/gcp_247_coverage_verify.json",
                           "sha256": [f for f in files_checked if f["role"] == "frozen_247_event_scope"][0]["actual"]},
        "calendar": {"n_days": len(cal), "first": cal[0], "last": cal[-1],
                     "data_cutoff": D.DATA_CUTOFF,
                     "sha256": hashlib.sha256(json.dumps(cal).encode()).hexdigest()},
        "minute_grid": {"path": "gcp_round_b_small_sample_v1/minute_grid.json",
                        "sha256": sha256_file(BASE / "minute_grid.json"),
                        "grid_1300_from": grid.get("grid_1300_from"),
                        "grid_n": len(grid.get("grid", [])), "grid_241_n": len(grid.get("grid_241", []))},
        "daily_parts": {"dir": str(DAILY_PARTS), "n_files": len(parts),
                        "total_bytes": sum(p.stat().st_size for p in parts),
                        "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                                  for p in parts]},
        "cache_registry": {
            "source": "gcp_247_phase1_inventory.json (已复验未变)",
            "dirs": inv["dirs"],
            "files": [{"file": r["file"], "bytes": r["bytes"], "sha256": r["sha256"],
                       "trusted": r.get("trusted")} for r in inv["files"]]},
        "budget_bytes_at_write": sum(f.stat().st_size for f in RUN.rglob("*") if f.is_file()),
    }
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"step1 written: {OUT.name} ({OUT.stat().st_size} bytes, {time.time()-t0:.1f}s)")
    print("manifest all_match:", all_match, "| cache recheck ok:", cache_ok)


if __name__ == "__main__":
    main()
