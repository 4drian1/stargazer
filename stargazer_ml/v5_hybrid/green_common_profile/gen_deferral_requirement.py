# -*- coding: utf-8 -*-
"""六条普通(ORDINARY)事件的顺延路径采集需求清单生成器 (v2: 基于实际 residual 轨道 + deferral_d0).

依据: 仍有残仓的轨道(remaining_shares>0) 的 实际首次退出指令日 deferral_d0 (非计划退出日),
交易所日历。顺延窗口 = deferral_d0+1 .. deferral_d0+5 交易日。扣除现有分钟缓存, 登记哈希与预算。
仅生成清单 + 登记(清单字节 SHA256 / 输入报告 / 执行器版本), 不发起采集。待批准后再按 needed 定向补采。
"""
import sys, os, json, hashlib, glob
from pathlib import Path
sys.path.insert(0, r"d:/stargazer/stargazer_ml/v5_hybrid/green_common_profile")
import dual_path_execution_v0_1_1 as D

HERE = Path(r"d:/stargazer/stargazer_ml/v5_hybrid/green_common_profile")
RECON_DIR = HERE / "gcp_round_b_small_sample_v1" / "execution_reconcile_v0_1_1"
ENGINE_PATH = HERE / "dual_path_execution_v0_1_1.py"
OUT = RECON_DIR / "deferral_requirement_list_v2.json"
MANIFEST_OUT = RECON_DIR / "deferral_requirement_list_v2.manifest.json"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 自动选取最新一次有限对账
runs = sorted(glob.glob(str(RECON_DIR / "run_*" / "limited_reconciliation.json")))
if not runs:
    raise SystemExit("未找到任何有限对账结果")
RUN = runs[-1]
run_path = Path(RUN)

cal = D.load_calendar()
ci = {d: i for i, d in enumerate(cal)}
run = json.load(open(RUN, encoding="utf-8"))

# 估算 by_pair 单文件平均体积 (预算)
by_pair_dir = HERE / "gcp_round_b_small_sample_v1" / "minute_collect_realtest" / "by_pair"
sizes = [os.path.getsize(os.path.join(by_pair_dir, f)) for f in os.listdir(by_pair_dir) if f.endswith(".parquet")]
avg_kb = (sum(sizes) / len(sizes) / 1024.0) if sizes else 10.0

reqs = []
seen = set()  # (code, day) 去重
for r in run:
    if r.get("role") != "ORDINARY":
        continue
    if r.get("class") != "INCOMPLETE":
        continue
    code = r["code"]; E = r["E"]
    for trk in ("H3", "H5"):
        t = r.get(trk, {})
        residual = t.get("remaining_shares", 0)
        if not residual or residual <= 0:
            continue
        if t.get("status") not in ("EXIT_DATA_MISSING", "EXIT_DEFERRED_CAP"):
            continue
        # FixC: 实际首次退出指令日 (非计划退出日); 提前止损时两者不同
        d0 = t.get("deferral_d0")
        if d0 is None:
            continue
        ei = ci[E]
        di = ci[d0]
        window = [cal[di + k] for k in range(1, D.DEFER_MAX_DAYS + 1) if di + k < len(cal)]
        cached, needed = [], []
        bd, prov = D.load_minute_code(code)
        for d in window:
            if bd is not None and d in bd and bd[d] is not None:
                day_files = [{"path": p["path"], "sha256": p["sha256"], "naming": p.get("naming"), "day": p.get("day")}
                             for p in prov if p.get("day") == d]
                cached.append({"date": d, "files": day_files})
            else:
                if (code, d) not in seen:
                    needed.append(d); seen.add((code, d))
        reqs.append({
            "code": code, "board": r.get("board"), "E": E, "setup": r.get("setup"),
            "track": trk, "signal_source": r.get("signal_source"),
            "deferral_d0": d0, "planned_exit_date": t.get("planned_exit_date"),
            "residual_shares": residual, "deferral_window": window,
            "cached_days": [c["date"] for c in cached],
            "cached_file_hashes": cached,
            "needed_days": needed,
            "needed_file_est_kb": round(avg_kb, 2),
            "budget_kb": round(avg_kb * len(needed), 2),
        })

total_needed = sum(len(x["needed_days"]) for x in reqs)
total_budget = round(sum(x["budget_kb"] for x in reqs), 2)
summary = {
    "generated_from_run": run_path.name,
    "ordinary_incomplete_count": len([r for r in run if r.get("role") == "ORDINARY" and r.get("class") == "INCOMPLETE"]),
    "residual_tracks_count": len(reqs),
    "defer_window_rule": "deferral_d0+1 .. deferral_d0+5 (deferral_d0 = actual first exit-instruction day, NOT planned_exit_date)",
    "avg_file_kb_est": round(avg_kb, 2),
    "total_needed_pairs": total_needed,
    "total_budget_kb": total_budget,
    "note": "仅清单+登记, 未发起采集. 待批准后再按 needed_days 定向补采 (有限重试/独立缓存/累计<=50MB/不替补/不扩围).",
    "requirements": reqs,
}
json.dump(summary, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
list_sha = sha256_of(OUT)
manifest = {
    "list_file": OUT.name,
    "list_sha256": list_sha,
    "input_report": {"path": str(run_path), "sha256": sha256_of(run_path)},
    "executor_version": {"file": ENGINE_PATH.name, "sha256": sha256_of(ENGINE_PATH)},
    "collection_constraints": {
        "limited_retries": True, "independent_cache": True,
        "cumulative_cap_bytes": 50 * 1024 * 1024, "no_substitute": True, "no_expansion": True,
    },
    "status": "PENDING_APPROVAL",
}
json.dump(manifest, open(MANIFEST_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("RUN:", run_path.name)
print("RESIDUAL TRACKS:", len(reqs))
for x in reqs:
    print(f"  {x['code']}@{x['E']} {x['track']} d0={x['deferral_d0']} (plan={x['planned_exit_date']}) "
          f"residual={x['residual_shares']} window={x['deferral_window']} cached={x['cached_days']} "
          f"needed={x['needed_days']} budget={x['budget_kb']}KB")
print(f"TOTAL needed_pairs={total_needed}  budget={total_budget}KB")
print("list_sha256:", list_sha)
print("written:", OUT)
print("manifest:", MANIFEST_OUT)
