import os

ROOT = r"d:/stargazer"
OLD_FLAT = "live_monitor_fingerprints_20261107_v167_flow.json.gz"
NEW_FLAT = "live_monitor_fingerprints_20261107_v168_flow.json.gz"
OLD_TAROT = "tarot_live_manifest_v155_20261107_rel.json.gz"
NEW_TAROT = "tarot_live_manifest_v156_20261107_rel.json.gz"

TARGETS = [
    "migration_support/demo_storage_recovery_20261107_v1/install_verified.py",
    "stargazer_desktop/sector_rule_scan.py",
    "stargazer_desktop/on_demand_model_worker.py",
    "stargazer_desktop/golden_stock_analysis.py",
    "stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py",
    "stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py",
    "stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py",
    "stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py",
    "stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py",
]

for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    t = open(p, encoding="utf-8").read()
    n = t.count(OLD_FLAT) + t.count(OLD_TAROT)
    t = t.replace(OLD_FLAT, NEW_FLAT).replace(OLD_TAROT, NEW_TAROT)
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(("updated %d ref: " % n) + rel)

# 复查：仓库运行时代码不得再引用旧清单名
import subprocess
left = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "build_candidate", "dist", "dist_candidate")]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            t = open(p, encoding="utf-8").read()
        except Exception:
            continue
        if OLD_FLAT in t or OLD_TAROT in t:
            left += 1
            print("STILL REFERENCED:", p)
print("remaining old refs:", left)
