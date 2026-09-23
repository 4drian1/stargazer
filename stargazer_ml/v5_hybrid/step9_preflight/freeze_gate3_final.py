"""freeze_gate3_final — FINAL Gate3 Freeze (PROVISIONAL_VALID -> FINAL_FROZEN).

仅在 ② full coverage / ④ real trace / ⑤ Gate4 三项全部通过后才执行。
冻结以下组件 (不可再改):
  GREEN-WATCH-v1.1 / Candidate Contract / Feature Registry / Label Contract /
  Execution Engine / Adjustment Chain / Market Proxy / Zone-State Engine /
  Split Contract / Cost Contract / 25-100 亿 Universe / Data Epoch & source snapshot.

用法: python freeze_gate3_final.py --confirm --evidence <json>
  --evidence: 指向包含 ②/④/⑤ 通过结论的 JSON (由前序步骤产物构成)。
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
MANIFEST = _HERE / "GATE3_MANIFEST.yaml"

FROZEN_COMPONENTS = [
    "GREEN-WATCH-v1.1",
    "Candidate Contract",
    "Feature Registry",
    "Label Contract",
    "Execution Engine",
    "Adjustment Chain",
    "Market Proxy",
    "Zone/State Engine",
    "Split Contract",
    "Cost Contract",
    "25-100 亿 Universe",
    "Data Epoch / source snapshot",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="必须显式确认才执行冻结")
    ap.add_argument("--evidence", default=None, help="前序 ②/④/⑤ 通过结论 JSON 路径")
    args = ap.parse_args()

    if not args.confirm:
        print("[freeze] DRY-RUN: 未加 --confirm, 不写盘。以下为将执行的操作:")
        print(f"  manifest : {MANIFEST}")
        print(f"  status   : PROVISIONAL_VALID -> FINAL_FROZEN")
        print(f"  frozen   : {len(FROZEN_COMPONENTS)} 个组件")
        for c in FROZEN_COMPONENTS:
            print(f"    - {c}")
        return

    if not MANIFEST.exists():
        print(f"[freeze] ERROR: {MANIFEST} 不存在")
        sys.exit(1)

    doc = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    prev = doc.get("freeze_status")
    print(f"[freeze] current freeze_status = {prev}")

    evidence = {}
    if args.evidence:
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))

    # ---- 强制顺序 (用户 2026-09-18 裁定 #6): 必须先 freeze.py 刷新哈希并通过 verify,
    #      否则会出现 "FINAL_FROZEN + stale component hashes" 的假 freeze ----
    sys.path.insert(0, str(_HERE))
    import freeze as _freeze  # noqa
    print("[freeze] running freeze.verify() (component_hashes 必须全部 CURRENT) ...")
    if not _freeze.verify():
        print("[freeze] ABORT: component_hashes 校验未通过。"
              "请先运行 `python freeze.py` 刷新哈希, 再 `python freeze.py --verify`。")
        sys.exit(2)
    if not evidence:
        print("[freeze] ABORT: 必须提供 --evidence (gate3_final_evidence.json)。")
        sys.exit(3)
    if evidence.get("ALL_GREEN") is not True:
        print(f"[freeze] ABORT: evidence.ALL_GREEN = {evidence.get('ALL_GREEN')} (必须为 true)。")
        sys.exit(4)
    g4v = ((evidence.get("gate4_real_harness") or {}).get("verdict"))
    if g4v != "PASS":
        print(f"[freeze] ABORT: Gate4 verdict = {g4v} (必须 PASS, 不接受 CONDITIONAL_PASS)。")
        sys.exit(5)

    doc["freeze_status"] = "FINAL_FROZEN"
    doc["freeze_status_note"] = ("② full coverage / ④ real trace / ⑤ Gate4 三项全部通过, "
                                 "由 PROVISIONAL_VALID 升为 FINAL_FROZEN (immutable)。")
    doc["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc["frozen_components"] = FROZEN_COMPONENTS
    doc["evidence"] = evidence
    doc["component_hashes_verified"] = True
    doc["component_hashes_verified_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc["snapshot_dir"] = "step9_final_frozen_snapshot/"
    doc["rollback_point"] = ("本仓库非 git: 唯一代码回滚点 = step9_final_frozen_snapshot/ "
                             "(含 SHA256SUMS.txt), 由 snapshot_step9_frozen.py 生成并只读化")
    doc["pending_components"] = {
        "split_contract": "FROZEN",
        "execution_engine": "FROZEN",
        "cost_model": "FROZEN (after A0-A4)",
    }

    MANIFEST.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[freeze] WROTE FINAL_FROZEN -> {MANIFEST}")
    print(f"[freeze] frozen components = {len(FROZEN_COMPONENTS)}")


if __name__ == "__main__":
    main()
