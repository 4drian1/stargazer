"""snapshot_step9_frozen — FINAL Gate3 时刻的真实代码回滚点 (用户 2026-09-18 裁定 #7).

问题: 本仓库不是 git 仓库 -> FINAL_FROZEN 若只有 YAML 意义, 就没有任何代码回滚点。
做法: 在冻结那一刻把全部被冻结的组件复制到
      step9_final_frozen_snapshot/  (通过 GATE3_MANIFEST.component_hashes 枚举, 不手抄名单)
      并生成 SHA256SUMS.txt; 复制后逐一复算 sha256 与源文件比对 (自证无复制损坏)。

只读化: 尽力去掉写权限 (Windows attrib +R); 失败不致命, 只在报告里记录。

运行:
  python snapshot_step9_frozen.py --confirm           # 需要 manifest 已是 FINAL_FROZEN
  python snapshot_step9_frozen.py                     # DRY-RUN (只打印将复制的清单)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
V5 = Path("D:/stargazer/stargazer_ml/v5_hybrid")
MANIFEST = _HERE / "GATE3_MANIFEST.yaml"
SNAP = _HERE / "step9_final_frozen_snapshot"

# 除 component_hashes 之外, 一并快照的"实验世界描述与结论"文件
EXTRA_DESCRIPTIVE = [
    _HERE / "GATE3_MANIFEST.yaml",
    _HERE / "GATE3_MANIFEST.yaml",  # 去重后保留
    _HERE / "gate3_final_evidence.json",
    _HERE / "gate4_real_report.json",
    _HERE / "trace_04_report.json",
    _HERE / "coverage_02_summary.json",
    _HERE / "scenario_cert.json",
    _HERE / "float_mv_probe_summary.json",
    _HERE / "DATA_GATE1A_FULL.yaml",
    _HERE / "freeze_gate3_final.py",
    _HERE / "snapshot_step9_frozen.py",
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_of(key: str) -> Path:
    """把 component_hashes 的 key 映射回源文件 (与 freeze.py COMPONENT_FILES 同构)。"""
    import freeze as F  # noqa
    p = F.COMPONENT_FILES.get(key)
    if p is None:
        return None
    return Path(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--require-final", action="store_true", default=True,
                    help="要求 manifest 已 FINAL_FROZEN (默认开启)")
    args = ap.parse_args()

    doc = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    status = doc.get("freeze_status")
    hashes = doc.get("component_hashes") or {}

    targets = {}
    for key in sorted(hashes):
        src = _source_of(key)
        if src is None:
            print(f"[snap] WARN: key={key} 无源映射, 跳过")
            continue
        if not src.exists():
            print(f"[snap] WARN: key={key} 源缺失 {src}")
            continue
        targets[key] = src
    for p in EXTRA_DESCRIPTIVE:
        if p.exists():
            targets[f"descriptive/{p.name}"] = p

    print(f"[snap] freeze_status = {status}")
    print(f"[snap] 待快照 {len(targets)} 个文件 -> {SNAP}")
    for k, v in targets.items():
        print(f"    {k:32s} {v}")

    if not args.confirm:
        print("[snap] DRY-RUN: 未加 --confirm, 不写盘。")
        return
    if args.require_final and status != "FINAL_FROZEN":
        print(f"[snap] ERROR: freeze_status={status} != FINAL_FROZEN; 拒绝生成快照 "
              "(先跑 freeze_gate3_final.py --confirm)")
        sys.exit(1)

    if SNAP.exists():
        print(f"[snap] 目标已存在, 移入 .previous_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        SNAP.rename(SNAP.with_name(f"{SNAP.name}.previous_"
                                   f"{datetime.now().strftime('%Y%m%d%H%M%S')}"))
    SNAP.mkdir(parents=True)

    lines = []
    bad = []
    for k, src in targets.items():
        dst = SNAP / k
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        h_src, h_dst = _sha(src), _sha(dst)
        lines.append(f"{h_dst}  {k}")
        if h_src != h_dst:
            bad.append(k)
        if "component_hashes" in doc and h_dst != hashes.get(k) and k in hashes:
            print(f"[snap] WARN: 复制后哈希 != manifest 记录 ({k})")
    body = ("# SHA256SUMS — FINAL Gate3 frozen snapshot\n"
            f"# generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"# freeze_status: {status}\n"
            f"# source_data_snapshot_id: {doc.get('source_data_snapshot_id')}\n"
            + "\n".join(sorted(lines)) + "\n")
    (SNAP / "SHA256SUMS.txt").write_text(body, encoding="utf-8")

    # 尽力只读化
    ro_fail = []
    for k in targets:
        try:
            subprocess.run(["attrib", "+R", str(SNAP / k)], check=False,
                           capture_output=True, shell=False)
        except Exception as e:  # noqa
            ro_fail.append(f"{k}: {e}")
    for f in SNAP.rglob("*"):
        if f.is_file():
            try:
                os.chmod(f, 0o444)
            except Exception:
                pass

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_dir": str(SNAP),
        "freeze_status": status,
        "source_data_snapshot_id": doc.get("source_data_snapshot_id"),
        "n_files": len(targets),
        "files": sorted(targets),
        "copy_hash_mismatch": bad,
        "sha256sums": "SHA256SUMS.txt",
        "readonly_errors": ro_fail,
        "note": "非 git 仓库 -> 这是 FINAL Gate3 的唯一代码回滚点; 校验: (Get-FileHash) 或 sha256sum -c",
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[snap] DONE files={len(targets)} mismatches={len(bad)} -> {SNAP}")
    if bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
