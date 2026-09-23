"""freeze — Gate 3 (Step 9 Preflight): L0 + GREEN-WATCH-v1 Freeze.

对现有证据文件做哈希快照, 生成不可变冻结清单。任何后续重新哈希不匹配
即判定 Gate3 FAIL (有人动了 L0 或候选引擎定义)。

L0 = V0_NEGATIVE_BASELINE (Step 6 证据), 历史失败证据不得重算覆盖。
GREEN-WATCH-v1-FROZEN = 冻结候选生成函数 (green_watch_v0), 与 standard_alignment 记录一致。

运行:
    python freeze.py        # 生成 L0_freeze.yaml / green_watch_v1_frozen.yaml
    python freeze.py --verify  # 重新哈希校验, 不匹配则退出码非0
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
V5 = Path("D:/stargazer/stargazer_ml/v5_hybrid")
STEP6 = V5 / "green_exit_v1/step6_rescue"
GW_V0 = V5 / "green_watch_v0"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tree(root: Path, patterns=("*.csv", "*.parquet", "*.json", "*.md", "*.py")) -> dict:
    out = {}
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if p.is_file():
                rel = p.relative_to(root)
                out[str(rel)] = _sha(p)
    return out


def build_l0_manifest() -> dict:
    out_dir = STEP6 / "out"
    files = _hash_tree(out_dir) if out_dir.exists() else {}
    verdict = (STEP6 / "step6_verdict.md")
    diag = (STEP6 / "diagnostics_60.md")
    meta = {
        "name": "L0_V0_NEGATIVE_BASELINE",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(STEP6),
        "rule": "禁止因新 Label Contract 重算并覆盖 Step 6 失败证据。",
        "artifacts": {
            "step6_verdict.md": _sha(verdict) if verdict.exists() else None,
            "diagnostics_60.md": _sha(diag) if diag.exists() else None,
            "out/": files,
        },
        "replay_note": "如需新合同下可比对照, 产生 A0_REPLAY_LABEL_V2, 不得与 L0_ORIGINAL 混用。",
    }
    return meta


GREEN_WATCH_FROZEN_DEF = {
    "universe_eligibility": "市值25-100亿 / ST-VOL>0 / 价域 / Low60-notna (N8 代理)",
    "setup_definitions": "BOTTOM_SETUP / INTRADAY_DIP_SETUP (standard_alignment 记录)",
    "bottom_score_formula": "build_pool.py 现有评分逻辑 (含 N1-N9 偏离, 冻结不动)",
    "intraday_dip_score_formula": "build_pool.py 现有评分逻辑",
    "subtype_definition": "build_pool.py subtype 分支",
    "zone_generation_rule": "LATR_ZONE 未开工, v1 由 draft 生成器用冻结公式 (T1/T2)",
    "candidate_status_rule": "PASS / NEAR_MISS / CONTROL / INVALIDATED (N9: CONTROL 待补)",
    "primary_setup_rule": "双分数取大者, 平票归 BOTTOM (口径 A)",
    "leader_definition_version": "板块等权指数代理 (N7 缺口, 真实映射接入前保持)",
    "rule_version": "GREEN-WATCH-v0-backfill (T4: 通过复验后才升 v1 命名)",
}


def build_gw_manifest() -> dict:
    srcs = {
        "build_pool.py": GW_V0 / "build_pool.py",
        "label_engine_pool.py": GW_V0 / "label_engine_pool.py",
        "standard_alignment_v1_20260917.md": GW_V0 / "standard_alignment_v1_20260917.md",
    }
    hashes = {k: (_sha(v) if v.exists() else None) for k, v in srcs.items()}
    return {
        "name": "GREEN-WATCH-v1-FROZEN",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(GW_V0),
        "rule": "A0-A8 期间禁止修改母池定义; 调阈值须另开 GREEN-WATCH-v1.1 实验。",
        "source_hashes": hashes,
        "frozen_definition": GREEN_WATCH_FROZEN_DEF,
    }


# ---- Gate 3 完整实验世界冻结 (扩展) ----
MARKET_PROXY_PRIMARY = "leave_one_out_float_weighted"
MARKET_PROXY_EQUAL = "equal_weighted (sensitivity only)"
import glob as _glob
from data_source_provenance import DAILY_PARTS
from trading_calendar import data_epoch_fingerprint  # ③.1 源快照指纹
MINUTE_START = 20250102
# DATA_EPOCH 由 canonical adapter 探测真实日线最大日期 (P0 ① provenance 探针证 20260917),
# 不再硬编码 (原 2026-09-16 与 HTTP 源真相差 1 天, 致 Gate3 INVALIDATED)。
try:
    from data_source_provenance import discover_data_epoch
    _ep = discover_data_epoch()
    DATA_EPOCH = _ep["epoch"]
    MINUTE_END = _ep["daily_max"]
except Exception:
    DATA_EPOCH = "2025-01-02..2026-09-17"
    MINUTE_END = 20260917
PIT_SECTOR_STATUS = "BLOCKED"
A5_PLUS_ALLOWED = False

# 参与 FINAL_GATE3 component_hashes 校验的全部组件 (单一事实来源; build/verify 共用)
COMPONENT_FILES = {
    # —— green_watch_v0: 候选母池 / 标签引擎 / 口径对齐 ——
    "green_watch_build_pool": GW_V0 / "build_pool.py",
    "green_watch_label_engine": GW_V0 / "label_engine_pool.py",
    "green_watch_alignment": GW_V0 / "standard_alignment_v1_20260917.md",
    # —— step9_preflight: 契约组件 (2026-09-18 纳入, 随 FINAL Gate3 一起不可变) ——
    "feature_registry": _HERE / "feature_registry.yaml",
    "feature_registry_py": _HERE / "feature_registry.py",
    "label_contract": _HERE / "label_contract_v1.yaml",
    "adjustment_chain": _HERE / "adjustment_chain.py",
    "market_proxy": _HERE / "market_proxy_v1.py",
    "execution_engine": _HERE / "execution_engine.py",
    "zone_state_engine": _HERE / "zone_state_engine.py",
    "latr_builder": _HERE / "latr_builder.py",
    "trading_calendar": _HERE / "trading_calendar.py",
    "gate4_harness": _HERE / "gate4_harness.py",
    "data_source_provenance": _HERE / "data_source_provenance.py",
    # —— 数据可用性制度登记 (2026-09-18 用户裁定 #3: FLOAT_MV_DATA_REGIME_CHANGE) ——
    "data_regimes": _HERE / "data_regimes.py",
}


def _registry_lists():
    """从 feature_registry 拉取启用/禁用特征集合 (Gate2 集成检查 #2/#3/#4 的来源)。"""
    try:
        import feature_registry as fr
        return {
            "enabled_features": sorted(fr.core_features_9a()),
            "blocked_features": sorted(fr.disabled_until_pit_sector()),
            "optional_disabled_features": sorted(fr.optional_disabled()),
        }
    except Exception as e:  # noqa
        return {"registry_error": str(e)}


def build_full_manifest() -> dict:
    comps = COMPONENT_FILES
    hashes = {k: (_sha(v) if v.exists() else None) for k, v in comps.items()}
    # 尚未在 Step9A 前存在的组件: 待 LATR Builder / A0-A4 创建后冻结
    pending = {
        "split_contract": "PENDING (LATR Builder 后冻结)",
        "execution_engine": "BUILT 2026-09-17 (provisional; mutable until final Gate3 freeze)",
        "cost_model": "PENDING (A0-A4 后冻结)",
    }
    return {
        "name": "GATE3_EXPERIMENT_WORLD_FREEZE",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freeze_status": "PROVISIONAL_VALID",
        "freeze_status_note": ("③⑥②④⑤ 全部完成并重新 Gate3/Gate4 全绿后, "
                               "再生成最终 immutable freeze (此版仅供中间验证)."),
        "rule": ("A0-A4 期间 Candidate/Label/Split/Execution/Adjustment/Cost/Portfolio "
                 "全部不可修改; 仅 Feature Group 可变化。"),
        "component_hashes": hashes,
        "pending_components": pending,
        "market_proxy_v1": {
            "primary": MARKET_PROXY_PRIMARY,
            "equal_weight_role": MARKET_PROXY_EQUAL,
            "note": "Primary=leave-one-out float-weighted (更接近资本权重暴露); "
                    "equal-weight 仅作 sensitivity diagnostic, 不据结果挑选。",
        },
        "minute_start": MINUTE_START,
        "data_epoch": DATA_EPOCH,
        "source_data_snapshot_id": data_epoch_fingerprint(),
        "bars_glob": DAILY_PARTS + "/part_*.parquet",
        "daily_parts_file_count": len(_glob.glob(DAILY_PARTS + "/part_*.parquet")),
        "data_epoch_max_date": int(str(DATA_EPOCH).split("..")[-1].replace("-", "")) if ".." in DATA_EPOCH else None,
        "pit_sector_status": PIT_SECTOR_STATUS,
        "a5_plus_allowed": A5_PLUS_ALLOWED,
        "features": _registry_lists(),
        "baselines": {
            "L0_ORIGINAL": "Step 6 原始失败证据 (L0_freeze.yaml), 禁止覆盖",
            "A0_REPLAY_LABEL_V2": "新合同下重算可比 Step9 baseline, 待 A0 生成, 不覆盖 L0",
        },
    }


def write_manifests(force: bool = False):
    # 安全护栏: FINAL_FROZEN 是不可变清单, 禁止被例行重跑静默降级回 PROVISIONAL_VALID
    mpath = _HERE / "GATE3_MANIFEST.yaml"
    if mpath.exists():
        try:
            cur = (yaml.safe_load(mpath.read_text(encoding="utf-8")) or {}).get("freeze_status")
        except Exception:
            cur = None
        if cur == "FINAL_FROZEN" and not force:
            raise RuntimeError(
                "[freeze] 拒绝覆盖: GATE3_MANIFEST.yaml 已是 FINAL_FROZEN (immutable)。"
                "若确需重开实验世界, 必须显式 --force-refreeze 并另开实验编号。")
    l0 = build_l0_manifest()
    gw = build_gw_manifest()
    full = build_full_manifest()
    ( _HERE / "L0_freeze.yaml").write_text(
        yaml.safe_dump(l0, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ( _HERE / "green_watch_v1_frozen.yaml").write_text(
        yaml.safe_dump(gw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ( _HERE / "GATE3_MANIFEST.yaml").write_text(
        yaml.safe_dump(full, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("wrote L0_freeze.yaml / green_watch_v1_frozen.yaml / GATE3_MANIFEST.yaml")


def assert_a5_blocked():
    """运行时硬门控: A5+ 探索性脚本必须被拒绝 (除非 manifest 明确放开)。"""
    if not A5_PLUS_ALLOWED:
        raise RuntimeError(
            "[Gate3] A5+ Sector/leader 扩展被冻结拒绝 (PIT_SECTOR_STATUS=BLOCKED). "
            "先完成 Gate 1B PIT Sector 验收再放开。")


def _verify_tree(root: Path, expected: dict) -> bool:
    ok = True
    for rel, h in expected.items():
        p = root / rel
        if not p.exists() or _sha(p) != h:
            print(f"  MISMATCH: {rel}")
            ok = False
    return ok


def _verify_full(expected: dict) -> bool:
    ok = True
    comps = COMPONENT_FILES
    for k, h in expected.get("component_hashes", {}).items():
        if h is None:
            continue
        p = comps.get(k)
        if p is None or not p.exists() or _sha(p) != h:
            print(f"  MISMATCH (full): {k}")
            ok = False
    if expected.get("a5_plus_allowed") is not False:
        print("  MISMATCH: a5_plus_allowed != false")
        ok = False
    if expected.get("pit_sector_status") != "BLOCKED":
        print("  MISMATCH: pit_sector_status != BLOCKED")
        ok = False
    return ok


def verify() -> bool:
    l0 = yaml.safe_load((_HERE / "L0_freeze.yaml").read_text(encoding="utf-8"))
    gw = yaml.safe_load((_HERE / "green_watch_v1_frozen.yaml").read_text(encoding="utf-8"))
    full = (yaml.safe_load((_HERE / "GATE3_MANIFEST.yaml").read_text(encoding="utf-8"))
            if (_HERE / "GATE3_MANIFEST.yaml").exists() else None)
    ok = True
    out_dir = STEP6 / "out"
    if not _verify_tree(out_dir, l0["artifacts"].get("out/", {})):
        ok = False
    for k, v in gw["source_hashes"].items():
        p = GW_V0 / k
        if not p.exists() or _sha(p) != v:
            print(f"  MISMATCH: {k}")
            ok = False
    if full is not None and not _verify_full(full):
        ok = False
    print(f"Gate3 verify = {ok}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--refuse-a5", action="store_true",
                    help="演示 A5+ 拒绝 (应抛 RuntimeError)")
    ap.add_argument("--force-refreeze", action="store_true",
                    help="(危险) 允许覆盖已 FINAL_FROZEN 的清单")
    args = ap.parse_args()
    if args.refuse_a5:
        assert_a5_blocked()
    if args.verify:
        sys.exit(0 if verify() else 1)
    write_manifests(force=args.force_refreeze)
