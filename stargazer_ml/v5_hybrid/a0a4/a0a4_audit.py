"""a0a4_audit — 红队审计 (只读, 不训练/不重训).

目标 (用户 2026-09-19 红队指令):
  1) MODEL_ROLE_FEATURE_PASS: 审计 S0 轨是否混入了 D 日盘中(A3/A4)特征.
  2) TRAINING_TIME_CAUSALITY_PASS: 用数据集自带的 feature_asof_ts / signal_time /
     fill_ts / pclose_feature_asof_ts 逐行证明因果链.
  3) 导出 A1..A4 × {s0,timing} 实际特征矩阵 (因 harness 按 tier 选特征,
     s0 与 timing 共用同一 feats, 导出会显示两者完全相同 — 这正是 bug 证据).

纪律: 只读 a14_universe.parquet 与已落盘 metrics; 不修改任何训练逻辑.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

OUT = _HERE / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # ---- 1. 每个 tier 的实际特征列 (harness 按 tier 选, s0/timing 共用) ----
    tiers = {t: [f for f, _ in C.TIER_FEATURES[t]] for t in ("a1", "a2", "a3", "a4")}
    A1 = set(f for f, _ in C.A1_CORE)
    A2 = set(f for f, _ in C.A2_ZONE)
    A3 = set(f for f, _ in C.A3_DIP)
    A4 = set(f for f, _ in C.A4_RECOVERY)
    S0_SAFE = A1 | A2                      # P_CLOSE 可用 (pool 继承 / 前日 zone / 前日 ATR)
    DDAY = A3 | A4                         # D 日盘中 intraday -> S0/Timing 均不应在信号前使用

    # ---- 2. 导出特征矩阵 (s0 与 timing 同一份, 证明共用) ----
    exported = {}
    for t in tiers:
        for track in ("s0", "timing"):
            name = f"{t.upper()}_{track}_features.txt"
            (OUT / name).write_text("\n".join(tiers[t]) + "\n", encoding="utf-8")
            exported[f"{t}/{track}"] = tiers[t]

    # ---- 3. MODEL_ROLE_FEATURE_PASS (S0 轨是否含 D 日盘中特征) ----
    role = {}
    for t, feats in tiers.items():
        used = set(feats)
        banned = sorted(used & DDAY)
        role[t] = {
            "n_features": len(feats),
            "banned_dday_features_in_s0": banned,
            "MODEL_ROLE_FEATURE_PASS_S0": (len(banned) == 0),
        }

    # ---- 4. TRAINING_TIME_CAUSALITY_PASS (逐行 feature_asof vs signal/fill/pclose) ----
    df = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    causality = {"n_rows": int(len(df))}
    if "feature_asof_ts" in df.columns:
        fa = pd.to_numeric(df["feature_asof_ts"], errors="coerce")
        pc = pd.to_numeric(df.get("pclose_feature_asof_ts"), errors="coerce")
        sig = pd.to_numeric(df.get("signal_time"), errors="coerce")
        fill = pd.to_numeric(df.get("fill_ts"), errors="coerce")
        causality["feature_asof_example"] = int(fa.dropna().iloc[0]) if fa.notna().any() else None
        causality["signal_time_example"] = int(sig.dropna().iloc[0]) if sig.notna().any() else None
        causality["fill_ts_example"] = int(fill.dropna().iloc[0]) if fill.notna().any() else None
        causality["pclose_feature_asof_example"] = int(pc.dropna().iloc[0]) if pc.notna().any() else None
        causality["frac_feature_asof_gt_signal"] = (
            float((fa > sig).mean()) if sig.notna().any() else None)   # Timing 因果违反比例
        causality["frac_feature_asof_gt_pclose"] = (
            float((fa > pc).mean()) if pc.notna().any() else None)     # S0 因果违反比例
        # S0 要求 feature_asof <= P_CLOSE; Timing 要求 <= signal
        causality["TRAINING_TIME_CAUSALITY_PASS_TIMING"] = (
            bool((fa <= sig).all()) if sig.notna().any() else None)
        causality["TRAINING_TIME_CAUSALITY_PASS_S0"] = (
            bool((fa <= pc).all()) if pc.notna().any() else None)
    else:
        causality["note"] = "数据集无 feature_asof_ts 列 (应补全 lineage)"

    # ---- 5. 已落盘 S0 IC 现状 (供报告引用, 非重新计算) ----
    s0_ic = {}
    for t in ("a1", "a2", "a3", "a4"):
        mpath = C.TIER_DIRS[t] / "metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            s0 = m.get("tracks", {}).get("s0", {})
            s0_ic[t] = s0.get("ic_mean")
    timing_ic = {}
    for t in ("a1", "a2", "a3", "a4"):
        mpath = C.TIER_DIRS[t] / "metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            ti = m.get("tracks", {}).get("timing", {})
            timing_ic[t] = ti.get("ic_mean")

    result = {
        "verdict": "AUDIT_REQUIRED",
        "MODEL_ROLE_FEATURE_PASS": {t: role[t]["MODEL_ROLE_FEATURE_PASS_S0"] for t in tiers},
        "role_detail": role,
        "TRAINING_TIME_CAUSALITY": causality,
        "s0_ic_existing": s0_ic,
        "timing_ic_existing": timing_ic,
        "note_s0_safe_bound": sorted(S0_SAFE),
        "note_dday_banned": sorted(DDAY),
    }
    (OUT / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印摘要 ----
    print("=" * 70)
    print("MODEL_ROLE_FEATURE_PASS (S0 轨是否混入 D 日盘中特征)")
    print("=" * 70)
    for t in tiers:
        ok = "PASS" if role[t]["MODEL_ROLE_FEATURE_PASS_S0"] else "FAIL <-- 泄漏"
        print(f"  {t.upper():4} S0: {ok}  (banned={role[t]['banned_dday_features_in_s0']})")
    print()
    print("=" * 70)
    print("TRAINING_TIME_CAUSALITY_PASS (逐行 feature_asof vs signal/pclose)")
    print("=" * 70)
    for k, v in causality.items():
        print(f"  {k}: {v}")
    print()
    print("已落盘 IC (仅供引用, S0 A3/A4 在此审计后视为无效):")
    print(f"  S0 : " + ", ".join(f"{t.upper()}={s0_ic.get(t)}" for t in tiers))
    print(f"  Tim: " + ", ".join(f"{t.upper()}={timing_ic.get(t)}" for t in tiers))
    print()
    print(f"特征矩阵已导出到 {OUT}/*.txt (注意: s0 与 timing 文件内容完全相同,")
    print("           即 harness 按 tier 选特征, 两轨共用 -> 这是泄漏的根因)")
    print(f"完整结果: {OUT / 'audit.json'}")


if __name__ == "__main__":
    main()
