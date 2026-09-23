"""a0a4_gates — 红队审计修复的 3 个新硬 Gate (只读, 可独立 TDD 运行).

Gate 1 MODEL_ROLE_FEATURE_PASS:
  S0 轨只用 role_allowed 含 S0 的特征; Timing 轨只用 role_allowed 含 TIMING 的。
Gate 2 TRAINING_TIME_CAUSALITY_PASS (全量, 非抽样):
  Timing: feature_asof_max_ts <= signal_ts AND signal_ts < actual_fill_ts
  S0    : feature_asof_max_ts <= pclose_feature_asof_ts
  violation_count 必须 == 0。
Gate 3 TIMING_TRIGGER_SEMANTICS_PASS (全量):
  signal_ts == 首次 ARMED (FIRST_MODEL_SCORE_REQUEST) enter_ts;
  且 reclaim_ts / second_test_ts / zone_touch_ts 等事件时间 <= signal_ts (无未来倒灌)。
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


def _parse_armed(transitions) -> float:
    if not isinstance(transitions, str):
        return np.nan
    try:
        tr = json.loads(transitions)
    except Exception:
        return np.nan
    for t in tr:
        if t.get("state") == "ARMED":
            return float(t.get("state_enter_ts"))
    return np.nan


# ---------------------------------------------------------------- Gate 1
def model_role_feature_pass():
    out, overall = {}, True
    for tier in ("a1", "a2", "a3", "a4"):
        out[tier] = {}
        for track in C.LABEL_TRACKS:
            feats = C.track_features(tier, track)
            if feats is None:
                out[tier][track] = {"applicable": False, "status": "N/A_S0_TIER"}
                continue
            viol = [f for f in feats
                    if track.upper() not in C.LINEAGE_REGISTRY.get(f, {}).get("role_allowed", [])]
            ok = len(viol) == 0
            overall &= ok
            out[tier][track] = {"applicable": True, "pass": ok, "violations": viol}
    return {"detail": out, "pass": bool(overall)}


# ---------------------------------------------------------------- Gate 2
def training_time_causality_pass(df: pd.DataFrame):
    tdf = (df[df["timing_eligible"] == 1].copy() if "timing_eligible" in df
           else df.iloc[0:0])
    sdf = (df[df["s0_eligible"] == 1].copy() if "s0_eligible" in df
           else df.iloc[0:0])
    if len(tdf):
        fa = pd.to_numeric(tdf.get("feature_asof_max_ts", tdf.get("feature_asof_ts")), errors="coerce")
        sig = pd.to_numeric(tdf.get("signal_time"), errors="coerce")
        fill = pd.to_numeric(tdf.get("fill_ts"), errors="coerce")
        timing_viol = int((~((fa <= sig) & (sig < fill)).fillna(False)).sum())
    else:
        timing_viol = None
    if len(sdf):
        fa = pd.to_numeric(sdf.get("feature_asof_max_ts", sdf.get("feature_asof_ts")), errors="coerce")
        pc = pd.to_numeric(sdf.get("pclose_feature_asof_ts"), errors="coerce")
        s0_viol = int((~(fa <= pc).fillna(False)).sum())
    else:
        s0_viol = None
    t_pass = (timing_viol == 0) if timing_viol is not None else None
    s_pass = (s0_viol == 0) if s0_viol is not None else None
    return {"timing_violation_count": timing_viol, "s0_violation_count": s0_viol,
            "timing_pass": t_pass, "s0_pass": s_pass,
            "pass": ((t_pass if t_pass is not None else True) and
                     (s_pass if s_pass is not None else True))}


# ---------------------------------------------------------------- Gate 3
def timing_trigger_semantics_pass(df: pd.DataFrame):
    tdf = (df[df["timing_eligible"] == 1].copy() if "timing_eligible" in df
           else df.iloc[0:0])
    if "state_transitions_json" not in tdf.columns or len(tdf) == 0:
        return {"applicable": False, "reason": "no state_transitions_json / no timing rows"}
    sig = pd.to_numeric(tdf.get("signal_time"), errors="coerce")
    armed = tdf["state_transitions_json"].apply(_parse_armed)
    valid = armed.notna() & sig.notna()
    n_mismatch = int((armed[valid] != sig[valid]).sum())
    n_event_leak = 0
    for ev in ("reclaim_ts", "second_test_ts", "zone_touch_ts"):
        if ev in tdf.columns:
            e = pd.to_numeric(tdf[ev], errors="coerce")
            n_event_leak += int(((e.notna()) & (e > sig)).sum())
    return {"n_checked": int(valid.sum()), "signal_mismatch_count": n_mismatch,
            "future_event_leak_count": n_event_leak,
            "pass": (n_mismatch == 0 and n_event_leak == 0)}


def run_all_gates(df: pd.DataFrame):
    return {
        "MODEL_ROLE_FEATURE_PASS": model_role_feature_pass(),
        "TRAINING_TIME_CAUSALITY_PASS": training_time_causality_pass(df),
        "TIMING_TRIGGER_SEMANTICS_PASS": timing_trigger_semantics_pass(df),
    }


if __name__ == "__main__":
    df = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    g = run_all_gates(df)
    print(json.dumps(g, ensure_ascii=False, indent=2, default=str))
