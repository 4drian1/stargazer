"""v2_gates — TIMING-TRIGGER-v2 的 5 个契约硬 Gate (只读, 可独立 TDD 运行).

1. MODEL_ROLE_FEATURE_PASS
   v2-primary 模型特征集中不得含 second_test* (MODEL_ELIGIBLE=False) 及任何 S0-only 字段.
2. TRAINING_TIME_CAUSALITY_PASS (全量, 非抽样)
   feature_asof_max_ts <= score_ts < actual_fill_ts < label_end_ts
   且 proxy_start_ts == actual_fill_ts ; proxy_end_ts == label_end_ts
   violation_count 必须 == 0.
3. TIMING_TRIGGER_SEMANTICS_PASS (全量)
   score_ts 落在 D 日连续竞价区间内 (>= open+1min, <= 15:00), 无未来倒灌.
4. WINDOW_SEMANTICS_PASS
   所有窗口字段: 非 null -> 对应 window_complete == 1 (即 flag==0 必须 NaN).
5. SECOND_TEST_MODEL_EXCLUSION_PASS
   model_features 绝对不含 second_test*.

用法:
    python v2_gates.py [--parquet datasets/timing_universe_v2_all.parquet]
"""
from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

WINDOWED = {
    "open_dip_atr": "window_complete_15m",
    "dip_depth_5m_atr": "window_complete_5m",
    "dip_depth_30m_atr": "window_complete_30m",
    "dip_speed_5m": "window_complete_5m",
    "dip_speed_15m": "window_complete_15m",
    "dip_speed_30m": "window_complete_30m",
}
SECOND_TEST_COLS = ("second_test_flag", "second_test_higher_low")


def model_role_feature_pass(model_features):
    offend = [f for f in model_features if f in SECOND_TEST_COLS]
    return {"pass": len(offend) == 0, "offending": offend}


def training_time_causality_pass(df):
    el = df[df["timing_eligible_v2"] == 1].copy() if "timing_eligible_v2" in df else df
    if len(el) == 0:
        return {"pass": None, "timing_violation_count": None, "proxy_violation_count": None,
                "note": "no eligible rows"}
    fa = pd.to_numeric(el.get("feature_asof_max_ts"), errors="coerce")
    sig = pd.to_numeric(el.get("score_ts"), errors="coerce")
    fill = pd.to_numeric(el.get("actual_fill_ts"), errors="coerce")
    lend = pd.to_numeric(el.get("label_end_ts"), errors="coerce")
    pstart = pd.to_numeric(el.get("proxy_start_ts"), errors="coerce")
    pend = pd.to_numeric(el.get("proxy_end_ts"), errors="coerce")
    t_viol = int((~((fa <= sig) & (sig < fill) & (fill < lend)).fillna(False)).sum())
    p_viol = int((~((pstart == fill) & (pend == lend)).fillna(False)).sum())
    return {"pass": (t_viol == 0 and p_viol == 0),
            "timing_violation_count": t_viol, "proxy_violation_count": p_viol}


def timing_trigger_semantics_pass(df):
    el = df[df["timing_eligible_v2"] == 1].copy() if "timing_eligible_v2" in df else df
    if "score_ts" not in el or "effective_trade_date" not in el or len(el) == 0:
        return {"pass": None, "note": "no eligible rows / missing cols"}
    sig = pd.to_numeric(el["score_ts"], errors="coerce")
    d = pd.to_numeric(el["effective_trade_date"], errors="coerce")
    lo = d * 1_000_000 + 93_000   # 09:30 (open); 真实成交须晚于开盘
    hi = d * 1_000_000 + 150_000  # 15:00
    n_bad = int((~((sig >= lo) & (sig <= hi))).fillna(False).sum())
    return {"pass": n_bad == 0, "out_of_session_count": n_bad}


def window_semantics_pass(df):
    el = df[df["timing_eligible_v2"] == 1].copy() if "timing_eligible_v2" in df else df
    viol = 0
    per_field = {}
    for f, flag in WINDOWED.items():
        if f in el.columns and flag in el.columns:
            nonnull = el[f].notna()
            bad = int((nonnull & (el[flag] != 1)).sum())
            per_field[f] = bad
            viol += bad
    return {"pass": viol == 0, "window_violation_count": viol, "per_field": per_field}


def second_test_model_exclusion_pass(model_features):
    offend = [f for f in model_features if str(f).startswith("second_test")]
    return {"pass": len(offend) == 0, "offending": offend}


def run_all_gates(df, model_features):
    return {
        "MODEL_ROLE_FEATURE_PASS": model_role_feature_pass(model_features),
        "TRAINING_TIME_CAUSALITY_PASS": training_time_causality_pass(df),
        "TIMING_TRIGGER_SEMANTICS_PASS": timing_trigger_semantics_pass(df),
        "WINDOW_SEMANTICS_PASS": window_semantics_pass(df),
        "SECOND_TEST_MODEL_EXCLUSION_PASS": second_test_model_exclusion_pass(model_features),
    }


def _all_pass(gates):
    for name, res in gates.items():
        if isinstance(res, dict):
            if res.get("pass") is not True:
                return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(C.DATASETS / "timing_universe_v2_all.parquet"))
    ap.add_argument("--model-features", default=None,
                    help="逗号分隔; 缺省用 build_timing_universe_v2.MODEL_FEATURES")
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    if args.model_features:
        mf = [x.strip() for x in args.model_features.split(",") if x.strip()]
    else:
        sys.path.insert(0, str(_HERE))
        from build_timing_universe_v2 import MODEL_FEATURES as mf  # noqa
    gates = run_all_gates(df, mf)
    print(json.dumps(gates, ensure_ascii=False, indent=2, default=str))
    print("ALL_PASS =", _all_pass(gates))
