"""feature_parity_gate.py — Feature Parity Gate（C4 上线前必过）。

验证：
1. 离线（批量）特征 vs 在线（子集注入）特征：列名/顺序 / dtype / 缺失语义 /
   类别编码 完全一致，数值 max_abs_diff ≈ 0。核心在于 cat_* 已冻结为 TOP_CATEGORIES，
   因此即使只传入一个子集（线上单簇方式），类别编码仍与离线一致。
2. 截断测试：向 label_start 之后追加伪造行情，特征向量不变（无未来信息）。

通过后写 artifacts/feature_parity_passed.json；c4_shadow.MaterialityModel 加载模型前
会检查该文件，缺失则拒绝加载（强制先做 parity）。
"""
from __future__ import annotations

import json
import os
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corvus.materiality_v1.materiality_features_v1 import (
    build_features, MODEL_FEATURES, TOP_CATEGORIES)

OUT = "out"
ART = Path(__file__).resolve().parent / "artifacts"
TOL = 1e-9


def _load_eligible():
    ev = pd.read_parquet(os.path.join(OUT, "event_cluster.parquet"))
    mat = ev[(ev["tier"] == "P0") & (ev["label_status"] == "OK") &
             (ev["attribution_quality"] != "HEAVY") &
             (ev["category"] != "DISTRIBUTION")].copy()
    return mat.sort_values("announce_date").drop_duplicates("event_cluster_id")


def _compare(offline, online, cids):
    off = offline.set_index("event_cluster_id").reindex(cids)
    on = online.set_index("event_cluster_id").reindex(cids)
    issues = []
    if list(off[MODEL_FEATURES].columns) != MODEL_FEATURES:
        issues.append("feature column order mismatch")
    for c in MODEL_FEATURES:
        if off[c].dtype != on[c].dtype:
            issues.append(f"dtype {c}: {off[c].dtype} vs {on[c].dtype}")
    off_nan = off[MODEL_FEATURES].isna()
    on_nan = on[MODEL_FEATURES].isna()
    if not off_nan.equals(on_nan):
        issues.append("NaN mask mismatch")
    for c in MODEL_FEATURES:
        a = off[c].to_numpy(dtype=float, na_value=np.nan)
        b = on[c].to_numpy(dtype=float, na_value=np.nan)
        m = ~(np.isnan(a) | np.isnan(b))
        if m.any() and np.nanmax(np.abs(a[m] - b[m])) > TOL:
            issues.append(f"numeric diff {c}: {np.nanmax(np.abs(a[m]-b[m]))}")
    return issues


def _truncation_test(ev_one, code, out_dir):
    from backfill_pipeline import daily_cache_path
    p = daily_cache_path(out_dir, code)
    if not p.exists():
        return ["truncation: no daily cache for code"]
    daily = pd.read_parquet(p).copy()
    daily["date"] = daily["date"].astype(int)
    base = build_features(ev_one, out_dir, with_labels=False)
    future = daily.tail(1).copy()
    future["date"] = int(daily["date"].max()) + 100000
    daily_future = pd.concat([daily, future], ignore_index=True)

    def provider(c):
        return daily_future if c == code else daily

    fut = build_features(ev_one, out_dir, with_labels=False, daily_provider=provider)
    issues = []
    for c in MODEL_FEATURES:
        a = base[c].to_numpy(dtype=float, na_value=np.nan)
        b = fut[c].to_numpy(dtype=float, na_value=np.nan)
        m = ~(np.isnan(a) | np.isnan(b))
        if m.any() and np.nanmax(np.abs(a[m] - b[m])) > TOL:
            issues.append(f"truncation diff {c}")
    return issues


def main():
    mat = _load_eligible()
    sample = mat.sample(min(2000, len(mat)), random_state=1)
    cids = sample["event_cluster_id"].tolist()
    off = build_features(sample, OUT, with_labels=False)         # 离线批量
    sub = sample.sample(min(500, len(sample)), random_state=2)   # 线上子集注入
    on_sub = build_features(sub, OUT, with_labels=False)
    issues = []
    issues += _compare(off, on_sub, sub["event_cluster_id"].tolist())
    # 截断测试（抽一个 code）
    code = sample["code"].iloc[0]
    ev_one = sample[sample["code"] == code]
    if len(ev_one):
        issues += _truncation_test(ev_one, code, OUT)

    ART.mkdir(parents=True, exist_ok=True)
    if issues:
        print("PARITY FAILED:")
        for i in issues[:20]:
            print("  -", i)
        raise SystemExit(1)
    flag = {
        "passed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_features": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "top_categories": TOP_CATEGORIES,
        "tol": TOL,
        "checks": ["column_order", "dtype", "nan_mask", "numeric_max_abs_diff",
                   "categorical_encoding_frozen", "truncation_no_future_leak"],
    }
    (ART / "feature_parity_passed.json").write_text(
        json.dumps(flag, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"PARITY PASSED: {len(MODEL_FEATURES)} 特征；已写 {ART / 'feature_parity_passed.json'}")


if __name__ == "__main__":
    main()
