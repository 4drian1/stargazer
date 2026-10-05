"""train_final.py — 冻结 materiality_head_v1 最终历史训练（全部 eligible P0 2019–2025）。

契约（已冻结，不再调参）：
  feature set / model family=LGBM / hyperparameters / training domain /
  label contract / time semantics

训练域（冻结）：
  P0 & label_valid==True & training_eligible==True & non-HEAVY & cluster-level
  & category!=DISTRIBUTION & 排除 corp_action_within_horizon (CORPORATE_ACTION_WINDOW)

产出：
  artifacts/lgbm_materiality_s{0,1,2}.txt  (3 seed 平均)
  artifacts/feature_medians.json          (线上缺失填充参考)
  artifacts/model_manifest.json           (版本 1.0.0 + 各文件 SHA256 + 训练元数据)

最终 production/shadow 模型使用全部合格历史数据，是部署阶段正常动作；
四折 OOT 继续作为“配置在历史时间外能工作”的证据。训练结束后不再回头调模型。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from corvus.materiality_v1.materiality_features_v1 import (
    build_features, MODEL_FEATURES, TOP_CATEGORIES)

ART = Path(__file__).resolve().parent / "artifacts"
OUT = "out"
EVENTS = os.path.join(OUT, "event_cluster.parquet")

FROZEN_CONFIG = {
    "model_family": "LightGBM",
    "objective": "binary",
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "class_weight": "balanced", "seeds": [0, 1, 2],
}
TRAIN_DOMAIN = ("tier==P0 & label_valid==True & training_eligible==True & "
                "attribution_quality!=HEAVY & category!=DISTRIBUTION & "
                "~corp_action_within_horizon")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ART.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(EVENTS)
    lv = ev["label_valid"] if "label_valid" in ev.columns else pd.Series(True, index=ev.index)
    te = ev["training_eligible"] if "training_eligible" in ev.columns else pd.Series(True, index=ev.index)
    ca = ev["corp_action_within_horizon"].fillna(False).astype(bool) \
        if "corp_action_within_horizon" in ev.columns else pd.Series(False, index=ev.index)
    cond = ((ev["tier"] == "P0") & (lv == True) & (te == True) &
            (ev["attribution_quality"] != "HEAVY") & (ev["category"] != "DISTRIBUTION") &
            (~ca.fillna(False)))
    mat = ev[cond].copy()
    mat = mat.sort_values("announce_date").drop_duplicates("event_cluster_id")
    print(f"[train_final] 训练域 eligible: {len(mat)} 簇")

    feat = build_features(mat, OUT, with_labels=True)
    data = feat.dropna(subset=["materiality"]).copy()
    data["y"] = data["materiality"].astype(int)
    X = data[MODEL_FEATURES].copy()
    medians = X.median()
    Xf = X.fillna(medians).values
    y = data["y"].values
    pos_rate = float(y.mean())
    print(f"[train_final] 训练样本 {len(Xf)}，正率 {pos_rate:.3f}")

    for s in FROZEN_CONFIG["seeds"]:
        m = lgb.LGBMClassifier(
            n_estimators=FROZEN_CONFIG["n_estimators"],
            learning_rate=FROZEN_CONFIG["learning_rate"],
            num_leaves=FROZEN_CONFIG["num_leaves"],
            min_child_samples=FROZEN_CONFIG["min_child_samples"],
            subsample=FROZEN_CONFIG["subsample"],
            colsample_bytree=FROZEN_CONFIG["colsample_bytree"],
            class_weight=FROZEN_CONFIG["class_weight"],
            random_state=s, verbose=-1)
        m.fit(Xf, y)
        m.booster_.save_model(str(ART / f"lgbm_materiality_s{s}.txt"))
    (ART / "feature_medians.json").write_text(
        json.dumps({k: (None if pd.isna(v) else float(v)) for k, v in medians.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    # artifact 路径登记为相对 manifest 所在目录（artifacts/）的可移植路径；
    # loader（c4_shadow.MaterialityModel._resolve）兼容历史绝对路径 manifest。
    model_files = [f"lgbm_materiality_s{s}.txt" for s in FROZEN_CONFIG["seeds"]]
    fc_path = Path(__file__).resolve().parent / "feature_contract.json"
    manifest = {
        "version": "1.0.0",
        "head": "CORVUS_MATERIALITY_HEAD_V1",
        "status": "FROZEN",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "events_file": EVENTS,
        "train_domain": TRAIN_DOMAIN,
        "train_years": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "n_train": int(len(Xf)),
        "positive_rate": pos_rate,
        "model_files": model_files,
        "model_sha256": {os.path.basename(p): sha256_file(ART / p) for p in model_files},
        "feature_medians_file": "feature_medians.json",
        "feature_medians_sha256": sha256_file(ART / "feature_medians.json"),
        "training_data_sha256": sha256_file(EVENTS),
        "feature_contract_file": "../feature_contract.json",
        "feature_contract_sha256": sha256_file(fc_path) if fc_path.exists() else None,
        "config_sha256": hashlib.sha256(
            json.dumps(FROZEN_CONFIG, sort_keys=True).encode()).hexdigest(),
        "config": FROZEN_CONFIG,
        "feature_names": MODEL_FEATURES,
        "top_categories": TOP_CATEGORIES,
        "calibration_status": "HISTORICAL_OOT_VALIDATED_RAW_PROBA",
        "code_commit": git_commit(),
        "label_contract": "materiality=|ret_1/3/5d_pc| 任一 > 2x 前250日收益std",
        "time_semantics": ["POST_CLOSE_NEXT_DAY", "WEEKEND_NEXT_DAY", "RESUME_FIRST_TRADE"],
        "assume_premarket": "FORBIDDEN_IN_FROZEN_V1",
    }
    (ART / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[train_final] 已写 model_manifest.json (v1.0.0, n={len(Xf)}, pos={pos_rate:.3f})")


if __name__ == "__main__":
    main()
