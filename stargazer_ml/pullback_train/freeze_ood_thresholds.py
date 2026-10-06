# -*- coding: utf-8 -*-
"""冻结 T7 OOD（训练分布外）警告阈值（2026-09-09 评审拍板）。

口径：OOD = 模型训练支持范围之外，评分可信度低，仅观察不生正式买入建议。
主规则：runup_10d > 训练池 P99.5；辅规则：高热指标中 ≥3 个同时 > 训练池 P99。
训练池 = 2025 建模池（load_pool(2025, 全部7dp)，与消融一致）。

输出：ood_thresholds.json（冻结值，供影子监控读取）+ ood_thresholds.csv（全分位留档）。
"""
import json
import os

import numpy as np
import pandas as pd

from ablation_t7 import BG, load_pool

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
HEAT = ("runup_5d", "runup_10d", "runup_20d", "pos20", "pos60", "rsi14",
        "rsi_over75_10d", "limit_up_10d", "limit_up_20d", "turnover_sum_5d",
        "turnover_sum_10d", "turnover_max_5d", "ht_down_10d",
        "wide_range_10d", "stage_low_high_20d")
DPS7 = (945, 1000, 1030, 1330, 1400, 1430, 1450)


def main():
    df = load_pool(2025, DPS7)
    # 事件级去重（同一信号日多个决策点重复同一背景值，分位按事件算）
    ev = df.drop_duplicates(["code", "date"])
    print(f"建模池 {len(df):,} 行 / 事件级 {len(ev):,} 个 (code,date)")
    rows, rules = [], {}
    for c in HEAT:
        s = ev[c].dropna()
        q = {p: float(s.quantile(p)) for p in (0.95, 0.99, 0.995)}
        rows.append({"feature": c, "n": int(s.shape[0]), "p95": q[0.95],
                     "p99": q[0.99], "p99_5": q[0.995], "max": float(s.max())})
        rules[c] = {"p99_5": q[0.995], "max": float(s.max())}
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT_DIR, "ood_thresholds.csv"), index=False)
    frozen = {
        "_meta": {
            "frozen_at": "2026-09-09",
            "pool": "2025 modeling pool (load_pool(2025, 7dp), event-level dedup)",
            "n_events": int(len(ev)),
            "primary_rule": "OOD if runup_10d > p99_5",
            "secondary_rule": "OOD if >=3 heat features > their p99_5",
            "action": "OUT_OF_DISTRIBUTION: display only, no formal buy suggestion",
        },
        "features": rules,
    }
    with open(os.path.join(OUT_DIR, "ood_thresholds.json"), "w",
              encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
