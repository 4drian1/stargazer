"""v2_rowflow_audit — 解释 timing_universe_v2_train (4453) -> A1-A4 OOF (2475) 的缩水。

要证明: 缩水来自 walk-forward 的 split 角色分配 (test 行才进 OOF),
而非 dropna / 特征完整性 / 公共字段 NaN 造成的隐藏 sample selection。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))
import a0a4_train as T  # noqa (build_split / split_hash)

df = pd.read_parquet(C.DATASETS / "timing_universe_v2_train.parquet")
N = len(df)
print(f"[audit] n_train (timing_universe_v2_train.parquet) = {N}")

assign, folds = T.build_split(df)
print("\n[audit] folds:")
for f in folds:
    print(f"  fold{f['fold']}: test_lo={f['test_lo']} test_hi={f['test_hi']} "
          f"train_hi={f['train_hi']} (purge={C.PURGE_TRADING_DAYS}d)")

print("\n[audit] role counts (build_split):")
print(assign["role"].value_counts().to_string())

print("\n[audit] fold assignment counts:")
print(assign["fold"].value_counts().sort_index().to_string())

test_n = int((assign["fold"] > 0).sum())
nontest = N - test_n
print(f"\n[audit] OOF population (fold>0, test rows that get predicted) = {test_n}")
print(f"[audit] non-test (train_only + unused_tail, used for training but never OOF) = {nontest}")
print(f"[audit] 4453 -> 2475 gap = {N - 2475}  ; test_n==2475 ? {test_n == 2475}")

print("\n[audit] per test-fold n (what ablation iterates as te):")
tot = 0
for f in folds:
    m = int((assign["fold"] == f["fold"]).sum())
    tot += m
    print(f"  fold{f['fold']}: test n = {m}")
print(f"  sum of test folds = {tot}")

# ---- 证明没有隐藏 dropna / 特征完整性删行 ----
print("\n[audit] v2_ablation.py 中 dropna/drop 出现次数 (应为 0):")
src = (Path(__file__).parent / "v2_ablation.py").read_text(encoding="utf-8")
print(f"  'dropna' = {src.count('dropna')}  'drop(' = {src.count('drop(')}  '.drop(' = {src.count('.drop(')}")

# 验证: 对 test 行, 即使特征有 NaN, lightgbm 仍逐行预测 -> OOF n 不变
# 这里只检查 test 行在关键 id/label 列上是否完整 (不应有 NaN 导致 merge 丢行)
test_ids = assign[assign["fold"] > 0]["sample_id"]
tr = df[df["sample_id"].isin(test_ids)]
print(f"\n[audit] test rows with non-null sample_id/label: "
      f"sample_id={tr['sample_id'].notna().sum()} "
      f"timing_return_h1={tr['timing_return_h1'].notna().sum()}/{len(tr)}")
print(f"[audit] -> OOF n 由 split 决定, 与特征 NaN 无关 (lightgbm 保留 NaN 行预测)")
print("\n[audit] CONCLUSION:",
      "LEGIT walk-forward split" if test_n == 2475 else "INVESTIGATE further")
