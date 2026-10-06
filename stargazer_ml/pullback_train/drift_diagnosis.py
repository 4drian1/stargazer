# -*- coding: utf-8 -*-
"""漂移诊断：adj_ready 子类 2024(验证) vs 2025(参考) 失效归因。

只读分析，不改任何文件。结论排序：市场状态 / 特征漂移 / 标签关系反转。
口径：adj_ready + eligible_s1.2（冻结 K）+ 非删失，板块内外分开看。
"""
import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
PRED_DIR = os.path.join(OUT_DIR, "preds")

PSI_FEATS = ["market_vol_state", "eq_ret20", "amount_ratio20", "quality_score",
             "sector_ret5", "pullback_from_high5", "rsi14", "turnover_ratio20",
             "boll_pct_b", "atr_ref"]
DECILE_FEATS = ["quality_score", "rsi14", "amount_ratio20", "pullback_from_high5",
                "sector_ret5", "turnover_ratio20"]


def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    """PSI：|期望占比-实际占比|求和，基于 a(基准=2024) 的分位分箱。"""
    a, b = a.dropna(), b.dropna()
    if len(a) < 100 or len(b) < 100:
        return np.nan
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return np.nan
    qa = np.histogram(a, qs)[0] / len(a)
    qb = np.histogram(b, qs)[0] / len(b)
    qa, qb = np.clip(qa, 1e-4, None), np.clip(qb, 1e-4, None)
    return float(np.sum((qa - qb) * np.log(qa / qb)))


def decile_returns(df: pd.DataFrame, feat: str, ycol: str) -> pd.DataFrame:
    d = df[[feat, ycol]].dropna()
    if len(d) < 500:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d[feat], 10, labels=False, duplicates="drop")
    return d.groupby("bin")[ycol].agg(["mean", "count"]).round(4)


def main() -> None:
    ev = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"))
    m = ev[(ev["subtype"] == "adj_ready") & (~ev["is_censored"])
           & (ev["eligible_s1.2"])].copy()
    m["yr"] = (m["date"] // 10000).astype(int)
    y24, y25 = m[m["yr"] == 2024], m[m["yr"] == 2025]
    lab, ret = "outcome_s1.2_t2.0", "net_ret_s1.2_t2.0"

    print("=" * 70)
    print("① 标签分布与收益（adj_ready, eligible_s1.2, t2.0）")
    for tag, d in (("2024", y24), ("2025", y25)):
        print(f"\n--- {tag}  n={len(d)} ---")
        print(d[lab].value_counts(normalize=True).round(3).to_string())
        print("net_ret 分位(10/25/50/75/90%):",
              (d[ret].quantile([.1, .25, .5, .75, .9]).round(4) * 100).tolist())
        print("裸EV:", f"{d[ret].mean():+.3%}",
              "| TARGET均:", f"{d.loc[d[lab]=='TARGET', ret].mean():+.3%}",
              "| STOP均:", f"{d.loc[d[lab]=='STOP', ret].mean():+.3%}",
              "| TIMEOUT均:", f"{d.loc[d[lab]=='TIMEOUT', ret].mean():+.3%}")

    print("\n" + "=" * 70)
    print("② 市场状态与事件时间分布")
    for tag, d in (("2024", y24), ("2025", y25)):
        print(f"\n--- {tag} ---")
        print("market_vol_state:", d["market_vol_state"].value_counts(normalize=True)
              .round(3).to_dict() if "market_vol_state" in d else "n/a")
        for c in ("eq_ret5", "eq_ret20", "sh_ret5"):
            if c in d:
                print(f"{c} 中位: {d[c].median():+.3%}")
        mm = d.assign(mo=d["date"].astype(str).str[:6]).groupby("mo").size()
        print("月度事件数:", mm.to_dict())

    print("\n" + "=" * 70)
    print("③ 特征 PSI（基准=2024，>0.25 显著漂移）")
    rows = []
    for f in PSI_FEATS:
        if f not in m:
            continue
        rows.append((f, psi(y24[f], y25[f]),
                     float(y24[f].median()), float(y25[f].median())))
    psi_df = pd.DataFrame(rows, columns=["feature", "PSI", "med2024", "med2025"]) \
               .sort_values("PSI", ascending=False)
    print(psi_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("④ 特征十分位 × net_ret 分层（2024 vs 2025 找反转）")
    for f in DECILE_FEATS:
        if f not in m:
            continue
        d24, d25 = decile_returns(y24, f, ret), decile_returns(y25, f, ret)
        if d24.empty or d25.empty:
            continue
        joined = d24[["mean"]].rename(columns={"mean": "r2024"}) \
            .join(d25[["mean"]].rename(columns={"mean": "r2025"}), how="outer")
        corr = joined["r2024"].corr(joined["r2025"], method="spearman")
        print(f"\n[{f}] 分层收益单调性相关(2024 vs 2025) = {corr:+.2f}")
        print((joined * 100).round(2).to_string())

    print("\n" + "=" * 70)
    print("⑤ 模型预测分布与 Top 构成（preds/，若有）")
    for fn in ("val_pred.parquet", "test2025_pred.parquet"):
        p = os.path.join(PRED_DIR, fn)
        if not os.path.exists(p):
            print(f"{fn}: 不存在，跳过")
            continue
        pr = pd.read_parquet(p)
        pr["yr"] = (pr["date"] // 10000).astype(int)
        print(f"\n--- {fn} ---")
        print("EV 分布(10/50/90分位):",
              pr["EV"].quantile([.1, .5, .9]).round(4).tolist())
        top = pr[pr["EV"] >= pr["EV"].quantile(0.9)]
        print(f"Top10%: n={len(top)}, 实际净收益均={top['net_ret'].mean():+.3%}, "
              f"target_atr构成={top['target_atr'].value_counts(normalize=True).round(2).to_dict()}")


if __name__ == "__main__":
    main()
