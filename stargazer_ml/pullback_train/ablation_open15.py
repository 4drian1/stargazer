# -*- coding: utf-8 -*-
"""盘中路线第6步：M0–M3 ATR开盘冲击-承接特征组增量消融（2025开发数据）。

消融结构（固定事件、固定决策点9:45 vwap口径、固定标签t2.0、固定group_id
时序切分，只变特征）：
  M0 = 当前盘中模型v1特征（12个，minute_four_way.FEATS）
  M1 = M0 + ATR开盘冲击组（open15_* 10个）
  M2 = M1 + 收回/VWAP交互（shock_recovery_ix、downside_vwap_ix）
  M3 = M2 + 板块日级相对强弱（m3_sector_ret1/excess1/h2c/score；
       板块指数分钟数据stockdb无覆盖，M3按评审降级为日级强弱并记录缺口）

时间折（描述性，双向）：
  fold1: train=20250101-20250831, test=20250901-20251231
  fold2: train=20250901-20251231, test=20250101-20250831
（2025全年为开发数据，本消融只证明工程与特征可行性，不宣称样本外有效。）

过滤：entry_type=vwap、censored=0、net_ret_t2.0 非空、price_scale_valid=True
（跨除权窗口事件排除，§18.9；过滤规模在报告中记录）。

验收指标（每个fold×版本）：测试折 Top30%EV 净收益均值（vs M0）、Top30%
STOP率（不恶化）、Brier、EV-净收益 Spearman、按 src(A2/B0/LEGACY) 分层。
另输出 open15_tr_atr_ratio 预定义分箱 × Recovery高低 × src 的收益/STOP
/MFE/MAE 报告（不做硬门槛）。
报告写入 §18.10；分箱明细落盘 open15_bins_report.csv。
"""
import json
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
BINS_CSV = os.path.join(OUT_DIR, "open15_bins_report.csv")
PF_CSV = os.path.join(OUT_DIR, "minute_path_features_2025.parquet")
KT = "2.0"
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
M1_ADD = ["open15_tr_atr_ratio", "open15_gap_atr_ratio",
          "open15_range_atr_ratio", "open15_downside_atr_ratio",
          "open15_upside_atr_ratio", "open15_recovery_ratio",
          "open15_close_location", "open15_lower_wick_ratio",
          "open15_vwap_reclaim", "open15_volume_ratio"]
M2_ADD = ["shock_recovery_ix", "downside_vwap_ix"]
M3_ADD = ["m3_sector_ret1", "m3_sector_excess1", "m3_sector_h2c",
          "m3_sector_score"]
# T1：盘中路径结构特征（path_features.py 产出，左连 minute_samples）
T1_ADD = ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
          "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows"]
MODELS = {"M0": M0, "M1": M0 + M1_ADD, "M2": M0 + M1_ADD + M2_ADD,
          "M3": M0 + M1_ADD + M2_ADD + M3_ADD, "T1": M0 + T1_ADD}
FOLDS = [("fold1", 20250101, 20250831, 20250901, 20251231),
         ("fold2", 20250901, 20251231, 20250101, 20250831)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_sector_features(df):
    """信号日(T)日级板块强弱：code→12关注板块映射 + sector_daily + 市场。"""
    cons = json.load(open(os.path.join(OUT_DIR, "board_cons.json"),
                          encoding="utf-8"))
    code2board = {}
    for b, codes in cons.items():
        for c in codes:
            code2board.setdefault(c, b)  # 多板块取第一个，固定不事后择优
    sd = pd.read_parquet(os.path.join(OUT_DIR, "sector_daily.parquet"),
                         columns=["date", "sector", "close", "high", "pct_chg"])
    sd = sd[(sd["date"] >= 20250101) & (sd["date"] <= 20251231)].copy()
    sd["pre_close"] = sd["close"] / (1 + sd["pct_chg"] / 100.0)
    sd["m3_sector_h2c"] = (sd["high"] - sd["close"]) / sd["pre_close"]
    sd = sd.rename(columns={"pct_chg": "m3_sector_ret1"})[
        ["date", "sector", "m3_sector_ret1", "m3_sector_h2c"]]
    mk = pd.read_parquet(os.path.join(OUT_DIR, "market_daily.parquet"),
                         columns=["date", "sh000001_pct_chg"])
    mk = mk.rename(columns={"sh000001_pct_chg": "mkt_ret1"})
    sd = sd.merge(mk, on="date", how="left")
    sd["m3_sector_excess1"] = sd["m3_sector_ret1"] - sd["mkt_ret1"]
    sc = pd.read_parquet(os.path.join(OUT_DIR, "sector_day_scores.parquet"),
                         columns=["sector_name", "date", "sector_gate_score"])
    sc = sc.rename(columns={"sector_name": "sector",
                            "sector_gate_score": "m3_sector_score"})
    sd = sd.merge(sc, on=["sector", "date"], how="left")
    df = df.copy()
    df["sector"] = df["code"].map(code2board)
    df = df.merge(sd.drop(columns=["mkt_ret1"]), on=["sector", "date"],
                  how="left")
    return df


def brier(y_true, proba, classes):
    idx = {c: i for i, c in enumerate(classes)}
    oh = np.zeros_like(proba)
    for r, y in enumerate(y_true):
        oh[r, idx[y]] = 1.0
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def run_fold(tr, te, feats):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=31,
                             learning_rate=0.05, min_child_samples=100,
                             subsample=0.9, colsample_bytree=0.9, seed=7,
                             n_estimators=400, verbosity=-1, n_jobs=8)
    clf.fit(tr[feats], tr["y"])
    proba = clf.predict_proba(te[feats])
    ev = np.zeros(len(te))
    for i, cl in enumerate(clf.classes_):
        gain = tr.loc[tr["y"] == cl, "nr"].mean()
        ev += proba[:, i] * gain
    te = te.copy()
    te["EV"] = ev
    thr = te["EV"].quantile(0.7)
    top = te[te["EV"] >= thr]
    oc_top = top[f"outcome_t{KT}"]
    oc_all = te[f"outcome_t{KT}"]
    spear = te[["EV", "nr"]].corr(method="spearman").iloc[0, 1]
    return {"n_train": len(tr), "n_test": len(te),
            "all_mean": float(te["nr"].mean()),
            "all_stop": float((oc_all == "STOP").mean()),
            "top_mean": float(top["nr"].mean()),
            "top_stop": float((oc_top == "STOP").mean()),
            "top_n": len(top),
            "brier": brier(te["y"].values, proba, clf.classes_),
            "spearman": float(spear)}


def bins_report(df):
    """open15_tr_atr_ratio 预定义分箱 × Recovery × src（9:45 vwap口径）。"""
    d = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
           & df["nr"].notna() & df["price_scale_valid"]].copy()
    bins = [-np.inf, 0.15, 0.20, 0.25, 0.35, np.inf]
    labels = ["<15%", "15-20%", "20-25%", "25-35%", ">35%"]
    d["shock_bin"] = pd.cut(d["open15_tr_atr_ratio"], bins=bins, labels=labels)
    d["rec_grp"] = np.where(d["open15_recovery_ratio"] >= 0.7, "rec>=0.7",
                            np.where(d["open15_recovery_ratio"] < 0.4,
                                     "rec<0.4", "0.4<=rec<0.7"))
    rows = []
    for (b, rg, src), g in d.groupby(["shock_bin", "rec_grp", "src"],
                                     observed=True):
        oc = g[f"outcome_t{KT}"]
        rows.append({"shock_bin": b, "rec_grp": rg, "src": src, "n": len(g),
                     "mean_nr": g["nr"].mean(),
                     "stop": (oc == "STOP").mean(),
                     "tgt": (oc == "TARGET").mean(),
                     "timeout": (oc == "TIMEOUT").mean(),
                     "mfe_atr": (g["mfe"] / g["atr_pct"]).median(),
                     "mae_atr": (g["mae"] / g["atr_pct"]).median()})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    df = pd.read_parquet(SAMPLES)
    # ---- T1：左连盘中路径结构特征（path_features.py 产出，on code/dp_date/dp）----
    pf = pd.read_parquet(PF_CSV)
    df = df.merge(pf, on=["code", "dp_date", "dp"], how="left")
    df = df[(df["dp"] == 945) & (df["entry_type"] == "vwap")].copy()
    log(f"T1 左连：pf 表 {len(pf):,} 行；9:45 vwap 样本 pf_ret1m 命中 "
        f"{df['pf_ret1m'].notna().sum():,}/{len(df):,}")
    n_raw = len(df)
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
        log("warn: 样本无除权标志列，视为全有效")
    ok = df["price_scale_valid"].astype(bool)
    log(f"9:45 vwap 样本 {n_raw:,}；price_scale_valid=False 排除 "
        f"{int((~ok).sum()):,}；censored 排除 "
        f"{int((df['censored'] != 0).sum()):,}")
    df = df[ok & (df["censored"] == 0) & df["nr"].notna()].copy()
    log(f"入模样本: {len(df):,}（src: {df['src'].value_counts().to_dict()}）")

    df = add_sector_features(df)
    for c in M1_ADD + M2_ADD + M3_ADD + M0 + T1_ADD:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    log(f"M3板块特征覆盖: sector映射 "
        f"{df['sector'].notna().mean():.1%}, score非空 "
        f"{df['m3_sector_score'].notna().mean():.1%}")

    # ---- 消融 ----
    results = []
    for fname, ftr0, ftr1, t0d, t1d in FOLDS:
        tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
        te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
        for ver, feats in MODELS.items():
            r = run_fold(tr, te, feats)
            results.append({"fold": fname, "model": ver, **r})
            log(f"{fname} {ver}: top_mean={r['top_mean']:+.3%} "
                f"top_stop={r['top_stop']:.1%} all={r['all_mean']:+.3%} "
                f"brier={r['brier']:.4f} spear={r['spearman']:+.4f}")
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR, "ablation_m0_m3.csv"), index=False)
    # ---- T1 专门摘要（fold1/fold2，仅 T1 行，不依赖 src_res）----
    t1_res = res[res["model"] == "T1"].copy()
    t1_res.to_csv(os.path.join(OUT_DIR, "ablation_T1.csv"), index=False)

    # ---- src 分层（M3 vs M0，fold1）----
    src_rows = []
    ftr0, ftr1, t0d, t1d = FOLDS[0][1:]
    tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
    te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
    for src in ("A2", "B0", "LEGACY"):
        tes = te[te["src"] == src]
        trs = tr[tr["src"] == src]
        if len(tes) < 500:
            continue
        for ver in ("M0", "T1", "M3"):
            r = run_fold(trs, tes, MODELS[ver])
            src_rows.append({"fold": "fold1", "src": src, "model": ver, **r})
            log(f"fold1 {src} {ver}: top={r['top_mean']:+.3%} "
                f"stop={r['top_stop']:.1%}")
    src_res = pd.DataFrame(src_rows)
    src_res.to_csv(os.path.join(OUT_DIR, "ablation_m0_m3_by_src.csv"),
                   index=False)
    # T1 分来源摘要（依赖 src_res）
    t1_src = src_res[src_res["model"] == "T1"].copy()
    t1_src.to_csv(os.path.join(OUT_DIR, "ablation_T1_by_src.csv"), index=False)

    # ---- T1 vs M0 对比（fold1/fold2）----
    def _row(fold, model):
        sub = res[(res["fold"] == fold) & (res["model"] == model)]
        return None if len(sub) == 0 else sub.iloc[0]
    t1cmp = []
    for fld in ("fold1", "fold2"):
        m0 = _row(fld, "M0")
        t1 = _row(fld, "T1")
        if m0 is not None and t1 is not None:
            t1cmp.append((fld, m0, t1))

    # ---- 跨月覆盖（fold1，25H2 各月 T1 vs M0）----
    ftr0, ftr1, t0d, t1d = FOLDS[0][1:]
    tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
    te = df[(df["date"] >= t0d) & (df["date"] <= t1d)].copy()
    te["month"] = te["date"] // 100
    month_rows = []
    for mo in sorted(te["month"].unique()):
        tem = te[te["month"] == mo]
        for ver in ("M0", "T1"):
            r = run_fold(tr, tem, MODELS[ver])
            month_rows.append({"month": int(mo), "model": ver, **r})
    month_res = pd.DataFrame(month_rows)
    month_res.to_csv(os.path.join(OUT_DIR, "ablation_T1_by_month.csv"),
                     index=False)

    # ---- 分箱报告 ----
    br = bins_report(df)
    br.to_csv(BINS_CSV, index=False, float_format="%.4f")

    # ---- 汇总报告 §18.10 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.10 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.10 ATR开盘冲击-承接特征组消融 M0–M3（第6步）\n",
         "设计：固定事件（LEGACY∪A2∪B0，9:45 vwap，目标2.0ATR，Ks按来源）、"
         "固定group_id时序切分，只变特征。M0=盘中模型v1十二特征；"
         "M1=+open15 ATR开盘冲击组（10个，分母=信号日冻结ATR_ref）；"
         "M2=+收回/VWAP交互2个；M3=+板块日级强弱4个"
         "（**板块指数分钟数据stockdb无覆盖，M3按评审降级为日级强弱，"
         "open15_relative_sector_ret 留空，缺口记录**）。\n",
         f"- 样本：9:45 vwap 原始 {n_raw:,}，除权窗口排除后入模 {len(df):,}"
         f"（{df['src'].value_counts().to_dict()}）",
         "- 时间折（描述性双向）：fold1 train=25H1/test=25H2；"
         "fold2 反向。2025为开发数据，不宣称样本外有效。\n",
         "| fold | 版本 | Top30%净收益 | Top30%STOP | 全体净收益 | "
         "Brier | EV-收益Spearman |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['fold']} | {r['model']} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} | {r['all_mean']:+.2%} | "
                 f"{r['brier']:.4f} | {r['spearman']:+.3f} |")
    L.append("")
    L.append("src分层（fold1，M0 vs T1 vs M3）：")
    L.append("| src | 版本 | n_test | Top30%净收益 | Top30%STOP | Brier |")
    L.append("|---|---|---:|---:|---:|---:|")
    for _, r in src_res.iterrows():
        L.append(f"| {r['src']} | {r['model']} | {r['n_test']:,} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} |")
    L.append("")
    L.append(f"- open15分箱×Recovery×src 明细：{BINS_CSV}"
             "（预定义分箱 <15/15-20/20-25/25-35/>35% × rec≥0.7/0.4-0.7/<0.4"
             " × A2/B0/LEGACY；报告 n/净收益/STOP/TARGET/TIMEOUT/MFE/MAE_ATR）")
    L.append("- 验收口径：两fold方向一致 + Top30%净收益>M0 + STOP不恶化 + "
             "Brier不恶化 + 非单一来源贡献；未过验收则特征组只留观察不入建议。")
    L.append("\n判读边界：本节为特征可行性证据；真正前瞻验证从2026-09后的"
             "分钟样本开始。")
    # ---- 新增 §18.11：T1 路径结构特征增量（第7步）----
    L.append("\n### 18.11 T1 盘中路径结构特征增量（第7步）\n")
    L.append("设计：在 M0（12 分钟特征）之上**仅加 7 个路径结构特征**"
             "（pf_ret1m/3m/5m、pf_decline_speed、pf_decel、"
             "pf_low2_minus_low1、pf_has_two_lows），复用 M0 的事件/DP(9:45)/"
             "标签(t2.0)/切分，只变特征一个变量。特征由 path_features.py 产出，"
             "窗口对齐标签引擎 [09:30, DP]，ATR 单位=价格差/(sig_close×atr_pct)，"
             "局部低点仅取 DP 窗口内（无前视）；09:45 窗口仅 15 分钟，双底多为 "
             "NaN，用 pf_has_two_lows 标志不填充。")
    L.append("")
    L.append("| fold | 对比 | Top30%净收益 | ΔvsM0 | Top30%STOP | ΔSTOP | "
             "全体净收益 | Brier | ΔBrier | Spearman |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fld, m0, t1 in t1cmp:
        L.append(f"| {fld} | M0 | {m0['top_mean']:+.2%} | — | "
                 f"{m0['top_stop']:.1%} | — | {m0['all_mean']:+.2%} | "
                 f"{m0['brier']:.4f} | — | {m0['spearman']:+.3f} |")
        L.append(f"| {fld} | T1 | {t1['top_mean']:+.2%} | "
                 f"{t1['top_mean']-m0['top_mean']:+.2%} | "
                 f"{t1['top_stop']:.1%} | {t1['top_stop']-m0['top_stop']:+.1%} | "
                 f"{t1['all_mean']:+.2%} | {t1['brier']:.4f} | "
                 f"{t1['brier']-m0['brier']:+.4f} | {t1['spearman']:+.3f} |")
    L.append("")
    L.append("src分层（fold1，M0 vs T1）：")
    L.append("| src | 版本 | n_test | Top30%净收益 | Top30%STOP | Brier |")
    L.append("|---|---|---:|---:|---:|---:|")
    for _, r in t1_src.iterrows():
        L.append(f"| {r['src']} | {r['model']} | {r['n_test']:,} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} |")
    L.append("")
    L.append("跨月覆盖（fold1/test=25H2，各月 T1 vs M0）：")
    L.append("| 月份 | 版本 | n_test | Top30%净收益 | Top30%STOP | Brier |")
    L.append("|---|---|---:|---:|---:|---:|")
    for _, r in month_res.iterrows():
        L.append(f"| {r['month']} | {r['model']} | {r['n_test']:,} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} |")
    L.append("")
    L.append("- 验收（定性）：fold1 Top30%净收益≥M0 且 STOP不恶化 且 Brier不恶化"
             " 且 三来源(A2/B0/LEGACY)方向一致 且无方向反转；fold2 仅作稳定性诊断、"
             "不用于升级裁决。明细见 ablation_T1.csv / ablation_T1_by_src.csv。")
    L.append("- 判读边界：2025 为开发数据，T1 仅证明路径特征增量可行；"
             "真正前瞻验证从 2026-09 后分钟样本开始。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
