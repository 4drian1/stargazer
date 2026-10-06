# -*- coding: utf-8 -*-
"""T5 双头 OOF 合成 v3（2026-09-09 冻结方案，§18.31）。

输入为两个头各自通过闸门的冻结版本（--risk-ver / --opp-ver，AM/PM 斜杠
分隔，如 'R6/O6'；基座与特征组定义复用 ablation_risk / ablation_opp
的常量，避免口径漂移）。机会头主标签=o_t1_bt_5d（3d/5d 经济账本口径）。

流程（fold1 为部署方向；2026 复核只能否定，不据此调参）：
1. 同一建模池（与消融脚本同过滤口径），AM/PM 各头分别：
   - 训练期（2025-01~08）group_id GroupKFold(5) OOF 概率；
   - OOF 上 Isotonic 校准，应用到测试期预测；
2. 两个头在测试期逐行拼接。
3. EV 合成：训练期 OOF 校准概率上分别拟合
   EV_3d = f(P_risk, P_opp, atr_pct, dp) 与 EV_5d（同一形式，两套
   回归目标 o_net_ret_3d / o_net_ret_5d）。EV_3d 与 EV_5d 分开保留，
   不主观合成一个分数。两个概率不人工赋权、不简单相减。
4. 二维行动映射（阈值=训练期 OOF 中位数）：
   BUY_CANDIDATE=低风险×高机会；WAIT=高风险×高机会；
   WATCH=低风险×低机会；REJECT=高风险×低机会。仅建议不自动下单。
5. 组合闸门：BUY 池 5d 实际净收益>0 且>同池 M0 基线、3d 不恶化、
   STOP(5d) 不恶化。

--recheck2026：冻结版本在 2026（训练=2025 全年，测试 2026-01~08）上的
时间复核，只能否定。输出 combine_t5_*_{tag}.* + §18.31。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from ablation_risk import D0F, M0, T1F, T2AM, T2PM, T6R
from ablation_opp import T4_V2, T6O, V3

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
S25 = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
S26 = os.path.join(OUT_DIR, "minute_labels/minute_samples_2026.parquet")
D0 = os.path.join(OUT_DIR, "d0_daily_features_{y}.parquet")
PF = os.path.join(OUT_DIR, "minute_path_features_{y}.parquet")
ST = os.path.join(OUT_DIR, "minute_second_test_{y}.parquet")
TPM = os.path.join(OUT_DIR, "t2pm_features_{y}.parquet")
T3 = os.path.join(OUT_DIR, "minute_t3_vwap_{y}.parquet")
T4 = os.path.join(OUT_DIR, "minute_t4_proxy_{y}.parquet")
T6 = os.path.join(OUT_DIR, "t6_features_{y}.parquet")
LBL = os.path.join(OUT_DIR, "dual_head_labels_{y}.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
HEADS = {"AM": (1000, 1030), "PM": (1330, 1400, 1430, 1450)}
LGB_PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05,
                  min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
                  seed=7, n_estimators=400, verbosity=-1, n_jobs=8)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def risk_set(ver, head):
    t2 = T2AM if head == "AM" else T2PM
    d = {"0": [], "1": T1F, "2": t2, "6": T6R,
         "12": T1F + t2, "126": T1F + t2 + T6R}
    base = M0 + D0F if ver.startswith("RD") else M0
    suf = ver[2:] if ver.startswith("RD") else ver[1:]
    return base + d[suf]


def opp_set(ver):
    d = {"0": [], "3": V3, "4": T4_V2, "6": T6O,
         "34": V3 + T4_V2, "346": V3 + T4_V2 + T6O}
    base = M0 + D0F if ver.startswith("OD") else M0
    suf = ver[2:] if ver.startswith("OD") else ver[1:]
    return base + d[suf]


def load_pool(year, dps):
    y = "2025" if year == 2025 else "2026"
    df = pd.read_parquet(S25 if year == 2025 else S26)
    df = df[df["dp"].isin(dps)].copy()
    files = [(D0.format(y=y), ["code", "date"]),
             (PF.format(y=y), ["code", "dp_date", "dp"]),
             (ST.format(y=y), ["code", "dp_date", "dp"]),
             (TPM.format(y=y), ["code", "dp_date", "dp"]),
             (T3.format(y=y), ["code", "dp_date", "dp"]),
             (T4.format(y=y), ["code", "dp_date", "dp"]),
             (T6.format(y=y), ["code", "dp_date", "dp"]),
             (LBL.format(y=y), ["code", "dp_date", "dp"])]
    for p, on in files:
        m = pd.read_parquet(p)
        m["code"] = m["code"].astype(str)
        df["code"] = df["code"].astype(str)
        df = df.merge(m, on=on, how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()
            & df["r_sbr_t2"].notna() & df["o_t1_bt_5d"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    return df


def brier_bin(y, p):
    return float(((p - y) ** 2).mean())


def fit_head(tr, feats, ycol):
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold
    y = tr[ycol].astype(float).values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(tr))
    for tri, vai in gkf.split(tr, y, groups=tr["group_id"]):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(tr.iloc[tri][feats], y[tri])
        oof[vai] = m.predict_proba(tr.iloc[vai][feats])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(tr[feats], y)
    return m, cal, oof


def head_metrics(y, p_raw, p_cal):
    from sklearn.metrics import average_precision_score
    return {"brier_raw": brier_bin(y, p_raw),
            "brier_cal": brier_bin(y, p_cal),
            "pr_auc": float(average_precision_score(y, p_cal))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-ver", required=True,
                    help="AM/PM 版本，斜杠分隔，如 'R6/R6'（§18.28 冻结）")
    ap.add_argument("--opp-ver", required=True,
                    help="AM/PM 版本，斜杠分隔，如 'O6/O6'（§18.29 冻结）")
    ap.add_argument("--recheck2026", action="store_true")
    a = ap.parse_args()
    risk_v = dict(zip(("AM", "PM"), a.risk_ver.split("/")))
    opp_v = dict(zip(("AM", "PM"), a.opp_ver.split("/")))
    t0 = time.time()

    tr_years, te_year = [2025], 2025
    if a.recheck2026:
        tr_years, te_year = [2025], 2026
    te_frames, tr_oof = [], {}
    head_info = {}
    for head, dps in HEADS.items():
        trs = [load_pool(y, dps) for y in tr_years]
        tr = pd.concat(trs, ignore_index=True)
        te = load_pool(te_year, dps)
        if not a.recheck2026:
            tr = tr[tr["date"] <= 20250831]
            te = te[(te["date"] >= 20250901) & (te["date"] <= 20251231)]
        else:
            te = te[te["date"] <= 20260831]
        ycol_r, ycol_o = "r_sbr_t2", "o_t1_bt_5d"
        mr, calr, oofr = fit_head(tr, risk_set(risk_v[head], head), ycol_r)
        mo, calo, oofo = fit_head(tr, opp_set(opp_v[head]), ycol_o)
        pr_te = calr.predict(mr.predict_proba(te[risk_set(risk_v[head], head)])[:, 1])
        po_te = calo.predict(mo.predict_proba(te[opp_set(opp_v[head])])[:, 1])
        info_r = head_metrics(te[ycol_r].values,
                              mr.predict_proba(te[risk_set(risk_v[head], head)])[:, 1],
                              pr_te)
        info_o = head_metrics(te[ycol_o].values,
                              mo.predict_proba(te[opp_set(opp_v[head])])[:, 1],
                              po_te)
        head_info[head] = {"risk": info_r, "opp": info_o}
        log(f"[{head}] risk {risk_v[head]}: brier {info_r['brier_raw']:.4f}->"
            f"{info_r['brier_cal']:.4f} prAUC={info_r['pr_auc']:.3f} | "
            f"opp {opp_v[head]}: {info_o['brier_raw']:.4f}->"
            f"{info_o['brier_cal']:.4f} prAUC={info_o['pr_auc']:.3f}")
        tro = tr.copy()
        tro["P_risk"] = calr.predict(oofr)
        tro["P_opp"] = calo.predict(oofo)
        tro["head"] = head
        tr_oof[head] = tro
        tec = te.copy()
        tec["P_risk"] = pr_te
        tec["P_opp"] = po_te
        tec["head"] = head
        te_frames.append(tec)
    te_all = pd.concat(te_frames, ignore_index=True)
    tro_all = pd.concat([d[["P_risk", "P_opp", "atr_pct", "dp", "nr",
                            "r_sbr_t2", "o_t1_bt_5d", "y5", "head",
                            "o_net_ret_3d", "o_net_ret_5d", "o_stop_5d"]]
                         for d in tr_oof.values()], ignore_index=True)

    # ---- EV 合成：EV_3d 与 EV_5d 分开保留，不主观合成一个分数 ----
    import lightgbm as lgb
    ev_feats = ["P_risk", "P_opp", "atr_pct", "dp"]
    ev_stats = {}
    for hor, tgt in (("3d", "o_net_ret_3d"), ("5d", "o_net_ret_5d")):
        reg = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05,
                                min_child_samples=200, subsample=0.9,
                                colsample_bytree=0.9, seed=7,
                                n_estimators=200, verbosity=-1, n_jobs=8)
        reg.fit(tro_all[ev_feats], tro_all[tgt])
        te_all[f"ev_hat_{hor}"] = reg.predict(te_all[ev_feats])
        hi = te_all[te_all[f"ev_hat_{hor}"] >= te_all[f"ev_hat_{hor}"].quantile(0.7)]
        from sklearn.metrics import r2_score
        ev_stats[hor] = {
            "hi_nr": float(hi[tgt].mean()),
            "hi_n": len(hi),
            "pool_nr": float(te_all[tgt].mean()),
            "r2": float(r2_score(te_all[tgt], te_all[f"ev_hat_{hor}"]))}
        log(f"EV_{hor}: Top30%实际 {ev_stats[hor]['hi_nr']:+.3%} vs 池 "
            f"{ev_stats[hor]['pool_nr']:+.3%} R2={ev_stats[hor]['r2']:.4f}")

    # ---- 二维行动映射（阈值=训练期 OOF 中位数） ----
    thr_r = float(tro_all["P_risk"].median())
    thr_o = float(tro_all["P_opp"].median())

    def act(r):
        hi_risk = r["P_risk"] >= thr_r
        hi_opp = r["P_opp"] >= thr_o
        if not hi_risk and hi_opp:
            return "BUY_CANDIDATE"
        if hi_risk and hi_opp:
            return "WAIT"
        if not hi_risk and not hi_opp:
            return "WATCH"
        return "REJECT"
    te_all["action"] = te_all.apply(act, axis=1)
    q = te_all.groupby("action").agg(
        n=("nr", "size"), nr5=("o_net_ret_5d", "mean"),
        nr3=("o_net_ret_3d", "mean"),
        stop=("r_sbr_t2", "mean"), opp=("o_t1_bt_5d", "mean"),
        stop5d=("o_stop_5d", "mean")).reset_index()
    log(q.to_string(index=False))

    tag = "recheck2026" if a.recheck2026 else "fold1"
    te_all.to_parquet(os.path.join(OUT_DIR,
                                   f"combine_t5_pool_{tag}.parquet"),
                      index=False)
    q.to_csv(os.path.join(OUT_DIR, f"combine_t5_quadrants_{tag}.csv"),
             index=False)

    # ---- 报告 §18.31 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.31 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = [f"\n### 18.31 T5 双头 OOF 合成 v3（{tag}，risk={a.risk_ver} "
         f"opp={a.opp_ver}（AM/PM））\n",
         "机会头主标签=o_t1_bt_5d（3d/5d 经济账本口径）。流程：各头训练期"
         "(2025-01~08) group_id GroupKFold(5) OOF → Isotonic 校准 → 测试期"
         "预测；EV_3d 与 EV_5d 分别=f(P_risk,P_opp,atr_pct,dp) 在训练期OOF上"
         "拟合（LGBM回归），两套EV分开保留不合成；行动映射阈值=训练期OOF"
         "中位数（待人工确认），模型只给建议不自动下单。\n",
         "头校准表现（测试期）：\n",
         "| 头 | 标签 | Brier(原始→校准) | PR-AUC |",
         "|---|---|---:|---:|"]
    for head in HEADS:
        ir, io = head_info[head]["risk"], head_info[head]["opp"]
        L.append(f"| {head} | r_sbr_t2 | {ir['brier_raw']:.4f}→"
                 f"{ir['brier_cal']:.4f} | {ir['pr_auc']:.3f} |")
        L.append(f"| {head} | o_t1_bt_5d | {io['brier_raw']:.4f}→"
                 f"{io['brier_cal']:.4f} | {io['pr_auc']:.3f} |")
    L += ["\nEV 合成（测试期，Top30%预测EV 的实际净收益 vs 同池基线）："]
    for hor in ("3d", "5d"):
        s = ev_stats[hor]
        L.append(f"- EV_{hor}: Top30%实际 **{s['hi_nr']:+.3%}**（n={s['hi_n']:,}）"
                 f" vs 池 {s['pool_nr']:+.3%}；R²={s['r2']:.4f}")
    L += ["\n行动映射（测试期，阈值 P_risk≥" f"{thr_r:.3f} / P_opp≥{thr_o:.3f}）：\n",
          "| 行动 | n | 净收益5d | 净收益3d | STOP(T+2) | 机会t1_bt_5d | STOP5d |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in q.iterrows():
        L.append(f"| {r['action']} | {int(r['n']):,} | {r['nr5']:+.3%} | "
                 f"{r['nr3']:+.3%} | {r['stop']:.1%} | {r['opp']:.1%} | "
                 f"{r['stop5d']:.1%} |")
    buy = q[q["action"] == "BUY_CANDIDATE"]
    if len(buy):
        b = buy.iloc[0]
        conds = {"nr5>0": b["nr5"] > 0,
                 "nr5>池": b["nr5"] > ev_stats["5d"]["pool_nr"],
                 "nr3不劣": b["nr3"] >= ev_stats["3d"]["pool_nr"] - 1e-4,
                 "stop5d不恶化": b["stop5d"] <= te_all["o_stop_5d"].mean() + 1e-4}
        verdict = "通过" if all(conds.values()) else "未通过"
        L.append(f"\n组合闸门：BUY_CANDIDATE 5d {b['nr5']:+.3%}、3d {b['nr3']:+.3%}、"
                 f"STOP5d {b['stop5d']:.1%}（池 {te_all['o_stop_5d'].mean():.1%}）→ "
                 f"**{verdict}**（" + "、".join(f"{k}={'✓' if v else '✗'}"
                 for k, v in conds.items()) + f"）。仅 {tag} 口径。")
    L.append(f"\n明细：combine_t5_pool_{tag}.parquet / quadrants CSV。"
             f"用时 {time.time()-t0:.0f}s")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
