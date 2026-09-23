"""外围行情风险基线（Global Context Model，F-v2.2 复杂度控制版）。

样本单位 = 一行（trade_date × strategy_id），主模型只用 tarot.hermit.a2@v1；
legacy 池只做方向一致性推断对照，不与 a2 拼接训练。

F-v2.2 对比四组（统一沿用 F-v2.0 的时点/冻结产物口径）：
- C1   固定外围规则（阈值写死，未用 2024 调）
- C2-L Ridge 回归（标准化 + median 缺失填补，alpha=1.0 设计期冻结，全特征）
- C2-S 精简特征 LightGBM（约 30 个特征，按经济逻辑预先冻结，见 SMALL_FEATURES，
       绝不用 2024/2025 重要性倒推）
- C2-F 现状 156 特征 LightGBM

冻结产物三件套（F-v2.0 修复 2，对 C2-L/C2-S/C2-F 各自独立执行）：
- 2019–2021：训练模型（此后再不重训）；
- 2022–2023：冻结模型预测 → isotonic 校准 → 冻结高风险阈值
  （前 EMBARGO 个交易日因标签窗口与训练期重叠而剔除）；
- 2024/2025/2026：始终同一套冻结模型 + 校准器 + 阈值（与模型同尺度）。
C0 先验与 C1 校准/阈值同样只用 2019–2021 训练期冻结。

主标签 y = stop_rate_3d（报告中写作 loss3_touch_rate，语义：池内 entry*(1-3%)
固定 3% 触碰口径的止损触碰率，不是 A2 实际 ATR 止损）。

用法（在 corvus/ 目录下）：
    PYTHONIOENCODING=utf-8 python -m corvus.global_baseline
可重复运行，幂等（预测 parquet 与报告整体重写）。
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .db import get_conn, init_db
from .global_features import INSTRUMENTS, build_features, feature_columns
from .global_report import render_report

MAIN_SID = "tarot.hermit.a2@v1"
LEGACY_SID = "tarot.hermit.legacy@v1"
MODEL_SIDS = [MAIN_SID, LEGACY_SID]   # C0/C1 两个池都出（先验/规则与池无关）；ML 模型同样套用到两池

PERIODS = {
    "train": ("2019-01-01", "2021-12-31"),     # 训练（冻结模型）
    "calib": ("2022-01-01", "2023-12-31"),     # 校准 + 冻结阈值（前 5 日 embargo）
    "dev2024": ("2024-01-01", "2024-12-31"),
    "stab2025": ("2025-01-01", "2025-12-31"),
    "stress2026": ("2026-01-01", "2026-08-31"),
}

EMBARGO = 5            # 校准期前 5 个交易日剔除（标签窗口 t+1..t+5 与训练期重叠）
HIGH_RISK_Q = 0.80     # 高风险阈值 = 冻结模型 2022–2023 原始分 80 分位（冻结）
PRAUC_Q = 0.70         # 报告用「高风险日」定义：真实 loss3_touch_rate > 全期 70 分位（冻结常量）

LGB_PARAMS = dict(objective="regression", n_estimators=800, learning_rate=0.02,
                  num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=1.0, random_state=42,
                  verbose=-1, n_jobs=4)
RIDGE_ALPHA = 1.0      # Ridge 正则强度（设计期冻结，未用 2024 调）

# C2-S 精简特征集（F-v2.2 按经济逻辑预先冻结，绝不用 2024/2025 重要性倒推；
# SPX/NDX/DJI 大量同源衍生不重复保留，DJI 仅通过 us_three_down_cnt 间接体现）
SMALL_FEATURES = [
    # 美股核心：SPX/NDX 的 1/5/20 日收益 + 20 日波动
    "SPX_ret_1", "SPX_ret_5", "SPX_ret_20", "SPX_vol_20",
    "NDX_ret_1", "NDX_ret_5", "NDX_ret_20", "NDX_vol_20",
    # VIX：水平 + 1/5 日变化
    "VIX_lvl", "VIX_ret_1", "VIX_ret_5",
    # 港股：HSI/HSTECH 的 1/5/20 日收益 + 波动
    "HSI_ret_1", "HSI_ret_5", "HSI_ret_20", "HSI_vol_20",
    "HSTECH_ret_1", "HSTECH_ret_5", "HSTECH_ret_20", "HSTECH_vol_20",
    # 日经期货：1/5 日收益
    "N225_ret_1", "N225_ret_5",
    # 美债：水平、单日 bp 变化、期限利差
    "US2Y_lvl", "us2y_chg1_bp", "US10Y_lvl", "us10y_chg1_bp", "ust_2s10s",
    # 商品：WTI/黄金的 1/5/20 日收益
    "WTI_ret_1", "WTI_ret_5", "WTI_ret_20",
    "GOLD_ret_1", "GOLD_ret_5", "GOLD_ret_20",
    # 跨市场聚合 + 数据质量
    "us_three_down_cnt", "sync_crash_cnt", "us_down_max", "n_missing", "max_stale",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRED_PATH = DATA_DIR / "global_baseline_predictions.parquet"
FEAT_PATH = DATA_DIR / "global_baseline_features.parquet"
REPORT_PATH = DATA_DIR / "reports" / "global_baseline_c0c1c2.md"


def load_labels(conn, sid):
    """冻结日标签，trade_date 统一为 YYYY-MM-DD；CENSORED_TAIL 全部排除。"""
    df = pd.read_sql_query(
        """select trade_date, strategy_id, candidate_count, entry_valid_rate,
                  stop_rate_3d, loss_rate_3d, loss_rate_5d,
                  mean_net_return_3d, mean_net_return_5d, left_tail_return_5d,
                  n_valid_3d, n_valid_5d
           from tarot_day_labels
           where strategy_id = ? and label_status = 'OK'""",
        conn, params=(sid,))
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    return df.sort_values("trade_date").reset_index(drop=True)


def assign_period(trade_date):
    for name, (s, e) in PERIODS.items():
        if s <= trade_date <= e:
            return name
    return "other"


def train_ml_model(feat, y, train_dates, calib_dates, cols, kind):
    """F-v2.2 通用冻结产物三件套（C2-L/C2-S/C2-F 各自独立执行）：
    1) 2019–2021 训练模型（此后再不重训）；
    2) 冻结模型预测 2022–2023，isotonic 校准器只在这些预测上拟合；
    3) 高风险阈值 = 冻结模型 2022–2023 原始分的 80 分位（与模型同尺度）。
    kind='lgb'（LightGBM，原生容忍 NaN）或 'ridge'（median 填补+标准化+Ridge）。
    返回 dict(model, cal, cut, ca_raw, cols, kind)。"""
    X = feat.set_index("trade_date")
    m_tr = y.loc[train_dates].notna()
    if kind == "lgb":
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X.loc[train_dates, cols][m_tr], y.loc[train_dates][m_tr])
    else:  # ridge：median 填补（填补器只在训练期拟合）+ 标准化 + Ridge
        model = make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(X.loc[train_dates, cols][m_tr], y.loc[train_dates][m_tr])

    ca_raw = pd.Series(model.predict(X.loc[calib_dates, cols]),
                       index=pd.Index(calib_dates))
    m_ca = y.loc[ca_raw.index].notna()
    cal = IsotonicRegression(out_of_bounds="clip").fit(ca_raw[m_ca].values,
                                                       y.loc[ca_raw.index][m_ca].values)
    cut = float(ca_raw[m_ca].quantile(HIGH_RISK_Q))
    return {"model": model, "cal": cal, "cut": cut, "ca_raw": ca_raw,
            "cols": cols, "kind": kind}


def c1_rule_score(f):
    """C1 固定规则基线：阈值全部写死在代码里（冻结于实验设计阶段，未用 2024 数据调）。
    规则分 = 触发条数加权（权重反映预设的严重度）。"""
    s = pd.Series(0.0, index=f.index)
    us3 = ((f["SPX_ret_1"] < -0.015) & (f["NDX_ret_1"] < -0.015)
           & (f["DJI_ret_1"] < -0.015)).fillna(False)
    vix = ((f["VIX_lvl"] > 25) & (f["VIX_ret_1"] > 0.15)).fillna(False)
    u10 = (f["us10y_chg1_bp"] > 10).fillna(False)
    wti = (f["WTI_ret_1"].abs() > 0.04).fillna(False)
    hsi = (f["HSI_ret_1"] < -0.02).fillna(False)
    sync = (f["sync_crash_cnt"] >= 3).fillna(False)
    s += us3 * 2.0 + vix * 2.0 + u10 * 1.0 + wti * 1.0 + hsi * 1.0 + sync * 2.0
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="外围行情风险基线 C0/C1/C2")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    init_db(args.db)

    conn = get_conn(args.db)
    try:
        print("[1/6] 加载冻结日标签...", flush=True)
        labels = {sid: load_labels(conn, sid) for sid in MODEL_SIDS}
        master = sorted(set(labels[MAIN_SID]["trade_date"]) | set(labels[LEGACY_SID]["trade_date"]))
        print(f"      a2={len(labels[MAIN_SID])} 天 legacy={len(labels[LEGACY_SID])} 天", flush=True)

        print("[2/6] 构建外围特征（8:30 快照可见性过滤）...", flush=True)
        feat = build_features(master, conn)
        FEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(FEAT_PATH, index=False)
        print(f"      {feat.shape[0]} 天 × {feat.shape[1] - 1} 特征，缓存 {FEAT_PATH}", flush=True)

        print("[3/6] C0 常数先验（只用 2019–2021 训练期冻结）...", flush=True)
        lab = labels[MAIN_SID].set_index("trade_date")
        train_dates = sorted(d for d in master if assign_period(d) == "train")
        calib_all = sorted(d for d in master if assign_period(d) == "calib")
        # 校准期前 EMBARGO 个交易日剔除：其标签窗口（t+1..t+5）与训练期重叠
        calib_dates = calib_all[EMBARGO:]
        y = lab["stop_rate_3d"]
        prior = float(y.loc[[d for d in train_dates if d in y.index]].mean())
        print(f"      训练期基础概率 prior={prior:.4f}；校准期 {len(calib_all)} 天，"
              f"embargo 剔除前 {EMBARGO} 天 → 用 {len(calib_dates)} 天", flush=True)

        print("[4/6] C1 固定规则基线...", flush=True)
        fidx = feat.set_index("trade_date")
        rule = c1_rule_score(fidx)
        r_tr = rule.loc[train_dates]
        m1 = y.reindex(r_tr.index).notna()
        c1_cal = IsotonicRegression(out_of_bounds="clip").fit(r_tr[m1], y.reindex(r_tr.index)[m1])
        c1_cut = float(rule.loc[train_dates].quantile(HIGH_RISK_Q))
        print(f"      规则分阈值(冻结)={c1_cut}", flush=True)

        print("[5/6] C2-L/C2-S/C2-F 冻结产物三件套（2019–2021 训练 → 2022–2023 校准+冻结阈值）...", flush=True)
        full_cols = feature_columns(feat)
        missing_small = [c for c in SMALL_FEATURES if c not in feat.columns]
        assert not missing_small, f"精简特征集有未实现列: {missing_small}"
        model_specs = {   # F-v2.2 四组对比中的三个 ML 版本（C1 在 [4/6]）
            "C2-S": {"cols": SMALL_FEATURES, "kind": "lgb"},
            "C2-L": {"cols": full_cols, "kind": "ridge"},
            "C2-F": {"cols": full_cols, "kind": "lgb"},
        }
        models = {}
        for name, spec in model_specs.items():
            art = train_ml_model(feat, y, train_dates, calib_dates,
                                 spec["cols"], spec["kind"])
            y_ca = y.loc[art["ca_raw"].index]
            m_ca = y_ca.notna()
            art["calib_spearman"] = float(art["ca_raw"][m_ca].corr(y_ca[m_ca], method="spearman"))
            models[name] = art
            print(f"      {name}({spec['kind']}, {len(spec['cols'])}特征): "
                  f"阈值={art['cut']:.4f} 校准期Spearman={art['calib_spearman']:.4f}", flush=True)

        print("[6/6] 逐日预测落盘...", flush=True)
        FOLD_TAG = {"train": "train_insample", "calib": "calib",
                    "dev2024": "frozen_apply", "stab2025": "frozen_apply",
                    "stress2026": "frozen_apply"}
        rows = []
        for sid in MODEL_SIDS:
            dates = [d for d in labels[sid]["trade_date"] if d in fidx.index]
            sub = fidx.loc[dates]
            per = pd.Series([assign_period(d) for d in dates], index=dates)
            # C0：常数先验（无排序能力，高风险标志恒 0）
            rows.append(pd.DataFrame({
                "trade_date": dates, "strategy_id": sid, "model": "C0",
                "score_raw": prior, "score_cal": prior, "high_risk_flag": 0,
                "fold": "-", "period": per.values}))
            # C1：规则分（校准后参与概率指标）
            s1 = rule.loc[dates]
            rows.append(pd.DataFrame({
                "trade_date": dates, "strategy_id": sid, "model": "C1",
                "score_raw": s1.values, "score_cal": cal_predict(c1_cal, s1.values),
                "high_risk_flag": (s1.values >= c1_cut).astype(int),
                "fold": "-", "period": per.values}))
            # C2-L/C2-S/C2-F：所有期间都用各自冻结模型打分（训练期为样本内，仅供诊断）
            for name, art in models.items():
                s2 = pd.Series(art["model"].predict(sub.loc[:, art["cols"]]), index=dates)
                rows.append(pd.DataFrame({
                    "trade_date": dates, "strategy_id": sid, "model": name,
                    "score_raw": s2.values, "score_cal": cal_predict(art["cal"], s2.values),
                    "high_risk_flag": (s2.values >= art["cut"]).astype(int),
                    "fold": per.map(FOLD_TAG).values,
                    "period": per.values}))
        preds = pd.concat(rows, ignore_index=True)
        preds.to_parquet(PRED_PATH, index=False)
        print(f"      {len(preds)} 行 → {PRED_PATH}", flush=True)

        ctx = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "main_sid": MAIN_SID, "legacy_sid": LEGACY_SID,
            "labels": labels, "preds": preds, "rule": rule, "feat": feat,
            "periods": PERIODS, "prior": prior,
            "c0_monthly": None, "c0_yearly": None,
            "c1_cut": c1_cut, "models": models,
            "q70": float(y.quantile(PRAUC_Q)),
            "feature_cols": full_cols,
            "train_n": len(train_dates), "calib_n": len(calib_dates), "embargo": EMBARGO,
            "high_risk_q": HIGH_RISK_Q, "prauc_q": PRAUC_Q,
            "instruments": INSTRUMENTS, "ridge_alpha": RIDGE_ALPHA,
            "small_features": SMALL_FEATURES,
        }
        # C0 月度/年度基础分布（只用训练期）
        tr_y = y.loc[[d for d in train_dates if d in y.index]]
        c0m = tr_y.copy()
        c0m.index = pd.to_datetime(c0m.index)
        ctx["c0_monthly"] = c0m.groupby(c0m.index.strftime("%Y-%m")).agg(["mean", "std", "count"])
        ctx["c0_yearly"] = c0m.groupby(c0m.index.strftime("%Y")).agg(["mean", "std", "count"])

        print("生成报告...", flush=True)
        md = render_report(ctx)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(md, encoding="utf-8")
        print(f"报告 → {REPORT_PATH}", flush=True)
    finally:
        conn.close()


def cal_predict(model, values):
    """isotonic 预测，容忍 NaN（fold0 无 OOF 分的位置保持 NaN）。"""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    m = ~np.isnan(v)
    if m.any():
        out[m] = model.predict(v[m])
    return out


if __name__ == "__main__":
    main()
