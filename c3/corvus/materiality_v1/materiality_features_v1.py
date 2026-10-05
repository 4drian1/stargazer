"""materiality_features_v1.py — Corvus materiality_head_v1 冻结特征契约与提取器。

这是 C3 walk-forward / train_final / C4 在线打分 三者**唯一共享**的特征实现。
任何一侧（训练或线上）都不得自行重实现特征，避免 feature drift。

特征全部满足 available_at 前可见：
- 日k派生（事件日及之前 20/250 日）：log mv、turnover20、vol20、dist_high_250
- 公告元数据：gap_return、volume_shock、n_announcements_in_cluster、sigma250
- 标题关键词命中 kw_*（来自 _taxonomy.KEYWORD_MATERIAL_WEIGHT）
- 事件族 one-hot cat_* —— **冻结为 TOP_CATEGORIES 常量**（不再按输入动态计算，
  否则线上单簇提取时类目编码会与离线错位，正是要防的 drift）

时间因果性：所有派生只用 label_start 及其之前的数据；label_start 之后的行情
绝不进入特征（见 feature_parity_gate 的截断测试）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _stockdb import DEFAULT_BASE, load_daily, make_opener
from _taxonomy import board_of, title_keyword_features, KEYWORD_MATERIAL_WEIGHT
from backfill_pipeline import daily_cache_path


# ---- 冻结特征契约常量 ----------------------------------------------------
CATEGORY_TOP = 12
# 冻结 top-12 事件族（由 canonical 训练域 value_counts().head(12) 确定，离线/在线共用）
TOP_CATEGORIES = [
    "BUYBACK", "SHAREHOLDER_REDUCTION", "EARNINGS_PREVIEW", "RESTRUCTURING",
    "LITIGATION_DEFAULT", "EARNINGS_REVISION", "MAJOR_CONTRACT",
    "SHAREHOLDER_INCREASE", "SHARE_UNLOCK", "REGULATORY_PENALTY",
    "INDUSTRY_POLICY", "DELISTING_RISK",
]

RAW_NUMERIC = [
    "is_st_at_event", "log_float_mv", "log_total_mv", "turnover20", "vol20",
    "dist_high_250", "gap_return", "volume_shock", "n_announcements_in_cluster",
    "sigma250",
]
KEYWORD_FEATURES = [f"kw_{k}" for k in KEYWORD_MATERIAL_WEIGHT]  # 顺序由 _taxonomy 固定
CATEGORY_FEATURES = [f"cat_{c}" for c in TOP_CATEGORIES]
# 模型输入特征顺序（冻结）：原始数值 + 关键词 + 冻结类目 one-hot
MODEL_FEATURES = RAW_NUMERIC + KEYWORD_FEATURES + CATEGORY_FEATURES


def _default_daily_provider(code: str, out_dir, base):
    p = daily_cache_path(out_dir, code)
    if p.exists():
        return pd.read_parquet(p)
    return load_daily(make_opener(), base, code)


def build_features(ev: pd.DataFrame, out_dir="out", base=DEFAULT_BASE,
                   with_labels: bool = True, daily_provider=None) -> pd.DataFrame:
    """逐票补特征。返回 DataFrame，列顺序 = id 列 + MODEL_FEATURES + (标签列)。

    daily_provider: 可选 (code) -> DataFrame，便于在线/测试注入行情源。
    默认从回填 daily_cache 或 stockdb 读取。
    """
    provider = daily_provider or (lambda code: _default_daily_provider(code, out_dir, base))
    feats = []
    for code, g in ev.groupby("code", sort=False):
        try:
            daily = provider(code)
        except Exception:
            daily = None
        if daily is None or daily.empty:
            continue
        daily = daily.copy()
        daily["date"] = daily["date"].astype(int)
        pos = {d: i for i, d in enumerate(daily["date"])}
        for _, e in g.iterrows():
            idx = pos.get(int(e["label_start"]))
            if idx is None:
                continue
            row = daily.iloc[idx]
            prev20 = daily.iloc[max(0, idx - 20): idx]
            prev250 = daily.iloc[max(0, idx - 250): idx]
            f = {
                "event_cluster_id": e["event_cluster_id"],
                "announce_date": int(e["announce_date"]),
                "code": code,
                "board": board_of(code),
                "is_st_at_event": bool(row.get("is_st")) if not pd.isna(row.get("is_st")) else False,
                "log_float_mv": float(np.log(float(row["float_mv"]))) \
                    if not pd.isna(row.get("float_mv")) and float(row.get("float_mv") or 0) > 0 else np.nan,
                "log_total_mv": float(np.log(float(row["total_mv"]))) \
                    if not pd.isna(row.get("total_mv")) and float(row.get("total_mv") or 0) > 0 else np.nan,
                "turnover20": float(pd.to_numeric(prev20.get("turnover"),
                                                  errors="coerce").mean()) if len(prev20) >= 10 else np.nan,
                "vol20": float((pd.to_numeric(prev20["pct_chg"],
                                              errors="coerce") / 100.0).std()) \
                    if len(prev20) >= 10 and "pct_chg" in prev20.columns else np.nan,
                "dist_high_250": float(row["close"] / float(prev250["high"].max()) - 1) \
                    if len(prev250) >= 60 else np.nan,
                "gap_return": e.get("gap_return"),
                "volume_shock": e.get("volume_shock"),
                "n_announcements_in_cluster": int(e["n_announcements_in_cluster"]),
                "sigma250": e.get("sigma250"),
            }
            kw = title_keyword_features(e["title"])
            for k in KEYWORD_MATERIAL_WEIGHT:
                f[f"kw_{k}"] = kw.get(k, 0.0)
            # 冻结类目编码（不依赖输入分布，消除线上 drift）
            cat = e["category"]
            for c in TOP_CATEGORIES:
                f[f"cat_{c}"] = float(cat == c)
            feats.append(f)
    feat = pd.DataFrame(feats)
    if with_labels:
        feat = feat.merge(
            ev[["event_cluster_id", "materiality", "direction_3d", "ret_3d_pc",
                "attribution_quality", "tier", "label_status"]],
            on="event_cluster_id", how="left")
    return feat
