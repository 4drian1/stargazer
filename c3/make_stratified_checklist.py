"""make_stratified_checklist.py — 生成 C3 分层人工核对清单（Step 1）。

做法：
1. 载入全量 event_cluster.parquet（训练域口径：P0 & OK & 非 HEAVY & 非 DISTRIBUTION）；
2. 用落盘 LGBM（3 seed 平均）给每个事件打 materiality 预测分；
3. 按 年份 × 预测分位 × 特殊情况 分层抽样，输出人工可核对的清单；
4. 同时把全局预测存 out/materiality_pred.parquet（供 Step2/3 walk-forward）。

已知限制（见 DEVIATION D1）：巨潮个股接口仅日期粒度，available_at 无法判断“靠近收盘”，
需人工对照 cninfo 时间戳。清单里专门标注此事。
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _stockdb import DEFAULT_BASE
from corvus.materiality_v1.materiality_features_v1 import build_features, MODEL_FEATURES


def numeric_cols_of(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns
            if c.startswith(("kw_", "cat_", "log_", "turnover", "vol",
                             "dist_", "gap_", "volume_", "n_ann", "sigma"))
            or c in ("is_st_at_event",)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="out/event_cluster.parquet")
    ap.add_argument("--out", default="out")
    ap.add_argument("--per-cell", type=int, default=2, help="年份×分位 每格抽样数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ev = pd.read_parquet(args.events)
    mat = ev[(ev["tier"] == "P0") & (ev["label_status"] == "OK") &
             (ev["attribution_quality"] != "HEAVY") &
             (ev["category"] != "DISTRIBUTION")].copy()
    mat = mat.sort_values("announce_date").drop_duplicates("event_cluster_id")
    print(f"[checklist] 训练域事件 {len(mat)}")

    feat = build_features(mat, args.out, base=DEFAULT_BASE)
    print(f"[checklist] 特征 {len(feat)} 行 x {feat.shape[1]} 列")

    # 预测
    import json as _json, lightgbm as lgb
    from pathlib import Path as _Path
    mp = _Path(__file__).resolve().parent / "corvus" / "materiality_v1" / "artifacts" / "model_manifest.json"
    if not mp.exists():
        raise SystemExit(f"未找到冻结模型 manifest：{mp}（先运行 train_final.py）")
    mf = _json.loads(mp.read_text(encoding="utf-8"))
    paths = mf["model_files"]
    boosters = [lgb.Booster(model_file=p) for p in paths]
    nc = MODEL_FEATURES
    X = feat[nc].fillna(feat[nc].median()).values
    preds = np.column_stack([b.predict(X) for b in boosters])
    feat["pred"] = preds.mean(axis=1)
    print(f"[checklist] 预测完成 ({len(paths)} seed 平均)")

    # 合并标签列
    merged = feat[["event_cluster_id", "pred"]].merge(
        ev[["event_cluster_id", "code", "name", "announce_date", "title", "category",
            "label_start", "label_start_basis", "entry_price", "gap_return",
            "materiality", "attribution_quality", "label_status", "raw_url",
            "overlap_count"]],
        on="event_cluster_id", how="left")
    merged["year"] = (merged["announce_date"].astype("int64") // 10000).astype(int)

    # 落盘全局预测（供 Step2/3）
    merged.to_parquet(os.path.join(args.out, "materiality_pred.parquet"), index=False)
    print(f"[checklist] 已存 out/materiality_pred.parquet ({len(merged)} 行)")

    def band(p):
        if p < 0.1: return "0.0-0.1"
        if p < 0.3: return "0.1-0.3"
        if p < 0.5: return "0.3-0.5"
        if p < 0.7: return "0.5-0.7"
        if p < 0.9: return "0.7-0.9"
        return "0.9-1.0"
    merged["band"] = merged["pred"].map(band)

    # ---- 主分层表：年份 × 分位 ----
    main_rows = []
    for yr in range(2019, 2026):
        sub = merged[merged["year"] == yr]
        for b in ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]:
            cell = sub[sub["band"] == b]
            take = cell.sample(min(args.per_cell, len(cell)), random_state=args.seed) \
                if len(cell) else pd.DataFrame()
            main_rows.append(take)
    main_df = pd.concat(main_rows) if main_rows else pd.DataFrame()

    # ---- 特殊情况专项 ----
    special = {}
    special["事件重叠 HEAVY (attribution_quality=HEAVY)"] = \
        ev[(ev["attribution_quality"] == "HEAVY") & (ev["label_status"] == "OK")] \
        .drop_duplicates("event_cluster_id").sample(min(6, len(ev)), random_state=args.seed)
    special["事件重叠 MODERATE"] = \
        ev[(ev["attribution_quality"] == "MODERATE") & (ev["label_status"] == "OK")] \
        .drop_duplicates("event_cluster_id").sample(min(4, _n(ev, "MODERATE")), random_state=args.seed)
    special["NO_PRICE_DATA 边界 (label_status)"] = \
        ev[ev["label_status"] == "NO_PRICE_DATA"].sample(min(6, _n(ev, "NO_PRICE_DATA")), random_state=args.seed)
    # 停牌/复牌
    if "RESUME_FIRST_TRADE" in set(ev["label_start_basis"].dropna().astype(str)):
        special["停牌/复牌 (RESUME_FIRST_TRADE)"] = \
            ev[ev["label_start_basis"].astype(str) == "RESUME_FIRST_TRADE"].sample(min(5, _n(ev, "RESUME_FIRST_TRADE")), random_state=args.seed)
    # 除权疑似：gap 极大（>11.5%）且非单纯涨跌停
    big_gap = ev[(ev["label_status"] == "OK") & (ev["gap_return"].abs() > 0.115)]
    special["除权疑似 (|gap|>11.5%)"] = big_gap.sample(min(4, len(big_gap)), random_state=args.seed)
    # 实体/事件族歧义：UNMAPPED 或标题很短
    ambig = ev[(ev["category"] == "UNMAPPED") & (ev["label_status"] == "OK")]
    special["实体/事件族歧义 (UNMAPPED)"] = ambig.sample(min(4, len(ambig)), random_state=args.seed)

    # ---- 输出 markdown ----
    lines = []
    lines += ["# C3 完整性/标签人工核对清单（分层抽样）", ""]
    lines += ["- 生成: 自动分层（`make_stratified_checklist.py`）",
              f"- 训练域事件基数: {len(mat)}",
              f"- 主表抽样: 年份(2019-2025) × 预测分位(6档) × {args.per_cell} 例",
              "- 预测分位: 落盘 LGBM(3 seed 平均) 的 materiality 预测概率",
              ""]
    lines += ["## ⚠️ 关键限制（对照前必读）", ""]
    lines += ["1. **available_at 粒度**：本包公告时间仅日期粒度（巨潮个股接口只给日期，见 DEVIATION D1）。",
              "   因此“公告是否靠近收盘发布”**无法自动判断**，需人工打开 cninfo 链接看原始时间戳。",
              "2. **核对链路**（每一条都按此走一遍）：",
              "   原公告/新闻 → available_at(=label_start) → entity(代码/简称) → event family(category) →",
              "   第一个合法可见/可交易价格点(entry_price/gap_return) → materiality 标签(是否|ret|超2σ)。",
              "3. **materiality 标签口径**：label_start 起 1/3/5 日收益任一绝对值 > 该股前250日收益std的2倍。",
              "4. 顶部 `核对结果` 列请填 **TP/FP/TN/FN**（相对你认为的真实标签）：",
              "   - 预测/标签为“materiality=True”且你确认属实 → TP；预测True但你认定非 → FP；",
              "   - 预测/标签“False”且你确认非 → TN；预测False但你认定是 → FN。",
              ""]
    lines += ["## 主表：年份 × 预测分位", ""]
    lines += _table_header()
    for _, r in main_df.sort_values(["year", "band"]).iterrows():
        lines.append(_row(r))
    lines.append("")

    for title, sub in special.items():
        if len(sub) == 0:
            continue
        lines += ["", f"## 特殊情况：{title}（{len(sub)} 例）", ""]
        lines += _table_header()
        for _, r in sub.iterrows():
            rr = r.copy()
            rr["pred"] = merged.set_index("event_cluster_id")["pred"].get(r["event_cluster_id"], np.nan) \
                if "event_cluster_id" in sub.columns else np.nan
            lines.append(_row(rr))
        lines.append("")

    out_path = os.path.join(args.out, "completeness_checklist.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[checklist] 已写 {out_path}（主表 {len(main_df)} 行 + 特殊情况若干）")


def _n(ev, key):
    if key == "MODERATE":
        return int((ev["attribution_quality"] == "MODERATE").sum())
    if key == "NO_PRICE_DATA":
        return int((ev["label_status"] == "NO_PRICE_DATA").sum())
    if key == "RESUME_FIRST_TRADE":
        return int((ev["label_start_basis"].astype(str) == "RESUME_FIRST_TRADE").sum())
    return 0


def _table_header():
    return ["| # | 年份 | 代码 | 简称 | 公告日 | 标题 | 事件族 | 实体 | available_at | basis | 首价(entry/gap) | materiality标签 | 模型预测 | 特殊标记 | 链接 | 核对结果 | 备注 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def _as_float(x):
    try:
        if isinstance(x, (pd.Series, np.ndarray)):
            x = x.ravel()[0]
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _row(r):
    cid = r.get("event_cluster_id", "")
    code = str(r.get("code", ""))
    name = str(r.get("name", "") or "")
    ad = r.get("announce_date", "")
    title = str(r.get("title", "")).replace("|", "/")[:60]
    cat = str(r.get("category", ""))
    ls = r.get("label_start", "")
    basis = str(r.get("label_start_basis", "") or "")
    entry = r.get("entry_price", "")
    gap = r.get("gap_return", "")
    mat_lbl = r.get("materiality", "")
    pred = _as_float(r.get("pred", np.nan))
    pred_s = f"{pred:.3f}" if pred is not None else "NA"
    special = []
    if str(r.get("attribution_quality", "")) in ("HEAVY", "MODERATE"):
        special.append(f"overlap={r.get('overlap_count', '')}")
    if str(r.get("label_status", "")) == "NO_PRICE_DATA":
        special.append("NO_PRICE_DATA")
    if basis == "RESUME_FIRST_TRADE":
        special.append("停牌复牌")
    if pd.notna(gap) and abs(float(gap)) > 0.115:
        special.append("大gap/除权疑")
    if cat == "UNMAPPED":
        special.append("事件族未映射")
    url = str(r.get("raw_url", "") or "")
    return (f"| | {r.get('year', '')} | {code} | {name} | {ad} | {title} | {cat} | {code} | {ls} | {basis} | "
            f"{entry}/{gap} | {mat_lbl} | {pred_s} | {';'.join(special)} | {url} |  |  |")


if __name__ == "__main__":
    main()
