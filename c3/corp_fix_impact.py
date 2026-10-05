"""corp_fix_impact.py — 量化除权守卫修复(含 label_start 当天)对标签的影响。

对比：
  V1 = out/events_v1_pre_corp_fix.parquet  (LABEL_VERSION = c3_pack_v1, 修复前)
  V2 = out/events_SUPERSEDED_V2_labeled.parquet (LABEL_VERSION = CORVUS_LABEL_V2_CORP_STARTDAY_FIX, 修复后)

输出 out/CORP_FIX_IMPACT_REPORT.json（+ 同款 md 摘要），含：
  n_total / n_corp_start_day / n_corp_horizon / n_label_changed /
  n_materiality_flip_true_to_false / n_materiality_flip_false_to_true
  by_year / by_event_family / by_score_bucket

同时把 V1 副本的 label_version 显式改写为 CORVUS_LABEL_V1_PRE_CORP_FIX 以便审计。
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd


def main():
    v1_path = "out/events_v1_pre_corp_fix.parquet"
    v2_path = "out/events_SUPERSEDED_V2_labeled.parquet"

    # 强制非数值列转 object，彻底绕开 pyarrow 列的 take/merge 内存错误
    def _no_arrow(df):
        for c in list(df.columns):
            dt = df[c].dtype
            if not (pd.api.types.is_integer_dtype(dt)
                    or pd.api.types.is_float_dtype(dt)):
                df[c] = df[c].astype("object")
        return df

    v1 = _no_arrow(pd.read_parquet(v1_path))
    v2 = _no_arrow(pd.read_parquet(v2_path))
    # events_SUPERSEDED_V2_labeled.parquet 含大量精确完全重复行（约 3.28x，元数据列不同导致物理重复）
    # 先剥掉 parquet dataset 的 __ 元数据列，再按逻辑列去重做 1:1 比对
    n1_raw, n2_raw = len(v1), len(v2)
    v1 = v1.drop(columns=[c for c in v1.columns if c.startswith("__")])
    v2 = v2.drop(columns=[c for c in v2.columns if c.startswith("__")])
    # events_SUPERSEDED_V2_labeled.parquet 粒度是"每条公告一行"，标签为 cluster 级(同 cluster 内 materiality 100% 一致)。
    # 按 event_cluster_id 去重到 cluster 级别做 1:1 比对，避免翻转计数被 3.28x 公告冗余虚增。
    v1 = v1.drop_duplicates(subset=["event_cluster_id"])
    v2 = v2.drop_duplicates(subset=["event_cluster_id"])
    print(f"[corp_fix_impact] 去重: V1 {n1_raw}->{len(v1)}, V2 {n2_raw}->{len(v2)}")

    # 把 V1 副本的 label_version 显式改写为 V1 名（原为 c3_pack_v1）
    v1 = v1.copy()
    v1["label_version"] = "CORVUS_LABEL_V1_PRE_CORP_FIX"
    v1.to_parquet(v1_path, index=False)

    base_cols = ["event_cluster_id", "announce_date", "category",
                 "materiality", "label_status"]
    corp_cols = ["corp_action_on_label_start", "corp_action_within_horizon",
                 "corp_action_type", "corp_action_date"]
    # V1 是修复前副本，没有新 corp 审计列 —— 补占位(False/None)，merge 仍用 V2 真值
    a = v1[base_cols].copy()
    for c in corp_cols:
        a[c] = False if c in ("corp_action_on_label_start",
                              "corp_action_within_horizon") else None
    b = v2[base_cols + corp_cols].copy()
    m = a.merge(b, on="event_cluster_id", suffixes=("_v1", "_v2"))

    n_total = int(len(m))
    n_corp_start_day = int(m["corp_action_on_label_start_v2"].fillna(False).sum())
    n_corp_horizon = int(m["corp_action_within_horizon_v2"].fillna(False).sum())

    both = m.dropna(subset=["materiality_v1", "materiality_v2"])
    changed = both[both["materiality_v1"] != both["materiality_v2"]]
    n_label_changed = int(len(changed))
    t2f = changed[(changed["materiality_v1"] == True) & (changed["materiality_v2"] == False)]
    f2t = changed[(changed["materiality_v1"] == False) & (changed["materiality_v2"] == True)]
    n_t2f = int(len(t2f))
    n_f2t = int(len(f2t))

    # 分数桶（用 V1 全局预测，特征在 V1/V2 间不变，故桶口径一致）
    if os.path.exists("out/materiality_pred.parquet"):
        pred = pd.read_parquet("out/materiality_pred.parquet")
        pred = pred.groupby("event_cluster_id")["pred"].first().reset_index()
        m = m.merge(pred, on="event_cluster_id", how="left")

        def bucket(p):
            if pd.isna(p):
                return "NA"
            if p < 0.3:
                return "0.0-0.3"
            if p < 0.5:
                return "0.3-0.5"
            if p < 0.7:
                return "0.5-0.7"
            return "0.7-1.0"
        m["score_bucket"] = m["pred"].map(bucket)
        by_score = {}
        for sb in ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0", "NA"]:
            sub = m[m["score_bucket"] == sb]
            ch = sub.dropna(subset=["materiality_v1", "materiality_v2"])
            ch = ch[ch["materiality_v1"] != ch["materiality_v2"]]
            by_score[sb] = {
                "n": int(len(sub)),
                "n_corp_start_day": int(sub["corp_action_on_label_start_v2"].fillna(False).sum()),
                "n_label_changed": int(len(ch)),
                "flip_t2f": int(((ch["materiality_v1"] == True) & (ch["materiality_v2"] == False)).sum()),
                "flip_f2t": int(((ch["materiality_v1"] == False) & (ch["materiality_v2"] == True)).sum()),
            }
    else:
        by_score = {}

    by_year = {}
    m["year"] = (m["announce_date_v1"].astype("int64") // 10000).astype(int)
    for y, sub in m.groupby("year"):
        ch = sub.dropna(subset=["materiality_v1", "materiality_v2"])
        ch = ch[ch["materiality_v1"] != ch["materiality_v2"]]
        by_year[int(y)] = {
            "n": int(len(sub)),
            "n_corp_start_day": int(sub["corp_action_on_label_start_v2"].fillna(False).sum()),
            "n_label_changed": int(len(ch)),
            "flip_t2f": int(((ch["materiality_v1"] == True) & (ch["materiality_v2"] == False)).sum()),
            "flip_f2t": int(((ch["materiality_v1"] == False) & (ch["materiality_v2"] == True)).sum()),
        }

    by_fam = {}
    for c, sub in m.groupby("category_v1"):
        ch = sub.dropna(subset=["materiality_v1", "materiality_v2"])
        ch = ch[ch["materiality_v1"] != ch["materiality_v2"]]
        by_fam[str(c)] = {
            "n": int(len(sub)),
            "n_corp_start_day": int(sub["corp_action_on_label_start_v2"].fillna(False).sum()),
            "n_label_changed": int(len(ch)),
            "flip_t2f": int(((ch["materiality_v1"] == True) & (ch["materiality_v2"] == False)).sum()),
            "flip_f2t": int(((ch["materiality_v1"] == False) & (ch["materiality_v2"] == True)).sum()),
        }
    by_fam = dict(sorted(by_fam.items(), key=lambda kv: -kv[1]["n_label_changed"])[:25])

    report = {
        "label_version_v1": "CORVUS_LABEL_V1_PRE_CORP_FIX",
        "label_version_v2": "CORVUS_LABEL_V2_CORP_STARTDAY_FIX",
        "fix": "corp guard window extended to include label_start day (idx..idx+5)",
        "n_total": n_total,
        "n_corp_start_day": n_corp_start_day,
        "n_corp_horizon": n_corp_horizon,
        "n_label_changed": n_label_changed,
        "n_materiality_flip_true_to_false": n_t2f,
        "n_materiality_flip_false_to_true": n_f2t,
        "by_year": by_year,
        "by_event_family_top25": by_fam,
        "by_score_bucket": by_score,
    }
    with open("out/CORP_FIX_IMPACT_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)

    # md 摘要
    lines = ["# 除权守卫修复影响报告（CORP FIX IMPACT）", ""]
    lines.append(f"- V1 (修复前): CORVUS_LABEL_V1_PRE_CORP_FIX")
    lines.append(f"- V2 (修复后): CORVUS_LABEL_V2_CORP_STARTDAY_FIX")
    lines.append(f"- 修复: corp 守卫窗口包含 label_start 当天 (idx..idx+5)")
    lines.append("")
    lines.append(f"- n_total: {n_total}")
    lines.append(f"- n_corp_start_day (V2 标记 label_start 当天除权): {n_corp_start_day}")
    lines.append(f"- n_corp_horizon (含 T..T+5 任一除权): {n_corp_horizon}")
    lines.append(f"- **n_label_changed (materiality 翻转): {n_label_changed}**")
    lines.append(f"  - True→False: {n_t2f}")
    lines.append(f"  - False→True: {n_f2t}")
    lines.append("")
    lines.append("### by_year (label_changed / corp_start_day)")
    lines.append("| 年 | n | corp_start_day | changed | T2F | F2T |")
    lines.append("|---|---|---|---|---|---|")
    for y, r in sorted(by_year.items()):
        lines.append(f"| {y} | {r['n']} | {r['n_corp_start_day']} | {r['n_label_changed']} | "
                     f"{r['flip_t2f']} | {r['flip_f2t']} |")
    if by_score:
        lines.append("")
        lines.append("### by_score_bucket (V1 全局预测分位)")
        lines.append("| 分位 | n | corp_start_day | changed | T2F | F2T |")
        lines.append("|---|---|---|---|---|---|")
        for sb, r in by_score.items():
            lines.append(f"| {sb} | {r['n']} | {r['n_corp_start_day']} | {r['n_label_changed']} | "
                         f"{r['flip_t2f']} | {r['flip_f2t']} |")
    lines.append("")
    lines.append("### by_event_family top (按 changed 降序)")
    lines.append("| 事件族 | n | corp_start_day | changed | T2F | F2T |")
    lines.append("|---|---|---|---|---|---|")
    for c, r in by_fam.items():
        lines.append(f"| {c} | {r['n']} | {r['n_corp_start_day']} | {r['n_label_changed']} | "
                     f"{r['flip_t2f']} | {r['flip_f2t']} |")
    with open("out/CORP_FIX_IMPACT_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[corp_fix_impact] 已写 out/CORP_FIX_IMPACT_REPORT.json / .md")
    print(f"  n_total={n_total} n_corp_start_day={n_corp_start_day} "
          f"n_label_changed={n_label_changed} T2F={n_t2f} F2T={n_f2t}")


if __name__ == "__main__":
    main()
