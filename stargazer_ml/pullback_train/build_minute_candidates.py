# -*- coding: utf-8 -*-
"""盘中路线第1步：2025年分钟候选事件清单。

范围（评审确认，门控前全集，防选择偏差）：
- A2全部 adj_ready 合格事件（subtype∈{adj_ready,both} 且 eligible_s1.2，
  含正EV/负EV/模型拒绝/未进Top——事件表在模型评分之前，天然满足）；
- B0全部规则候选（events_b_b0_cooled，含不合格——盘中模型管时机不管仓位）；
- LEGACY全部规则候选（events_legacy_cooled，desktop A模式复现，冷却后与
  A2/B0同口径；近期采用"旧模型盘后选股+新模型盘中确认"，缺LEGACY候选
  会使盘中模型对旧模型独有候选分布外推）；
- 三来源按 (code,date) 去重，source_modes 保留来源；
- 北交所（8/4/92开头）无分钟覆盖 → 标记 minute_unavailable，不作负样本。

输出 minute_candidates_2025.parquet：
  code,date,source_modes,a2_event_id,b0_event_id,legacy_event_id,
  atr_ref,entry_qfq,ks_a2,ks_b0,ks_legacy,minute_unavailable
同时输出候选股票清单与事件数统计到报告 §18.4。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
OUT = os.path.join(OUT_DIR, "minute_candidates_2025.parquet")
Y0, Y1 = 20250101, 20251231


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_bse(code: str) -> bool:
    return code.startswith(("8", "4", "92"))


def main():
    t0 = time.time()
    # ---- A2: adj_ready 合格（eligible_s1.2），2025
    a2 = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"),
                         columns=["event_id", "date", "code", "subtype",
                                  "eligible_s1.2", "atr_ref", "close"])
    a2 = a2.drop_duplicates("event_id")
    a2 = a2[(a2["date"] >= Y0) & (a2["date"] <= Y1)
            & a2["subtype"].isin(["adj_ready", "both"])
            & a2["eligible_s1.2"].astype(bool) & a2["atr_ref"].notna()
            & (a2["atr_ref"] > 0)]
    a2 = a2.rename(columns={"event_id": "a2_event_id", "close": "entry_qfq"})
    log(f"A2 adj_ready eligible 2025: {len(a2):,} events, {a2['code'].nunique()} codes")

    # ---- B0: 全部规则候选，2025
    b0 = pd.read_parquet(os.path.join(OUT_DIR, "events_b_b0_cooled.parquet"),
                         columns=["event_id", "date", "code", "atr_ref", "close"])
    b0 = b0[(b0["date"] >= Y0) & (b0["date"] <= Y1) & b0["atr_ref"].notna()
            & (b0["atr_ref"] > 0)]
    b0 = b0.rename(columns={"event_id": "b0_event_id", "close": "entry_qfq_b0",
                            "atr_ref": "atr_ref_b0"})
    log(f"B0 规则候选 2025: {len(b0):,} events, {b0['code'].nunique()} codes")

    # ---- LEGACY: 全部规则候选（冷却口径与A2/B0一致），2025
    lg = pd.read_parquet(os.path.join(OUT_DIR, "events_legacy_cooled.parquet"),
                         columns=["event_id", "date", "code", "atr_ref", "close"])
    lg = lg[(lg["date"] >= Y0) & (lg["date"] <= Y1) & lg["atr_ref"].notna()
            & (lg["atr_ref"] > 0)]
    lg = lg.rename(columns={"event_id": "legacy_event_id", "close": "entry_qfq_lg",
                            "atr_ref": "atr_ref_lg"})
    log(f"LEGACY 规则候选 2025: {len(lg):,} events, {lg['code'].nunique()} codes")

    # ---- 合并：A2 左表 + B0 + LEGACY 按 (code,date) 并入
    b0_key = b0.drop_duplicates(["code", "date"])[
        ["code", "date", "b0_event_id", "atr_ref_b0", "entry_qfq_b0"]]
    lg_key = lg.drop_duplicates(["code", "date"])[
        ["code", "date", "legacy_event_id", "atr_ref_lg", "entry_qfq_lg"]]
    df = a2.merge(b0_key, on=["code", "date"], how="outer").merge(
        lg_key, on=["code", "date"], how="outer")
    # B0/LEGACY-only 行：atr_ref/entry 用对应来源值补齐
    df["atr_ref"] = df["atr_ref"].fillna(df["atr_ref_b0"]).fillna(df["atr_ref_lg"])
    df["entry_qfq"] = df["entry_qfq"].fillna(df["entry_qfq_b0"]).fillna(
        df["entry_qfq_lg"])
    df = df.drop(columns=["atr_ref_b0", "entry_qfq_b0", "atr_ref_lg", "entry_qfq_lg"])
    df["subtype"] = df["subtype"].fillna(pd.Series(
        np.where(df["b0_event_id"].notna(), "b0_only", "legacy_only"),
        index=df.index))
    has_a2 = df["a2_event_id"].notna()
    has_b0 = df["b0_event_id"].notna()
    has_lg = df["legacy_event_id"].notna()
    df["source_modes"] = [",".join(m for m, h in zip(("A2", "B0", "LEGACY"),
                                                     (a, b, l)) if h)
                          for a, b, l in zip(has_a2, has_b0, has_lg)]
    df["ks_a2"] = np.where(has_a2, "1.2", "")
    df["ks_b0"] = np.where(has_b0, "1.5", "")
    df["ks_legacy"] = np.where(has_lg, "1.5", "")
    df["minute_unavailable"] = df["code"].map(is_bse)
    df = df.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    log(f"合并后: {len(df):,} 个 (code,date) 候选事件, "
        f"{df['code'].nunique()} 只股票; 分布 {df['source_modes'].value_counts().to_dict()}")

    df.to_parquet(OUT, index=False)
    log(f"wrote {OUT}")

    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.4 分钟候选清单")
    if idx != -1:
        old = old[:idx] + "\n"
    n_bse = int(df["minute_unavailable"].sum())
    L = ["\n### 18.4 分钟候选清单（第1步，门控前全集）\n",
         "范围：A2全部 adj_ready 合格事件（eligible_s1.2，含正负EV/被拒/未进Top——"
         "事件表在模型评分之前）∪ B0全部规则候选 ∪ LEGACY全部规则候选"
         "（events_legacy_cooled，与A2/B0同为冷却后口径）。按(code,date)去重，"
         "source_modes 保留来源；北交所标记 minute_unavailable 不作负样本。\n",
         f"- 候选事件：{len(df):,} 个（{df['source_modes'].value_counts().to_dict()}）",
         f"- 涉及股票：{df['code'].nunique():,} 只（北交所 {n_bse:,} 事件无分钟覆盖）",
         "  其中 LEGACY 独有事件 "
         f"{int((has_lg & ~has_a2 & ~has_b0).sum()):,} 个"
         f"（{df.loc[has_lg & ~has_a2 & ~has_b0, 'code'].nunique()} 只）",
         f"- 落盘：minute_candidates_2025.parquet",
         f"- 拉取窗口：信号日前约25个交易日（基线，2025年1月信号会截断→完整性标志）"
         f"+ 信号日 + 信号后5个交易日（含2026年1月初）"]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
