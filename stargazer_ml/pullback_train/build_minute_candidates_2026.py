# -*- coding: utf-8 -*-
"""2026时间复核第1步：2026候选事件清单（A2∪B0∪LEGACY，门控前全集）。

口径与 build_minute_candidates.py（2025版）完全一致，仅窗口改为
20260101~20260831（复核窗口；8月末信号的T+6标签完整性由引擎censored标志
处理，复核脚本剔除）。输出 minute_candidates_2026.parquet。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
OUT = os.path.join(OUT_DIR, "minute_candidates_2026.parquet")
Y0, Y1 = 20260101, 20260831


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_bse(code: str) -> bool:
    return code.startswith(("8", "4", "92"))


def main():
    t0 = time.time()
    # ---- A2: adj_ready 合格（eligible_s1.2），2026
    a2 = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"),
                         columns=["event_id", "date", "code", "subtype",
                                  "eligible_s1.2", "atr_ref", "close"])
    a2 = a2.drop_duplicates("event_id")
    a2 = a2[(a2["date"] >= Y0) & (a2["date"] <= Y1)
            & a2["subtype"].isin(["adj_ready", "both"])
            & a2["eligible_s1.2"].astype(bool) & a2["atr_ref"].notna()
            & (a2["atr_ref"] > 0)]
    a2 = a2.rename(columns={"event_id": "a2_event_id", "close": "entry_qfq"})
    log(f"A2 adj_ready eligible 2026: {len(a2):,} events, {a2['code'].nunique()} codes")

    # ---- B0: 全部规则候选，2026
    b0 = pd.read_parquet(os.path.join(OUT_DIR, "events_b_b0_cooled.parquet"),
                         columns=["event_id", "date", "code", "atr_ref", "close"])
    b0 = b0[(b0["date"] >= Y0) & (b0["date"] <= Y1) & b0["atr_ref"].notna()
            & (b0["atr_ref"] > 0)]
    b0 = b0.rename(columns={"event_id": "b0_event_id", "close": "entry_qfq_b0",
                            "atr_ref": "atr_ref_b0"})
    log(f"B0 规则候选 2026: {len(b0):,} events, {b0['code'].nunique()} codes")

    # ---- LEGACY: 全部规则候选，2026
    lg = pd.read_parquet(os.path.join(OUT_DIR, "events_legacy_cooled.parquet"),
                         columns=["event_id", "date", "code", "atr_ref", "close"])
    lg = lg[(lg["date"] >= Y0) & (lg["date"] <= Y1) & lg["atr_ref"].notna()
            & (lg["atr_ref"] > 0)]
    lg = lg.rename(columns={"event_id": "legacy_event_id", "close": "entry_qfq_lg",
                            "atr_ref": "atr_ref_lg"})
    log(f"LEGACY 规则候选 2026: {len(lg):,} events, {lg['code'].nunique()} codes")

    # ---- 合并
    b0_key = b0.drop_duplicates(["code", "date"])[
        ["code", "date", "b0_event_id", "atr_ref_b0", "entry_qfq_b0"]]
    lg_key = lg.drop_duplicates(["code", "date"])[
        ["code", "date", "legacy_event_id", "atr_ref_lg", "entry_qfq_lg"]]
    df = a2.merge(b0_key, on=["code", "date"], how="outer").merge(
        lg_key, on=["code", "date"], how="outer")
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
    log(f"合并后: {len(df):,} 个 (code,date) 候选事件, {df['code'].nunique()} 只; "
        f"分布 {df['source_modes'].value_counts().to_dict()}")
    df.to_parquet(OUT, index=False)
    log(f"wrote {OUT} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
