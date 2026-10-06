# -*- coding: utf-8 -*-
"""T6 审计：模板审计（600313/600479）+ 无前视审计（截断重算一致性）。

模板审计：2026-09-04 600313、2026-09-07/08 600479 全部 dp 应产出非空 T6
特征且方向合理。注意：这三段数据已参与假设形成，仅作模板素材（E0），
不作验证证据。

无前视审计：随机抽 N 个 (code,dp_date,dp)，把该日分钟截断到 dp（丢弃
t > dp*100 的分钟），重算 T6 并归一，与全量 parquet 行逐列比对——
必须完全一致（T6 只允许使用 ≤DP 已完成数据；容忍 1e-6 相对误差）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minute_label_engine import load_code  # noqa: E402
import t6_features as T  # noqa: E402

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
FEATS = os.path.join(OUT_DIR, "t6_features_2026.parquet")
SAMPLES = os.path.join(OUT_DIR, "minute_labels/minute_samples_2026.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def truncated_cm(cm, day_i, dp):
    out = dict(cm)
    keep = ~((out["day_idx"] == day_i) & (out["t"].astype(int) > dp * 100))
    for k in ("day_idx", "t", "o", "h", "l", "c", "v", "a"):
        out[k] = out[k][keep]
    return out


def main():
    rng = np.random.default_rng(7)
    fe = pd.read_parquet(FEATS)
    fe["code"] = fe["code"].astype(str)
    t6cols = [c for c in fe.columns if c.startswith("t6_")]
    log(f"特征文件 {fe.shape}，t6 列 {len(t6cols)} 个")

    L = ["\n### 18.30 账本验证与 T6 审计（无前视一致性）\n",
         "**模板审计**（E0 素材，非证据）：\n",
         "| code | date | dp | dist_mid_s | width | slope | run_min | cross | vs_break | vs_reclaim |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for code, d in (("600313", 20260904), ("600479", 20260907),
                    ("600479", 20260908)):
        sub = fe[(fe["code"] == code) & (fe["dp_date"] == d)]
        if not len(sub):
            L.append(f"| {code} | {d} | - | 无数据 | | | | | | |")
            log(f"[模板] {code} {d}: 缺数据")
            continue
        n_nan = sub[t6cols].isna().mean().max()
        log(f"[模板] {code} {d}: {len(sub)} 行, 最大列NaN率 {n_nan:.1%}")
        for dp in (945, 1000, 1030, 1330, 1400, 1430, 1450):
            r = sub[sub["dp"] == dp]
            if not len(r):
                continue
            r = r.iloc[0]
            vb = r["t6_volsur_break"]
            vr = r["t6_volsur_reclaim"]
            L.append(
                f"| {code} | {d} | {dp} | {r['t6_dist_mid_s_atr']:+.2f} | "
                f"{r['t6_width_atr']:.2f} | {r['t6_mid_slope_atr_h']:+.3f} | "
                f"{r['t6_below_now_min']:.0f} | {r['t6_last_cross_dir']:.0f} | "
                f"{vb:.2f} | {vr:.2f} |")

    # ---- 无前视审计 ----
    sp = pd.read_parquet(SAMPLES)
    sp["code"] = sp["code"].astype(str)
    atr_of = {(r["code"], int(r["dp_date"])): float(r["atr_pct"])
              for _, r in sp[(sp["entry_type"] == "vwap")]
              [["code", "dp_date", "atr_pct"]].drop_duplicates().iterrows()}
    pool = fe[fe["dp"].isin((1000, 1030, 1400, 1430))].sample(
        150, random_state=rng)
    mism = 0
    cur, cm = None, None
    for _, r in pool.iterrows():
        code, ddate, dp = r["code"], int(r["dp_date"]), int(r["dp"])
        if code != cur:
            cur = code
            cm = load_code(code)
        if cm is None:
            continue
        hit = {int(d): i for i, d in enumerate(cm["days"])}
        if ddate not in hit:
            continue
        cm_t = truncated_cm(cm, hit[ddate], dp)
        f2 = T.compute_code(cm_t, {hit[ddate]})
        if not len(f2):
            mism += 1
            log(f"[无前视] {code} {ddate} dp{dp}: 截断后无特征")
            continue
        atr_map = {ddate: atr_of.get((code, ddate), np.nan)}
        n2 = T.normalize(f2, atr_map)
        n2 = n2[n2["dp"] == dp]
        if not len(n2):
            mism += 1
            log(f"[无前视] {code} {ddate} dp{dp}: 归一后缺失")
            continue
        row2 = n2.iloc[0]
        ok = True
        for c in t6cols:
            v1, v2 = float(r[c]), float(row2[c])
            if np.isfinite(v1) and np.isfinite(v2):
                if abs(v1 - v2) > 1e-6 * max(1.0, abs(v2)):
                    ok = False
                    break
            elif not (np.isnan(v1) and np.isnan(v2)):
                ok = False
                break
        if not ok:
            mism += 1
            log(f"[无前视] 不一致: {code} {ddate} dp{dp}")
    L.append(f"\n**无前视审计**：抽查 {len(pool)} 行，截断重算不一致 {mism} 行"
             f"（必须为0）。抽查范围 dp∈{{1000,1030,1400,1430}}。")
    log(f"[无前视] 不一致 {mism}/{len(pool)}")

    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.30 ")
    if idx != -1:
        old = old[:idx] + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log("done")


if __name__ == "__main__":
    main()
