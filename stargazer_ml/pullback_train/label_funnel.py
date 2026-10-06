# -*- coding: utf-8 -*-
"""标签漏斗审计（评审要求，§18.24）。

训练前最后检查：按 2025H1/H2、2026、AM/PM×决策点、src 分层统计：
- 主标签 r_sbr_t2 / o_rbs_t2 正例率（可训练性：若风险主标签率<2%需停下重估）；
- 互斥校验：r_sbr_t2=1 且 o_rbs_t2=1 必须为 0（同bar悲观STOP已保证）；
- 触及/执行/锁死：r_stop_touched_t2 vs r_stop_by_t2（差值=T+2窗口内
  触及止损但全部落在跌停锁死bar、无法按止损价成交的数量）；
- r_stop_locked_only_t2 比例；
- 同日辅助标签 r_same_day_extreme_adverse_move 正例率；
- 30/60分钟辅助标签正例率（仅在未截断子集 cens30/cens60=False 上统计，
  截断样本绝不记失败）；
- cens30/cens60 截断比例（预期 dp1430/1450 截断集中）。
输入：两年 dual_head_labels_*.parquet + minute_samples（取 src/date/过滤口径）。
"""
import os
import time

import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
S25 = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
S26 = os.path.join(OUT_DIR, "minute_labels/minute_samples_2026.parquet")
L25 = os.path.join(OUT_DIR, "dual_head_labels_2025.parquet")
L26 = os.path.join(OUT_DIR, "dual_head_labels_2026.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
DPS = (945, 1000, 1030, 1330, 1400, 1430, 1450)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(samples_path, labels_path, tag):
    s = pd.read_parquet(samples_path)
    s = s[(s["dp"].isin(DPS)) & (s["entry_type"] == "vwap")
          & (s["censored"] == 0)][["code", "date", "dp_date", "dp", "src"]]
    s["code"] = s["code"].astype(str)
    l = pd.read_parquet(labels_path)
    l["code"] = l["code"].astype(str)
    df = l.merge(s, on=["code", "dp_date", "dp"], how="inner")
    df["period"] = tag
    df["half"] = tag + ("H1" if tag == "2025" else "")  # placeholder
    df.loc[df["period"] == "2025", "half"] = (
        df["date"] >= 20250601).map({False: "2025H1", True: "2025H2"})
    df["head"] = (df["dp"] < 1200).map({True: "AM", False: "PM"})
    return df


def block(df, keys):
    g = df.groupby(keys)
    out = pd.DataFrame({
        "n": g.size(),
        "sbr_t2": g["r_sbr_t2"].mean(),
        "rbs_t2": g["o_rbs_t2"].mean(),
        "both1": g.apply(lambda x: ((x["r_sbr_t2"] == 1)
                                    & (x["o_rbs_t2"] == 1)).sum()),
        "touched": g["r_stop_touched_t2"].mean(),
        "exec": g["r_stop_by_t2"].mean(),
        "locked_only": g["r_stop_locked_only_t2"].mean(),
        "same_day_adv": g["r_same_day_extreme_adverse_move"].mean(),
        "cens30": g["cens30"].mean(),
        "cens60": g["cens60"].mean(),
    }).reset_index()
    # 30/60分钟辅助标签：只在未截断子集上
    u30 = df[~df["cens30"]].groupby(keys)["r_new_low_30m"].mean() \
        .rename("new_low_30m_unc")
    u60 = df[~df["cens60"]].groupby(keys)["o_reclaim03_60m"].mean() \
        .rename("reclaim03_60m_unc")
    out = out.merge(u30, on=keys, how="left") \
             .merge(u60, on=keys, how="left")
    return out


def fmt(b, keys):
    cols = ["n", "sbr_t2", "rbs_t2", "both1", "touched", "exec",
            "locked_only", "same_day_adv", "cens30", "cens60",
            "new_low_30m_unc", "reclaim03_60m_unc"]
    head = "| " + " | ".join(keys + cols) + " |"
    sep = "|" + "---|" * (len(keys) + len(cols))
    lines = [head, sep]
    for _, r in b.iterrows():
        row = [str(r[k]) for k in keys]
        for c in cols:
            v = r[c]
            if c == "n" or c == "both1":
                row.append(f"{int(v):,}")
            elif pd.isna(v):
                row.append("-")
            else:
                row.append(f"{v:.1%}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main():
    t0 = time.time()
    df = pd.concat([load(S25, L25, "2025"), load(S26, L26, "2026")],
                   ignore_index=True)
    log(f"合并后 {len(df):,} 行（建模口径 inner join）")
    df.to_parquet(os.path.join(OUT_DIR, "label_funnel_pool.parquet"),
                  index=False)

    L = ["\n### 18.24 双头标签口径与漏斗审计（训练前检查）\n",
         "主标签口径（评审钉死）：窗口=entry(T+1盘中VWAP)→T+2收盘（最早可卖日），"
         "含隔夜跳空；可执行止损=触及且非跌停锁死bar（T+2跌停无法成交不计执行，"
         "见locked_only列）；同bar双触悲观记STOP；两主标签互斥（both1必须=0）；"
         "T+2收盘后信息不进标签。30/60分钟辅助标签只在未截断样本上统计"
         "（cens30/cens60=False），截断绝不记失败。\n",
         "**总池**："]
    tot = block(df, ["period"])
    L += fmt(tot, ["period"])
    L.append("\n**按 期间×头×决策点**：")
    L += fmt(block(df, ["period", "head", "dp"]), ["period", "head", "dp"])
    L.append("\n**按 期间×src**：")
    L += fmt(block(df, ["period", "src"]), ["period", "src"])
    L.append("\n**按 期间×half**：")
    L += fmt(block(df, ["period", "half"]), ["period", "half"])

    # 互斥性硬校验
    both = int(((df["r_sbr_t2"] == 1) & (df["o_rbs_t2"] == 1)).sum())
    miss = df[["r_sbr_t2", "o_rbs_t2"]].isna().any(axis=1).sum()
    L.append(f"\n互斥校验：r_sbr_t2=1 且 o_rbs_t2=1 的行数 = **{both}**"
             f"（必须为0）；主标签缺失 {miss:,} 行。")
    low = df.groupby(["period", "head"])["r_sbr_t2"].mean().min()
    L.append(f"风险主标签率最低分层 = {low:.1%}"
             + ("（<2%，需停下重估窗口）" if low < 0.02 else "（可训练）"))
    L.append("\n明细：label_funnel_pool.parquet")

    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.24 ")
    if idx != -1:
        old = old[:idx] + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
