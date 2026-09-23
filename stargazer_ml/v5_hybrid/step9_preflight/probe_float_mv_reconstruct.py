"""probe_float_mv_reconstruct — 任务4: 2026-07-01 前缺失 float_mv 能否用 PIT float_share*close 重建?

背景 (用户 2026-09-18 裁定):
  canonical 日线中 float_mv 有效占比在 2026-07-01 从 ~72.9% 阶跃到 ~99.96%,
  而 GREEN-WATCH Universe 定义依赖 PIT float_mv (25亿~100亿) -> 候选母池可见范围
  在 7/1 前后发生制度性变化 (DATA_AVAILABILITY_REGIME)。
  若 PIT float_share 在 7/1 前可得, 则可重建 float_mv_reconstructed_pit 补齐覆盖,
  而不必把 data epoch 缩到 2026-07-01 (那样只剩两个多月无法训练)。

输出:
  float_mv_probe_by_date.csv   逐日: n_rows / n_float_mv / n_float_share / n_reconstructable
  float_mv_probe_summary.json  分年 & 分段 (pre/post 2026-07-01) 的覆盖率与 ratio 分布
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_source_provenance import DAILY_PARTS  # noqa

_HERE = Path(__file__).resolve().parent
REGIME_DATE = 20260701


def main():
    d = ds.dataset(DAILY_PARTS, format="parquet")
    cols = ["date", "code", "name", "close", "float_share", "float_mv", "total_share", "total_mv"]
    num_cols = ["close", "float_share", "float_mv", "total_share", "total_mv"]
    df = d.to_table(columns=cols).to_pandas()
    df["date"] = df["date"].astype(int)
    print(f"[probe] rows={len(df)} dates={df['date'].nunique()} "
          f"range={df['date'].min()}..{df['date'].max()}")

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["has_fmv"] = df["float_mv"].notna() & (df["float_mv"] > 0)
    df["has_fs"] = df["float_share"].notna() & (df["float_share"] > 0)
    df["has_close"] = df["close"].notna() & (df["close"] > 0)
    df["reconstructable"] = (~df["has_fmv"]) & df["has_fs"] & df["has_close"]
    df["fs_close"] = df["float_share"] * df["close"]

    both = df["has_fmv"] & df["has_fs"] & df["has_close"]
    df["ratio"] = np.where(both, df["fs_close"] / df["float_mv"], np.nan)
    df["resid"] = np.where(both, (df["fs_close"] - df["float_mv"]).abs() / df["float_mv"], np.nan)

    g = df.groupby("date")
    by_date = pd.DataFrame({
        "n_rows": g.size(),
        "n_float_mv": g["has_fmv"].sum(),
        "n_float_share": g["has_fs"].sum(),
        "n_reconstructable": g["reconstructable"].sum(),
        "ratio_median": g["ratio"].median(),
        "resid_p95": g["resid"].quantile(0.95),
    }).reset_index()
    by_date["fmv_cov"] = (by_date["n_float_mv"] / by_date["n_rows"]).round(6)
    by_date["fmv_cov_after_reconstruct"] = (
        (by_date["n_float_mv"] + by_date["n_reconstructable"]) / by_date["n_rows"]).round(6)
    by_date.to_csv(_HERE / "float_mv_probe_by_date.csv", index=False, encoding="utf-8")

    df["year"] = df["date"] // 10000
    df["seg"] = np.where(df["date"] < REGIME_DATE, f"pre_{REGIME_DATE}", f"post_{REGIME_DATE}")

    def agg(sub):
        return {
            "n_rows": int(len(sub)),
            "float_mv_cov": round(float(sub["has_fmv"].mean()), 6),
            "float_share_cov": round(float(sub["has_fs"].mean()), 6),
            "reconstructable_share": round(float(sub["reconstructable"].mean()), 6),
            "cov_after_reconstruct": round(float((sub["has_fmv"] | sub["reconstructable"]).mean()), 6),
            "ratio_median": (round(float(sub.loc[both.loc[sub.index], "ratio"].median()), 6)
                             if bool(both.loc[sub.index].any()) else None),
            "resid_p95": (round(float(sub.loc[both.loc[sub.index], "resid"].quantile(0.95)), 6)
                          if bool(both.loc[sub.index].any()) else None),
            "n_still_missing_after_reconstruct": int((~sub["has_fmv"] & ~sub["reconstructable"]).sum()),
        }

    summary = {
        "regime_date": REGIME_DATE,
        "overall": agg(df),
        "by_segment": {k: agg(v) for k, v in df.groupby("seg")},
        "by_year": {str(k): agg(v) for k, v in df.groupby("year")},
        "load_bearing_question": "pre-2026-07-01 缺失 float_mv 的股票, PIT float_share 是否可得?",
    }

    # ---- 附加: 缺失主体是谁? 对 GREEN-WATCH 候选池 / Market Proxy 的实际影响 ----
    from data_source_provenance import get_universe  # noqa
    univ = {str(c).zfill(6) for c in get_universe()}
    df["_code6"] = df["code"].astype(str).str.zfill(6)
    pre = df[df["date"] < REGIME_DATE]
    ever = pre.groupby("_code6")["has_fmv"].max()
    never = set(ever[ever == 0].index)
    names = (df[df["_code6"].isin(never)].dropna(subset=["name"])
             .groupby("_code6")["name"].last())
    etf_like = int(sum(1 for n in names.values if "ETF" in str(n) or "基金" in str(n)))

    # Market Proxy 权重构成: float_mv>0 的 P 日权重中, 「regime 前从未有 float_mv」的标的占比
    share = {}
    for dt in (20260630, 20260701):
        s = df[(df["date"] == dt) & df["has_fmv"]]
        tot = float(s["float_mv"].sum())
        e = float(s[s["_code6"].isin(never)]["float_mv"].sum())
        share[str(dt)] = {"n_fmv": int(len(s)), "n_never_fmv": int(s["_code6"].isin(never).sum()),
                          "weight_share": round(e / tot, 6) if tot else None}

    summary["missing_instrument_analysis"] = {
        "n_never_fmv_codes_pre_regime": len(never),
        "n_of_those_in_get_universe": len(never & univ),
        "n_name_looks_like_etf_or_fund": etf_like,
        "sample_names": list(names.head(12).values),
        "market_proxy_weight_composition": share,
        "conclusion": ("regime 前缺失 float_mv 的主体是 ETF/基金等非股票工具 (name 含 ETF/基金); "
                       "GREEN-WATCH 候选池用 total_mv 25-100亿 过滤股票, 母池规模在 2026-06-30→07-01 "
                       "无台阶; 但 Market Proxy 的 universe 谓词是 float_mv>0 (规格写的是 eligible 股票), "
                       "regime 后把 ~2,020 只 ETF 计入代理权重 -> 见 weight_share"),
    }
    ( _HERE / "float_mv_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    boundary = by_date[(by_date["date"] >= 20260620) & (by_date["date"] <= 20260710)]
    print("\n=== boundary dates (20260620-20260710) ===")
    print(boundary[["date", "n_rows", "n_float_mv", "n_float_share", "n_reconstructable",
                    "fmv_cov", "fmv_cov_after_reconstruct", "ratio_median"]].to_string(index=False))
    print("\n=== by segment ===")
    for k, v in summary["by_segment"].items():
        print(f"{k:16s} {json.dumps(v, ensure_ascii=False)}")
    print("\n=== by year ===")
    for k, v in summary["by_year"].items():
        print(f"{k:6s} {json.dumps(v, ensure_ascii=False)}")
    print(f"\n[probe] wrote float_mv_probe_by_date.csv / float_mv_probe_summary.json")


if __name__ == "__main__":
    main()
