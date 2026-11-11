"""TDX 日线适配器 vs stockdb 日线 parquet 交叉验证（迁移第2步小样本）。

校验目标（PLAN 第2步对照项）：
- 量/额/股本/市值/名称/ST 必须一致（这些是“不改复权”也能量化的字段）。
- 价格：TDX 是原始日线，stockdb 是前复权(qfq)；两者差异即累计复权量，
  用于量化“必须先解析权息资料才能做价格级对照”，不直接判失败。

规则：固定抽查代码 + 固定随机种子小样本；容差预先登记。
"""
import os
import glob
import random
import datetime as dt
import pandas as pd
from tdx_daily_source_v1 import get_daily, to_tdx_file

SDB_PARTS = os.environ.get("STARGAZER_STOCKDB_PARTS", r"D:/a股数据/data_daily_parts")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "migration_support",
                   "tdx_vs_stockdb_daily_check.json")
SEED = 20261030
SAMPLE_PER_MKT = 6
TDX_WINDOW = ("20240101", "20261030")   # 近两年，便于看到复权累计


def load_sdb(codes6, start, end):
    files = sorted(glob.glob(os.path.join(SDB_PARTS, "part_*.parquet")))
    cols = ["date", "code", "name", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "is_st", "total_share", "float_share",
            "total_mv", "float_mv"]
    dfs = []
    for f in files:
        use = [c for c in cols if c in pd.read_parquet(f).columns]
        d = pd.read_parquet(f, columns=use)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["code"].astype(str).str.zfill(6).isin(set(codes6))]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[(df["date"] >= int(start)) & (df["date"] <= int(end))]
    return df


def compare_one(our_code, sdb_df):
    tdx, meta = get_daily(our_code, TDX_WINDOW[0], TDX_WINDOW[1])
    rec = {"our_code": our_code, "tdx_file": meta["tdx_file"], "ok": False,
           "tdx_rows": meta["raw_rows"], "sdb_rows": 0, "qfq_status": meta["qfq_status"],
           "float_mv_source": meta["float_mv_source"], "name_source": meta["name_source"]}
    if tdx is None:
        rec["error"] = "tdx raw missing"
        return rec
    key6 = our_code.upper().replace(".", "")
    s = sdb_df[sdb_df["code"] == key6]
    rec["sdb_rows"] = len(s)
    if len(s) == 0:
        rec["error"] = "sdb missing"
        return rec
    m = tdx.merge(s[["date", "open", "high", "low", "close", "volume", "amount",
                     "float_share", "float_mv", "is_st", "name"]],
                  on="date", how="inner", suffixes=("_t", "_s"))
    if len(m) == 0:
        rec["error"] = "no date overlap"
        return rec
    rec["overlap"] = len(m)
    # 量/额：原始口径应一致（TDX 原始 vs stockdb 原始？ stockdb 是 qfq 但量/额不受复权影响）
    rec["volume_max_abs"] = float((m["volume_t"] - m["volume_s"]).abs().max())
    rec["amount_max_abs"] = float((m["amount_t"] - m["amount_s"]).abs().max())
    # 股本/市值
    if m["float_share_t"].notna().any():
        rec["float_share_max_abs"] = float((m["float_share_t"] - m["float_share_s"]).abs().max())
    if m["float_mv_t"].notna().any():
        rec["float_mv_max_abs"] = float((m["float_mv_t"] - m["float_mv_s"]).abs().max())
    # 名称/ST（参照一致性）
    rec["name_match"] = bool((m["name_t"] == m["name_s"]).all())
    rec["st_match"] = bool((m["is_st_t"] == m["is_st_s"]).all())
    # 价格：TDX 原始 vs stockdb 前复权 -> 量化复权累计量（中位数比率）
    r = (m["close_s"] / m["close_t"]).replace([float("inf"), -float("inf")], float("nan")).dropna()
    rec["price_adj_ratio_median"] = float(r.median()) if len(r) else None
    rec["price_adj_ratio_min"] = float(r.min()) if len(r) else None
    rec["price_adj_ratio_max"] = float(r.max()) if len(r) else None
    rec["ok"] = True
    return rec


def main():
    fixed = ["600000", "000001.SH", "300845", "399006.SZ", "002313",
             "688651", "000001", "002933"]   # 000001 两次分别测沪指(SH)与平安银行(SZ)
    # 随机小样本（股票，每市场若干）
    sdb_all = load_sdb([], TDX_WINDOW[0], TDX_WINDOW[1])
    rnd = random.Random(SEED)
    sample = []
    for mkt, pref in (("sh", "6"), ("sz", "0"), ("sz", "3")):
        pool = sorted({c for c in sdb_all["code"] if c.startswith(pref)})
        sample += rnd.sample(pool, min(SAMPLE_PER_MKT, len(pool)))
    all_codes = fixed + sample
    sdb_df = load_sdb(set(c[-6:] if len(c) >= 6 else c for c in all_codes), *TDX_WINDOW)
    results = [compare_one(c, sdb_df) for c in all_codes]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        import json
        json.dump({"seed": SEED, "window": TDX_WINDOW, "results": results}, f, ensure_ascii=False, indent=2)
    # 控制台摘要
    print(f"窗口 {TDX_WINDOW}  测试 {len(results)} 只")
    ok = [r for r in results if r.get("ok")]
    print(f"可对照(有重叠): {len(ok)}")
    print(f"量最大绝对差: max={max((r.get('volume_max_abs',0) for r in ok), default=0):.0f}")
    print(f"额最大绝对差: max={max((r.get('amount_max_abs',0) for r in ok), default=0):.0f}")
    print(f"float_share最大绝对差: max={max((r.get('float_share_max_abs',0) for r in ok), default=0):.0f}")
    print(f"float_mv最大绝对差: max={max((r.get('float_mv_max_abs',0) for r in ok), default=0):.0f}")
    nm = [r for r in ok if not r.get("name_match")]; st = [r for r in ok if not r.get("st_match")]
    print(f"名称不一致: {len(nm)}   ST不一致: {len(st)}")
    rr = [r for r in ok if r.get("price_adj_ratio_median") is not None]
    if rr:
        med = sum(r["price_adj_ratio_median"] for r in rr) / len(rr)
        print(f"价格复权累计中位比率(=stockdb_qfq/TDX_raw): mean={med:.4f} "
              f"-> 说明价格级对照必须先解析权息资料")
    print("写入:", OUT)


if __name__ == "__main__":
    main()
