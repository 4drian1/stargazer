"""verify_daily_vs_backfill.py — 校验增量(--date)与回填(backfill)逐字段口径一致。

用法：python verify_daily_vs_backfill.py [YYYYMMD D]
默认比对 pool_daily.parquet 中 rule_version=GREEN-WATCH-v0-daily 的目标日行
与 pool_daily.parquet.bak_20260917（回填冻结产物）中同一日期的行。
除 rule_version、pre_close、close 外所有列必须完全一致（NaN 视为相等）。
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
POOL = BASE / "pool_daily.parquet"
BAK = BASE / "pool_daily.parquet.bak_20260917"
SKIP = {"rule_version", "pre_close", "close", "date", "code"}  # date/code 是连接键


def series_equal(a, b):
    """NaN 感知逐元素比较（含 float 精确相等；口径一致应逐位相同）。"""
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        a = a.astype("float64")
        b = b.astype("float64")
        both_nan = a.isna() & b.isna()
        eq = (a == b) | both_nan
    else:
        a_na, b_na = a.isna(), b.isna()
        eq = ((a == b) | (a_na & b_na)).fillna(False)
    return eq


def main(date=None):
    t0 = time.time()
    cols_new = pd.read_parquet(POOL, columns=["date", "rule_version"])
    if date is None:
        dates = sorted(cols_new.loc[cols_new["rule_version"] == "GREEN-WATCH-v0-daily",
                                    "date"].unique())
        if not dates:
            raise SystemExit("pool_daily.parquet 中没有 daily 行，先跑 --date")
        date = int(dates[-1])
    date = int(date)
    print(f"比对日期 {date}：daily(新) vs backfill(备份) …")

    daily = pd.read_parquet(POOL)
    daily = daily[(daily["date"] == date) &
                  (daily["rule_version"] == "GREEN-WATCH-v0-daily")].copy()
    bak = pd.read_parquet(BAK)
    bak = bak[bak["date"] == date].copy()
    print(f"  daily 行 {len(daily)}，backfill 行 {len(bak)}")

    # code 集合与主键对齐
    d_codes, b_codes = set(daily["code"]), set(bak["code"])
    print(f"  仅 daily 有: {len(d_codes - b_codes)}，仅 backfill 有: {len(b_codes - d_codes)}")
    common = sorted(d_codes & b_codes)
    dk = daily.set_index("code").loc[common]
    bk = bak.set_index("code").loc[common]

    compare_cols = [c for c in bak.columns if c not in SKIP]
    bad_cols, n_bad_cells = [], 0
    for c in compare_cols:
        if c not in dk.columns:
            bad_cols.append(f"{c}(缺失)")
            continue
        eq = series_equal(dk[c].reset_index(drop=True), bk[c].reset_index(drop=True))
        if not bool(eq.all()):
            bad_cols.append(c)
            n_bad_cells += int((~eq).sum())
    print(f"  全量比对列数 {len(compare_cols)}，不一致列数 {len(bad_cols)}")
    if bad_cols:
        print("  不一致列:", bad_cols, f"（共 {n_bad_cells} 个单元格）")
        for c in bad_cols[:10]:
            if c.endswith("(缺失)"):
                continue
            eq = series_equal(dk[c].reset_index(drop=True), bk[c].reset_index(drop=True))
            idx = np.where(~eq.to_numpy())[0][:5]
            for i in idx:
                print(f"    {c} code={common[i]}: daily={dk[c].iloc[i]!r} "
                      f"bak={bk[c].iloc[i]!r}")

    # 抽样逐字段比对（>=1000 行；行数少时全抽）
    rng = np.random.RandomState(42)
    n_samp = min(1000, len(common))
    samp = sorted(rng.choice(len(common), size=n_samp, replace=False).tolist())
    samp_codes = [common[i] for i in samp]
    sd = daily.set_index("code").loc[samp_codes]
    sb = bak.set_index("code").loc[samp_codes]
    mism = 0
    for c in compare_cols:
        if c not in sd.columns:
            mism += n_samp
            continue
        eq = series_equal(sd[c].reset_index(drop=True), sb[c].reset_index(drop=True))
        mism += int((~eq).sum())
    print(f"  抽样 {n_samp} 行 × {len(compare_cols)} 列 逐字段比对：不一致单元格 {mism}")

    # pre_close/close 附加校验：与日线源数据对账
    import glob
    import pyarrow.dataset as ds
    dset = ds.dataset(sorted(glob.glob("D:/a股数据/data_daily_parts/part_*.parquet")),
                      format="parquet")
    src = dset.to_table(columns=["date", "code", "pre_close", "close"],
                        filter=(ds.field("date") == date)).to_pandas()
    src["code"] = src["code"].astype(str).str.zfill(6)
    src = src.set_index("code")
    chk = daily.set_index("code")[["pre_close", "close"]]
    ok = 0
    for code, row in chk.iterrows():
        if code in src.index:
            s = src.loc[code]
            if np.isclose(row["pre_close"], s["pre_close"]) and \
               np.isclose(row["close"], s["close"]):
                ok += 1
    print(f"  pre_close/close 与日线源对账：{ok}/{len(chk)} 行一致")
    print(f"用时 {time.time()-t0:.0f}s → 结论：{'PASS' if not bad_cols and not mism else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
