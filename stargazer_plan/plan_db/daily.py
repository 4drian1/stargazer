"""个股日线读取（plan_db/daily.py）。

数据源固定为 D:/a股数据/data_daily_parts/part_*.parquet（按 code 过滤读取）。
日线存在增量延迟：窗口未走完一律 CENSORED_TAIL，绝不填 0（规格第 4 节）。
"""
import glob

import pandas as pd

PART_GLOB = r"D:/a股数据/data_daily_parts/part_*.parquet"

_COLS = ["date", "open", "high", "low", "close"]


def load_daily(code):
    """读取单只个股的全部日线，返回按日期升序的 DataFrame（date 为 Timestamp）。

    多个 part 文件可能有重叠，按 date 去重；找不到该股返回空 DataFrame。
    """
    frames = []
    for path in sorted(glob.glob(PART_GLOB)):
        try:
            df = pd.read_parquet(path, columns=_COLS,
                                 filters=[("code", "=", str(code))])
        except Exception:
            continue
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_COLS)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


def prev_close_on_or_before(daily_df, day):
    """day（Timestamp/date）之前最近一个交易日的收盘；日线未覆盖到则返回 None。"""
    if daily_df is None or len(daily_df) == 0:
        return None
    d = pd.Timestamp(day)
    hist = daily_df[daily_df["date"] < d]
    if len(hist) == 0:
        return None
    return float(hist["close"].iloc[-1])
