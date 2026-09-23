"""行情读取（mquote.py）——分钟线/昨收，可独立部署（无 PG/psycopg 依赖）。

从 cli.py 原样搬来 load_minute / load_pre_close / _next_trade_day：
akshare/pandas 延迟 import，监控机只需 akshare+pandas 即可运行。

日线 parquet 目录缺省 D:/a股数据/data_daily_parts，可用环境变量
STARGAZER_DAILY_PARTS 覆盖（监控机/便携环境指向自己的数据目录）。
"""
import os
from pathlib import Path

MINUTE_CACHE = Path(__file__).resolve().parent / "data" / "minute_cache"
BARS_PARTS = Path(os.environ.get("STARGAZER_DAILY_PARTS",
                                r"D:/a股数据/data_daily_parts"))


def load_minute(code, day, *, force=False):
    """生产分钟线：复用 corvus 同款新浪源口径（ak.stock_zh_a_minute，近8交易日），
    缓存在 stargazer_plan/data/minute_cache/。失败返回 None（DATA_MISSING 语义）。

    force=True 时忽略已有 parquet 缓存直接重新拉（盘中监控用）；缓存写回逻辑不变。"""
    import pandas as pd
    cache = MINUTE_CACHE / f"{code}_{day}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    if code.startswith(("4", "8", "92")):
        return None
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        import akshare as ak
        df = ak.stock_zh_a_minute(symbol=f"{prefix}{code}", period="1", adjust="")
        if df is None or df.empty:
            return None
        df["dt"] = pd.to_datetime(df["day"])
        day_next = _next_trade_day(day)
        df = df[(df["dt"] >= pd.Timestamp(f"{day} 09:00"))
                & (df["dt"] <= pd.Timestamp(f"{day_next} 15:30"))]
        if df.empty:
            return None
        for c in ("open", "close", "high", "low", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"])
        MINUTE_CACHE.mkdir(parents=True, exist_ok=True)
        df[["dt", "open", "close", "high", "low", "volume"]].to_parquet(cache, index=False)
        return df[["dt", "open", "close", "high", "low", "volume"]].reset_index(drop=True)
    except Exception:
        return None


def _next_trade_day(day):
    """下一交易日（用日线 parquet 日历；找不到就退化为 +1 自然日，仅用于拉数窗口）。"""
    import pandas as pd
    cal = set()
    for f in sorted(BARS_PARTS.glob("part_*.parquet")):
        cal.update(int(x) for x in pd.read_parquet(f, columns=["date"])["date"].unique())
    future = sorted(d for d in cal if d > int(day))
    return str(future[0]) if future else str(int(day) + 1)


def load_pre_close(code, day):
    """事件日前一交易日收盘（日线 parquet）。"""
    import pandas as pd
    for f in sorted(BARS_PARTS.glob("part_*.parquet")):
        df = pd.read_parquet(f, filters=[("code", "=", code)], columns=["date", "close"])
        df = df[df["date"] < int(day)]
        if len(df):
            return float(df["close"].iloc[-1])
    return None
