"""trading_calendar — 统一交易日历 (GREEN-WATCH/pool 与 LATR loader 同源).

约束 (用户 ③.1-1): effective_trade_date 只能来自统一交易日历。
日历由 canonical daily parts (data_daily_parts) 的 date 列推导 —— 与 build_pool 的
BARS_GLOB 指向同一份数据, 保证 plan_date -> effective_trade_date 在两段代码完全一致,
不会出现"文件路径相同、内容版本变了"的复现问题。
"""
import os
import glob
import hashlib
import pyarrow.dataset as ds
from data_source_provenance import DAILY_PARTS

_CAL = None


def _glob():
    return sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))


def load_calendar():
    """返回 sorted list[int] 交易日 (来自 canonical daily parts 全部 date)。"""
    global _CAL
    if _CAL is None:
        files = _glob()
        dates = ds.dataset(files, format="parquet") \
            .to_table(columns=["date"]).column("date").to_pylist()
        _CAL = sorted({int(d) for d in dates})
    return _CAL


def next_trading_day(d):
    """返回 d 之后第一个交易日 (str YYYYMMDD); 无则 None。"""
    d = int(d)
    cal = load_calendar()
    later = [x for x in cal if x > d]
    return str(later[0]) if later else None


def data_epoch_fingerprint():
    """source_data_snapshot_id: 文件路径 + 文件数 + 总字节 + 最新 mtime + data epoch。

    不逐行 hash (17M 行代价大), 用元数据指纹识别 '文件路径相同但内容版本变了'。
    """
    files = _glob()
    if not files:
        return "NO_DAILY_PARTS"
    sizes = [os.path.getsize(f) for f in files]
    mtimes = [os.path.getmtime(f) for f in files]
    cal = load_calendar()
    epoch = max(cal) if cal else 0
    h = hashlib.sha256()
    for f, s in zip(files, sizes):
        h.update(f.encode("utf-8"))
        h.update(str(s).encode("utf-8"))
    digest = h.hexdigest()[:16]
    return (f"daily_parts@{len(files)}files_s{sum(sizes)}_"
            f"m{max(mtimes):.0f}_epoch{epoch}_h{digest}")
