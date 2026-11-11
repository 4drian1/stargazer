"""日历接口（依据合同重建，非找回原件）。

合同：复用现有统一交易日历函数（tdx_daily_source_v2.calendar_days），
绝不重新实现日历。若真实日历函数不可用，明确抛错（由路由标记为 CALENDAR_UNAVAILABLE），
而非伪造日历。

注意：本模块为 ui_next 新接口的独立命名实现（避免与生产已有模块重名）。
"""
import sys
from pathlib import Path

_DESKTOP = Path(__file__).resolve().parents[1] / "stargazer_desktop"
if str(_DESKTOP) not in sys.path:
    sys.path.insert(0, str(_DESKTOP))


def _raw_calendar_days():
    """真实日历源：桌面端统一交易日历（akshare calendar.json 派生）。"""
    import tdx_daily_source_v2
    return sorted(tdx_daily_source_v2.calendar_days())


def trading_days(source_fn=None):
    fn = source_fn or _raw_calendar_days
    return fn()


def is_trading_day(date: int, source_fn=None) -> bool:
    return date in set(trading_days(source_fn=source_fn))


def next_trading_day(date: int, source_fn=None):
    """返回 date 之后的第一个交易日；date 自身是否交易日无关。"""
    for d in trading_days(source_fn=source_fn):
        if d > date:
            return d
    return None


def calendar_source_path():
    """日历真实来源文件（akshare calendar.json），用于响应 source_path 如实标注。"""
    import tdx_daily_source_v2
    return tdx_daily_source_v2.calendar_file()
