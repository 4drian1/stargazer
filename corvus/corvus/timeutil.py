"""统一时间工具（1.3）：所有采集器不准各自裸调 datetime 解析来源时间。

parse_source_time() 集中处理来源时间字符串，返回四元组：
    published_at    带时区 ISO8601（北京时间 +08:00 或保留原时区）；解析失败 NULL
    tz_inference    时区推断方式文本
    time_parse_status  OK / PARTIAL
    source_time_raw 原文（调用方自行存库，此处原样返回）

规则：
- 带时区 ISO → OK，保留原时区。
- 无时区 naive → 按北京时间 +08:00 解释，tz_inference 注明。
- 相对时间（如"昨天 15:30"）→ 可换算但不可靠，PARTIAL。
- 完全无法解析 → PARTIAL，published_at 为 None。
"""
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")

_RE_RELATIVE = re.compile(r"^(昨天|今日|今天|前天)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_RE_NAIVE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$")


def now_iso():
    """当前时刻，带时区 ISO8601（北京时间）。"""
    return datetime.now(BEIJING).isoformat()


def parse_source_time(raw):
    """解析来源时间文本。raw 为 None/空串也走同一出口。"""
    if raw is None or str(raw).strip() == "":
        return None, None, "PARTIAL", raw

    s = str(raw).strip()

    # 1) ISO8601 带时区（fromisoformat 支持 +08:00 / Z）
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.isoformat(), "来源自带时区，原样保留", "OK", raw
    except ValueError:
        pass

    # 2) 无时区 naive：按北京时间解释
    m = _RE_NAIVE.match(s)
    if m:
        y, mo, d, h, mi, sec = (int(g) for g in m.groups("0"))
        try:
            dt = datetime(y, mo, d, h, mi, sec, tzinfo=BEIJING)
            return dt.isoformat(), "naive→+08:00 按北京时间解释", "OK", raw
        except ValueError:
            return None, "naive 日期非法", "PARTIAL", raw

    # 3) 相对时间："昨天 15:30" 等，换算但标 PARTIAL
    m = _RE_RELATIVE.match(s)
    if m:
        day_word, h, mi, sec = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4) or 0)
        offset = {"前天": -2, "昨天": -1, "今日": 0, "今天": 0}[day_word]
        base = datetime.now(BEIJING) + timedelta(days=offset)
        try:
            dt = base.replace(hour=h, minute=mi, second=sec, microsecond=0)
            return dt.isoformat(), "相对时间换算→naive→+08:00 按北京时间解释", "PARTIAL", raw
        except ValueError:
            return None, "相对时间时分非法", "PARTIAL", raw

    return None, f"无法解析: {s[:40]}", "PARTIAL", raw


def to_beijing_iso(dt):
    """datetime → 北京时间带时区 ISO8601。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BEIJING)
    return dt.astimezone(BEIJING).isoformat()
