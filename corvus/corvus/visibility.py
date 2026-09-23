"""时间可见性规则（防前视执行层）。

规格第 5 节：
    规则：available_at ≤ 快照时点 且 通过工具注册表"当时是否可见"校验，缺一不可。
    核心规则不是"统一北京时间"，而是"这个数在当时是否已经可见"。

判定公式：
    event_time 所在交易日 session_close（换算北京时间）+ available_delay ≤ decision_time
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")

# 快照时刻表（北京时间，规格第 5 节）
SNAPSHOTS = {
    "POST_MARKET":   "15:00",   # 盘后快照：复盘、次日观察池
    "OVERNIGHT":     "05:30",   # 隔夜快照：美股收盘后更新外围状态（北京时间清晨）
    "PREMARKET_830": "08:30",   # 盘前快照：隔夜已收盘市场 + 夜间新闻
    "PREMARKET_915": "09:15",   # 盘前快照冻结
    "AUCTION_925":   "09:25",   # 竞价快照：A股竞价反馈 + 日韩盘中（香港未开盘）
    "INTRADAY_1000": "10:00",   # 盘中快照
    "INTRADAY_1030": "10:30",
    "INTRADAY_1330": "13:30",
}

# 国内时段品种：9:15/9:25 盘前快照只能用前一交易日已完成数据（见 instruments.yaml notes）
DOMESTIC_INSTRUMENTS = {"IRON_ORE", "COAL", "A50_FUT"}
# 港股：9:30 前当日数据不可见（香港未开盘）
HK_INSTRUMENTS = {"HSI", "HSTECH", "HSCE"}
# 日韩：9:25 时只有部分盘中信息——按 session_close 自然推导，此处仅 notes 标注
PARTIAL_INTRADAY_NOTES = {
    "N225": "9:25 竞价快照时仅有当日部分盘中信息，按 session_close 规则推导可见性",
    "KS11": "9:25 竞价快照时仅有当日部分盘中信息，按 session_close 规则推导可见性",
}


def _parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def _to_beijing(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BEIJING)
    return dt.astimezone(BEIJING)


def compute_available_at(event_time, instrument_registry_row):
    """按注册表规则计算 available_at（北京时间）。

    event_time + 所在交易日 session_close(换算北京时间) + available_delay。
    拿不到 session 信息（tz/close 缺失）时保守取 event_time + 1 天。
    """
    event_bj = _to_beijing(_parse_time(event_time))
    try:
        tz_name = instrument_registry_row["exchange_timezone"]
        close_str = instrument_registry_row["session_close"]
        delay_min = int(instrument_registry_row.get("available_delay_minutes", 0))
        local_tz = ZoneInfo(tz_name)
        # event_time 的当地日期 + session_close（当地）
        local_dt = event_bj.astimezone(local_tz)
        hh, mm = close_str.split(":")
        close_local = local_dt.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        close_bj = close_local.astimezone(BEIJING)
        return close_bj + timedelta(minutes=delay_min)
    except Exception:
        return event_bj + timedelta(days=1)


def special_rule_check(instrument_id, event_time, decision_time):
    """特殊规则，写死为可测试的纯函数。返回 (bool 是否放行, str 说明)。"""
    event_bj = _to_beijing(_parse_time(event_time))
    decision_bj = _to_beijing(_parse_time(decision_time))
    same_day = event_bj.date() == decision_bj.date()
    hhmm = decision_bj.strftime("%H:%M")

    if instrument_id in DOMESTIC_INSTRUMENTS and same_day and hhmm in ("08:30", "09:15", "09:25"):
        # 国内时段品种：盘前/竞价快照当日 bar 尚未完成，只放行前一交易日数据
        return False, "国内时段品种盘前快照只允许前一交易日已完成数据"
    if instrument_id in HK_INSTRUMENTS and same_day and hhmm < "09:30":
        return False, "港股 9:30 前当日数据不可见（香港未开盘）"
    return True, ""


def is_available(observation_row, decision_time, instrument_registry_row):
    """判定"这个数在当时是否已经可见"。

    observation_row: 至少含 event_time、available_at（可为 None，则现场计算）
    decision_time:   决策/快照时刻（北京时间）
    instrument_registry_row: instruments.yaml 中该序列的注册表行
    """
    decision_bj = _to_beijing(_parse_time(decision_time))

    # 规则一：available_at ≤ decision_time
    available_at = observation_row.get("available_at")
    if available_at is None:
        available_at = compute_available_at(observation_row["event_time"], instrument_registry_row)
    else:
        available_at = _to_beijing(_parse_time(available_at))
    if available_at > decision_bj:
        return False

    # 规则二：session_close 自然推导 + 特殊规则
    computed = compute_available_at(observation_row["event_time"], instrument_registry_row)
    if computed > decision_bj:
        return False

    instrument_id = instrument_registry_row["instrument_id"]
    ok, _ = special_rule_check(instrument_id, observation_row["event_time"], decision_time)
    return ok


def _row(**kw):
    return kw


def _self_test():
    """单元自检：若干防前视断言。"""
    import yaml

    cfg = yaml.safe_load(open(__file__.replace("visibility.py", "../config/instruments.yaml"),
                              encoding="utf-8"))
    reg = {i["instrument_id"]: i for i in cfg["instruments"]}

    # 昨夜美股收盘数据对 8:30 盘前快照可用（美股 16:00 ET 收盘 = 次日 04:00/05:00 北京）
    us_event = "2026-09-10T00:00:00+08:00"  # 2026-09-09 美股交易日
    obs = _row(event_time=us_event, available_at=None)
    assert is_available(obs, "2026-09-10T08:30:00+08:00", reg["SPX"]) is True, \
        "昨夜美股收盘数据应对 8:30 快照可用"
    # 同理对隔夜快照（05:30）也可用：16:00 ET + 60min = 05:00 北京（夏令时）
    assert is_available(obs, "2026-09-10T05:30:00+08:00", reg["SPX"]) is True

    # 当日盘中 10:00 才能用昨夜美股？不——昨夜数据本来就可用；这里验证当天美股（未收盘）不可用
    today_us = "2026-09-10T00:00:00+08:00"
    obs_today = _row(event_time="2026-09-10T00:00:00+08:00", available_at=None)
    # 修正：event_time 需对应美股交易日，这里用 9/10 当天对应的美股交易日是 9/9（北京时区日期错位），
    # 为避免时区日期错位干扰，直接验证"未来 bar"不可见：
    obs_future = _row(event_time="2026-09-11T00:00:00+08:00", available_at=None)
    assert is_available(obs_future, "2026-09-10T08:30:00+08:00", reg["SPX"]) is False, \
        "未来日期的 bar 不可见"

    # 港股当日盘中 bar 对 9:25 竞价快照不可用（香港未开盘，session_close 未达成）
    obs_hk_intraday = _row(event_time="2026-09-11T09:00:00+08:00", available_at=None)
    assert is_available(obs_hk_intraday, "2026-09-11T09:25:00+08:00", reg["HSI"]) is False, \
        "港股当日盘中数据对 9:25 快照不可用"
    # 特殊规则直接断言：港股同日 9:30 前不可见，9:30 起放行当日盘中信息
    ok_925, _ = special_rule_check("HSI", "2026-09-11T09:00:00+08:00", "2026-09-11T09:25:00+08:00")
    ok_930, _ = special_rule_check("HSI", "2026-09-11T09:00:00+08:00", "2026-09-11T09:30:00+08:00")
    assert ok_925 is False and ok_930 is True
    # 港股已完成交易日 bar 对次日 13:30 盘中快照可用
    obs_hk = _row(event_time="2026-09-10T16:00:00+08:00", available_at=None)
    assert is_available(obs_hk, "2026-09-11T13:30:00+08:00", reg["HSI"]) is True

    # 铁矿石当日盘中 bar 对 9:15 盘前快照不可用（国内时段，当日 session 未完成）
    obs_io = _row(event_time="2026-09-11T11:30:00+08:00", available_at=None)
    assert is_available(obs_io, "2026-09-11T09:15:00+08:00", reg["IRON_ORE"]) is False, \
        "铁矿石当日盘中数据对 9:15 盘前快照不可用"
    # 特殊规则直接断言：盘前/竞价快照只放行前一交易日已完成数据
    ok_io, _ = special_rule_check("IRON_ORE", "2026-09-11T11:30:00+08:00", "2026-09-11T09:15:00+08:00")
    assert ok_io is False
    # 前一交易日铁矿石 bar 对 9:15 可用
    obs_io_prev = _row(event_time="2026-09-09T15:00:00+08:00", available_at=None)
    assert is_available(obs_io_prev, "2026-09-10T09:15:00+08:00", reg["IRON_ORE"]) is True

    # 日韩 9:25 时只有部分盘中信息：当日 bar 按 session_close 推导不可用（notes 标注）
    obs_jp = _row(event_time="2026-09-11T11:00:00+08:00", available_at=None)
    assert is_available(obs_jp, "2026-09-11T09:25:00+08:00", reg["N225"]) is False
    assert is_available(obs_jp, "2026-09-11T10:00:00+08:00", reg["N225"]) is False
    # 前一交易日完成的日经 bar 对 9:25 竞价快照可用
    obs_jp_prev = _row(event_time="2026-09-10T15:00:00+08:00", available_at=None)
    assert is_available(obs_jp_prev, "2026-09-11T09:25:00+08:00", reg["N225"]) is True

    # FRED 序列有发布延迟：9/9 的值美东 9/9 午后发布，北京时间 9/10 08:00 起可见
    obs_dgs = _row(event_time="2026-09-09T16:00:00+08:00", available_at=None)
    assert is_available(obs_dgs, "2026-09-10T05:30:00+08:00", reg["US10Y"]) is False, \
        "隔夜快照 05:30 早于 FRED 发布延迟，不可见"
    assert is_available(obs_dgs, "2026-09-10T08:30:00+08:00", reg["US10Y"]) is True
    # 9/10 当日值次日才发布：9/10 8:30 不可见，9/11 8:30 可见
    obs_dgs2 = _row(event_time="2026-09-10T16:00:00+08:00", available_at=None)
    assert is_available(obs_dgs2, "2026-09-10T08:30:00+08:00", reg["US10Y"]) is False
    assert is_available(obs_dgs2, "2026-09-11T08:30:00+08:00", reg["US10Y"]) is True

    print("visibility 自检：全部断言通过")


if __name__ == "__main__":
    _self_test()
