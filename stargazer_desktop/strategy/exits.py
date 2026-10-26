"""风控引擎(v2.2三层止损+修正案+止盈阶梯+v2.3吊灯草案)。
优先级: 逻辑止损 > 结构止损 > 移动止盈。均线信号仅在启动后(曾浮盈>=8%)有效。"""
import pandas as pd

ALERT_LADDER = [(5, "+5%: 复核板块五阶段位置"), (10, "+10%: 数新闻密度,刷屏=准备出货"),
                (15, "+15%: 止损线上移到成本价"), (20, "+20%: 对照止盈线分批卖,板块过半收紧")]


def check_position(pos: dict, df: pd.DataFrame, sector_ignited: bool = False):
    """pos: {entry, entry_date, low250, catalyst_date(可空), logic}
    df: 日K(最新在末尾)。sector_ignited: 板块是否已点火(已点火则时间止损失效)。
    返回 {gain, held_days, signals:[...], action: 建议}"""
    d = df.reset_index(drop=True)
    c = d["close"].iloc[-1]
    entry = pos["entry"]
    gain = (c / entry - 1) * 100
    held = len(d[d["date"] > pos["entry_date"]])
    since = d[d["date"] >= pos["entry_date"]]
    if since.empty:  # 买入日晚于最新K线(如非交易日登记)，用最后一天兜底避免NaN
        since = d.tail(1)
    max_close_gain = (since["close"].max() / entry - 1) * 100
    ma5 = d["close"].iloc[-5:].mean()
    ma10 = d["close"].iloc[-10:].mean()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift(1)).abs(),
                    (d["low"] - d["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr22 = tr.iloc[-22:].mean()
    chandelier = since["close"].max() - 3 * atr22

    signals, action = [], "持有"
    # 1. 结构止损(任何阶段)
    if c < pos["low250"]:
        signals.append(f"结构止损触发: 收盘{c:.2f} < 250日低点{pos['low250']:.2f}")
        action = "清仓"
    # 2. 时间止损(带两个例外: 催化日历 pending / 板块已点火)
    elif held >= 8 and gain < 2 and not sector_ignited:
        import datetime as dt
        catalyst_pending = pos.get("catalyst_date") and pos["catalyst_date"] >= dt.date.today().isoformat()
        if not catalyst_pending:
            signals.append(f"时间止损: {held}个交易日题材无动静且未点火(当前{gain:+.1f}%)")
            action = "换股"
        else:
            signals.append(f"时间止损豁免: 催化日{pos['catalyst_date']}未到, 继续等待")
    # 3. 移动止盈(仅启动后)
    if action == "持有" and max_close_gain >= 8:
        if c < ma10:
            signals.append(f"移动止盈: 曾浮盈{max_close_gain:+.1f}%, 收盘破MA10({ma10:.2f})")
            action = "清仓"
        elif c < ma5:
            signals.append(f"移动止盈: 曾浮盈{max_close_gain:+.1f}%, 收盘破MA5({ma5:.2f})")
            action = "减半"
        elif c < chandelier:
            signals.append(f"吊灯止损(草案): 收盘{c:.2f} < 吊灯线{chandelier:.2f}")
            if action == "持有":
                action = "减半"
    # 止盈提醒阶梯
    for thr, msg in ALERT_LADDER:
        if gain >= thr:
            signals.append(f"阶梯提醒 {msg}")
    return {"gain": round(gain, 2), "held_days": held, "max_close_gain": round(max_close_gain, 2),
            "ma5": round(ma5, 2), "ma10": round(ma10, 2), "chandelier": round(chandelier, 2),
            "signals": signals, "action": action}


def check_watch(watch_price_map: dict):
    """观察名单单日-5%预警。watch_price_map: {code: (name, pct)}"""
    return [(c, n, p) for c, (n, p) in watch_price_map.items() if p <= -5]
