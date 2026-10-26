"""策略二: 超卖价值回归三层漏斗(绿盘低吸)。
第一层池子(好公司底线)由调用方保证; 此处判定 错杀幅度+超卖+止跌结构。
新增观察档: K线贴近MA5/MA10(±3%内)，即便轻微跌破(跌幅不大)也纳入观察。"""
import numpy as np
import pandas as pd


def oversold_check(df: pd.DataFrame):
    """返回 (是否纳入, 明细dict)。
    强触发: 幅度(单日<=-5%或5日<=-12%) + 超卖(RSI14<30或RSI6<20或BIAS20<-8) + 止跌结构(长下影/阳包阴/缩量十字星)
    观察档: K线贴近MA5/MA10(±3%内)，即便轻微跌破(跌幅不大)也纳入观察名单"""
    if df is None or len(df) < 60:
        return False, {}
    d = df.reset_index(drop=True)
    n = len(d)
    c = d["close"].iloc[-1]
    pct1 = (c / d["close"].iloc[-2] - 1) * 100
    pct5 = (c / d["close"].iloc[-6] - 1) * 100
    diff = d["close"].diff()
    def rsi(p):
        dd = diff.iloc[n - p - 1:]
        up, dn = dd.clip(lower=0).mean(), (-dd.clip(upper=0)).mean()
        return 100 - 100 / (1 + up / dn) if dn > 0 else 100.0
    ma5 = d["close"].iloc[n - 5:].mean()
    ma10 = d["close"].iloc[n - 10:].mean()
    ma20 = d["close"].iloc[n - 20:].mean()
    bias20 = (c / ma20 - 1) * 100
    gap5 = (c / ma5 - 1) * 100
    gap10 = (c / ma10 - 1) * 100
    # 观察档: 贴近MA5/MA10(±3%)，即便轻微跌破也可纳入观察
    near_ma = bool(-3 <= gap5 <= 3 and -3 <= gap10 <= 3)
    # 止跌结构
    o, h, l = d["open"].iloc[-1], d["high"].iloc[-1], d["low"].iloc[-1]
    body = abs(c - o)
    lower = min(o, c) - l
    long_lower = lower > 2 * max(body, 0.01)
    prev_o, prev_c = d["open"].iloc[-2], d["close"].iloc[-2]
    yang_engulf = c > prev_o and prev_c < prev_o and c > o
    v5, v20 = d["vol"].iloc[n - 5:].mean(), d["vol"].iloc[n - 20:].mean()
    doji_shrink = body / c * 100 < 0.5 and v5 < v20
    detail = {"pct1": round(float(pct1), 2), "pct5": round(float(pct5), 2),
              "rsi14": round(float(rsi(14)), 1), "rsi6": round(float(rsi(6)), 1),
              "bias20": round(float(bias20), 1), "gap5%": round(float(gap5), 2),
              "gap10%": round(float(gap10), 2), "靠近均线(±3%)": bool(near_ma),
              "长下影": bool(long_lower), "阳包阴": bool(yang_engulf), "缩量十字星": bool(doji_shrink)}
    amplitude = bool(pct1 <= -5 or pct5 <= -12)
    oversold = bool(rsi(14) < 30 or rsi(6) < 20 or bias20 < -8)
    bottoming = bool(long_lower or yang_engulf or doji_shrink)
    strict = bool(amplitude and oversold and bottoming)
    # 观察档: 不要求强触发，只要贴近MA5/MA10即纳入观察
    include = bool(strict or near_ma)
    tier = "触发" if strict else ("观察" if near_ma else "未触发")
    detail["判定"] = {"幅度": amplitude, "超卖": oversold, "止跌": bottoming,
                      "贴近均线": near_ma, "档位": tier}
    return include, detail
