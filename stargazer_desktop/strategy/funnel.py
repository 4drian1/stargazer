"""四层漏斗的技术层(第三层)+v2.3因子+吸筹信号。全部在最新一根K线上评估。
合格线(v2.3): pos<=25, al<=30, aa<=2.8, vr<=1.05, d20<=15
草案因子: 换手率分位<=10%, BBW挤压分位<=10%, 涨停基因>=2"""
import numpy as np
import pandas as pd


def indicators(df: pd.DataFrame, fmc_yi: float = None, price: float = None):
    """df: 最新在末尾的日K(>=260行)。返回指标字典。"""
    if df is None or len(df) < 250:
        return None
    d = df.reset_index(drop=True)
    n = len(d)
    c = d["close"].iloc[-1]
    w = d.iloc[n - 250:]
    lo, hi = w["low"].min(), w["high"].max()
    pos = (c - lo) / (hi - lo) * 100 if hi > lo else 100
    al = (c / lo - 1) * 100
    last20 = d.iloc[n - 20:]
    pc = last20["close"].shift(1).fillna(last20["open"])
    aa = (((last20["high"] - last20["low"]) / pc) * 100).mean()
    v20 = d["vol"].iloc[n - 20:].mean()
    vr = d["vol"].iloc[n - 5:].mean() / v20 if v20 > 0 else 9
    d20 = (c / d["close"].iloc[n - 21] - 1) * 100
    # 均线与ATR
    ma = {f"ma{p}": d["close"].iloc[n - p:].mean() for p in (5, 10, 20, 60)}
    tr = np.maximum(d["high"] - d["low"],
                    np.maximum(abs(d["high"] - d["close"].shift(1)), abs(d["low"] - d["close"].shift(1))))
    atr14 = tr.iloc[n - 14:].mean()
    atr22 = tr.iloc[n - 22:].mean()
    # RSI
    def rsi(p):
        diff = d["close"].diff().iloc[n - p - 1:]
        up = diff.clip(lower=0).mean()
        dn = (-diff.clip(upper=0)).mean()
        return 100 - 100 / (1 + up / dn) if dn > 0 else 100.0
    bias20 = (c / ma["ma20"] - 1) * 100
    # v2.3 因子
    pct_chg = d["close"].pct_change() * 100
    limit_cnt = int((pct_chg.iloc[n - 250:] >= 9.8).sum())
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    bbw = ((mid + 2 * std) - (mid - 2 * std)) / mid * 100
    bbw_pct = float((bbw.iloc[n - 250:] <= bbw.iloc[-1]).mean() * 100)
    out = {"price": round(c, 2), "pos": round(pos, 1), "al": round(al, 1), "aa": round(aa, 2),
           "vr": round(vr, 2), "d20": round(d20, 1), "low250": round(lo, 2), **{k: round(v, 2) for k, v in ma.items()},
           "atr14": round(atr14, 3), "atr22": round(atr22, 3),
           "rsi14": round(rsi(14), 1), "rsi6": round(rsi(6), 1), "bias20": round(bias20, 1),
           "limit_cnt": limit_cnt, "bbw_pct": round(bbw_pct, 1)}
    # 换手率分位(需要流通市值)
    if fmc_yi and price:
        shares = fmc_yi * 1e8 / price  # 流通股数
        to = d["vol"] * 100 / shares * 100  # 日换手率%(腾讯量单位为手)
        to20 = to.rolling(20).mean()
        out["turnover_pct"] = round(float((to20.iloc[n - 250:] <= to20.iloc[-1]).mean() * 100), 1)
        out["turnover20"] = round(float(to20.iloc[-1]), 2)
    # 吸筹信号
    up_vol = d["vol"].iloc[n - 20:][pct_chg.iloc[n - 20:] > 0].mean()
    dn_vol = d["vol"].iloc[n - 20:][pct_chg.iloc[n - 20:] < 0].mean()
    out["ud_vol_ratio"] = round(up_vol / dn_vol, 2) if dn_vol and dn_vol > 0 else None
    obv = (np.sign(d["close"].diff().fillna(0)) * d["vol"]).cumsum()
    px_chg20 = d["close"].iloc[-1] / d["close"].iloc[n - 21] - 1
    obv_chg20 = (obv.iloc[-1] - obv.iloc[n - 21]) / (abs(obv.iloc[n - 250:]).mean() + 1e-9)
    out["obv_diverge"] = bool(px_chg20 <= 0 and obv_chg20 > 0)
    body = (d["close"] - d["open"]).abs().iloc[n - 20:]
    lower = (pd.concat([d["open"], d["close"]], axis=1).min(axis=1) - d["low"]).iloc[n - 20:]
    out["lower_shadow_cnt"] = int((lower > 2 * body.clip(lower=0.01)).sum())
    mfm = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / (d["high"] - d["low"]).replace(0, np.nan)
    mfv = mfm.fillna(0) * d["vol"]
    out["cmf20"] = round(float(mfv.iloc[n - 20:].sum() / d["vol"].iloc[n - 20:].sum()), 3) if d["vol"].iloc[n - 20:].sum() > 0 else None
    # 吸筹评级: 0-4星
    stars = sum([bool(out["ud_vol_ratio"] and out["ud_vol_ratio"] > 1.1), out["obv_diverge"],
                 out["lower_shadow_cnt"] >= 3, bool(out["cmf20"] and out["cmf20"] > 0)])
    out["accum_stars"] = stars
    # 横盘天数：从最新交易日往前数的连续横盘日数（v2.4.2口径）。
    # 横盘日：|日涨跌幅|<=2% 或 日振幅<=3%，二选一满足即算（计数不断）；
    # 只满足一条的日子记为"争议日"，随结果输出交人工复核。
    pc_all = d["close"].shift(1)
    daily_amp = (d["high"] - d["low"]) / pc_all * 100
    daily_chg = (d["close"] / pc_all - 1).abs() * 100
    calm_or = (daily_amp <= 3) | (daily_chg <= 2)
    disputed = (daily_amp <= 3) ^ (daily_chg <= 2)  # 只满足一条=争议日
    consol, disput = 0, 0
    for c, ds in zip(calm_or.iloc[::-1], disputed.iloc[::-1]):
        if bool(c):
            consol += 1
            if bool(ds):
                disput += 1
        else:
            break
    out["consol_days"] = consol
    out["consol_disputed"] = disput
    # 出货形态警示：近60日一波拉升>=30% + 位置>50% + 近7日振幅收敛 = 拉升后高位横盘，警惕A杀
    w60 = d.iloc[n - 60:]
    runup60 = (w60["close"].max() / w60["close"].iloc[0] - 1) * 100
    aa7 = float(daily_amp.iloc[n - 7:].mean())
    out["runup60"] = round(float(runup60), 1)
    out["aa7"] = round(aa7, 2)
    out["distrib_risk"] = bool(runup60 >= 30 and pos > 50 and aa7 < 3.5)
    return out


def funnel_pass(m: dict):
    """第三层合格判定 + 未过项列表"""
    fails = []
    if m["pos"] > 25: fails.append(f"位置{m['pos']}%>25%")
    if m["al"] > 30: fails.append(f"离低点{m['al']}%>30%")
    if m["aa"] > 2.8: fails.append(f"振幅{m['aa']}>2.8")
    if m["vr"] > 1.05: fails.append(f"量比{m['vr']}>1.05")
    if m["d20"] > 15: fails.append(f"20日涨{m['d20']}%>15%")
    if m.get("consol_days", 0) < 7: fails.append(f"横盘{m.get('consol_days', 0)}天<7天")
    # v2.4.3: 换手率下限——横盘7天以上的票20日换手中位数约0.83%，0.8%恰砍中位数，
    # 防"死透的僵尸票"（吸筹远未结束，点火遥遥无期）。与v2.3"换手率分位<=10%"不矛盾：
    # 分位=相对自己全年够冷，下限=绝对值别死透，甜区="冷门但活着"。
    t20 = m.get("turnover20")
    if t20 is not None and t20 < 0.8: fails.append(f"换手{t20}%<0.8%")
    # v2.4.3: 吸筹信号至少2星才进通过名单
    if m.get("accum_stars", 0) < 2: fails.append(f"吸筹{m.get('accum_stars', 0)}星<2星")
    return len(fails) == 0, fails


def layer2_pass(info: dict):
    """第二层硬指标: 3-30元, 10-100亿流通(已在取数时排除ST/北交所)"""
    return 3 <= info["price"] <= 30 and 10 <= info["fmc"] <= 100


def send_to_scoring(info: dict):
    """第四层送检范围: PE<0 或 PE>60"""
    pe = info.get("pe")
    if pe is None:
        return True
    return pe < 0 or pe > 60
