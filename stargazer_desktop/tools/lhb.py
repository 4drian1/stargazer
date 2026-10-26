"""龙虎榜数据层：东方财富数据中心公开接口，无需key。

两个接口：
- 日榜   RPT_DAILYBILLBOARD_DETAILS   某日所有上榜股票 + 上榜原因 + 买卖净额
- 席位   RPT_BILLBOARD_DAILYDETAILSBUY 某股当日上榜席位明细（含BUY/SELL/NET，
         买榜卖榜内容相同只是排序不同，取一份自行分列即可）
注意：必须带 Referer: https://data.eastmoney.com/ 否则被拦。
"""
import json
import time
import urllib.parse
import urllib.request

UA = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.eastmoney.com/",
}
BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _get(params, tries=3, timeout=12):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=timeout).read()
            return json.loads(raw.decode("utf-8"))
        except Exception:
            time.sleep(0.5 * (i + 1))
    return None


def lhb_daily(date, page_size=500):
    """某日龙虎榜股票总榜。返回 list[dict]，按净买额降序；失败返回 None。

    字段：code/name/explain(上榜原因)/chg_pct/net_amt/buy_amt/sell_amt/accum_amount(当日成交额)/net_ratio
    """
    d = _get({
        "reportName": "RPT_DAILYBILLBOARD_DETAILS",
        "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,EXPLAIN,CHANGE_RATE,"
                   "BILLBOARD_NET_AMT,BILLBOARD_BUY_AMT,BILLBOARD_SELL_AMT,"
                   "ACCUM_AMOUNT,DEAL_NET_RATIO",
        "filter": f"(TRADE_DATE='{date}')",
        "pageNumber": 1,
        "pageSize": page_size,
        "sortColumns": "BILLBOARD_NET_AMT",
        "sortTypes": -1,
        "source": "WEB",
        "client": "WEB",
    })
    rows = ((d or {}).get("result") or {}).get("data")
    if rows is None:
        return None
    out = []
    seen = set()
    for x in rows:
        code = x.get("SECURITY_CODE")
        if code in seen:
            continue  # 同一股票因多个上榜类型出现多行，数据相同，保留净买额最大的一行
        seen.add(code)
        out.append({
            "code": code,
            "name": x.get("SECURITY_NAME_ABBR"),
            "explain": x.get("EXPLAIN"),
            "chg_pct": x.get("CHANGE_RATE"),
            "net_amt": x.get("BILLBOARD_NET_AMT"),
            "buy_amt": x.get("BILLBOARD_BUY_AMT"),
            "sell_amt": x.get("BILLBOARD_SELL_AMT"),
            "accum_amount": x.get("ACCUM_AMOUNT"),
            "net_ratio": x.get("DEAL_NET_RATIO"),
        })
    return out


def lhb_stock_seats(code, date, page_size=50):
    """某股某日上榜席位明细。返回 list[dict]；未上榜返回 []；失败返回 None。

    字段：seat(营业部名称)/buy/sell/net/explanation(上榜原因)/rise_prob_3day(历史3日上涨概率%)
    """
    d = _get({
        "reportName": "RPT_BILLBOARD_DAILYDETAILSBUY",
        "columns": "ALL",
        "filter": f"(TRADE_DATE='{date}')(SECURITY_CODE=\"{code}\")",
        "pageNumber": 1,
        "pageSize": page_size,
        "sortColumns": "BUY",
        "sortTypes": -1,
        "source": "WEB",
        "client": "WEB",
    })
    if d is None:
        return None  # 网络层失败
    if not d.get("success"):
        return []  # 东财空结果返回 success=False + "返回数据为空"，即当日未上榜
    rows = ((d.get("result") or {}).get("data")) or []
    out = []
    for x in rows:
        out.append({
            "seat": x.get("OPERATEDEPT_NAME"),
            "buy": x.get("BUY"),
            "sell": x.get("SELL"),
            "net": x.get("NET"),
            "explanation": x.get("EXPLANATION"),
            "rise_prob_3day": x.get("RISE_PROBABILITY_3DAY"),
        })
    return out
