"""行情数据层：腾讯报价/K线 + 东财板块。全部公开接口，无需key。"""
import json
import time
import urllib.request

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}


def _get(url, tries=3, timeout=10, encoding=None):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=timeout).read()
            return raw.decode(encoding) if encoding else raw
        except Exception:
            time.sleep(0.4 * (i + 1))
    return None


def _get_json(url, tries=3):
    raw = _get(url, tries=tries)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def secid(code: str) -> str:
    """6位代码 -> 腾讯前缀代码"""
    if code.startswith(("60", "68")):
        return "sh" + code
    return "sz" + code


# ---------------- 腾讯实时报价 ----------------
def quotes(codes):
    """codes: 6位代码列表(<=8个一批)。返回 {code: {name,price,pct,open,high,low,prev,amount_wan,time}}"""
    out = {}
    for i in range(0, len(codes), 6):  # 每批<=6，防空返回
        batch = codes[i:i + 6]
        q = ",".join(secid(c) for c in batch)
        txt = _get(f"https://qt.gtimg.cn/q={q}", encoding="gbk")
        if not txt:
            continue
        for line in txt.strip().split(";"):
            if "=" not in line:
                continue
            body = line.split("=", 1)[1].strip('"').split("~")
            if len(body) < 45 or not body[1]:
                continue
            try:
                code = body[2]
                out[code] = {
                    "name": body[1], "price": float(body[3]), "prev": float(body[4]),
                    "open": float(body[5]), "pct": float(body[32]),
                    "high": float(body[33]), "low": float(body[34]),
                    "amount_wan": float(body[37]), "time": body[30],
                    "float_mv_yi": float(body[44]), "total_mv_yi": float(body[45]),
                }
            except (ValueError, IndexError):
                continue
        time.sleep(0.1)
    return out


# ---------------- 腾讯美股/隔夜外盘 ----------------
US_WATCH = [
    ("usDJI", "道琼斯"), ("usIXIC", "纳斯达克"), ("usINX", "标普500"),
    ("usSOXX", "费城半导体ETF"), ("usHXC", "纳斯达克金龙中国"),
    ("usNVDA", "英伟达"), ("usTSM", "台积电"),
]


def us_quotes():
    """隔夜美股关键指数/风向标 [{code,name,price,pct,time}]。腾讯行情，字段与A股同构。"""
    q = ",".join(c for c, _ in US_WATCH)
    txt = _get(f"https://qt.gtimg.cn/q={q}", encoding="gbk")
    out = []
    if not txt:
        return out
    for line in txt.strip().split(";"):
        if "=" not in line:
            continue
        body = line.split("=", 1)[1].strip('"').split("~")
        if len(body) < 45 or not body[1] or not body[3]:
            continue
        try:
            out.append({"code": body[2], "name": body[1], "price": float(body[3]),
                        "pct": float(body[32]), "time": body[30]})
        except (ValueError, IndexError):
            continue
    return out


# ---------------- 大盘指数 ----------------
def retail_flow(code: str, days=10):
    """新浪口径资金流向（散户/小单观察因子，东财push2his已不可用后的替代源）。
    返回 [{date, chg_pct, main_net_wan主力净额万, retail_net_wan散户净额万}]，最新在前；失败None。
    口径说明：netamount=主力净额(元)，r0_net=散户净额(元)。仅作观察因子，不进漏斗门槛。"""
    txt = _get("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"MoneyFlow.ssl_qsfx_zjlrqs?page=1&num={days}&sort=opendate&asc=0&daima={secid(code)}")
    if not txt:
        return None
    try:
        rows = json.loads(txt)
    except (json.JSONDecodeError, ValueError):
        return None
    out = []
    for r in rows:
        try:
            out.append({"date": r["opendate"],
                        "chg_pct": round(float(r.get("changeratio") or 0) * 100, 2),
                        "main_net_wan": round(float(r.get("netamount") or 0) / 1e4, 1),
                        "retail_net_wan": round(float(r.get("r0_net") or 0) / 1e4, 1)})
        except (ValueError, TypeError, KeyError):
            continue
    return out or None


def stock_industry(code: str):
    """个股所属东财行业板块名（push2delay f127），失败返回None。"""
    em = ("1." if code.startswith("6") else "0.") + code
    d = _get_json(f"https://push2delay.eastmoney.com/api/qt/stock/get?secid={em}&fields=f57,f127")
    try:
        return d["data"].get("f127")
    except (TypeError, KeyError, AttributeError):
        return None


INDEX_WATCH = [
    ("sh000001", "上证指数"), ("sz399001", "深证成指"), ("sz399006", "创业板指"),
    ("sh000688", "科创50"), ("sh000300", "沪深300"), ("sz399303", "国证2000"),
]


def index_quotes(tries=3):
    """主要指数 [{code,name,price,pct,amount_yi}]。"""
    q = ",".join(c for c, _ in INDEX_WATCH)
    txt = _get(f"https://qt.gtimg.cn/q={q}", tries=tries, encoding="gbk")
    out = []
    for line in (txt or "").strip().split(";"):
        if "=" not in line:
            continue
        f = line.split("=", 1)[1].strip('"').split("~")
        if len(f) < 45 or not f[1] or not f[3]:
            continue
        try:
            out.append({"code": f[2], "name": f[1], "price": float(f[3]),
                        "pct": float(f[32]), "time": f[30],
                        "amount_yi": round(float(f[37]) / 1e4, 1)})
        except (ValueError, IndexError):
            continue
    return out


# ---------------- 涨跌停池（东财） ----------------
_EM_UT = "7eea3edcaed734bea9cbfc24409ed989"


def limit_pool(kind="up", date=None):
    """涨停/跌停池 {total, pool:[{code,name,pct,lbc连板数,zbc炸板次数,fbt首封时间,fund_yi封单亿,hy行业,amount_yi}]}"""
    import datetime as dt
    date = date or dt.date.today().strftime("%Y%m%d")
    ep = "getTopicZTPool" if kind == "up" else "getTopicDTPool"
    d = _get_json(f"https://push2ex.eastmoney.com/{ep}?ut={_EM_UT}&dpt=wz.ztzt"
                  f"&Pageindex=0&pagesize=500&sort=fbt%3Aasc&date={date}")
    if not d or not d.get("data"):
        return {"total": 0, "pool": []}
    pool = []
    for x in d["data"].get("pool") or []:
        pool.append({"code": x.get("c"), "name": x.get("n"),
                     "pct": round(x.get("zdp") or 0, 2),
                     "lbc": (x.get("zttj") or {}).get("ct", x.get("lbc", 1)),
                     "zbc": x.get("zbc", 0),
                     "fbt": str(x.get("fbt", "")),
                     "fund_yi": round((x.get("fund") or 0) / 1e8, 2),
                     "hy": x.get("hybk", ""),
                     "amount_yi": round((x.get("amount") or 0) / 1e8, 2)})
    return {"total": d["data"].get("tc", len(pool)), "pool": pool}


# ---------------- 腾讯历史K线（前复权） ----------------
def kline(code: str, start="2021-01-01", end=None, qfq=True):
    """返回 DataFrame[date,open,close,high,low,vol]，按日期升序。自动分段。"""
    import datetime as dt
    end = end or dt.date.today().isoformat()
    chunks = [(start, "2022-12-31"), ("2023-01-01", "2024-12-31"), ("2025-01-01", end)]
    rows = []
    for s, e in chunks:
        if s > end:
            continue
        e = min(e, end)
        if s > e:
            continue
        adj = "qfq" if qfq else ""
        raw = _get_json(
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={secid(code)},day,{s},{e},800,{adj}")
        if not raw:
            continue
        k = raw["data"].get(secid(code), {})
        lst = k.get("qfqday") or k.get("day") or []
        for r in lst:
            try:
                rows.append([r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
            except (ValueError, IndexError):
                continue
        time.sleep(0.12)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "vol"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


# ---------------- 东财板块 ----------------
def board_list():
    """全部板块 {名称: 代码}"""
    out, pn = {}, 1
    while True:
        d = _get_json("https://push2delay.eastmoney.com/api/qt/clist/get?"
                      f"pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f12&fs=m:90+t:2,m:90+t:3&fields=f12,f14")
        if not d or not d.get("data") or not d["data"].get("diff"):
            break
        for it in d["data"]["diff"]:
            out[it["f14"]] = it["f12"]
        if len(d["data"]["diff"]) < 100:
            break
        pn += 1
        time.sleep(0.1)
    return out


def board_list_sorted():
    """全部板块按当日涨跌幅降序：[(名称, 代码, pct)]。用于先粗筛再细扫。"""
    out, pn = [], 1
    while True:
        d = _get_json("https://push2delay.eastmoney.com/api/qt/clist/get?"
                      f"pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:90+t:2,m:90+t:3"
                      "&fields=f12,f14,f3")
        if not d or not d.get("data") or not d["data"].get("diff"):
            break
        for it in d["data"]["diff"]:
            try:
                pct = float(it["f3"]) if it.get("f3") not in (None, "-") else 0.0
            except (TypeError, ValueError):
                pct = 0.0
            out.append((it["f14"], it["f12"], pct))
        if len(d["data"]["diff"]) < 100:
            break
        pn += 1
        time.sleep(0.1)
    return out


def board_constituents(bk: str, exclude_st=True, exclude_bj=True):
    """板块成分 {code: {name,price,pct,pe,fmc(亿),amount_yi(亿)}}"""
    out, pn = {}, 1
    while True:
        d = _get_json("https://push2delay.eastmoney.com/api/qt/clist/get?"
                      f"pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f12&fs=b:{bk}&fields=f12,f14,f2,f3,f9,f21,f6")
        if not d or not d.get("data") or not d["data"].get("diff"):
            break
        for it in d["data"]["diff"]:
            c = it["f12"]
            if not it["f2"] or it["f2"] == "-":
                continue
            if exclude_st and "ST" in it["f14"]:
                continue
            if exclude_bj and c.startswith(("8", "4", "920")):
                continue
            # B股排除：900/200开头，或名称带"Ｂ"(全角)
            if c.startswith(("900", "200")) or "Ｂ" in it["f14"]:
                continue
            out[c] = {
                "name": it["f14"], "price": float(it["f2"]),
                "pct": it["f3"] if it["f3"] != "-" else 0.0,
                "pe": it["f9"] if it["f9"] != "-" else None,
                "fmc": it["f21"] / 1e8 if it["f21"] and it["f21"] != "-" else 0.0,
                "amount_yi": round(it["f6"] / 1e8, 2) if it.get("f6") and it["f6"] != "-" else 0.0,
            }
        if len(d["data"]["diff"]) < 100:
            break
        pn += 1
        time.sleep(0.15)
    return out


def board_fund_flow(top=15):
    """板块主力资金流：返回 (流入榜, 流出榜)，元素 (名称, 涨跌幅%, 主力净流入亿, 净占比%)"""
    def _fetch(po):
        d = _get_json("https://push2delay.eastmoney.com/api/qt/clist/get?"
                      f"pn=1&pz={top}&po={po}&np=1&fltt=2&invt=2&fid=f62&fs=m:90+t:2,m:90+t:3"
                      "&fields=f12,f14,f3,f62,f184")
        if not d or not d.get("data") or not d["data"].get("diff"):
            return []
        return [(it["f14"], it["f3"] if it["f3"] != "-" else 0,
                 round(it["f62"] / 1e8, 2) if it["f62"] != "-" else 0,
                 round(it["f184"], 2) if it["f184"] != "-" else 0)
                for it in d["data"]["diff"]]
    return _fetch(1), _fetch(0)
