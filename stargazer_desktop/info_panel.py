"""info_panel.py — 信息面板采集（桌面版，路线 #13：SHADOW 展示，无 LLM，无决策权）。

移植自 D:/stargazer/info_panel/collect.py 并加资讯层：
  快讯源 = ak.stock_info_global_sina（新浪全球财经快讯，机器转述未经核实，如实标注）
  打标   = 宏观/商品词表（金/油/美元/美联储/美股）+ 申万行业词表 → 供面板按"计划相关"过滤

分析层仍只有两种（全页面明标）：①规则标注（冻结词表粗表）②materiality 正式分
（仅 A股公告、T+1 收盘后 mature）。资讯 = 事实层（转述源，未经核实标注）。
所有依赖 lazy import，桌面端缺 akshare/psycopg 时对应区块降级为缺口，不拖垮主程序。
产物落 data/info_panel/snapshots|notes。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
import paths as _paths  # 桌面端路径规范：打包后可写数据在 exe 同级，不在 _MEI 临时目录
DATA = Path(_paths.APP_DIR) / "data" / "info_panel"
def _sibling(name):
    """桌面端 exe/源码同级或上级目录查找（本机与监控机目录形态不同）。

    自 APP_DIR 向上扫描含目标子目录的项目根，不再硬编码 D:/stargazer；
    找不到时返回最近候选（缺口会在 gaps 区明示）。"""
    bases = [Path(_paths.APP_DIR).parent, *Path(_paths.APP_DIR).parents]
    for b in bases:
        p = b / name
        if p.exists():
            return p
    return Path(_paths.APP_DIR).parent / name  # 兜底缺省（缺口会在 gaps 区明示）


C3 = Path(os.environ.get("STARGAZER_C3_ROOT") or "") if os.environ.get(
    "STARGAZER_C3_ROOT") else _sibling("c3")
PLAN = Path(os.environ.get("STARGAZER_PLAN_ROOT") or "") if os.environ.get(
    "STARGAZER_PLAN_ROOT") else _sibling("stargazer_plan")
PIT = _sibling("stargazer_ml") / "v5_hybrid/opportunity_v2/o5_sector_lag/pit_membership.parquet"
sys.path.insert(0, str(PLAN))

KEYWORD_DIRECTION = {  # 与 c3/_taxonomy.py 冻结粗表一致（仅展示用）
    "预增": 1, "扭亏": 1, "中标": 1, "回购": 1, "增持": 1, "收购": 1,
    "资产注入": 1, "补贴": 1, "涨价": 1,
    "预亏": -1, "预降": -1, "减持": -1, "退市": -1, "立案": -1, "处罚": -1,
    "诉讼": -1, "违约": -1, "逾期": -1, "停产": -1, "下修": -1, "警示函": -1,
    "监管函": -1, "解禁": -1, "降价": -1, "退市风险": -1, "终止上市": -1,
    "火灾": -1, "爆炸": -1, "事故": -1,
}
DIRECTION_TEXT = {1: "偏正", -1: "偏负", None: "中性/未知"}
CATEGORY_PRIOR = {
    "SHAREHOLDER_REDUCTION": "减持：历史先验短期偏空",
    "EARNINGS_PREVIEW": "业绩预增/预亏分歧大，看具体方向",
    "EARNINGS_REVISION": "业绩修正（下修偏空）",
    "RESTRUCTURING": "重组/控制权：事件驱动高波动",
    "LITIGATION_DEFAULT": "诉讼/违约：偏空风险事件",
    "BUYBACK": "回购：温和偏多信号",
    "SHAREHOLDER_INCREASE": "增持：温和偏多信号",
    "SHARE_UNLOCK": "解禁：供给压力偏空",
    "REGULATORY_PENALTY": "监管处罚/立案：偏空风险事件",
    "INDUSTRY_POLICY": "行业政策：方向取决于内容",
    "DELISTING_RISK": "退市风险：高危",
    "MAJOR_CONTRACT": "重大合同/中标：偏多",
    "EQUITY_CHANGE": "权益变动/举牌：看方向",
}
# 快讯打标词表：宏观商品 + 申万行业（值为展示标签）
MACRO_TAGS = {
    "黄金": "贵金属", "金价": "贵金属", "白银": "贵金属", "白银价格": "贵金属",
    "原油": "原油", "油价": "原油", "WTI": "原油", "布伦特": "原油",
    "OPEC": "原油", "欧佩克": "原油", "产油国": "原油",
    "美元": "宏观货币", "美联储": "宏观货币", "降息": "宏观货币", "加息": "宏观货币",
    "通胀": "宏观货币", "CPI": "宏观货币", "非农": "宏观货币", "关税": "宏观货币",
    "美股": "美股", "纳斯达克": "美股", "标普": "美股", "道指": "美股",
}
SECTOR_TAGS = {
    "医药": "医药生物", "生物": "医药生物", "药品": "医药生物", "疫苗": "医药生物",
    "化工": "基础化工", "化学": "基础化工", "塑料": "基础化工",
    "钢铁": "钢铁", "钢材": "钢铁",
    "有色": "有色金属", "铜价": "有色金属", "铝": "有色金属", "稀土": "有色金属", "黄金": "有色金属",
    "煤炭": "煤炭", "煤价": "煤炭",
    "电力": "电力设备", "光伏": "电力设备", "风电": "电力设备", "储能": "电力设备",
    "新能源": "电力设备", "锂电池": "电力设备", "电池": "电力设备",
    "芯片": "电子", "半导体": "电子", "集成电路": "电子", "光刻": "电子", "存储": "电子",
    "地产": "房地产", "楼市": "房地产", "房地产": "房地产",
    "银行": "银行", "券商": "非银金融", "证券": "非银金融", "保险": "非银金融",
    "汽车": "汽车", "新能源车": "汽车", "整车": "汽车",
    "白酒": "食品饮料", "食品": "食品饮料", "消费": "食品饮料", "饮料": "食品饮料",
    "农业": "农林牧渔", "生猪": "农林牧渔", "猪价": "农林牧渔", "粮食": "农林牧渔",
    "军工": "国防军工", "国防": "国防军工", "武器": "国防军工",
    "石油": "石油石化", "天然气": "石油石化", "石化": "石油石化", "炼油": "石油石化",
    "基建": "建筑装饰", "建筑": "建筑装饰", "建材": "建筑材料", "水泥": "建筑材料",
    "玻璃": "建筑材料", "机械": "机械设备", "机器人": "机械设备", "工程机械": "机械设备",
    "家电": "家用电器", "空调": "家用电器",
    "通信": "通信", "5G": "通信", "算力": "通信", "光模块": "通信",
    "AI": "计算机", "人工智能": "计算机", "软件": "计算机", "大模型": "计算机",
    "游戏": "传媒", "影视": "传媒", "传媒": "传媒",
    "纺织": "纺织服饰", "服装": "纺织服饰", "零售": "商贸零售", "电商": "商贸零售",
    "旅游": "社会服务", "酒店": "社会服务", "免税": "社会服务",
    "航空": "交通运输", "航运": "交通运输", "快递": "交通运输", "物流": "交通运输",
    "铁路": "交通运输", "燃气": "公用事业", "水电": "公用事业",
}


def _log(m: str):
    print(f"[info_panel {dt.datetime.now():%H:%M:%S}] {m}", flush=True)


# ---------------------------------------------------------------- 计划股票
def latest_frozen() -> dict:
    fs = sorted(PLAN.glob("frozen_*.json"))
    if not fs:
        return {"trading_date": None, "candidates": [], "source": None}
    d = json.loads(fs[-1].read_text(encoding="utf-8"))
    cands = []
    for s in d.get("sessions") or []:
        for c in s.get("candidates") or []:
            cands.append({"code": str(c["code"]).zfill(6), "name": c.get("name") or "",
                          "rules": c.get("rules") or {}, "session_tag": s.get("session_tag")})
    return {"trading_date": d.get("trading_date"), "candidates": cands,
            "source": fs[-1].name, "exported_at": d.get("exported_at")}


def stock_quote(code: str, day: str) -> dict:
    q = {"code": code, "price": None, "pct_chg": None, "time": None, "status": "OK"}
    try:
        import akshare as ak
        prefix = "sh" if code.startswith("6") else "sz"
        df = ak.stock_zh_a_minute(symbol=f"{prefix}{code}", period="1", adjust="")
        df["dt"] = df["day"].astype(str)
        today = df[df["dt"].str[:10] == f"{day[:4]}-{day[4:6]}-{day[6:]}"]
        if today.empty:
            q["status"] = "NO_DATA_TODAY"
            return q
        last = today.iloc[-1]
        q["price"] = float(last["close"])
        q["time"] = str(last["dt"])[11:16]
        try:
            from mquote import load_pre_close
            pc = load_pre_close(code, day)
            if pc:
                q["pct_chg"] = round((q["price"] / float(pc) - 1) * 100, 2)
            else:
                q["status"] = "NO_PRE_CLOSE"
        except Exception:
            q["status"] = "NO_PRE_CLOSE_MODULE"
    except Exception as e:
        q["status"] = f"ERROR:{type(e).__name__}"
    return q


# ---------------------------------------------------------------- 板块
def sector_map(day: str) -> dict:
    if not PIT.exists():
        return {}
    import pandas as pd
    m = pd.read_parquet(PIT)
    d = int(day)
    m = m[(m["effective_from"].astype(int) <= d) &
          ((m["effective_to"].isna()) | (m["effective_to"].astype(int) > d))]
    m = m.sort_values("effective_from").drop_duplicates("symbol", keep="last")
    return {str(r["symbol"]).zfill(6): {"code_l1": str(r["index_code_l1"]),
                                        "name_l1": str(r["industry_name_l1"])}
            for _, r in m.iterrows()}


def sw_realtime() -> dict:
    try:
        import akshare as ak
        df = ak.index_realtime_sw(symbol="一级行业")
        out = {}
        for _, r in df.iterrows():
            pct = (float(r["最新价"]) / float(r["昨收盘"]) - 1) * 100 \
                if float(r["昨收盘"]) else None
            out[str(r["指数代码"])] = {"name": r["指数名称"], "pct_chg": round(pct, 2),
                                     "price": float(r["最新价"])}
        return out
    except Exception as e:
        _log(f"申万实时 FAIL: {type(e).__name__}: {str(e)[:60]}")
        return {}


# ---------------------------------------------------------------- 外盘宏观
def us_index(symbol: str, name: str) -> dict:
    try:
        import akshare as ak
        df = ak.index_us_stock_sina(symbol=symbol)
        last, prev = df.iloc[-1], df.iloc[-2]
        return {"name": name, "close": float(last["close"]),
                "pct_chg": round((float(last["close"]) / float(prev["close"]) - 1) * 100, 2),
                "date": str(last["date"]), "status": "OK"}
    except Exception as e:
        return {"name": name, "status": f"ERROR:{type(e).__name__}"}


def commodity(symbol: str, name: str) -> dict:
    try:
        import akshare as ak
        df = ak.futures_foreign_hist(symbol=symbol)
        last, prev = df.iloc[-1], df.iloc[-2]
        return {"name": name, "close": float(last["close"]),
                "pct_chg": round((float(last["close"]) / float(prev["close"]) - 1) * 100, 2),
                "date": str(last["date"])[:10], "status": "OK"}
    except Exception as e:
        return {"name": name, "status": f"ERROR:{type(e).__name__}"}


# ---------------------------------------------------------------- 资讯层
def global_news() -> list:
    """新浪全球财经快讯（20 条）→ 宏观/行业打标。转述源，未经核实，如实标注。"""
    try:
        import akshare as ak
        df = ak.stock_info_global_sina()
    except Exception as e:
        _log(f"全球快讯 FAIL: {type(e).__name__}: {str(e)[:60]}")
        return []
    out = []
    for _, r in df.iterrows():
        content = str(r["内容"])
        macro = sorted({tag for kw, tag in MACRO_TAGS.items() if kw in content})
        sectors = sorted({tag for kw, tag in SECTOR_TAGS.items() if kw in content})
        out.append({"time": str(r["时间"]), "content": content[:300],
                    "macro_tags": macro, "sector_tags": sectors})
    return out


# ---------------------------------------------------------------- 公告 + 模型分
def recent_events(days_back: int = 3) -> list:
    out = []
    today = dt.date.today()
    for i in range(days_back + 1):
        d = (today - dt.timedelta(days=i)).strftime("%Y%m%d")
        p = C3 / "out" / "daily_events" / f"events_{d}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def pg_scores(cluster_ids: list) -> dict:
    if not cluster_ids:
        return {}
    try:
        import psycopg
        conn = psycopg.connect(
            "host=localhost port=5432 dbname=corvus user=corvus password=123456",
            connect_timeout=5)
        cur = conn.execute(
            "SELECT event_cluster_id, p_materiality, prediction_ts, materiality_model_version "
            "FROM corvus_c4.c4_shadow_prediction WHERE event_cluster_id = ANY(%s)",
            (list(cluster_ids),))
        res = {r[0]: {"p_materiality": r[1], "prediction_ts": str(r[2]),
                      "model_version": r[3]} for r in cur.fetchall()}
        conn.close()
        return res
    except Exception as e:
        _log(f"PG 读取 FAIL: {type(e).__name__}: {str(e)[:60]}")
        return {}


# ---------------------------------------------------------------- 主装配
def collect(day: str | None = None) -> dict:
    t0 = dt.datetime.now()
    day = day or dt.date.today().strftime("%Y%m%d")
    payload = {"snapshot_ts": t0.isoformat(timespec="seconds"), "day": day,
               "sections": {}, "gaps": [], "labels": {
                   "rule": "规则标注（冻结词表粗表，非训练模型输出）",
                   "model": "materiality_head_v1 正式分（仅 A股公告、T+1 收盘后 mature）",
                   "facts": "纯事实（无模型覆盖）",
                   "news": "快讯转述源（新浪全球财经快讯，未经核实，仅供阅读）"}}

    plan = latest_frozen()
    stocks = plan["candidates"]
    payload["plan"] = {"trading_date": plan["trading_date"], "source": plan["source"],
                       "n": len(stocks)}
    if not stocks:
        payload["gaps"].append("无冻结计划（frozen_*.json 缺失）")

    smap = sector_map(plan["trading_date"] or day)
    sw = sw_realtime()
    if not sw:
        payload["gaps"].append("申万一级实时行情缺失（板块只显示归属名）")

    events = recent_events()
    by_code: dict[str, list] = {}
    for ev in events:
        by_code.setdefault(str(ev["code"]).zfill(6), []).append(ev)
    scores = pg_scores([ev["event_cluster_id"] for ev in events])

    stock_rows = []
    for s in stocks:
        code = s["code"]
        q = stock_quote(code, day)
        sec = smap.get(code)
        sw_info = sw.get(sec["code_l1"]) if sec else None
        evs = []
        for ev in by_code.get(code, []):
            sc = scores.get(ev["event_cluster_id"])
            kw = ev.get("matched_keyword")
            evs.append({
                "announce_date": ev["announce_date"],
                "tier": ev["tier"], "category": ev["category"],
                "titles": [t[:40] for t in ev.get("titles") or []][:3],
                "matched_keyword": kw,
                "direction_hint": DIRECTION_TEXT.get(ev.get("direction_hint"), "中性/未知"),
                "category_prior": CATEGORY_PRIOR.get(ev["category"]),
                "p_materiality": (round(sc["p_materiality"], 4) if sc else None),
                "score_note": (f"评分于 {sc['prediction_ts'][:16]}（T+1 收盘正式分, "
                               f"v{sc['model_version']}）" if sc else
                               "待 T+1 收盘后评分（路线 A）"),
                "url": (ev.get("raw_urls") or [""])[0],
            })
        note = None
        if q.get("pct_chg") is not None and sw_info and sw_info.get("pct_chg") is not None:
            diff = round(q["pct_chg"] - sw_info["pct_chg"], 2)
            note = f"个股{'强于' if diff >= 0 else '弱于'}板块 {abs(diff)}pp（规则计算）"
        stock_rows.append({**s, "quote": q,
                           "sector": (sec or {}).get("name_l1"),
                           "sector_pct": (sw_info or {}).get("pct_chg"),
                           "events": evs, "rule_note": note})
    payload["sections"]["stocks"] = stock_rows

    seen = {}
    for s in stock_rows:
        if s["sector"]:
            seen.setdefault(s["sector"], {"name": s["sector"],
                                          "pct_chg": s["sector_pct"], "stocks": []})
            seen[s["sector"]]["stocks"].append(f"{s['code']} {s['name']}")
    payload["sections"]["sectors"] = sorted(
        seen.values(), key=lambda x: (x["pct_chg"] is None, x["pct_chg"]))

    news = global_news()
    plan_sectors = set(seen)
    payload["sections"]["news"] = {
        "macro": [n for n in news if n["macro_tags"]],
        "sector_related": [{"sector": sec, "items": [n for n in news if sec in n["sector_tags"]]}
                           for sec in sorted(plan_sectors)
                           if any(sec in n["sector_tags"] for n in news)],
        "source": "新浪全球财经快讯（机器转述，未经核实）",
    }
    if not news:
        payload["gaps"].append("快讯源不可用（资讯区为空）")

    payload["sections"]["macro"] = {
        "us": [us_index(".IXIC", "纳斯达克"), us_index(".INX", "标普500"),
               us_index(".DJI", "道琼斯")],
        "commodities": [commodity("XAU", "伦敦金"), commodity("XAG", "白银"),
                        commodity("CL", "WTI原油"), commodity("OIL", "布伦特原油")],
        "label": "纯事实（无模型覆盖）——缺口登记：外盘→A股影响模型未立项",
    }
    payload["elapsed_s"] = round((dt.datetime.now() - t0).total_seconds(), 1)
    return payload


def save_snapshot(payload: dict) -> Path:
    d = DATA / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def append_note(text: str, snapshot_ts: str) -> Path:
    d = DATA / "notes"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{dt.date.today():%Y%m%d}.md"
    with f.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {dt.datetime.now():%H:%M}（快照 {snapshot_ts}）\n{text}\n")
    return f
