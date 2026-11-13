"""服务层：把数据层+策略层粘成 agent 可调用的函数。所有函数返回可JSON序列化的dict。"""
from contextlib import closing
import json
import os
from pathlib import Path
import time

import pandas as pd

from tools import market, news, financials, research, lhb
from strategy import funnel, scorer, exits, oversold

try:
    import paths
    BASE = paths.APP_DIR  # 可写目录（打包后=exe旁，watchlist 存这里）
except ImportError:
    BASE = os.path.dirname(os.path.abspath(__file__))


# ---------- 观察名单持久化 ----------
def wl_path(cfg):
    p = cfg["agent"].get("watchlist_file", "data/watchlist.json")
    if os.path.isabs(p):
        return p
    try:
        import paths
        return paths.data_file(p)  # 打包模式下首启自动从包内复制种子数据
    except ImportError:
        return os.path.join(BASE, p)


def wl_load(cfg):
    p = wl_path(cfg)
    d = {"positions": [], "watch": []}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            try:
                d.update(json.load(f) or {})
            except json.JSONDecodeError:
                pass
    return d


def wl_save(cfg, wl):
    os.makedirs(os.path.dirname(wl_path(cfg)), exist_ok=True)
    with open(wl_path(cfg), "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=1)
    # W2b：持仓/观察名单变化 → 只对新增代码做增量准备（幂等，sha 去重）。
    try:
        import postclose_prepared_v1
        postclose_prepared_v1.on_watchlist_saved(cfg)
    except Exception:
        pass  # 增量准备不得阻断名单保存本身


_boards_cache = {"ts": 0.0, "data": {}}


def _boards():
    """板块清单缓存1小时，避免每次扫描都拉999个板块。"""
    if not _boards_cache["data"] or time.time() - _boards_cache["ts"] > 3600:
        d = market.board_list()
        if d:
            _boards_cache.update(ts=time.time(), data=d)
    return _boards_cache["data"]


def _find_board(name_or_code):
    boards = _boards()
    if name_or_code in boards:  # 精确名称
        return boards[name_or_code], name_or_code
    if name_or_code in boards.values():  # 板块代码
        return name_or_code, [k for k, v in boards.items() if v == name_or_code][0]
    hits = {n: c for n, c in boards.items() if name_or_code in n}
    return hits, None


def _resolve_board(board_name):
    """返回 (bk, 名称, None) 或 (None, None, 错误dict)。"""
    found, bname = _find_board(board_name)
    if isinstance(found, dict):
        if not found:
            import difflib
            close = difflib.get_close_matches(board_name, list(_boards()), n=8, cutoff=0.25)
            return None, None, {"error": f"东财没有叫「{board_name}」的板块",
                                "相近板块": {n: _boards()[n] for n in close},
                                "提示": "请用相近板块的准确名称或BK代码重试一次"}
        return None, None, {"error": "板块名不唯一", "candidates": found}
    return found, bname, None


# ---------- 板块扫描（四层漏斗前三层+吸筹+v2.3因子） ----------
def scan_sector_rules(board_name, top=5):
    from sector_rule_scan import scan_sector_rules as scan
    return scan(board_name, top=top)


def scan_sector(board_name, max_stocks=80, with_score=False, year=2025, include_st=False):
    bk, bname, err = _resolve_board(board_name)
    if err:
        return err
    cons = market.board_constituents(bk, exclude_st=not include_st)
    pool = {c: v for c, v in cons.items() if funnel.layer2_pass(v)}
    note = ""
    if len(pool) > max_stocks:
        pool = dict(sorted(pool.items(), key=lambda kv: kv[1]["fmc"])[:max_stocks])
        note = f"成分过多, 截取流通市值最小{max_stocks}只"
    passed, failed = [], []
    for i, (code, info) in enumerate(pool.items()):
        df = market.kline(code)
        if df is None or len(df) < 260:
            continue
        m = funnel.indicators(df, info["fmc"], info["price"])
        if m is None:
            continue
        ok, fails = funnel.funnel_pass(m)
        rec = {"code": code, "name": info["name"], "price": info["price"],
               "fmc": round(info["fmc"], 1), "pe": info["pe"], **m}
        if ok:
            passed.append(rec)
        else:
            failed.append({"code": code, "name": info["name"], "fails": fails,
                           "vr": m["vr"], "pos": m["pos"]})
        if (i + 1) % 20 == 0:
            time.sleep(0.3)
    # v2.4.3: 通过名单按流通市值升序（市值口径放宽到100亿后，优先看小票）
    passed.sort(key=lambda r: r["fmc"])
    # v2.4.3 双篮子: A篮=烂票胚子(PE<0或>60或未知, 基本面烂才有重组/炒作空间, 走打分);
    # B篮=基本面正常(PE 0~60)但形态全过的"好公司横盘", 供人工复筛, 不自动打分
    basket_a = [r for r in passed if funnel.send_to_scoring(r)]
    basket_b = [r for r in passed if not funnel.send_to_scoring(r)]
    result = {"board": bname, "bk": bk, "pool": len(pool), "note": note, "含ST": include_st,
              "passed": passed, "basket_a": basket_a, "basket_b": basket_b,
              "failed_sample": failed[:30], "failed_count": len(failed)}
    if with_score and basket_a:
        result["scoring"] = score_stocks([r["code"] for r in basket_a], year=year)
    return result


# ---------- 垃圾度打分 ----------
def _pe_map(codes):
    """批量取PE TTM: ulist接口, f9需/100"""
    secids = ",".join(("1." if c.startswith("6") else "0.") + c for c in codes)
    d = market._get_json("https://push2delay.eastmoney.com/api/qt/ulist.np/get?"
                         f"secids={secids}&fields=f12,f9")
    out = {}
    try:
        for it in d["data"]["diff"]:
            out[it["f12"]] = None if it["f9"] == "-" else round(it["f9"] / 100, 2)
    except Exception:
        pass
    return out


def score_stocks(codes, year=2025):
    codes = codes[:20]
    pes = _pe_map(codes)
    rows = {}
    for code in codes:
        ind = financials.annual_indicators(code, year)
        if ind is None:
            continue
        ind["PE"] = pes.get(code)
        rows[code] = ind
        time.sleep(0.2)
    if not rows:
        return {"error": "财务数据全部获取失败"}
    df = pd.DataFrame(rows).T
    scored = scorer.score_pool(df)
    return {"year": year, "table": scored.reset_index(names="code").to_dict("records"),
            "用法": "总分越低基本面越差越是胚子; <40优选, 40-60结合标签形态, >60放弃"}


# ---------- 个股体检 ----------
def model_stock_analysis(code):
    """Read bound model results on demand; never disguise missing inference."""
    import re
    import assistant_workflow
    import plan_evidence
    if not isinstance(code, str) or not re.fullmatch(r'(?:00|30|60|68)\d{4}', code):
        raise ValueError('模型查询需要六位A股代码')
    try:
        snapshot = assistant_workflow.snapshot()
    except (OSError, ValueError):
        return {'code': code, 'status': 'UNAVAILABLE', 'reason': '尚无可绑定的当前选股快照；未调用训练或外部取数'}
    base = {'code': code, 'data_date': snapshot.get('data_date'),
            'mode': 'BOUND_MODEL_RESULT_LOOKUP', 'computed_now': False,
            'note': '读取已验收模型结果，不是本次重新推理；幅度非概率，未知不等于安全；无自动交易权限。'}
    if not any(r['code'] == code for r in snapshot.get('rows', [])):
        return {**base, 'status': 'NOT_COVERED', 'selection': {'status': 'NOT_IN_CURRENT_CANDIDATES'},
                'risk': {'status': 'MISSING'}, 'requires_feature_rebuild': True,
                'reason': '不在当前已绑定模型候选；须另建同规格特征后才可真实评分，本次没有偷偷启动取数或重建。'}
    evidence = plan_evidence.for_snapshot(snapshot)[code]
    return {**base, 'status': 'AVAILABLE' if evidence['risk']['status'] == 'AVAILABLE' else 'PARTIAL',
            'selection': evidence['selection'], 'risk': plan_evidence.prompt_risk(evidence['risk']),
            'risk_observations': plan_evidence.risk_lines(evidence['risk'])}


def model_stock_inference(code, model='both'):
    """Use bound evidence, or compute one frozen selection head without publishing a plan."""
    import subprocess
    from pathlib import Path
    import assistant_workflow
    import paths

    if model not in ('tarot', 'pyxis', 'both'):
        raise ValueError('模型范围无效')
    bound = model_stock_analysis(code)
    if bound['status'] in ('AVAILABLE', 'PARTIAL'):
        return bound
    if model == 'pyxis':
        return {**bound, 'inference_status': 'PYXIS_CONTEXT_NOT_BOUND',
                'inference_reason': '池外Pyxis需要同截止指数、板块及全板块个股分钟上下文；当前缓存不足以按单股同规格推理'}
    try:
        snapshot = assistant_workflow.snapshot()
        p, e = int(snapshot['data_date']), int(snapshot['monitor_date'])
    except (OSError, ValueError, KeyError):
        return {**bound, 'inference_status': 'NO_BOUND_DECISION_DATE'}
    app = Path(paths.APP_DIR).resolve()
    project = app.parent.parent if app.name == 'dist' else app.parent
    python = Path(os.environ.get('STARGAZER_PYTHON') or project / '_dev_venv/Scripts/python.exe')
    worker = Path(paths.resource('on_demand_model_worker.py'))
    if not python.is_file() or not worker.is_file():
        return {**bound, 'inference_status': 'LOCAL_WORKER_MISSING'}
    try:
        completed = subprocess.run([str(python), '-B', '-X', 'utf8', str(worker)],
            input=json.dumps({'code': code, 'P': p, 'E': e}), capture_output=True,
            text=True, encoding='utf-8', timeout=45,
            env={**os.environ, 'STARGAZER_PROJECT_ROOT': str(project)},
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        result = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {**bound, 'inference_status': 'LOCAL_WORKER_UNAVAILABLE'}
    if completed.returncode or result.get('status') == 'UNAVAILABLE':
        return {**bound, 'inference_status': 'LOCAL_WORKER_REFUSED',
                'inference_reason': result.get('reason', 'WORKER_EXIT')}
    selected = result['selection'].get('quality_state') == 'OK'
    return {**bound, 'bound_status': bound['status'],
            'status': 'PARTIAL_ON_DEMAND' if selected else bound['status'],
            'on_demand': result,
            'inference_status': 'SELECTION_ONLY' if selected else 'SELECTION_NOT_COMPUTABLE'}


def confirmed_plan():
    """Read the saved plan without creating a draft or changing reminders."""
    import trade_plan_v1
    plan = trade_plan_v1.load_plan()
    if not plan:
        return {'status': 'NO_CONFIRMED_PLAN'}
    facts = {}
    for code, item in (plan.get('model_evidence') or {}).items():
        selection, risk = item.get('selection') or {}, item.get('risk') or {}
        context = risk.get('context') or {}
        facts[code] = {'selection_score': selection.get('personal_score'),
                       'selection_ref': selection.get('ref'),
                       'risk_status': risk.get('status'), 'risk_asof': risk.get('asof_ts'),
                       'amp_232': context.get('amp_232'), 'amp_266': context.get('amp_266'),
                       'risk_horizon': context.get('horizon')}
    return {**{key: plan.get(key) for key in ('created_ts', 'data_date', 'monitor_date',
            'stocks', 'boards', 'manual_observations', 'assistant_plan_id')},
            'model_facts': facts}


def portfolio_positions(cfg):
    """Local recorded holdings; market prices require a separate live check."""
    return {'positions': active_positions(cfg), 'price_status': 'NOT_QUERIED'}


def stock_check(code):
    q = market.quotes([code]).get(code)
    df = market.kline(code)
    if df is None:
        return {"error": "K线获取失败"}
    info = {"fmc": None, "price": q["price"] if q else df["close"].iloc[-1]}
    m = funnel.indicators(df, None, None)
    ok, fails = funnel.funnel_pass(m)
    os_ok, os_detail = oversold.oversold_check(df)
    out = {"quote": q, "技术指标": m, "漏斗判定": {"pass": ok, "fails": fails},
           "超卖判定": {"trigger": os_ok, **os_detail},
            "参考止损": {"结构止损": m["low250"], "吊灯草案": round(m["price"] and df["close"].iloc[-20:].max() - 3 * m["atr22"], 2),
                         "ATR初始止损(策略二)": round(m["price"] - 2 * m["atr14"], 2)}}
    # 散户/小单资金流（新浪口径，观察因子，不进门槛）：横盘期散户持续流出=筹码向主力集中
    try:
        flow = market.retail_flow(code)
        if flow:
            d5 = flow[:5]
            out["散户资金流(新浪口径)"] = {
                "近5日": d5,
                "散户5日净额万": round(sum(x["retail_net_wan"] for x in d5), 1),
                "主力5日净额万": round(sum(x["main_net_wan"] for x in d5), 1),
            }
    except Exception:
        pass
    return out


# ---------- 持仓与观察名单检查 ----------
def check_portfolio(cfg):
    wl = wl_load(cfg)
    report = {"positions": [], "watch_alerts": []}
    for p in active_positions(cfg, wl["positions"]):
        df = market.kline(p["code"])
        if df is None:
            continue
        ignited = False
        if p.get("sector_bk"):
            cons = market.board_constituents(p["sector_bk"])
            if cons:
                pcts = [v["pct"] for v in cons.values()]
                import numpy as np
                arr = np.array(pcts, dtype=float)
                ignited = bool(arr.mean() >= 2.5 and (arr >= 5).mean() >= 0.2)
        r = exits.check_position(p, df, sector_ignited=ignited)
        r.update({"code": p["code"], "name": p.get("name"), "logic": p.get("logic"),
                  "sector_ignited": ignited})
        report["positions"].append(r)
    if wl["watch"]:
        codes = [w["code"] for w in wl["watch"]]
        qs = market.quotes(codes)
        for w in wl["watch"]:
            q = qs.get(w["code"])
            if not q:
                continue
            price = float(q.get("price") or 0)
            target = w.get("target_buy")
            stop = w.get("stop_loss")
            alert = {"code": w["code"], "name": q["name"], "price": price,
                     "pct": q["pct"], "reason": w.get("reason", ""),
                     "target_buy": target, "stop_loss": stop,
                     "strategy": w.get("strategy"), "source_date": w.get("source_date")}
            if target and price <= float(target):
                alert["type"] = "BUY_TRIGGER"
                alert["msg"] = f"价格 {price:.2f} 已触及目标买入价 {float(target):.2f}，等待人工确认"
                report["watch_alerts"].append(alert)
            elif stop and price <= float(stop):
                alert["type"] = "STOP_LOSS_BROKEN"
                alert["msg"] = f"价格 {price:.2f} 已跌破止损价 {float(stop):.2f}，建议放弃观察"
                report["watch_alerts"].append(alert)
            elif q["pct"] <= -5:
                df = market.kline(w["code"])
                trig, detail = oversold.oversold_check(df)
                alert["type"] = "OVERSOLD_DROP"
                alert["oversold_trigger"] = trig
                alert["detail"] = detail
                report["watch_alerts"].append(alert)
    return report


# ---------- 大盘概览（指数+情绪） ----------
def market_overview():
    idx = market.index_quotes()
    up = market.limit_pool("up")
    dn = market.limit_pool("down")
    total_amt = sum(i["amount_yi"] for i in idx if i["code"] in ("000001", "399001"))
    max_lbc = max((p["lbc"] for p in up["pool"]), default=0)
    broken = sum(1 for p in up["pool"] if p["zbc"] > 0)
    # 大小盘风格劈叉：沪深300(大盘) vs 国证2000(小盘)
    style = None
    pcts = {i["code"]: i["pct"] for i in idx}
    hs300, gz2000 = pcts.get("000300"), pcts.get("399303")
    if hs300 is not None and gz2000 is not None:
        diff = round(gz2000 - hs300, 2)
        if diff >= 1:
            tag = "小盘强（利好胚子策略，低位小票有资金）"
        elif diff <= -1:
            tag = "大盘护盘（小票失血，胚子策略谨慎）"
        else:
            tag = "大小盘均衡"
        style = {"沪深300": hs300, "国证2000": gz2000, "差值": diff, "判读": tag}
    return {"指数(名称,点位,涨跌%,成交额亿)": idx,
            "大小盘风格": style,
            "两市成交额(亿,上证+深成)": round(total_amt, 1),
            "情绪": {"涨停家数": up["total"], "跌停家数": dn["total"], "最高连板": max_lbc,
                     "炸板率%": round(broken / max(up["total"], 1) * 100, 1)},
            "涨停明细(按连板降序前20)": sorted(up["pool"], key=lambda p: -p["lbc"])[:20],
            "判读口径": "涨停多+炸板率低+连板高度打开=情绪好敢上仓位; 涨停少+炸板率高=情绪差多看少动; "
                       "成交额看增量资金; 指数大小盘劈叉(沪深300跌/国证2000涨)=结构行情做小票"}


# ---------- 板块解剖（广度+龙头） ----------
def sector_detail(board_name, top=10):
    bk, bname, err = _resolve_board(board_name)
    if err:
        return err
    cons = market.board_constituents(bk)
    if not cons:
        return {"error": "成分股拉取失败"}
    import numpy as np
    a = np.array([v["pct"] for v in cons.values()], dtype=float)
    leaders = sorted(cons.items(), key=lambda kv: -kv[1]["pct"])[:top]
    laggards = sorted(cons.items(), key=lambda kv: kv[1]["pct"])[:5]
    pack = lambda c, v: {"code": c, "name": v["name"], "pct": v["pct"], "price": v["price"],
                         "amount_yi": v.get("amount_yi", 0), "fmc": round(v["fmc"], 1)}
    return {"board": bname, "bk": bk, "成分数": len(cons),
            "上涨家数": int((a > 0).sum()), "下跌家数": int((a < 0).sum()),
            "平均涨幅%": round(float(a.mean()), 2),
            "涨超5%占比%": round(float((a >= 5).mean()) * 100, 1),
            "涨停家数(≥9.8%)": int((a >= 9.8).sum()),
            "点火判定(均涨>=2.5且涨5%+占比>=20%)": bool(a.mean() >= 2.5 and (a >= 5).mean() >= 0.2),
            "领涨龙头(前%d)" % top: [pack(c, v) for c, v in leaders],
            "领跌(前5)": [pack(c, v) for c, v in laggards],
            "用法": "龙头涨停/大涨+梯队跟随(广度够)=点火有效; 只有一两个龙头独涨、梯队不跟=假点火别追; "
                   "想细看某个龙头的位置/形态/止损位，拿代码再调 stock_check"}


# ---------- 板块资金流 ----------
def market_flow(top=12):
    inflow, outflow = market.board_fund_flow(top)
    return {"主力净流入榜(名称,涨幅%,净流入亿,净占比%)": inflow,
            "主力净流出榜": outflow,
            "用法": "看钱从哪来到哪去; 流入榜找低位新板块, 流出榜确认旧主线退潮"}


# ---------- 隔夜美股 ----------
def us_market():
    rows = market.us_quotes()
    return {"隔夜美股(名称,现价,涨跌幅%,行情时间)": [
                {"name": r["name"], "price": r["price"], "pct": r["pct"], "time": r["time"]}
                for r in rows],
            "判读口径": "看与A股题材的映射: 纳指/费半强→算力/半导体有援军; 金龙指数强→中概/港股映射; "
                       "英伟达台积电是AI链风向标。手册口径: 美股第二夜熄火=逻辑止损信号之一。"}


# ---------- 深度搜索（Tavily） ----------
def research_search(cfg, query, max_results=8, days=None):
    key = (cfg.get("tavily") or {}).get("api_key", "").strip()
    if not key:
        return {"error": "未配置 tavily.api_key（在推送设置里填）"}
    r = research.search(key, query, max_results=max_results, days=days)
    if r is None:
        return {"error": "tavily 请求失败（网络或key问题）"}
    return r


# ---------- 龙虎榜 ----------
# 常见知名游资席位关键词（主观名单，仅辅助识别，会随市场变迁过时）
HOT_SEAT_KEYS = {
    "章盟主": ["杭州庆春路", "章盟主"],
    "炒股养家": ["华鑫证券上海分公司", "上海宛平南路", "炒股养家"],
    "方新侠": ["兴业证券陕西分公司", "方新侠"],
    "作手新一": ["国泰君安南京太平南路", "作手新一"],
    "陈小群": ["东亚前海证券上海分公司", "陈小群"],
    "呼家楼": ["中信证券北京呼家楼", "呼家楼"],
    "佛山系": ["佛山", "光大证券佛山绿景路"],
    "量化打板": ["量化", "华泰证券总部"],
    "上塘路": ["财通证券杭州上塘路", "上塘路"],
    "桑田路": ["国盛证券宁波桑田路", "桑田路"],
}


def _tag_seats(seats):
    """给席位打标签：机构/北向/知名游资/普通营业部。返回 (标注后席位list, 命中的游资名list)"""
    tagged, hot_hits = [], set()
    for s in seats:
        name = s.get("seat") or ""
        tags = []
        if "机构专用" in name:
            tags.append("机构")
        if "股通专用" in name or "香港中央结算" in name:
            tags.append("北向")
        for who, keys in HOT_SEAT_KEYS.items():
            if any(k in name for k in keys):
                tags.append(f"知名游资:{who}")
                hot_hits.add(who)
                break
        if not tags:
            tags.append("营业部")
        tagged.append({**s, "tag": "/".join(tags)})
    return tagged, sorted(hot_hits)


def _latest_lhb_date(max_back=7):
    """从昨天往前找最近一个有龙虎榜数据的交易日（当天数据傍晚才出，盘中查当天必为空）。"""
    import datetime as dt
    d = dt.date.today()
    if dt.datetime.now().hour < 18:  # 傍晚前当天数据未发布，直接从昨天找
        d -= dt.timedelta(days=1)
    for _ in range(max_back):
        if d.weekday() < 5:
            rows = lhb.lhb_daily(d.isoformat(), page_size=1)
            if rows:
                return d.isoformat(), rows
        d -= dt.timedelta(days=1)
    return None, None


def lhb_today(top=20):
    """最近一个交易日的龙虎榜总览。"""
    date, probe = _latest_lhb_date()
    if not date:
        return {"error": "近7天未取到龙虎榜数据（接口异常）"}
    rows = lhb.lhb_daily(date)
    if rows is None:
        return {"error": "龙虎榜接口请求失败"}
    net_in = [r for r in rows if (r.get("net_amt") or 0) > 0]
    net_out = sorted([r for r in rows if (r.get("net_amt") or 0) < 0], key=lambda r: r["net_amt"])
    def pack(r):
        return {"code": r["code"], "name": r["name"], "chg_pct": r["chg_pct"],
                "net_yi": round((r["net_amt"] or 0) / 1e8, 2),
                "explain": r.get("explain")}
    return {"数据日期": date, "上榜家数": len(rows),
            "净买入前%d" % top: [pack(r) for r in net_in[:top]],
            "净卖出前10": [pack(r) for r in net_out[:10]],
            "判读口径": "龙虎榜看主力在哪打架: 净买额大+涨幅大=主力进攻方向; 净卖额大=出货方向。"
                       "想看某只是谁买的(机构/游资/北向)，拿代码调 lhb_stock 看席位。"
                       "注意数据是T日傍晚发布，盘中查到的是上一交易日。"}


def lhb_stock(code, date=None):
    """某股龙虎榜席位明细（谁买谁卖）。date 缺省=最近有数据的交易日。"""
    if not date:
        date, _ = _latest_lhb_date()
        if not date:
            return {"error": "近7天未取到龙虎榜数据（接口异常）"}
    seats = lhb.lhb_stock_seats(code, date)
    if seats is None:
        return {"error": "席位接口请求失败"}
    if not seats:
        return {"error": f"{code} 在 {date} 未上龙虎榜", "提示": "未上榜说明当日异动程度未达披露标准，属正常情况，不代表没资金关注"}
    tagged, hot_hits = _tag_seats(seats)
    buyers = sorted(tagged, key=lambda s: -(s.get("buy") or 0))[:5]
    sellers = sorted(tagged, key=lambda s: -(s.get("sell") or 0))[:5]
    def pack(s):
        return {"seat": s["seat"], "tag": s["tag"],
                "buy_wan": round((s.get("buy") or 0) / 1e4, 1),
                "sell_wan": round((s.get("sell") or 0) / 1e4, 1),
                "net_wan": round((s.get("net") or 0) / 1e4, 1)}
    inst_buy = sum(s.get("buy") or 0 for s in tagged if "机构" in s["tag"])
    north_buy = sum(s.get("buy") or 0 for s in tagged if "北向" in s["tag"])
    return {"code": code, "数据日期": date,
            "上榜原因": seats[0].get("explanation"),
            "买入席位前5": [pack(s) for s in buyers],
            "卖出席位前5": [pack(s) for s in sellers],
            "结构": {"机构买入合计_万": round(inst_buy / 1e4, 1),
                     "北向买入合计_万": round(north_buy / 1e4, 1),
                     "命中知名游资": hot_hits or "无"},
            "判读口径": "买方一线游资多+卖方散户席位=接力意愿强; 买方全是东财拉萨(散户营)=情绪尾声;"
                       "机构大买=中线背书但不是连板风格; 一家独大(买一占买入>40%)=次日容易被砸。"
                       "知名游资标注基于常见席位名单，仅供参考。"}


# ---------- 新闻 ----------
def news_digest(pages=2, keywords=None):
    items = news.flash(pages=pages)
    out = {"count": len(items), "latest": [{"time": n["time"], "content": n["content"][:300]} for n in items[:40]]}
    if keywords:
        cnt, ratio, hits = news.sector_news_density(items, keywords)
        out["密度"] = {"命中": cnt, "占比": ratio, "样本": [h["content"][:120] for h in hits[:8]],
                       "判读": "占比>=80%或绝对条数刷屏=共识已满只卖不买"}
    return out


# ---------- 观察名单管理 ----------
def wl_add_watch(cfg, code, reason="", target_buy=None, stop_loss=None, strategy=None, source_date=None):
    wl = wl_load(cfg)
    if any(w["code"] == code for w in wl["watch"]):
        return {"ok": False, "msg": "已在观察名单"}
    import datetime as dt
    item = {"code": code, "added": dt.date.today().isoformat(), "reason": reason}
    if target_buy is not None:
        item["target_buy"] = float(target_buy)
    if stop_loss is not None:
        item["stop_loss"] = float(stop_loss)
    if strategy:
        item["strategy"] = strategy
    if source_date:
        item["source_date"] = source_date
    wl["watch"].append(item)
    wl_save(cfg, wl)
    return {"ok": True, "watch_n": len(wl["watch"])}


def _norm_date(s):
    """把 2026-8-5 / 2026/8/5 之类宽松写法规范成 2026-08-05，保证和K线日期字符串可比较。"""
    return pd.to_datetime(str(s).strip()).strftime("%Y-%m-%d")


def wl_add_position(cfg, code, name, entry, entry_date, low250, logic="", catalyst_date=None, sector_bk=None):
    wl = wl_load(cfg)
    wl["positions"] = [p for p in wl["positions"] if p["code"] != code]
    wl["positions"].append({"code": code, "name": name, "entry": float(entry),
                            "entry_date": _norm_date(entry_date),
                            "low250": float(low250), "logic": logic,
                            "catalyst_date": _norm_date(catalyst_date) if catalyst_date else None,
                            "sector_bk": sector_bk})
    # Bind execution terms only to a manually recorded fill of the new confirmed plan.
    import trade_plan_v1
    plan = trade_plan_v1.load_plan() or {}
    if str(plan.get('monitor_date', '')).replace('-', '') == _norm_date(entry_date).replace('-', ''):
        stock = next((s for s in plan.get('stocks', []) if s['code'] == code), {})
        rule = stock.get('entry_rule') or {}
        if rule.get('version') == 'TAROT_A_SECOND_CASH_V2':
            distance = max(.03, min(.06, 1.5*float(rule['atr_ref'])/float(entry)))
            wl['positions'][-1]['tarot_execution'] = dict(version=rule['version'], stop_price=float(entry)*(1-distance),
                horizon_sessions=3, plan_id=plan.get('assistant_plan_id'), atr_ref=rule['atr_ref'])
    wl_save(cfg, wl)
    return {"ok": True, "positions_n": len(wl["positions"])}


def wl_remove(cfg, code):
    wl = wl_load(cfg)
    wl["watch"] = [w for w in wl["watch"] if w["code"] != code]
    wl["positions"] = [p for p in wl["positions"] if p["code"] != code]
    wl_save(cfg, wl)
    return {"ok": True}


def wl_remove_watch(cfg, code):
    """只移除观察名单条目；持仓、计划与成交账本一律不动。

    ``wl_remove`` 是聊天 /del 命令用的组合删除（同时删 watch 与 positions），
    不能复用为「删除观察」：同一只票可能同时出现在持仓/正式计划里，删观察
    绝不能连带抹掉持仓。本接口显式返回 untouched 计数，便于前端与验收核对。
    """
    code = str(code or "").strip()
    if not code:
        return {"ok": False, "error": "code 必填", "removed": 0}
    wl = wl_load(cfg)
    watch = list(wl.get("watch") or [])
    kept = [w for w in watch if str(w.get("code")) != code]
    removed = len(watch) - len(kept)
    if not removed:
        return {"ok": False, "error": "观察名单中未找到该代码", "removed": 0}
    wl["watch"] = kept
    wl_save(cfg, wl)
    return {"ok": True, "removed": removed, "watch_remaining": len(kept),
            "positions_untouched": len(wl.get("positions") or []),
            "code": code}


def wl_update_position(cfg, code, **fields):
    """编辑已有持仓的字段（名称/成本/买入日/止损/逻辑），找不到返回错误。"""
    wl = wl_load(cfg)
    for p in wl["positions"]:
        if p["code"] == code:
            if "name" in fields and fields["name"] is not None:
                p["name"] = str(fields["name"])
            if "entry" in fields and fields["entry"] not in (None, ""):
                p["entry"] = float(fields["entry"])
            if "entry_date" in fields and fields["entry_date"]:
                p["entry_date"] = _norm_date(fields["entry_date"])
            if "low250" in fields and fields["low250"] not in (None, ""):
                p["low250"] = float(fields["low250"])
            if "logic" in fields and fields["logic"] is not None:
                p["logic"] = str(fields["logic"])
            wl_save(cfg, wl)
            return {"ok": True, "position": p}
    return {"ok": False, "msg": "持仓中未找到该代码"}


def wl_update_watch(cfg, code, **fields):
    """编辑已有观察条目的目标价/止损/策略/理由。"""
    wl = wl_load(cfg)
    for w in wl["watch"]:
        if w["code"] == code:
            if "target_buy" in fields and fields["target_buy"] not in (None, ""):
                w["target_buy"] = float(fields["target_buy"])
            if "stop_loss" in fields and fields["stop_loss"] not in (None, ""):
                w["stop_loss"] = float(fields["stop_loss"])
            if "strategy" in fields and fields["strategy"] is not None:
                w["strategy"] = str(fields["strategy"])
            if "reason" in fields and fields["reason"] is not None:
                w["reason"] = str(fields["reason"])
            wl_save(cfg, wl)
            return {"ok": True, "watch": w}
    return {"ok": False, "msg": "观察名单中未找到该代码"}


# ---------- 交易记录数据库（供日后回测验证） ----------
import sqlite3 as _sqlite3


def trade_db_path(cfg):
    p = (cfg.get("agent") or {}).get("trades_db", "data/trades.db")
    if os.path.isabs(p):
        return p
    try:
        import paths
        return paths.data_file(p)
    except ImportError:
        return os.path.join(BASE, p)


def _trade_conn(cfg):
    p = trade_db_path(cfg)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    conn = _sqlite3.connect(p)
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,          -- 记录时间（ISO）
        trade_date TEXT,  -- 实际成交日期 YYYY-MM-DD
        action TEXT,      -- buy / sell
        code TEXT,        -- 6位代码
        name TEXT,        -- 股票名称
        price REAL,       -- 成交价
        shares INTEGER,   -- 股数（可选）
        amount REAL,      -- 金额 = price*shares（可选）
        reason TEXT,      -- 买入/卖出逻辑
        source TEXT,      -- chat / slash / manual
        raw TEXT         -- 原始用户消息（可追溯）
    )""")
    try:
        conn.execute("ALTER TABLE trades ADD COLUMN request_id TEXT")
    except _sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE trades ADD COLUMN plan_ref TEXT")
    except _sqlite3.OperationalError:
        pass
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_request_id "
                 "ON trades(request_id) WHERE request_id IS NOT NULL")
    return conn


def trade_add(cfg, **kw):
    """记录一条真实成交。字段宽松解析，忽略多余 key，避免 LLM 传参波动导致崩溃。"""
    action = (kw.get("action") or "").lower()
    if action not in ("buy", "sell"):
        return {"ok": False, "error": "action 必须是 buy 或 sell"}
    code = str(kw.get("code") or "").strip()
    name = str(kw.get("name") or "").strip()
    price = kw.get("price")
    if not code or not name or price in (None, ""):
        return {"ok": False, "error": "code/name/price 必填"}
    try:
        price = float(price)
    except (TypeError, ValueError):
        return {"ok": False, "error": "price 必须是数字"}
    shares = kw.get("shares")
    if shares not in (None, ""):
        try:
            shares = int(float(shares))
        except (TypeError, ValueError):
            shares = None
    import datetime as _dt
    trade_date = kw.get("trade_date") or _dt.date.today().isoformat()
    try:
        trade_date = _norm_date(trade_date)
    except Exception:
        trade_date = _dt.date.today().isoformat()
    amount = round(price * shares, 2) if shares else None
    conn = _trade_conn(cfg)
    request_id = str(kw.get("request_id") or "").strip() or None
    plan_ref = str(kw.get("plan_ref") or "").strip() or None
    if request_id:
        prior = conn.execute(
            "SELECT id, action, code, name, price, trade_date, plan_ref FROM trades WHERE request_id=?",
            (request_id,)).fetchone()
        if prior:
            conn.close()
            return {"ok": True, "duplicate": True, "id": prior[0], "action": prior[1],
                    "code": prior[2], "name": prior[3], "price": prior[4],
                    "trade_date": prior[5], "request_id": request_id,
                    "plan_ref": prior[6]}
    cursor = conn.execute(
        "INSERT INTO trades (ts, trade_date, action, code, name, price, shares, amount, reason, source, raw, request_id, plan_ref) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (_dt.datetime.now().isoformat(timespec="seconds"), trade_date, action, code, name,
         price, shares, amount, str(kw.get("reason") or ""), str(kw.get("source") or "chat"),
         str(kw.get("raw") or ""), request_id, plan_ref))
    trade_id = cursor.lastrowid
    conn.commit(); conn.close()
    result = {"ok": True, "duplicate": False, "id": trade_id, "action": action,
              "code": code, "name": name, "price": price,
              "trade_date": trade_date, "request_id": request_id,
              "plan_ref": plan_ref}
    try:
        from intraday_monitor import record_trade_context
        record_trade_context(cfg, trade_id, result)
    except Exception:
        pass  # 成交主记录优先；模型上下文缺失不能阻断人工成交日志
    return result


def trade_list(cfg, limit=50):
    conn = _trade_conn(cfg)
    rows = conn.execute(
        "SELECT id, ts, trade_date, action, code, name, price, shares, reason, source "
        "FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    cols = ["id", "ts", "trade_date", "action", "code", "name", "price", "shares", "reason", "source"]
    return [dict(zip(cols, r)) for r in rows]


def trade_export_csv(cfg, out_path=None, limit=200000):
    """导出为 CSV（utf-8-sig，Excel 中文不乱码），供 stargazer_ml 回测直接读取。"""
    rows = trade_list(cfg, limit=limit)
    if out_path is None:
        out_path = trade_db_path(cfg).replace(".db", "_export.csv")
    import csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["trade_date", "action", "code", "name", "price", "shares", "amount", "reason", "source"])
        for r in rows:
            w.writerow([r["trade_date"], r["action"], r["code"], r["name"], r["price"],
                        r["shares"] or "", round(r["price"] * (r["shares"] or 0), 2) if r["shares"] else "",
                        r["reason"] or "", r["source"] or ""])
    return out_path


def active_positions(cfg, positions=None):
    """Keep legacy manually registered positions until actual trades prove closure."""
    positions = list(wl_load(cfg).get('positions', []) if positions is None else positions)
    codes = sorted({str(row['code']) for row in positions})
    if not codes:
        return positions
    path = Path(trade_db_path(cfg)).resolve()
    if not path.is_file():
        return positions
    query = ('SELECT code, action, price, shares, trade_date FROM trades WHERE code IN ('
             + ','.join('?' for _ in codes) + ') ORDER BY trade_date,id')
    with closing(_sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        trades = db.execute(query, codes).fetchall()
    by_code = {code: [] for code in codes}
    for code, *row in trades:
        by_code[code].append(row)
    return [row for row in positions
            if _position_from_trades(by_code[str(row['code'])])['position_state'] != 'CLOSED']


def position_details(cfg, code, live_price=None):
    """真实合同：从已登记真实成交计算数量、加权成本、可卖（T+1）与盈亏。

    成本口径（部分卖出修复）：卖出按当时移动加权平均成本结转成本，而不是用
    卖出成交价冲减总成本。否则部分卖出会把剩余持仓的单位成本按卖价压低/抬高，
    剩余持仓的 pnl 与成本随即失真。卖出价与结转成本之差记入 ``realized_pnl``。

    余额语义：
      * 无任何成交记录 → quantity=None（未知，不用 0 冒充），state='NO_TRADES'
      * 有成交但已清空  → quantity=0（真实已清仓），state='CLOSED'
      * 仍有持仓         → quantity>0，state='OPEN'
    行情缺失 → 盈亏 None。
    """
    conn = _trade_conn(cfg)
    rows = conn.execute(
        "SELECT action, price, shares, trade_date FROM trades WHERE code=? ORDER BY trade_date, id",
        (str(code),)).fetchall()
    conn.close()
    return _position_from_trades(rows, live_price)


def _position_from_trades(rows, live_price=None):
    import datetime as _dt
    if not any(int(shares or 0) > 0 for _, _, shares, _ in rows):
        return dict(quantity=None, cost=None, sellable=None, pnl=None, pnl_pct=None,
                    position_state='NO_TRADES', realized_pnl=None)
    qty = 0
    cost_sum = 0.0
    realized = 0.0
    oversold = 0
    for action, price, shares, _ in rows:
        shares = int(shares or 0)
        if shares <= 0:
            continue
        px = float(price or 0)
        if action == "buy":
            qty += shares
            cost_sum += px * shares
        else:
            avg = cost_sum / qty if qty > 0 else None
            sold = min(shares, qty)
            if avg is not None and sold:
                # 结转平均成本，不用卖出成交价冲减成本池
                cost_sum -= avg * sold
                realized += (px - avg) * sold
            qty -= sold
            oversold += shares - sold  # 超卖不伪造负持仓，仅留痕
    if qty <= 0:
        return dict(quantity=0, cost=None, sellable=0, pnl=None, pnl_pct=None,
                    position_state='CLOSED', realized_pnl=round(realized, 2),
                    oversold_shares=oversold)
    cost = cost_sum / qty
    today = _dt.date.today().isoformat()
    sellable = sum((int(s or 0)) for a, _, s, d in rows if a == "buy" and str(d or "") < today)
    sellable = max(0, min(qty, sellable - sum(int(s or 0) for a, _, s, _ in rows if a == "sell")))
    pnl = pnl_pct = None
    if live_price is not None:
        try:
            pnl = round((float(live_price) - cost) * qty, 2)
            pnl_pct = round((float(live_price) / cost - 1) * 100, 2) if cost > 0 else None
        except (TypeError, ValueError, ZeroDivisionError):
            pnl = pnl_pct = None
    return dict(quantity=qty, cost=round(cost, 3), sellable=sellable, pnl=pnl, pnl_pct=pnl_pct,
                position_state='OPEN', realized_pnl=round(realized, 2),
                oversold_shares=oversold)
