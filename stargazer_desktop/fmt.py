"""斜杠命令输出的中文化格式化：把原始数据整理成直接能看懂的文字。"""
import math


def _num(v, nd=2, dash="—"):
    """NaN/None 显示为 —，其余保留 nd 位小数。"""
    if v is None:
        return dash
    try:
        f = float(v)
        if math.isnan(f):
            return dash
        return f"{f:.{nd}f}"
    except (TypeError, ValueError):
        return dash


def _pct(v, nd=1):
    s = _num(v, nd)
    return s if s == "—" else ("+" + s if float(s) > 0 else s) + "%"


# ---------- /check 持仓+观察名单 ----------
_ACTION_ICON = {"清仓": "🔴", "减半": "🟠", "换股": "🟡", "持有": "🟢"}


def check(report):
    lines = ["持仓风控检查", ""]
    pos = report.get("positions", [])
    lines.append(f"■ 持仓 {len(pos)} 只")
    if not pos:
        lines.append("  （空。买入后对我说“我X元买了XXXXXX，帮我登记”即可纳入风控）")
    for r in pos:
        act = r.get("action", "持有")
        icon = _ACTION_ICON.get(act, "⚪")
        lines.append(f"  {icon} {r.get('name') or ''} {r['code']} — 建议：{act}")
        lines.append(f"     浮盈 {_pct(r.get('gain'))}｜已持有 {r.get('held_days', '?')} 个交易日｜曾最高浮盈 {_pct(r.get('max_close_gain'))}")
        lines.append(f"     MA5 {_num(r.get('ma5'))}｜MA10 {_num(r.get('ma10'))}｜吊灯线 {_num(r.get('chandelier'))}")
        sigs = r.get("signals") or []
        if sigs:
            lines.append("     信号：")
            for s in sigs:
                lines.append(f"       · {s}")
        else:
            lines.append("     信号：无，纪律内运行")
        lines.append("")
    alerts = report.get("watch_alerts", [])
    lines.append("■ 观察名单")
    if alerts:
        for w in alerts:
            lines.append(f"  ⚠ {w.get('name') or ''} {w['code']} 今日 {_pct(w.get('pct'))}")
            lines.append(f"     超卖判定：{'触发（进入伏击区）' if w.get('oversold_trigger') else '未触发'}")
            d = w.get("detail") or {}
            if d:
                j = d.get("判定") or {}
                lines.append(f"     单日 {_pct(d.get('pct1'), 2)}｜5日 {_pct(d.get('pct5'), 2)}｜"
                             f"RSI14 {_num(d.get('rsi14'), 1)}｜BIAS20 {_num(d.get('bias20'), 1)}")
                struct = "、".join(k for k in ("长下影", "阳包阴", "缩量十字星") if d.get(k)) or "无"
                lines.append(f"     止跌结构：{struct}｜幅度 {'✓' if j.get('幅度') else '✗'} "
                             f"超卖 {'✓' if j.get('超卖') else '✗'} 止跌 {'✓' if j.get('止跌') else '✗'}（三项齐才算触发）")
    else:
        lines.append("  无异常（单日跌超 5% 才会列在这里）")
    return "\n".join(lines)


# ---------- /quote 个股体检 ----------
def quote(code, r):
    if r.get("error"):
        return f"体检失败：{r['error']}"
    q = r.get("quote") or {}
    m = r.get("技术指标") or {}
    fu = r.get("漏斗判定") or {}
    os_ = r.get("超卖判定") or {}
    name = q.get("name", "")
    lines = [f"个股体检：{name} {code}"]
    if q:
        lines.append(f"现价 {_num(q.get('price'))}（{_pct(q.get('pct'), 2)}）")
    lines.append("")
    ok = fu.get("pass")
    lines.append(f"■ 四层漏斗：{'✅ 通过' if ok else '❌ 未通过'}")
    if not ok and fu.get("fails"):
        lines.append(f"  卡在：{'、'.join(fu['fails'])}")
    lines.append(f"  位置 {_num(m.get('pos'),1)}%（≤25 过）｜离低点 {_pct(m.get('al'))}（≤30 过）｜"
                 f"20日振幅 {_num(m.get('aa'))}%（≤2.8 过）")
    consol_txt = f"横盘 {m.get('consol_days', 0)} 天（≥7 过）"
    if m.get("consol_disputed"):
        consol_txt += f" ⚠其中 {m['consol_disputed']} 天只满足单条件，需人工复核"
    lines.append(f"  量比 {_num(m.get('vr'))}（≤1.05 过）｜20日涨幅 {_pct(m.get('d20'))}（≤15 过）｜{consol_txt}")
    if m.get("distrib_risk"):
        lines.append(f"  🟥 出货形态警示：近60日拉升 {_pct(m.get('runup60'))} 后高位横盘（位置 {_num(m.get('pos'), 1)}%、"
                     f"近7日振幅 {_num(m.get('aa7'))}%），警惕二波A杀，这不是胚子是出货")
    stars = m.get("accum_stars", 0)
    lines.append(f"  吸筹信号：{'★' * stars}{'☆' * (4 - stars)}"
                 f"（涨跌量比 {_num(m.get('ud_vol_ratio'))}｜OBV背离 {'有' if m.get('obv_diverge') else '无'}｜"
                 f"长下影 {m.get('lower_shadow_cnt', 0)} 次｜CMF {_num(m.get('cmf20'), 3)}）")
    if m.get("turnover_pct") is not None:
        lines.append(f"  换手率：20日均 {_num(m.get('turnover20'))}%，处250日 {_num(m.get('turnover_pct'), 0)}% 分位")
    lines.append("")
    lines.append(f"■ 超卖判定：{'⚠ 触发' if os_.get('trigger') else '未触发'}")
    if os_.get("detail"):
        lines.append(f"  {os_['detail']}")
    sl = r.get("参考止损") or {}
    lines.append("")
    lines.append("■ 参考止损位（三档）")
    lines.append(f"  结构止损（250日低点）：{_num(sl.get('结构止损'))}  ← 登记持仓默认用这条")
    lines.append(f"  吊灯草案：{_num(sl.get('吊灯草案'))}")
    lines.append(f"  ATR初始止损（策略二）：{_num(sl.get('ATR初始止损(策略二)'))}")
    rf = r.get("散户资金流(新浪口径)")
    if rf:
        lines.append("")
        lines.append("■ 散户资金流（新浪口径，观察因子，不进门槛）")
        for x in rf.get("近5日", []):
            lines.append(f"  {x['date']}｜涨跌 {_pct(x['chg_pct'], 2)}｜散户净额 {_num(x['retail_net_wan'], 1)} 万｜主力净额 {_num(x['main_net_wan'], 1)} 万")
        lines.append(f"  5日合计：散户 {_num(rf.get('散户5日净额万'), 1)} 万｜主力 {_num(rf.get('主力5日净额万'), 1)} 万")
        lines.append("  判读：横盘期散户持续净流出+主力温和净流入=筹码集中（吸筹佐证）；散户大幅净流入=筹码派发，警惕。"
                     "庄股、极度冷门票该口径会失真，仅作参考。")
    return "\n".join(lines)


# ---------- /scan 板块扫描 ----------
def scan(data):
    if data.get("error"):
        lines = [f"扫描失败：{data['error']}"]
        if data.get("相近板块"):
            lines.append("你是不是想找：")
            for name, bk in data["相近板块"].items():
                lines.append(f"  · {name}（{bk}）")
        if data.get("candidates"):
            lines.append("匹配的板块有多个，请说具体点：")
            for name, bk in data["candidates"].items():
                lines.append(f"  · {name}（{bk}）")
        return "\n".join(lines)
    st_note = "含ST" if data.get("含ST") else "已排除ST"
    lines = [f"板块扫描：{data['board']}（{data['bk']}）｜入池 {data['pool']} 只（{st_note}）"]
    if data.get("note"):
        lines.append(f"（{data['note']}）")
    passed = data.get("passed", [])
    basket_a = data.get("basket_a")
    basket_b = data.get("basket_b")

    def _row(i, r):
        stars = "★" * r.get("accum_stars", 0) + "☆" * (4 - r.get("accum_stars", 0))
        lines.append(f"  {i}. {r['name']} {r['code']}｜现价 {_num(r.get('price'))}｜流通市值 {_num(r.get('fmc'), 1)} 亿｜吸筹 {stars}")
        consol_txt = f"横盘 {r.get('consol_days', 0)} 天"
        if r.get("consol_disputed"):
            consol_txt += f"（⚠{r['consol_disputed']} 天仅单条件，需人工复核）"
        lines.append(f"     位置 {_num(r.get('pos'), 1)}%｜量比 {_num(r.get('vr'))}｜20日涨幅 {_pct(r.get('d20'))}｜{consol_txt}｜250日低点 {_num(r.get('low250'))}")

    lines.append("")
    lines.append(f"■ 通过四层漏斗 {len(passed)} 只（按流通市值升序）")
    if basket_a is None:  # 兼容旧结构
        for i, r in enumerate(passed, 1):
            _row(i, r)
        if not passed:
            lines.append("  （无。板块内没有符合低位+缩量+未启动形态的票）")
    else:
        lines.append(f"▎A篮·烂票胚子 {len(basket_a)} 只（PE亏损/>60/未知，可送打分）")
        for i, r in enumerate(basket_a, 1):
            _row(i, r)
        if not basket_a:
            lines.append("  （无）")
        lines.append(f"▎B篮·好公司横盘 {len(basket_b)} 只（基本面正常，形态全过，供人工复筛）")
        for i, r in enumerate(basket_b, 1):
            _row(i, r)
        if not basket_b:
            lines.append("  （无）")
        if not passed:
            lines.append("  （板块内没有符合低位+缩量+未启动形态的票）")
    fc = data.get("failed_count", 0)
    if fc:
        # 剔除原因统计
        from collections import Counter
        cnt = Counter()
        for f in data.get("failed_sample", []):
            for x in f.get("fails", []):
                cnt[x] += 1
        lines.append("")
        lines.append(f"■ 被剔除 {fc} 只（样例前 {len(data.get('failed_sample', []))} 只的原因分布）")
        for reason, n in cnt.most_common():
            lines.append(f"  · {reason}：{n} 只")
    sc = data.get("scoring")
    if sc:
        lines.append("")
        if isinstance(sc, str):
            lines.append(f"■ 财务打分：{sc}")
        else:
            lines.append("■ 财务打分（送检标的，总分越低越是胚子）")
            lines.append(score(sc))
    return "\n".join(lines)


# ---------- /score 垃圾度打分 ----------
def score(data):
    if isinstance(data, dict) and data.get("error"):
        return f"打分失败：{data['error']}"
    rows = data.get("table", []) if isinstance(data, dict) else []
    lines = [f"垃圾度打分（{data.get('year', '')} 年报）｜总分 0-100，越低基本面越差越是胚子", ""]
    for r in rows:
        flags = r.get("旗标") or ""
        lines.append(f"  {r.get('code')}｜总分 {_num(r.get('总分'), 1)}"
                     + (f"｜{flags}" if flags else ""))
    lines.append("")
    lines.append("口径：<40 优选胚子｜40-60 结合标签形态｜>60 放弃")
    return "\n".join(lines)


# ---------- /flow 资金流 ----------
def flow(data):
    lines = ["板块主力资金流", "", "■ 净流入榜"]
    for row in data.get("主力净流入榜(名称,涨幅%,净流入亿,净占比%)", []):
        lines.append(f"  {row[0]}｜{_pct(row[1], 2)}｜净流入 {_num(row[2])} 亿｜占比 {_num(row[3])}%")
    lines.append("")
    lines.append("■ 净流出榜")
    for row in data.get("主力净流出榜", []):
        lines.append(f"  {row[0]}｜{_pct(row[1], 2)}｜净流出 {_num(abs(row[2]))} 亿｜占比 {_num(abs(row[3]))}%")
    lines.append("")
    lines.append("口径：流入榜找低位新主线，流出榜确认旧主线退潮")
    return "\n".join(lines)


# ---------- /news 快讯 ----------
def news(r):
    lines = [f"最新快讯（前 25 条，共拉取 {r.get('count', 0)} 条）", ""]
    for n in r.get("latest", [])[:25]:
        lines.append(f"[{n['time']}] {n['content'][:120]}")
    if "密度" in r:
        d = r["密度"]
        lines += ["", f"■ 关键词密度：{d['命中']} 条命中（占比 {d['占比']}）",
                  f"  判读：{d['判读']}"]
    return "\n".join(lines)
