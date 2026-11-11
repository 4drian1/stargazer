"""Three-layer index map: near watch zone / near resistance / big-picture.

Same-cutoff facts only. The near watch zone reads the REAL resonance_bands
field (node/lower/upper per band) — never stitched from min/max of unrelated
nodes. Above/below classification compares each node's price with the
same-cutoff close (the execution map has no above_close field). Same-day and
prior highs/lows are price facts, never re-labeled golden. Overlaps merge:
a node inside a band is the band's inner node, not an extra support. The
big-picture retracement map is background only. No invented integer levels.
"""

NAMES = {'sse': '沪指', 'cyb': '创业板'}


def _near_band(evidence, close):
    """Resonance bands by their REAL relation to the same-cutoff close: the
    band the close sits in comes first, then the nearest band below. Bands
    are never dropped just because the close is inside them."""
    bands = evidence.get('resonance_bands') or []
    current = [b for b in bands if float(b['lower']) <= close <= float(b['upper'])]
    if current:
        return current[0], True
    below = [b for b in bands if float(b['upper']) < close]
    if below:
        return max(below, key=lambda b: float(b['upper'])), False
    return None, False


def build_layers(name, evidence, cutoff):
    label = NAMES.get(name, name)
    if evidence.get('status') not in ('PRICE_FACTS_ONLY', 'PRICE_MAP_ONLY'):
        return None
    if evidence.get('asof') != cutoff:
        return None
    close = float(evidence['last_close'])
    band, band_is_current = _near_band(evidence, close)
    bands = evidence.get('resonance_bands') or []
    node = (evidence.get('critical_position') or {}).get('lower')
    node_in_band = bool(band and node and
                        float(band['lower']) <= node <= float(band['upper']))
    node_below = bool(node and node <= close)  # 高于现价的节点不写成"下方观察"
    session_low = evidence.get('session_low')
    rebound = evidence.get('outlook') or {}
    prior_low = rebound.get('low') if rebound.get('status') == 'PRICE_MAP_ONLY' else None
    prior_low_day = rebound.get('low_day')
    above_levels = [float(v['price']) for v in (evidence.get('levels') or [])
                    if float(v['price']) > close]
    above_bands = [float(b['upper']) for b in bands if float(b['lower']) > close]
    near_above = sorted(set(above_levels + above_bands))
    remote = [v for v in (rebound.get('levels') or [])] if rebound.get('status') == 'PRICE_MAP_ONLY' else []
    return dict(label=label, close=close, band=band, band_is_current=band_is_current,
                node=node, node_in_band=node_in_band, node_below=node_below,
                session_low=session_low, prior_low=prior_low, prior_low_day=prior_low_day,
                above_near=near_above, remote=remote)


def render_layers(layers_by_name):
    blocks = []
    for name in NAMES:
        L = layers_by_name.get(name)
        if not L:
            continue
        f = lambda v: f"{v:.2f}"
        lines = [f"{L['label']}收{f(L['close'])}"]
        if L['band']:
            pos = '当前所在带' if L.get('band_is_current') else '最近下方带'
            text = (f"下方观察：{f(float(L['band']['lower']))}—{f(float(L['band']['upper']))}"
                    f"试运行黄金带（{pos}；{L['band'].get('source_waves', '?')}波源/"
                    f"{L['band'].get('peak_groups', '?')}峰组）")
            if L['node_in_band']:
                text += f"，带内节点{f(L['node'])}"
            elif L.get('node_below') and L['node']:
                text += f"；节点{f(L['node'])}"
            if L['session_low'] is not None:
                text += f"；当日低点{f(L['session_low'])}（价格事实）"
            lines.append(text + '；几何参照，尚非确认支撑。')
        elif L.get('node_below') and L['node']:
            lines.append(f"下方观察：节点{f(L['node'])}；共振带未检出，不拼造。")
        else:
            lines.append("下方观察：本截止无真实执行地图节点，不补造整数位。")
        if L['prior_low']:
            day = f"（{L['prior_low_day']}）" if L['prior_low_day'] else ''
            lines.append(f"更下方：前低{f(L['prior_low'])}{day}（价格事实，非黄金位）。")
        if L['above_near']:
            lines.append("上方：" + '；'.join(f"近处压力{f(v)}" for v in L['above_near'][:3])
                         + "（执行地图节点，未收复前不作支撑）。")
        else:
            lines.append("上方：近处无真实压力节点。")
        if L['remote']:
            refs = '；'.join(f"{v['ratio']:.1%}反弹参照{f(v['price'])}（距当前{v['distance']:+.2%}）"
                            for v in L['remote'][:3])
            lines.append(f"大波段背景（远端参照，不覆盖近处地图）：{refs}；非上涨目标承诺。")
        blocks.append('\n'.join(lines))
    if not blocks:
        return ''
    return '指数三层地图（代码事实；近处=执行地图与价格事实，远端=整段跌幅反弹参照）\n' \
        + '\n\n'.join(blocks) \
        + '\n\n触位立即提醒；守位与修复分别标注实时试算与收盘确认（两根完整五分钟）。'


def render(facts):
    by_name = {}
    for name in NAMES:
        evidence = facts.get('index.' + name + '.map_input')
        if not evidence or evidence.get('status') != 'KNOWN':
            continue
        layers = build_layers(name, evidence.get('value') or {}, evidence.get('asof'))
        if layers:
            by_name[name] = layers
    return render_layers(by_name)
