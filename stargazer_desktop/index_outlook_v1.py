"""Numeric index maps rendered from same-cutoff evidence, never model prose."""


from datetime import datetime
from math import isfinite

NAMES={'sse':'沪指','szse':'深证成指','cyb':'创业板'}
OVERVIEW_INDICES = (('sse', '沪指'), ('szse', '深证成指'), ('cyb', '创业板指'))


def nearby_levels(name, short, price):
    nearby = dict(upper=None, lower=None, following=None)
    if isinstance(price, (int, float)) and isfinite(price) and price > 0:
        below = [node for node in reversed(short) if node <= price]
        if below:
            nearby['lower'] = below[0]
            target_gap = 20 if name == 'sse' else price * (20 / 3639)
            candidates = [node for node in below[1:] if target_gap * .5 <= below[0] - node <= target_gap * 1.75]
            nearby['following'] = min(candidates, key=lambda node: abs(below[0] - node - target_gap), default=None)
        nearby['upper'] = next((node for node in short if node > price), None)
    return nearby


def overview_cards(facts, cutoff, *, overlay=None, overlay_cutoff=None, fixed=None):
    cards = []
    overlay = overlay or {}
    for name, label in OVERVIEW_INDICES:
        def value(key):
            fact = facts.get(key) or {}
            if fact.get('status') == 'KNOWN' and fact.get('asof') == cutoff:
                return fact.get('value')
            return None
        display_quote = facts.get('overview.quote.'+name)
        quote = display_quote or facts.get('market.'+name) or {}
        quote_ok = bool(display_quote) or (quote.get('status') == 'KNOWN' and quote.get('evidence_kind') == 'QUOTE_SNAPSHOT'
                    and quote.get('bar_cutoff') == cutoff)
        evidence = value(f'index.{name}.map_input') or {}
        mapping = value(f'index.{name}.rebound_map') or {}
        card_cutoff = cutoff
        prepared = overlay.get(name) or {}
        if (prepared.get('status') == 'PRICE_FACTS_ONLY' and prepared.get('asof') == overlay_cutoff
                and (not evidence or not cutoff or overlay_cutoff > cutoff)):
            evidence, mapping, card_cutoff = prepared, {}, overlay_cutoff
        close = evidence.get('last_close') or mapping.get('close') or value('market.'+name+'.price')
        price = quote.get('price') if quote_ok else close
        levels = sorted({float(v['price']) for v in evidence.get('levels', [])
                         if isinstance(v.get('price'), (int, float)) and v['price'] > 0})
        above = [v for v in levels if close is not None and v > close]
        upper = above[0] if above else None
        critical = evidence.get('critical_position') or {}
        lower = critical.get('lower')
        following = critical.get('next_lower')
        reference = critical.get('reference_close')
        return_value = close / reference - 1 if close and reference else None
        if following is None and lower is not None:
            following = max((v for v in levels if v < lower), default=None)
        short = (prepared.get('short_wave_levels') if prepared.get('status') == 'PRICE_FACTS_ONLY'
                 and prepared.get('asof') == card_cutoff else evidence.get('short_wave_levels')) or []
        nearby = nearby_levels(name, short, price)
        closes = evidence.get('last_two_closes') or []
        times = evidence.get('last_two_times') or []
        adjacent = False
        if type(card_cutoff) is int and len(times) == 2 and times[-1] == card_cutoff:
            a, b = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in times]
            adjacent = (b-a).total_seconds() == 300
        danger = bool(lower is not None and adjacent and len(closes) == 2 and
                      all(c < lower for c in closes) and return_value is not None and return_value < -.02)
        missing_reason = None
        if upper is None and lower is None and overlay:
            if name not in overlay:
                missing_reason = '缺少本地五分钟 K 线'
            elif prepared.get('reason') == 'CURRENT_PREFIX_GAP_OR_CUTOFF_MISMATCH':
                missing_reason = '五分钟 K 线尚未补齐至当前截止'
        cards.append(dict(name=name, label=label, price=price, price_time=quote.get('asof') if quote_ok else card_cutoff if close else None,
                          map_time=card_cutoff if upper is not None or lower is not None or any(nearby.values()) else None,
                          map_note=missing_reason, nearby=nearby,
                          low=(quote.get('day_low') if quote_ok and isinstance(quote.get('day_low'), (int,float))
                               and 0 < quote['day_low'] <= price else None) or
                              evidence.get('session_low', mapping.get('session_low')), upper=upper,
                          lower=lower, following=following, danger=danger))
        if fixed is not None:
            saved = fixed.get(name) or {}
            cards[-1].update(nearby=saved.get('nearby') or nearby_levels(name, [], None),
                             map_time=saved.get('asof'), map_note='等待盘后黄金位' if not saved else None)
            if saved:
                quote_time = cards[-1]['price_time']
                if price is None:
                    cards[-1].update(price=saved.get('close'), price_time=saved.get('asof'),
                                     low=saved.get('session_low'))
                cards[-1].update(lower=saved.get('risk_support'),
                    danger=bool(quote_time and saved.get('risk_day') == int(quote_time) // 1000000
                                and saved.get('risk_active')))
    return cards


def facts_for(name,evidence,cutoff):
    mapping=evidence.get('outlook') or {}
    if mapping.get('status')!='PRICE_MAP_ONLY' or mapping.get('asof')!=cutoff:
        return {}
    label=NAMES.get(name,name);key='index.'+name+'.rebound_map'
    facts={key:dict(asof=cutoff,status='KNOWN',value=mapping,
        text=f"{label}窗口高点{mapping['high_day']} {mapping['high']:.2f}→其后低点{mapping['low_day']} {mapping['low']:.2f}；当前{mapping['close']:.2f}；仅价格地图，非双底或量能确认")}
    for i,level in enumerate(mapping['levels']):
        facts[f'index.{name}.rebound.{i}']=dict(asof=cutoff,status='KNOWN',value=level,
            text=f"{label}自低点向上{level['ratio']:.1%}反弹位{level['price']:.2f}，距当前{level['distance']:+.2%}；几何参照，非上涨目标承诺")
    return facts


def render(facts):
    lines=[]
    for name,label in (*NAMES.items(), ('star50', '科创50')):
        fact=facts.get('index.'+name+'.rebound_map') or {}
        m=fact.get('value') or {}
        if fact.get('status')!='KNOWN' or m.get('status')!='PRICE_MAP_ONLY':continue
        if m.get('asof')!=fact.get('asof'):continue
        lines += [f"{label}｜{'收盘' if m['session_complete'] else '完整K线截止价'} {m['close']:.2f}（较{m['reference_day']}收盘{m['return_from_previous_close']:+.2%}）",
            f"当日低点{m['session_low']:.2f}；下影{m['lower_shadow']:.2f}点；自日低回升{m['rebounce_from_session_low']:.2%}；距窗口高点{m['drawdown_from_high']:.2%}",
            f"黄金地图：{m['high_day']}高{m['high']:.2f}→{m['low_day']}低{m['low']:.2f}（最近{m['window_sessions']}个有效交易日窗口，按截止重建）",
            '反弹位：'+'；'.join(f"{v['ratio']:.1%} {v['price']:.2f}（距当前{v['distance']:+.2%}）" for v in m['levels'][:5])]
        lower = facts.get(f'index.{name}.band.lower_watch') or {}
        if lower.get('status') == 'KNOWN' and lower.get('asof') == fact.get('asof'):
            band = lower['value']
            lines.append(f"下方黄金观察带：{band['lower']:.2f}—{band['upper']:.2f}")
        above=[v for v in m['levels'] if v['above_close']]
        if above:
            first=above[0];next_level=above[1] if len(above)>1 else None
            text=f"后续关注：两根完整五分钟收在{first['price']:.2f}上方，再核对修复"
            if next_level:text+=f"；进一步参照{next_level['price']:.2f}"
            lines.append(text+f"；未收复则仍观察低点{m['low']:.2f}与该反弹位之间的运行，跌破低点须重评地图。")
        else:lines.append(f"后续关注：已越过当前地图高点，重新核对波段；低点{m['low']:.2f}仍只作历史参照。")
        lines += [f"低点状态：{'当日新形成，后续守位尚未验证' if m['low_is_current_session'] else '其后已观察'+str(m['low_following_sessions'])+'个交易日，仍不能据此确认双底'}。",
                  '量能：当前指数源仅价格，放量／缩量未核验，不判断资金进场；量价联合确认尚不成立。']
        if m.get('excluded_incomplete_days'):lines.append('地图剔除了不完整历史日，覆盖有限；缺日不填补。')
    if not lines:return ''
    return '指数价格与黄金地图（代码事实；非Pyxis幅度预测）\n'+'\n\n'.join(lines)+'\n\n次日验证：比较各指数守低与逐档收复是否同步；单边反弹不代表全市场修复。上述条件不自动改变既有闸门或个股买入规则。'
