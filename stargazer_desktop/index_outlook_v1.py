"""Numeric index maps rendered from same-cutoff evidence, never model prose."""


NAMES={'sse':'沪指','cyb':'创业板'}


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
    for name,label in NAMES.items():
        fact=facts.get('index.'+name+'.rebound_map') or {}
        m=fact.get('value') or {}
        if fact.get('status')!='KNOWN' or m.get('status')!='PRICE_MAP_ONLY':continue
        if m.get('asof')!=fact.get('asof'):continue
        lines += [f"{label}｜{'收盘' if m['session_complete'] else '完整K线截止价'} {m['close']:.2f}（较{m['reference_day']}收盘{m['return_from_previous_close']:+.2%}）",
            f"当日低点{m['session_low']:.2f}；下影{m['lower_shadow']:.2f}点；自日低回升{m['rebounce_from_session_low']:.2%}；距窗口高点{m['drawdown_from_high']:.2%}",
            f"黄金地图：{m['high_day']}高{m['high']:.2f}→{m['low_day']}低{m['low']:.2f}（最近{m['window_sessions']}个有效交易日窗口，按截止重建）",
            '反弹位：'+'；'.join(f"{v['ratio']:.1%} {v['price']:.2f}（距当前{v['distance']:+.2%}）" for v in m['levels'][:5])]
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
