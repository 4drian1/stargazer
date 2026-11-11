"""User's 2026-11-04 pullback observation policy. Pure facts, never orders.

Plan stocks keep their existing three-of-two rule. This policy has no fitted
parameters, no implicit capital allocation, and no use of later prices.
"""
from datetime import datetime
import math

VERSION='STRONG_PULLBACK_20261104_V1'
RATIOS=(.236,.382,.5,.618,.786)


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def fresh(quote,now,required=('price','prev')):
    try:
        t=datetime.strptime(str(quote['time']),'%Y%m%d%H%M%S')
        return (t.date()==now.date() and 0<=(now-t).total_seconds()<=120
                and all(finite(quote.get(k)) and quote[k]>0 for k in required))
    except (ValueError,KeyError,TypeError):return False


def market_state(quote,level,now,t0=None,calendar=()):
    result=dict(state='UNKNOWN',level=level,price=None,deviation=None,t0=t0,t_plus=None)
    if not finite(level) or level<=0 or not fresh(quote,now,required=('price',)):return result
    price=quote['price'];d=price/level-1
    state='RED' if price<level*.995-1e-10 else 'PROBE' if price<level else 'ABOVE'
    # Above a test level is a position fact, not a claim of a low-risk market.
    result.update(state=state,price=price,deviation=d,asof=quote['time'])
    day=int(now.strftime('%Y%m%d'));days=list(calendar)
    if t0 and int(str(t0)[:8]) in days and day in days:
        n=days.index(day)-days.index(int(str(t0)[:8]))
        if n>=0:result['t_plus']=n
    return result


def next_add_band(code,last_price,bands,entry_band):
    if not finite(last_price) or last_price<=0:return None
    gap=.06 if str(code).startswith(('30','68')) else .03
    choices=[b for b in bands if finite(b.get('lower')) and finite(b.get('upper'))
             and 0<b['lower']<=b['upper']<=last_price*(1-gap)+1e-10
             and b['upper']<entry_band['lower']]
    return max(choices,key=lambda b:b['upper']) if choices else None


def evaluate(row,quote,band,market,volume,board,now,position=None,daily_close=None):
    """Unknown != false. No actual position -> next add is a reference only.

    board is None unless an explicit same-day board approval is supplied.
    A stock's registered fills, never delivered notices, determine its cost.
    """
    result=dict(version=VERSION,state='WAIT',passed=[],waiting=[],blocked=[],
        market=market,trigger_price=quote.get('price'),quote_time=quote.get('time'),
        next_add=None,add_kind='REFERENCE_ONLY',research_target=.05,stop_fraction=.035,
        volume=volume,board=board,order=False)
    def gate(label,value):
        result['passed' if value is True else 'blocked' if value is False else 'waiting'].append(label)
    gate('行情有效',fresh(quote,now))
    gate('合格强势池',row.get('pool')=='MONITOR' and row.get('wave_return',0)>=.20-1e-12)
    gate('排除三顶',None if not row.get('structure') else row['structure']!='TRIPLE')
    gate('未发现已知ST或退市标记',row.get('risk_clear'))
    lo,hi=band.get('lower'),band.get('upper');p=quote.get('price')
    valid_band=all(finite(v) and v>0 for v in (lo,hi)) and lo<=hi
    gate('真实冻结黄金区',valid_band and row.get('map_known_before_open') is True)
    in_band=valid_band and finite(p) and lo<=p<=hi
    gate('现价仍在观察带',in_band)
    gate('大盘未低于测试位0.5%',None if market['state']=='UNKNOWN' else market['state']!='RED')
    pos=position or {};cost=pos.get('average_cost');last=pos.get('last_buy_price')
    entry=pos.get('entry_band') or band
    if not fresh(quote,now):return result
    if pos.get('quantity',0)>0:
        if not finite(cost) or cost<=0:
            result['waiting'].append('登记真实持仓均价');return result
        close_break=(now.hour>=15 and finite(daily_close) and finite(entry.get('lower')) and daily_close<entry['lower'])
        if pos.get('exit_latched') is True or p<=cost*.965+1e-10 or close_break:
            result.update(state='EXIT_TRIGGERED',exit_latched=True,add_kind='BLOCKED_BY_EXIT')
            result['waiting']=['核对逐笔T+1及可卖数量后执行退出'];return result
        if p>=cost*1.05-1e-10:
            result.update(state='PROFIT_5PCT',add_kind='BLOCKED_BY_EXIT')
            result['waiting']=['个人操作：可卖股份减半；研究：全部可卖股份按5%退出'];return result
    reference=last if finite(last) and last>0 else p
    if valid_band:result['next_add']=next_add_band(row['code'],reference,row.get('positions',[]),pos.get('last_band') or entry)
    if result['blocked']:return result
    if not fresh(quote,now):return result
    decline=1-p/quote['prev'];result['decline']=decline
    ratio=volume.get('ratio') if volume.get('status')=='KNOWN' else None
    standard=.045-1e-12<=decline<=.07+1e-12 and finite(ratio) and ratio<1
    if decline>.07+1e-12:
        gate('深跌时板块环境通过',board)
        tier='CAUTIOUS'
    elif standard:tier='STANDARD'
    elif .035-1e-12<=decline<=.07+1e-12:
        if ratio is None:result['waiting'].append('同时间段20日量比')
        tier='TRIAL'
    else:
        result['waiting'].append('相对昨收跌幅至少3.5%');return result
    # A standard stock in the index struggle zone remains a light probe.
    result['stock_tier']=tier
    result['state']=('WAIT' if result['waiting'] or result['blocked'] else
                     'CAUTIOUS' if tier=='CAUTIOUS' else
                     'PROBE' if market['state']=='PROBE' else tier)
    if pos.get('quantity',0)>0:
        gate('首笔为标准档',pos.get('first_tier')=='STANDARD')
        gate('补仓未满两次',pos.get('add_count',0)<2)
        gate('登记上一笔真实买价',finite(last) and last>0)
        gate('下一档满足3%或6%间距',result['next_add'] is not None)
        target=result['next_add']
        if target and target['lower']<=p<=target['upper'] and not result['waiting'] and not result['blocked']:
            result.update(state='ADD_CANDIDATE',add_kind='ACTUAL_POSITION_RECHECKED')
        else:result['add_kind']='WAIT_ACTUAL_POSITION_CONDITIONS'
    return result


def build_map(rows,asof):
    """Causal 60-session map; independent endpoint components counted once."""
    from strong_wave_facts_v1 import swings
    prefix=[r for r in rows if r['date']<=asof]
    if not prefix or prefix[-1]['date']!=asof:raise ValueError('PULLBACK_MAP_DATE')
    recent=prefix[-60:];sources=[];offset=len(prefix)-len(recent)
    for pivot in swings(prefix):
        i=pivot['index']-offset
        if i<0 or not pivot['low'] or not pivot['complete_window'] or i+3>=len(recent):continue
        j=max(range(i+1,len(recent)),key=lambda k:recent[k]['high'])
        low,high=recent[i]['low'],recent[j]['high']
        if high/low<1.2-1e-12 or len(recent)-1-j>30:continue
        wave=[recent[i]['date'],recent[j]['date']]
        for ratio in (*RATIOS,1.272,1.618,2.):
            price=high-(high-low)*ratio
            if price>0:sources.append(dict(price=price,ratio=ratio,wave=wave,
                confirmed_at=recent[i+3]['date'],high=high,low=low))
    groups=[]
    for source in sorted(sources,key=lambda s:s['price']):
        if not groups or source['price']/groups[-1][0]['price']-1>=.008:groups.append([])
        groups[-1].append(source)
    bands=[]
    for group in groups:
        components=[]
        for s in group:
            endpoints=set(s['wave']);overlap=[g for g in components if g&endpoints]
            for g in overlap:endpoints|=g;components.remove(g)
            components.append(endpoints)
        count=len(components)
        bands.append(dict(lower=group[0]['price'],upper=group[-1]['price'],sources=group,
            source_waves=count,primary=count>=3,independent_groups=count))
    return bands


def market_line(m):
    f=lambda x:'未知' if not finite(x) else f'{x:.4f}'.rstrip('0').rstrip('.')
    names=dict(RED='红灯：禁止新增买入',PROBE='黄闪：仅轻仓试探',
        ABOVE='在测试位上方；不等于低风险',UNKNOWN='测试位或实时报价待核实')
    age='T0尚未确认' if not m.get('t0') else ('T+'+str(m['t_plus']) if m.get('t_plus') is not None else 'T0已记录，交易日数待核实')
    deviation='未知' if not finite(m.get('deviation')) else f"{m['deviation']:+.2%}"
    return f"大盘：{names[m['state']]}｜测试位{f(m.get('level'))}｜偏离{deviation}｜{age}"


def render(result):
    m=result['market'];f=lambda x:'未知' if not finite(x) else f'{x:.4f}'.rstrip('0').rstrip('.')
    names=dict(STANDARD='标准候选',TRIAL='小仓试错候选',CAUTIOUS='谨慎试探候选',
        PROBE='大盘黄闪，仅轻仓试探候选',WAIT='观察等待',EXIT_TRIGGERED='退出已触发，禁止补仓',
        PROFIT_5PCT='达到5%退出研究线',ADD_CANDIDATE='补仓观察条件通过')
    lines=[f"规则：{names[result['state']]}｜触发价{f(result['trigger_price'])}",
        market_line(m)]
    if result['passed']:lines.append('满足：'+'、'.join(result['passed']))
    if result['blocked']:lines.append('未满足：'+'、'.join(result['blocked']))
    if result['waiting']:lines.append('等待：'+'、'.join(result['waiting']))
    b=result['next_add']
    lines.append('下一补仓观察：'+(f"{f(b['lower'])}—{f(b['upper'])}（须核对真实成交、间距和退出条件）" if b else '暂无合格补仓位'))
    lines.append('规则观察，未通过完整风险与AI流程，不生成正式买入建议。')
    return lines
