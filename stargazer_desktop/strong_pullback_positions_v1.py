"""Explicit actual fills and separate add-level notices. Never infer a fill."""
from datetime import datetime
import copy
import hashlib
import json
import threading

from strong_pullback_policy_v1 import finite, fresh, next_add_band, evaluate, render

LOCK=threading.RLock()


def load(base):
    path=base/'strong_pullback/positions.json'
    if not path.exists():return dict(fills=[],positions={},notices={})
    if path.stat().st_size>2*1024**2:raise ValueError('PULLBACK_FILL_LEDGER_LIMIT')
    return json.loads(path.read_bytes())


def save(base,value):
    from strong_sector_watch_v1 import save as bounded
    if len(json.dumps(value).encode())>2*1024**2:raise ValueError('PULLBACK_FILL_LEDGER_LIMIT')
    bounded(base/'strong_pullback/positions.json',value)


def apply_fill(old,fill,calendar,now,origin=None):
    """Pure, idempotent actual-fill accounting; invalid orders are not invented.

    The ledger records a real fill even if it violated an entry condition.
    Such a fill is not a new mechanical permission to average down.
    """
    value=copy.deepcopy(old);f=dict(fill)
    if not isinstance(f.get('id'),str) or not f['id'] or len(f['id'])>100:raise ValueError('FILL_ID_REQUIRED')
    for prior in value['fills']:
        if prior['id']==f['id']:
            if any(prior.get(k)!=f.get(k) for k in ('code','action','price','shares','time')):raise ValueError('FILL_ID_CONFLICT')
            return value
    code=f.get('code');qty=f.get('shares');price=f.get('price')
    if not isinstance(code,str) or len(code)!=6 or not code.isdigit():raise ValueError('FILL_CODE')
    if type(qty) is not int or qty<=0 or not finite(price) or price<=0:raise ValueError('ACTUAL_PRICE_AND_QUANTITY_REQUIRED')
    stamp=datetime.strptime(str(f['time']),'%Y%m%d%H%M%S');day=int(stamp.strftime('%Y%m%d'))
    if stamp>now or day not in calendar or not (93000<=int(f['time'])%1000000<=113000 or 130000<=int(f['time'])%1000000<=150000):
        raise ValueError('ACTUAL_FILL_TIME')
    previous=[r for r in value['fills'] if r['code']==code]
    if previous and f['time']<previous[-1]['time']:raise ValueError('FILL_CHRONOLOGY')
    pos=value['positions'].get(code)
    if f['action']=='buy':
        if not pos or pos['quantity']==0:
            if not origin or origin['row']['code']!=code:raise ValueError('FIRST_FILL_MUST_LINK_ALERT')
            if int(origin['packet']['asof'])>f['time']:raise ValueError('FILL_BEFORE_LINKED_ALERT')
            row=origin['row'];band=origin['band'];grade=origin['packet'].get('pullback_rule') or {}
            pos=dict(code=code,name=row['name'],row=row,entry_band=band,last_band=band,
                first_tier=grade.get('state','UNVERIFIED'),quantity=0,average_cost=0,add_count=0,lots=[],exit_latched=False)
            observed=datetime.strptime(str(origin['packet']['asof']),'%Y%m%d%H%M%S')
            if stamp.date()!=observed.date() or (stamp-observed).total_seconds()>120:
                pos['first_tier']='UNVERIFIED'
                pos['first_tier_reason']='成交时刻已超提醒证据时效，不把旧标准档自动延续到成交时刻'
            if not band['lower']<=price<=band['upper']:pos['off_rule_add']=True
        else:
            pos['add_count']+=1
            target=next_add_band(code,pos['last_buy_price'],pos['row']['positions'],pos['last_band'])
            # Off-rule actual purchases remain real. Without an actual mapped
            # next band, do not manufacture a lower anchor for future adds.
            if target and target['lower']<=price<=target['upper']:pos['last_band']=target
            else:pos['off_rule_add']=True
        old_qty=pos['quantity'];pos['quantity']+=qty
        pos['average_cost']=(pos['average_cost']*old_qty+price*qty)/pos['quantity']
        pos['last_buy_price']=price;pos['last_fill_id']=f['id']
        i=calendar.index(day)
        if i+1==len(calendar):raise ValueError('FILL_T1_CALENDAR_MISSING')
        pos['lots'].append(dict(shares=qty,sellable_day=calendar[i+1],fill_id=f['id']))
    elif f['action']=='sell':
        if not pos or qty>pos['quantity']:raise ValueError('SELL_EXCEEDS_REGISTERED_POSITION')
        if qty>sum(l['shares'] for l in pos['lots'] if l['sellable_day']<=day):raise ValueError('SELL_T1_UNAVAILABLE')
        left=qty
        for lot in pos['lots']:
            if lot['sellable_day']>day:continue
            take=min(left,lot['shares']);lot['shares']-=take;left-=take
        pos['quantity']-=qty;pos['last_fill_id']=f['id']
    else:raise ValueError('FILL_ACTION')
    value['positions'][code]=pos;value['fills'].append(f)
    return value


def origin(base,alert_id):
    alerts=json.loads((base/'strong_sector/alerts.json').read_bytes())
    alert=alerts.get(alert_id)
    if not alert:raise ValueError('LINKED_ALERT_NOT_FOUND')
    for path in (base/'strong_sector/pools').glob('*.json'):
        if path.name.endswith('.failed.json'):continue
        pool=json.loads(path.read_bytes())
        if pool.get('rows_sha256')!=alert['pool_id']:continue
        from strong_sector_watch_v1 import digest
        if digest(pool['rows'])!=pool['rows_sha256']:raise ValueError('FILL_MAP_SHA')
        row=next((r for r in pool['rows'] if r['code']==alert['code']),None)
        if row:return dict(row=row,packet=alert['packet'],band=alert['event']['band'])
    raise ValueError('LINKED_FROZEN_MAP_MISSING')


def register(app):
    from flask import request,jsonify
    from postclose_risk_runtime import directory
    @app.route('/api/strong_pullback/fills',methods=['GET','POST'])
    def fills_api():
        base=directory()
        try:
            with LOCK:
                value=load(base)
                if request.method=='POST':
                    body=request.get_json(force=True)
                    import akshare
                    from pathlib import Path
                    calendar=json.loads((Path(akshare.__file__).parent/'file_fold/calendar.json').read_bytes())
                    link=origin(base,body['alert_id']) if body.get('alert_id') else None
                    f={k:body.get(k) for k in ('id','code','action','price','shares','time')}
                    value=apply_fill(value,f,sorted(map(int,calendar)),datetime.now(),link);save(base,value)
            alerts_file=base/'strong_sector/alerts.json'
            alerts=json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
            return jsonify(ok=True,positions=list(value['positions'].values()),
                alerts=[dict(id=k,code=v['code'],name=v['packet'].get('name'),time=v['packet']['asof']) for k,v in alerts.items()],
                note='仅登记你已实际成交的价格和股数；提醒不等于成交，程序不下单。')
        except (OSError,ValueError,KeyError,TypeError) as error:return jsonify(ok=False,error=str(error)),400

    @app.route('/strong-pullback/fills')
    def fills_page():
        return r'''<!doctype html><meta charset="utf-8"><title>强势回调实际成交</title>
<style>body{background:#111;color:#ddd;font:16px sans-serif;max-width:900px;margin:35px auto}input,select,button{padding:9px;margin:7px}pre{white-space:pre-wrap}a{color:#9cf}</style>
<a href="/strong-pullback">板块环境</a> · <a href="/">工作台</a><h2>强势回调 · 实际成交登记</h2>
<p>仅填写已经完成的成交。首笔必须选择对应提醒；后续成交按同代码接续。同一次建仓或补仓的拆单成交先合并，登记实际均价与总股数。提醒后超过两分钟才成交，首笔档位记待核实，不自动启用补仓资格。此处单独记账，不代替券商成交单。</p>
<form><select id="alert"><option value="">后续成交（已有登记持仓）</option></select><br>
<input id="code" placeholder="6位股票代码" required><select id="action"><option value="buy">实际买入</option><option value="sell">实际卖出</option></select><br>
<input id="price" type="number" step="0.001" min="0.001" placeholder="实际成交价" required><input id="shares" type="number" min="1" step="1" placeholder="本笔实际股数" required><br>
<input id="time" type="datetime-local" step="1" required><button>登记实际成交</button></form><p id="msg"></p><pre id="positions"></pre>
<script>let token=crypto.randomUUID();const el=id=>document.getElementById(id);
async function load(){const d=await(await fetch('/api/strong_pullback/fills')).json();el('positions').textContent=(d.positions||[]).map(p=>p.code+' '+p.name+' ｜ '+p.quantity+'股 ｜ 均价'+p.average_cost.toFixed(3)+' ｜ 已补'+p.add_count+'次'+(p.exit_latched?' ｜ 退出已触发':'' )).join('\n')||'尚无实际成交登记';const a=el('alert');a.length=1;for(const r of d.alerts||[]){const o=document.createElement('option');o.value=r.id;o.textContent=r.code+' '+r.name+' '+r.time;o.dataset.code=r.code;a.add(o)}}
el('alert').onchange=()=>{const o=el('alert').selectedOptions[0];if(o.dataset.code)el('code').value=o.dataset.code};
document.querySelector('form').onsubmit=async e=>{e.preventDefault();const t=el('time').value.replaceAll(/[-:T]/g,'');const b={id:token,alert_id:el('alert').value,code:el('code').value,action:el('action').value,price:Number(el('price').value),shares:Number(el('shares').value),time:Number(t.length===12?t+'00':t)};const r=await(await fetch('/api/strong_pullback/fills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json();el('msg').textContent=r.ok?'已登记；后续只监控满足间距的下一真实黄金档位。':r.error;if(r.ok){token=crypto.randomUUID();await load()}};load();</script>'''


def observe(base,cfg,now,snapshot,market,indicator,broadcast,send,active,eligible_codes,refresh=None):
    """Separate identities per actual last fill and frozen next band.

    Every quote checks exit before eligibility. A notice never increases cost,
    quantity or add_count. Real additional fills are the only advancement.
    """
    with LOCK:
        value=load(base)
        positions=copy.deepcopy(value['positions'])
    for code,pos in positions.items():
        if pos['quantity']<=0:continue
        q=snapshot.get(code) or {}
        if not fresh(q,now):continue
        target=next_add_band(code,pos['last_buy_price'],pos['row']['positions'],pos['last_band'])
        band=target or pos['last_band']
        # Do not wait for indicators to latch an already-triggered risk exit.
        cost=pos['average_cost']
        if q['price']<=cost*.965+1e-10 or (now.hour>=15 and q['price']<pos['entry_band']['lower']):
            with LOCK:
                current=load(base);p=current['positions'].get(code)
                if p and p.get('last_fill_id')==pos['last_fill_id']:
                    p['exit_latched']=True;save(base,current)
            continue
        if code not in eligible_codes or pos.get('exit_latched') or pos.get('off_rule_add') or pos['first_tier']!='STANDARD' or pos['add_count']>=2:continue
        if not target or not target['lower']<=q['price']<=target['upper']:continue
        key=hashlib.sha256(json.dumps([code,pos['last_fill_id'],target],sort_keys=True).encode()).hexdigest()
        with LOCK:
            if key in load(base)['notices']:continue
        context=indicator(pos['row'],q)
        rule=evaluate(pos['row'],q,band,market,context.get('volume_ratio') or dict(status='UNKNOWN'),None,now,position=pos)
        if refresh is not None:
            try:
                latest=refresh([code,'sh000001']);q=latest.get(code) or {};checked=datetime.now()
                from strong_pullback_policy_v1 import market_state
                current_market=market_state(latest.get('sh000001') or {},market.get('level'),checked,market.get('t0'))
                rule=evaluate(pos['row'],q,band,current_market,context.get('volume_ratio') or dict(status='UNKNOWN'),None,checked,position=pos)
            except (OSError,ValueError,KeyError,TypeError):continue
        if rule['state']!='ADD_CANDIDATE':continue
        # Addition is allowed only in a current standard setup, outside the
        # index probe zone. A deep-drop/manual-board trial is not an add license.
        if rule.get('stock_tier')!='STANDARD' or rule['market']['state']!='ABOVE':continue
        lines=[f"【强势回调·补仓观察】{pos['name']} {code}",f"现价{q['price']:.3f}｜上次实际买价{pos['last_buy_price']:.3f}",*render(rule),
            f"登记持仓{pos['quantity']}股｜均价{cost:.3f}｜已补{pos['add_count']}次（最多2次）",
            '条件通过不等于已成交；补仓后请登记本笔实际成交，程序未下单。']
        title=f"强势回调·补仓观察 {code}";body='\n\n'.join(lines)
        with LOCK:
            current=load(base)
            if key in current['notices'] or current['positions'][code]['last_fill_id']!=pos['last_fill_id']:continue
            current['notices'][key]=dict(code=code,last_fill_id=pos['last_fill_id'],target=target,
                time=now.strftime('%Y%m%d%H%M%S'),price=q['price'],delivery='RECORDED')
            save(base,current)
        if not active():return
        ok=False;status='DISABLED'
        if cfg.get('push',{}).get('enabled'):
            try:ok,message=send(cfg,title,body);status='SENT' if ok else 'FAILED'
            except Exception:status='UNKNOWN'
        with LOCK:
            current=load(base);current['notices'][key]['delivery']=status;save(base,current)
        broadcast(dict(type='push',kind='strong_pullback_add',title=title,body=body,wechat=bool(ok),delivery=status))
