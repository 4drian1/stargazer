"""Explicit daily board approval. Never infer it from partial pool breadth."""
from datetime import datetime
import json
import threading

_LOCK=threading.RLock()


def confirm(path,day,name,flag,save):
    with _LOCK:
        old=json.loads(path.read_bytes()) if path.is_file() else {}
        boards=old.get('boards',{}) if old.get('day')==day else {}
        value=dict(day=day,boards={**boards,name:flag},confirmed_at=datetime.now().strftime('%Y%m%d%H%M%S'),
            basis='USER_CONFIRMED_NOT_MODEL_INFERRED')
        save(path,value);return value


def approved(base,day,names):
    path=base/'strong_pullback/board_approvals.json'
    try:value=json.loads(path.read_bytes())
    except (OSError,ValueError):return None
    if value.get('day')!=day:return None
    flags=[value.get('boards',{}).get(name) for name in names]
    return True if any(x is True for x in flags) else False if flags and all(x is False for x in flags) else None


def register(app):
    from flask import jsonify,request
    from postclose_risk_runtime import directory
    from strong_sector_watch_v1 import save

    @app.route('/api/strong_pullback/boards',methods=['GET','POST'])
    def board_environment():
        base=directory();day=int(datetime.now().strftime('%Y%m%d'))
        summary_file=base/'strong_sector/summary.json'
        summary=json.loads(summary_file.read_bytes()) if summary_file.is_file() else {}
        names=[r['name'] for r in summary.get('boards',[])]
        path=base/'strong_pullback/board_approvals.json'
        old=json.loads(path.read_bytes()) if path.is_file() else {}
        if request.method=='POST':
            body=request.get_json(silent=True) or {}
            if summary.get('pool_day')!=day or body.get('board') not in names or type(body.get('approved')) is not bool:
                return jsonify(ok=False,error='仅可确认当日监控板块；需明确通过或不通过'),400
            old=confirm(path,day,body['board'],body['approved'],save)
        return jsonify(day=day,summary=summary,approvals=old if old.get('day')==day else {},
            note='仅为冻结强势池内报价统计，不代表全板块；人工确认仅当天有效，不自动交易或再次推送')

    @app.route('/strong-pullback')
    def board_page():
        return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>强势回调 · 板块环境</title>
<style>body{background:#111;color:#ddd;font:16px sans-serif;max-width:850px;margin:40px auto;padding:20px}section{padding:16px;border:1px solid #444;margin:14px 0}button{padding:8px;margin:8px}a{color:#9cf}</style>
<a href="/">返回工作台</a> · <a href="/strong-pullback/fills">登记实际成交 / 查看补仓持仓</a><h2>强势回调 · 当日板块环境确认</h2><p id="note"></p><main></main>
<script>const main=document.querySelector('main');
async function load(){const d=await(await fetch('/api/strong_pullback/boards')).json();main.replaceChildren();document.querySelector('#note').textContent=d.note;
for(const r of d.summary.boards||[]){const s=document.createElement('section'),p=document.createElement('p');
const flag=(d.approvals.boards||{})[r.name];p.textContent=r.name+' ｜ 池内均涨跌 '+(r.mean_return==null?'未知':(r.mean_return*100).toFixed(2)+'%')+' ｜ 上涨 '+(r.up_count??'?')+'/'+r.valid_count+' ｜ 报价覆盖 '+r.valid_count+'/'+r.pool_count+' ｜ '+(flag===true?'人工通过':flag===false?'人工不通过':'未确认');s.append(p);
for(const [label,value] of [['确认环境通过',true],['环境不通过',false]]){const b=document.createElement('button');b.textContent=label;b.disabled=d.summary.pool_day!==d.day;b.onclick=async()=>{const q=await fetch('/api/strong_pullback/boards',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({board:r.name,approved:value})});if(!q.ok)alert((await q.json()).error);await load()};s.append(b)}main.append(s)}
if(!main.childNodes.length)main.textContent='当日监控报价尚未生成；先观察行情，不能用旧日统计代替。'}load();</script></html>'''
