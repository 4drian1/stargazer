#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stargazer desktop bridge for I34 (frozen I33 announcement-importance demo).

Wires i34_infer (zero-fit inference, frozen I33) + i34_desktop_integration
(panel renderer + DeepSeek context builder) into the running Flask server.

The corvus i34_demo package is located RELATIVE to this file (no hardcoded drive
letter); override with the I34_INFER_DIR environment variable. The desktop
config.yaml / deepseek_api are resolved from this (stargazer_desktop) directory.
"""
import os, sys, json, functools
from pathlib import Path

HERE = Path(__file__).resolve().parent          # stargazer_desktop
STARGAZER = HERE.parent                         # D:/stargazer

# EXE-aware: under PyInstaller onefile, data unpacks to sys._MEIPASS.
_FROZEN = getattr(sys, 'frozen', False)
if _FROZEN:
    # onefile layout: i34_demo/ and the frozen asset dirs are bundled at the bundle
    # root (flat), so i34_infer's own derived defaults resolve correctly.
    _MEIPASS = Path(sys._MEIPASS)
    I34_DIR = _MEIPASS / 'i34_demo'
    IMPACT_DIR = _MEIPASS
    DESKTOP_DIR = _MEIPASS
else:
    I34_DIR = Path(os.environ.get('I34_INFER_DIR') or
                   (STARGAZER / 'c3' / 'corvus' / 'impact_v1_1' / 'i34_demo'))
    IMPACT_DIR = I34_DIR.parent
    DESKTOP_DIR = HERE

if str(I34_DIR) not in sys.path:
    sys.path.insert(0, str(I34_DIR))

import i34_infer as I
import i34_desktop_integration as INT

# inject config: the impact_v1_1 directory (so i34_infer finds frozen extractors/model)
# and the artifact dir (persisted vocab), both resolved from the bundle when frozen
I.configure(impact_dir=str(IMPACT_DIR), artifacts_dir=str(I34_DIR))
# resolve desktop (deepseek_api / config.yaml) from this directory
INT.set_desktop_dir(str(DESKTOP_DIR))


def register_i34_routes(app, get_cfg):
    """Register /api/i34/* routes + the /i34 panel page. Call once at startup."""
    INT.register_i34_routes(app, get_cfg)
    _enrich_panel_items(app)
    _register_page(app)
    return app


def _enrich_panel_items(app):
    """展示字段 enrich（纯派生，不拼造）：原文URL、影响说明、披露时间标签。
    缺失一律 null/未提供；标题不冒充摘要；影响说明与事实分开，不承诺涨跌。"""
    original = app.view_functions.get('i34_panel')
    if original is None:
        # 路由函数名可能不同：按 endpoint 兜底找 /api/i34/panel 的 view
        for rule in app.url_map.iter_rules():
            if str(rule) == '/api/i34/panel':
                original = app.view_functions.get(rule.endpoint)
                break
    if original is None:
        return

    import functools
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        response = original(*args, **kwargs)
        try:
            payload = json.loads(response.get_data(as_text=True))
        except (ValueError, AttributeError):
            return response
        items = payload.get('items')
        if not isinstance(items, list):
            return response
        for item in items:
            if not isinstance(item, dict):
                continue
            url = item.get('source_url') or item.get('url')
            item['source_url'] = url if isinstance(url, str) and url.strip() else None
            item['source_url_note'] = None if item['source_url'] else '原文链接未提供（不以拼造补齐）'
            involved = [str(x) for x in (item.get('stocks') or [])][:6]
            sectors = [str(x) for x in (item.get('sectors') or [])][:6]
            scopes = involved + sectors
            item['impact_note'] = ('涉及：' + '、'.join(scopes) + '；潜在影响及成立条件见证据坐标，不构成涨跌承诺'
                                   if scopes else '涉及范围未提供；潜在影响不构成涨跌承诺')
            stamp = item.get('disclosure_time') or item.get('disclosed_at') or item.get('published_at')
            item['disclosure_label'] = str(stamp) if stamp else None
            # 事实摘要：无摘要字段时用真实证据首条摘录（标注摘录，非拟写摘要）
            if not item.get('summary'):
                evidence = item.get('evidence') or []
                candidates = []
                for ev in evidence:
                    snippet = str((ev or {}).get('snippet') or '').strip()
                    if len(snippet) >= 20 and any('一' <= ch <= '鿿' for ch in snippet):
                        candidates.append(snippet)
                if candidates:
                    excerpt = max(candidates, key=len)[:160]
                    item['summary'] = '摘录：' + excerpt
                    item['summary_kind'] = 'EXCERPT_NOT_AI_SUMMARY'
        response.set_data(json.dumps(payload, ensure_ascii=False))
        return response
    for rule in app.url_map.iter_rules():
        if str(rule) == '/api/i34/panel':
            app.view_functions[rule.endpoint] = wrapper
            break


def _register_page(app):
    @app.route('/i34')
    def i34_panel_page():
        return app.response_class(_PANEL_HTML, mimetype='text/html')


_PANEL_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corvus 统一信息页</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
  header{padding:14px 20px;background:#161a21;border-bottom:1px solid #2a2f3a;display:flex;align-items:center;gap:14px}
  header h1{font-size:16px;margin:0}
  .hint{font-size:12px;color:#8a93a3}
  main{padding:16px 20px;max-width:1100px;margin:0 auto}
  .card{border:1px solid #2a2f3a;border-radius:10px;padding:14px 16px;margin:12px 0;background:#14181f}
  .card h2{font-size:15px;margin:0 0 6px}
  .tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:6px;margin-right:6px}
  .HIGH{background:#5a1f1f;color:#ffb4b4}.MEDIUM{background:#5a4a1f;color:#ffe6a8}
  .LOW{background:#1f3a5a;color:#a8d4ff}.UNKNOWN{background:#3a3a3a;color:#cfcfcf}
  .meta{font-size:12px;color:#9aa3b2;margin:4px 0}
  .ev{font-size:12px;color:#cdd3dd;background:#0f1318;border-left:2px solid #3a4150;padding:4px 8px;margin:3px 0;border-radius:4px}
  .ev b{color:#ffd479}
  .verify{font-size:12px;color:#ffcaca;margin:2px 0}
  #reply{white-space:pre-wrap;background:#0f1318;padding:12px;border-radius:8px;margin-top:10px;border:1px solid #2a2f3a}
  button{cursor:pointer;background:#27406b;color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:13px}
  button:disabled{opacity:.5;cursor:default}
  .usage{font-size:11px;color:#7a8493}
</style>
</head>
<body>
<header>
  <h1>I34 公告重要性面板</h1>
  <span class="hint">冻结 I33 分类器（仅判“是否重要”）；重要性只排序，不删非 HIGH。政策/美股/金油未实际提供时标注未知。</span>
</header>
<main>
  <div id="conn" class="hint" style="padding:6px 10px;border-radius:8px;background:#14181f;margin-bottom:8px">连接状态检测中…</div>
  <div id="status" class="hint">加载中…</div>
  <div class="card" id="importCard">
    <h2>导入新公告（本地分类＋排队；不自动调用付费抽取/DeepSeek）</h2>
    <div class="meta">股票代码 <input id="ic" placeholder="如 000498" style="width:110px;background:#0f1318;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:4px 6px">
      　标题 <input id="it" placeholder="留空则取正文首行" style="width:420px;background:#0f1318;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:4px 6px"></div>
    <textarea id="ib" placeholder="粘贴公告正文…" style="width:100%;min-height:120px;margin-top:6px;background:#0f1318;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px"></textarea>
    <div style="margin-top:8px"><button id="ii" onclick="doImport()">导入</button>　<span id="ires" class="meta"></span></div>
  </div>
  <div id="cards"></div>
  <button id="ds" onclick="askDeepSeek()">让 DeepSeek 综合（独立来源另附）</button>
  <div id="reply" style="display:none"></div>
</main>
<script>
const $ = s => document.querySelector(s);
function tagClass(i){return ['HIGH','MEDIUM','LOW','UNKNOWN'].includes(i)?i:'UNKNOWN';}
function esc(s){
  return String(s ?? '').replace(/&/g,'&amp;')
    .replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
async function load(){
  const r = await fetch('/api/i34/panel'); const j = await r.json();
  const items = j.items||[];
  $('#status').textContent = items.length + ' 条公告';
  const merged = items.filter(i=>i.d2_status==='D2_INFO_LAYER_MERGED').length;
  const waiting = items.filter(i=>i.d2_status==='QUEUED_FOR_OFFLINE_D2').length;
  $('#conn').textContent = items.length
    ? '已接入 · 本地信息层（I33分类＋D2规则层，SHADOW）· 公告 '+items.length+' 条 · D2 等待 '+waiting+' / 完成 '+merged
    : '已接入 · 本地信息层 · 暂无公告记录（信息缺失≠无利空）';
  $('#conn').style.border = '1px solid #3a4150';
  const wrap = $('#cards'); wrap.innerHTML='';
  const card = it => {
    const el=document.createElement('div'); el.className='card';
    const date = it.authoritative_date ? ('权威日期(回执索引): '+it.authoritative_date+'（'+(it.receipt_status||'')+'）')
                                       : '权威日期：尚未匹配回执（正文首日期不可信，已弃用）';
    let ev = (it.evidence||[]).slice(0,6).map(e=>{
      const snip=esc((e.snippet||'').slice(0,40));
      return '<div class="ev">['+esc(e.kind)+'] '+esc(e.value)+' '+esc(e.unit)+' @['+e.start+':'+e.end+'] '+snip+'</div>';
    }).join('');
    let v = (it.items_to_verify||[]).map(x=>'<div class="verify">待核查：'+esc(x)+'</div>').join('');
    const d2 = it.info_layer || {};
    let d2html = '';
    const l0 = d2.l0_status || (it.gate_unknown ? 'GATED_UNKNOWN' : '');
    if (l0) d2html += '<div class="d2">L0门禁: '+esc(l0)+'</div>';
    const rules = d2.rule_judgments || [];
    if (rules.length) d2html += '<div class="d2">规则判定: '+rules.map(r=>esc(r.rule_id||'')+':'+esc(r.result||'')).join('； ')+'</div>';
    const evs = d2.event_statuses || [];
    if (evs.length) d2html += '<div class="d2">事项状态: '+evs.map(e=>esc(e.event_type||'')+':'+esc(e.status||'')).join('； ')+'</div>';
    const unk = d2.unknowns || [];
    if (unk.length) d2html += '<div class="d2">未知项('+unk.length+'): '+unk.map(u=>esc(u.item||u.rule_id||'')).join('； ')+'</div>';
    el.innerHTML = '<h2 class="'+tagClass(it.importance)+'">['+esc(it.importance)+'] '+esc(it.title||it.id)+'</h2>'
      + '<div class="meta">来源ID: '+esc(it.source_id)+'　股票: '+(it.stocks||[]).join(',')+'　板块: '+(it.sectors||[]).join(',')+'</div>'
      + '<div class="meta">'+date+'</div>'
      + '<div class="meta">输入SHA: '+(it.input_sha||'—')+'　门控UNKNOWN: '+it.gate_unknown+'</div>'
      + '<div class="meta">D2状态: <b>'+esc(it.d2_status||'—')+'</b>　正文版本: '+(it.body_version||1)+'　落盘SHA: '+esc(it.full_body_sha||'—')+'</div>'
      + ev + v + d2html;
    return el;
  };
  const imp = items.filter(i=>i.importance==='HIGH'||i.importance==='MEDIUM');
  const rest = items.filter(i=>!(i.importance==='HIGH'||i.importance==='MEDIUM'));
  imp.forEach(it=>wrap.appendChild(card(it)));
  if(rest.length){
    const det=document.createElement('details'); det.className='card';
    const sum=document.createElement('summary');
    sum.textContent='其他等级公告（'+rest.length+' 条）——重要性仅排序，未删除；展开查看';
    det.appendChild(sum);
    rest.forEach(it=>det.appendChild(card(it)));
    wrap.appendChild(det);
  }
}
async function doImport(){
  const code=$('#ic').value.trim(), title=$('#it').value.trim(), body=$('#ib').value;
  const out=$('#ires'), btn=$('#ii');
  if(!body.trim()){out.textContent='正文为空，未导入';return;}
  btn.disabled=true; out.textContent='导入中…';
  try{
    const r=await fetch('/api/i34/classify',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({code:code,title:title,body:body})});
    const j=await r.json();
    if(!j.ok){out.textContent='导入失败: '+(j.error||('HTTP '+r.status));}
    else{
      const rec=j.record||{};
      out.textContent='已导入：分类='+(rec.importance||'—')
        +'　落盘='+(rec.persisted?'已写入':'未写入(失败)')
        +'　D2='+(rec.d2_status||'—')
        +'　正文版本='+(rec.body_version||1)
        +'　input_sha='+(rec.input_sha||'—')
        +'　完整正文SHA='+(rec.full_body_sha||'—').slice(0,16);
      $('#ib').value='';
      await load();
    }
  }catch(e){out.textContent='请求失败: '+e;}
  btn.disabled=false;
}
async function askDeepSeek(){
  const b=$('#ds'); b.disabled=true; $('#reply').style.display='none';
  try{
    const r = await fetch('/api/i34/deepseek',{method:'POST'}); const j = await r.json();
    const res = j.result||{};
    let html = '<b>DeepSeek 回复</b>'+(res.model?'（'+esc(res.model)+'）':'');
    html += '\\n\\n'+esc(res.content||'（无内容）');
    if(res.usage) html += '\\n\\n[usage] prompt='+res.usage.prompt_tokens+' completion='+res.usage.completion_tokens+' total='+res.usage.total_tokens;
    $('#reply').innerHTML = html; $('#reply').style.display='block';
  }catch(e){ $('#reply').textContent='调用失败: '+e; $('#reply').style.display='block'; }
  b.disabled=false;
}
load();
</script>
</body>
</html>
"""
