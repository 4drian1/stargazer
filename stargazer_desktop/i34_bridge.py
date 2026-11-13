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
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime, timedelta, timezone

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
if _FROZEN:
    INT.set_info_shadow_source(str(IMPACT_DIR / 'model_a' / 'materiality_v1_build' /
                                  'materiality_d2_4_desktop_shadow_records_v0_1.jsonl'))


def _news_root():
    import paths
    base = Path(paths.APP_DIR)
    project = base.parents[1] if paths.FROZEN else base.parent
    return (project / 'corvus' / 'data').resolve()


def _news_document(root, raw_path):
    try:
        path = Path(raw_path).resolve()
        if path.is_relative_to(root / 'raw' / 'news') and path.stat().st_size <= 512 * 1024:
            document = json.loads(path.read_bytes())
            if isinstance(document, dict):
                return document
    except (OSError, ValueError, TypeError):
        pass
    return {}


def local_announcements(*, code=None, name=None, limit=12, data_dir=None):
    """Read the latest locally collected items and exact-code issuer announcements."""
    root = Path(data_dir).resolve() if data_dir is not None else _news_root()
    database = root / 'corvus.sqlite'
    if not database.is_file() or (code and not name):
        return []
    now = datetime.now(timezone(timedelta(hours=8)))
    after = (now - timedelta(days=30)).date().isoformat()
    sql = ("SELECT raw_id,source_id,title,url,published_at,raw_path FROM raw_news "
           "WHERE published_at>=? AND published_at<=? ")
    params = [after, now.isoformat(timespec='seconds')]
    if code:
        sql += "AND source_id IN ('CNINFO','SZSE','SSE') AND substr(title,1,?)=? "
        params.extend((len(name), name))
    else:
        sql += "AND source_id='EASTMONEY_NEWS' AND title IS NOT NULL AND title<>'' "
    sql += "ORDER BY published_at DESC, raw_id DESC LIMIT ?"
    params.append(limit * 4)
    items = []
    seen = set()
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=1)) as conn:
        rows = conn.execute(sql, params).fetchall()
    for raw_id, source, title, url, published, raw_path in rows:
        document = _news_document(root, raw_path)
        if code:
            codes = document.get('secCode') or document.get('SECURITY_CODE') or document.get('securityCode')
            if str(code) not in (codes if isinstance(codes, list) else [codes]):
                continue
        source_url = url or document.get('url')
        if not isinstance(source_url, str) or not source_url.startswith(('https://', 'http://')):
            source_url = None
        summary = document.get('内容') or document.get('summary') or document.get('body')
        summary = ' '.join(summary.split())[:280] if isinstance(summary, str) else None
        key = (title, published[:10])
        if key in seen:
            continue
        seen.add(key)
        body = document.get('内容')
        items.append(dict(id=raw_id, source=source, title=title, source_url=source_url,
                          disclosure_date=published, summary=summary or None,
                          local_detail_available=source == 'EASTMONEY_NEWS' and
                          isinstance(body, str) and bool(body.strip())))
        if len(items) >= limit:
            break
    return items


def register_i34_routes(app, get_cfg):
    """Register /api/i34/* routes + the /i34 panel page. Call once at startup."""
    from flask import jsonify

    @app.get('/api/i34/news/<int:raw_id>')
    def local_news_detail(raw_id):
        root = _news_root()
        database = root / 'corvus.sqlite'
        if database.is_file():
            try:
                with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=1)) as conn:
                    row = conn.execute(
                        "SELECT title,published_at,raw_path FROM raw_news WHERE raw_id=? AND source_id='EASTMONEY_NEWS'",
                        (raw_id,)).fetchone()
            except sqlite3.Error:
                return jsonify(error='本地快讯暂时无法读取'), 503
            if row:
                title, published, raw_path = row
                body = _news_document(root, raw_path).get('内容')
                if isinstance(body, str) and body.strip():
                    return jsonify(title=title, disclosure_date=published, body=body)
        return jsonify(error='本地快讯正文未找到'), 404

    INT.register_i34_routes(app, get_cfg)
    _enrich_panel_items(app)
    _register_page(app)
    return app


_CNINFO_LINKS = None


def _cninfo_detail_links():
    """真实官方原文链接（用户手动下载清单）。只返回清单中确实存在的 URL，
    按 (证券代码, 披露日期) 索引；绝不按代码/标题拼造 CNINFO URL。"""
    global _CNINFO_LINKS
    if _CNINFO_LINKS is not None:
        return _CNINFO_LINKS
    out = {}
    try:
        p = I34_DIR / 'i34_cninfo_download_links.txt'
        for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or ',' not in line:
                continue
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 5:
                continue
            code, day, url = parts[1], parts[2], parts[4]
            if code and day and url.startswith('https://www.cninfo.com.cn/'):
                out.setdefault((code, day), set()).add(url)
    except (OSError, ValueError):
        out = {}
    _CNINFO_LINKS = out
    return out


def _d2_asset_locator(item):
    """D2 资产定位：写清信息层来自哪个真实文件/版本，不假装链接到公告原文。"""
    layer = item.get('info_layer') or {}
    if not layer:
        return None
    return dict(
        layer_version=layer.get('layer_version') or 'D2_4_PACKAGE',
        date_source=item.get('date_source'),
        source_id=item.get('source_id'),
        input_sha=item.get('input_sha'),
        batch_file='materiality_d2_4_desktop_shadow_records_v0_1.jsonl',
        note='D2 为本地零训练规则层资产，不是 CNINFO 原文链接；按 source_id/input_sha 可回溯本地正文')


def _derive_summary(item):
    """事实摘要：只用真实证据原文摘录，绝不拟写、绝不用标题冒充。

    优先 D2 信息层的 evidence_quote（长文本事实摘录），其次冻结证据 snippet。
    均不可用时明确写「事实摘要未提供」，不留空字符串让人以为是加载失败。
    """
    if item.get('summary'):
        item['summary_kind'] = item.get('summary_kind') or 'PROVIDED'
        return
    layer = item.get('info_layer') or {}
    candidates = []
    for ev in layer.get('event_statuses') or []:
        quote = str((ev or {}).get('evidence_quote') or '').strip()
        if len(quote) >= 20 and any('一' <= ch <= '鿿' for ch in quote):
            candidates.append((bool((ev or {}).get('is_primary')), quote))
    for judgment in layer.get('rule_judgments') or []:
        quote = str((judgment or {}).get('evidence_quote') or '').strip()
        if len(quote) >= 20 and any('一' <= ch <= '鿿' for ch in quote):
            candidates.append((False, quote))
    for ev in item.get('evidence') or []:
        snippet = str((ev or {}).get('snippet') or '').strip()
        if len(snippet) >= 20 and any('一' <= ch <= '鿿' for ch in snippet):
            candidates.append((False, snippet))
    if candidates:
        item['summary'] = '事实摘录：' + max(candidates, key=lambda pair: (pair[0], len(pair[1])))[1][:180]
        item['summary_kind'] = 'EVIDENCE_EXCERPT_NOT_AI_SUMMARY'
    else:
        item['summary'] = None
        item['summary_kind'] = 'NOT_AVAILABLE'
        item['summary_note'] = '事实摘要未提供（证据均为短数值片段；不以标题冒充摘要，不调用模型生成）'


def derive_display_fields(item):
    """面板单条记录的展示派生。纯函数，便于离线验收。"""
    links = _cninfo_detail_links()
    code = str(item.get('code') or '')
    day = str(item.get('authoritative_date') or '')
    candidates = links.get((code, day)) or set()
    url = item.get('source_url') or item.get('url')
    if not isinstance(url, str) or not url.startswith(('https://', 'http://')):
        # The legacy receipt has a stock and date only. Refuse ambiguous matches.
        url = next(iter(candidates)) if len(candidates) == 1 else None
    item['source_url'] = url
    if item['source_url']:
        item['source_url_note'] = 'CNINFO 官方公告详情页（本地下载清单逐条核对）'
    else:
        item['source_url_note'] = '原文链接未提供（不以代码/标题拼造；可按下述资产定位回溯本地正文）'
    item['asset_locator'] = _d2_asset_locator(item)
    involved = [str(x) for x in (item.get('stocks') or [])][:6]
    sectors = [str(x) for x in (item.get('sectors') or [])][:6]
    scopes = involved + sectors
    item['impact_note'] = ('涉及：' + '、'.join(scopes) + '；潜在影响及成立条件见证据坐标，不构成涨跌承诺'
                           if scopes else '涉及范围未提供；潜在影响不构成涨跌承诺')
    # 披露日期：权威日期优先，并带上日期来源；无回执时如实说明，不取正文首日期。
    stamp = (item.get('disclosure_time') or item.get('disclosed_at') or
             item.get('published_at') or day)
    item['disclosure_date'] = str(stamp) if stamp else None
    item['disclosure_label'] = (str(stamp) + '（' + str(item.get('date_source') or '来源未标注') + '）'
                                if stamp else '披露日期尚未匹配回执（正文首日期不可信，已弃用）')
    _derive_summary(item)
    return item


def _enrich_panel_items(app):
    """展示字段 enrich（纯派生，不拼造）：原文URL、事实摘录、披露日期、资产定位。
    缺失一律 null + 明确说明；标题不冒充摘要；影响说明与事实分开，不承诺涨跌。"""
    original = app.view_functions.get('i34_panel')
    if original is None:
        for rule in app.url_map.iter_rules():
            if str(rule) == '/api/i34/panel':
                original = app.view_functions.get(rule.endpoint)
                break
    if original is None:
        return

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
            if isinstance(item, dict):
                derive_display_fields(item)
        # 按披露日期倒序（真实近期记录在前）；无权威日期的排最后并保留可见。
        items.sort(key=lambda x: (bool(x.get('disclosure_date')),
                                 str(x.get('disclosure_date') or '')), reverse=True)
        payload['items'] = items
        try:
            payload['recent_local'] = local_announcements()
        except sqlite3.Error as exc:
            payload['recent_local'] = []
            payload['recent_local_error'] = type(exc).__name__
        dated = [x for x in items if x.get('disclosure_date')]
        payload['display'] = dict(
            sort='disclosure_date_desc',
            dated_count=len(dated),
            undated_count=len(items) - len(dated),
            latest_disclosure_date=max((x['disclosure_date'] for x in dated), default=None),
            note='按披露日期展示；无回执日期的记录排最后。历史语料不得冒充近期公告。')
        response.set_data(json.dumps(payload, ensure_ascii=False))
        return response
    for rule in app.url_map.iter_rules():
        if str(rule) == '/api/i34/panel':
            app.view_functions[rule.endpoint] = wrapper
            break


_CHECK_STATE = None


def _check_state_path():
    try:
        import paths
        return Path(paths.data_file('i34_check_state.json'))
    except Exception:
        return HERE / 'i34_check_state.json'


def set_check_state_path(p):
    """测试钩子：隔离检查状态文件；传 None 恢复默认路径。"""
    global _CHECK_STATE
    _CHECK_STATE = Path(p) if p is not None else None


def check_new_status(mark=False):
    """只读扫描本地记录，返回上次检查时间/新增数/待处理数/失败原因。

    mark=True 只写「本次检查时间 + 已知 id 集合」，不触发任何 LLM、事实抽取、
    付费调用或网络请求。刷新/切页调用 mark=False，绝不改动状态。
    """
    state_file = _CHECK_STATE or _check_state_path()
    state = {}
    failure_reason = None
    try:
        if state_file.exists():
            got = json.loads(state_file.read_text(encoding='utf-8'))
            if isinstance(got, dict):
                state = got
    except (OSError, ValueError) as exc:
        failure_reason = 'CHECK_STATE_UNREADABLE:' + type(exc).__name__
    try:
        records = INT.all_records()
    except Exception as exc:
        records = []
        failure_reason = (failure_reason or '') + ';RECORDS_UNREADABLE:' + type(exc).__name__
    ids = {str(r.get('id') or r.get('source_id') or '') for r in records if isinstance(r, dict)}
    ids.discard('')
    known = set(state.get('known_ids') or [])
    new_ids = sorted(ids - known)
    pending = sum(
        1 for r in records if isinstance(r, dict) and
        (str(r.get('d2_status') or '').startswith('QUEUED') or
         (r.get('info_layer') or {}).get('l0_status') == 'ABSTAIN'))
    out = dict(ok=not failure_reason,
               last_check_at=state.get('last_check_at'),
               checked_at=datetime.now().isoformat(timespec='seconds'),
               total_records=len(ids), new_count=len(new_ids),
               new_ids=new_ids[:20], pending_count=pending,
               failure_reason=failure_reason,
               note='只读本地记录；未调用模型/抽取/DeepSeek，未产生费用')
    if mark:
        tmp = state_file.with_suffix('.tmp')
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(dict(last_check_at=out['checked_at'],
                                          known_ids=sorted(ids)), ensure_ascii=False),
                           encoding='utf-8')
            tmp.replace(state_file)
            out['state_written'] = True
        except OSError as exc:
            out['state_written'] = False
            out['failure_reason'] = ((failure_reason or '') +
                                     ';CHECK_STATE_WRITE_FAILED:' + type(exc).__name__).lstrip(';')
            out['ok'] = False
    else:
        out['state_written'] = False
    return out


def _register_page(app):
    @app.route('/api/i34/check_new', methods=['GET', 'POST'])
    def i34_check_new():
        from flask import jsonify, request
        # GET 纯读（刷新/切页）；仅显式 POST 才记录本次检查时间。
        mark = request.method == 'POST'
        return jsonify(check_new_status(mark=mark))

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
  <div class="card" id="checkCard">
    <h2>检查新信息（只读本地记录，不调用模型/抽取/DeepSeek）</h2>
    <div class="meta" id="checkStat">未检查</div>
    <div style="margin-top:6px"><button id="cb" onclick="checkNew(false)">刷新检查状态</button>
      　<button id="cm" onclick="checkNew(true)">标记本次已检查</button></div>
  </div>
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
async function checkNew(mark){
  const out=$('#checkStat');
  try{
    const r=mark
      ? await fetch('/api/i34/check_new',{method:'POST'})
      : await fetch('/api/i34/check_new');
    const j=await r.json();
    out.textContent='上次检查: '+(j.last_check_at||'尚未检查')
      +'　新增记录: '+j.new_count+'　待处理: '+j.pending_count
      +'　记录总数: '+j.total_records
      +(j.failure_reason?('　失败原因: '+j.failure_reason):'')
      +'　'+(j.note||'');
  }catch(e){out.textContent='检查失败: '+e;}
}
async function load(){
  const r = await fetch('/api/i34/panel'); const j = await r.json();
  const items = j.items||[];
  const disp=j.display||{};
  $('#status').textContent = items.length + ' 条公告'
    + (disp.latest_disclosure_date ? ('　最新披露日期: '+disp.latest_disclosure_date) : '')
    + (disp.undated_count ? ('　日期待回执: '+disp.undated_count+' 条') : '');
  checkNew(false);
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
    const sumText = it.summary
      ? '<div class="d2">事实摘要: '+esc(it.summary)+' <span class="usage">('+esc(it.summary_kind||'')+')</span></div>'
      : '<div class="d2">'+esc(it.summary_note||'事实摘要未提供')+'</div>';
    const linkHtml = it.source_url
      ? '<div class="meta">原文: <a href="'+esc(it.source_url)+'" target="_blank" rel="noopener">CNINFO 公告详情页</a>　'+esc(it.source_url_note||'')+'</div>'
      : '<div class="meta">原文链接: '+esc(it.source_url_note||'未提供')+'</div>';
    const asset = it.asset_locator||{};
    const assetHtml = asset.layer_version
      ? '<div class="usage">D2资产: '+esc(asset.layer_version)+' / '+esc(asset.batch_file||'')
        +'　source_id='+esc(asset.source_id||'')+'　input_sha='+esc((asset.input_sha||'').slice(0,16))+'</div>'
      : '';
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
      + '<div class="meta">披露日期: '+esc(it.disclosure_label||'未匹配回执')+'</div>'
      + '<div class="meta">D2状态: <b>'+esc(it.d2_status||'—')+'</b>　正文版本: '+(it.body_version||1)+'　落盘SHA: '+esc(it.full_body_sha||'—')+'</div>'
      + linkHtml + sumText
      + '<div class="d2">'+esc(it.impact_note||'')+'</div>'
      + assetHtml + ev + v + d2html;
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
