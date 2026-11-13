#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""I34 stock linkage v2: per-stock three-source read-only view over PRODUCTION outputs.

Fix history (user read-only review 2026-11-07, five findings):
  R1 risk false-available  — v1 called postclose_risk_runtime.find_pack() with
      PR.state() (a task status, not a selection snapshot) and treated the returned
      (pack, reason) tuple as truthy -> empty packs reported available=True.
      v2 delegates per-stock risk to the PRODUCTION path plan_evidence.for_snapshot()
      which handles find_pack's (pack|None, reason|None) contract and returns
      MISSING + reason. Missing risk can never present as available.
  R2 selection = research pool — pool_daily_v11.parquet is the GREEN-WATCH research
      pool, NOT the formal selection. v2 reads the FORMAL selection snapshot
      (assistant_workflow.snapshot(), data/pentacles_demo/snapshot.json) and the
      confirmed plan (assistant_plan.json + plan_readiness); the research pool is
      only shown as a SEPARATE, explicitly-labeled historical source when explicitly
      requested. Frozen-EXE paths must be injected (configure/set_desktop_dir);
      the module never guesses data paths from a temp extraction dir.
  R4 production registration — the arrival path is GUI import -> /api/i34/classify
      -> record_from_prediction() (production function, persisted); verified by
      acceptance without HTTP.
  R5 register dedup/persist — handled in i34_desktop_integration (identity pair
      (source_id, input_sha); changed body = new version; fail-closed persistence).

Discipline unchanged: read-only over existing results; no recomputation of
market-wide risk context per announcement; date/cutoff mismatches flagged
explicitly; SHADOW is not a real importance grade; no voting/adding/multiplying.
"""
import os
import sys
from collections import OrderedDict

_CFG = {
    'research_pool_parquet': os.environ.get('I34_POOL_PARQUET'),
    'include_research_pool': False,
    # 生产数据目录注入（仅 dev 进程读生产数据时用；frozen EXE 由 paths 原生解析）：
    # 指向 <install>/data（含 pentacles_demo/snapshot.json、assistant_plan.json）。
    'data_dir': None,
}
_PROVIDERS = {}  # 'snapshot' / 'evidence' / 'market' / 'confirmed_plan'


def configure(research_pool_parquet=None, include_research_pool=None, data_dir=None):
    """Inject paths (esp. frozen EXE: the desktop injects, never guess)."""
    if research_pool_parquet is not None:
        _CFG['research_pool_parquet'] = research_pool_parquet
    if include_research_pool is not None:
        _CFG['include_research_pool'] = bool(include_research_pool)
    if data_dir is not None:
        _CFG['data_dir'] = data_dir


def set_provider(name, fn):
    """Test hook: override 'snapshot' / 'evidence' / 'market' / 'confirmed_plan'."""
    _PROVIDERS[name] = fn


def _frozen():
    return getattr(sys, 'frozen', False)


def _desktop_sources():
    """Import the production desktop modules. Desktop dir comes from the
    established injection point (i34_desktop_integration._DESKTOP_DIR, env
    overridable); in a frozen EXE without injection we FAIL with an explicit
    reason instead of guessing paths from the temp extraction dir."""
    try:
        import i34_desktop_integration as INT
        d = str(INT._DESKTOP_DIR)
    except Exception as e:
        raise RuntimeError('DESKTOP_DIR_UNAVAILABLE:' + str(e)[:120])
    if not os.path.isdir(d):
        raise RuntimeError('DESKTOP_DIR_NOT_INJECTED:%s' % d)
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import assistant_workflow as AW
        import plan_evidence as PE
    except Exception as e:
        raise RuntimeError('DESKTOP_MODULES_UNAVAILABLE:' + str(e)[:120])
    return AW, PE


def _load_pentacles_json(name):
    """Read the PRODUCTION pentacles_demo file (user review fix #1: the acceptance
    previously read the dev-tree data dir and saw an empty 11-02 roster, while the
    real production plan lives in <install>/data). Same validation as
    assistant_workflow.snapshot(); read-only, never writes production data."""
    import json as _json
    p = os.path.join(_CFG['data_dir'], 'pentacles_demo', name)
    value = _json.loads(open(p, encoding='utf-8').read())
    if name == 'snapshot.json':
        if value.get('schema') != 'PAGE_OF_PENTACLES_DEMO_V1' \
                or not isinstance(value.get('rows'), list):
            raise ValueError('请先生成星币侍从观察名单')
    return value


def _snapshot_provider():
    if 'snapshot' in _PROVIDERS:
        return _PROVIDERS['snapshot']()
    if _CFG.get('data_dir'):
        return _load_pentacles_json('snapshot.json')
    AW, _ = _desktop_sources()
    return AW.snapshot()  # raises ValueError('请先生成星币侍从观察名单') if absent


def _evidence_provider(snapshot):
    if 'evidence' in _PROVIDERS:
        return _PROVIDERS['evidence'](snapshot)
    _, PE = _desktop_sources()
    return PE.for_snapshot(snapshot)


def _market_provider(codes, evidence):
    if 'market' in _PROVIDERS:
        return _PROVIDERS['market'](codes, evidence)
    _, PE = _desktop_sources()
    return PE.market_risk_summary(codes, evidence)


def _confirmed_plan_provider(snapshot):
    if 'confirmed_plan' in _PROVIDERS:
        return _PROVIDERS['confirmed_plan'](snapshot)
    try:
        import json as _json
        if _CFG.get('data_dir'):
            p = os.path.join(_CFG['data_dir'], 'pentacles_demo', 'assistant_plan.json')
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            confirmed = _json.loads(open(p, encoding='utf-8').read())
        else:
            AW, PE = _desktop_sources()
            import paths
            from pathlib import Path
            f = Path(paths.data_file('data/pentacles_demo/assistant_plan.json',
                                     seed_from_bundle=False))
            confirmed = _json.loads(f.read_bytes())
        match = (confirmed.get('assistant_plan_id') is not None
                 and str(confirmed.get('data_date')) == str(snapshot.get('data_date'))
                 and str(confirmed.get('monitor_date')) == str(snapshot.get('monitor_date')))
        return dict(available=True, confirmed=bool(match),
                    assistant_plan_id=confirmed.get('assistant_plan_id'),
                    data_date=confirmed.get('data_date'),
                    monitor_date=confirmed.get('monitor_date'),
                    selected_codes=confirmed.get('selected_codes'),
                    note='确认计划与当前快照同日同名单' if match
                    else '确认计划存在但与当前快照不匹配（旧计划）；不混用，以下并列展示')
    except FileNotFoundError:
        return dict(available=False, reason='NO_CONFIRMED_PLAN_FILE',
                    note='尚无确认计划文件；仅展示当前选股快照')
    except Exception as e:
        return dict(available=False, reason='CONFIRMED_PLAN_UNAVAILABLE',
                    detail=str(e)[:120])


# ---------------------------------------------------------------------------
# 历史研究池（GREEN-WATCH pool_daily_v11）：明确标注的独立历史来源，绝不当入围依据
# ---------------------------------------------------------------------------
def research_pool_provider(codes):
    path = _CFG.get('research_pool_parquet')
    if not path or not os.path.exists(path):
        return {c: dict(available=False, reason='RESEARCH_POOL_NOT_CONFIGURED')
                for c in codes}
    try:
        import pyarrow.parquet as pq
    except Exception:
        return {c: dict(available=False, reason='PYARROW_UNAVAILABLE') for c in codes}
    try:
        want = [str(c).zfill(6) for c in codes]
        cols = ['date', 'code', 'primary_setup', 'primary_tier', 'zone_distance_atr',
                'structure_intact_flag', 'structural_breakdown_flag']
        t = pq.read_table(path, columns=cols, filters=[('code', 'in', want)])
        rows = t.to_pydict()
        latest = {}
        for i in range(len(rows['code'])):
            code = str(rows['code'][i]).zfill(6)
            d = rows['date'][i]
            if code not in latest or (d is not None and d > latest[code][0]):
                latest[code] = (d, i)
        out = {}
        for c in want:
            if c in latest:
                i = latest[c][1]
                out[c] = dict(available=True, date=rows['date'][i],
                              primary_setup=rows['primary_setup'][i],
                              primary_tier=rows['primary_tier'][i],
                              zone_distance_atr=rows['zone_distance_atr'][i],
                              structure_intact_flag=rows['structure_intact_flag'][i],
                              structural_breakdown_flag=rows['structural_breakdown_flag'][i])
            else:
                out[c] = dict(available=False, reason='NOT_IN_RESEARCH_POOL')
        return out
    except Exception as e:
        return {c: dict(available=False, reason='RESEARCH_POOL_READ_FAILED',
                        detail=str(e)[:120]) for c in codes}


def _fmt_date(v):
    if v is None:
        return None
    s = str(v)
    if len(s) == 8 and s.isdigit():
        return '%s-%s-%s' % (s[:4], s[4:6], s[6:])
    return s


def _asof_day(risk):
    ts = risk.get('asof_ts')
    if ts and len(str(ts)) >= 8:
        return str(ts)[:8]
    return None


def _date_mismatches(info_dates, snapshot, risk, confirmed):
    mm = []
    known = {}
    for k, v in info_dates.items():
        if v:
            known['info@' + k] = _fmt_date(v)
    if snapshot:
        known['selection.data_date'] = _fmt_date(snapshot.get('data_date'))
    if risk and risk.get('status') == 'AVAILABLE':
        day = _asof_day(risk)
        if day:
            known['risk.asof_ts_day'] = _fmt_date(day)
    if confirmed and confirmed.get('available') and confirmed.get('data_date'):
        known['confirmed_plan.data_date'] = _fmt_date(confirmed['data_date'])
    vals = set(known.values())
    if len(known) >= 2 and len(vals) > 1:
        mm.append(OrderedDict([('type', 'DATE_MISMATCH'),
                               ('detail', '各来源日期不一致，仅并列参考，不得混用为同一时点'),
                               ('dates', known)]))
    elif len(known) < 2:
        mm.append(OrderedDict([('type', 'DATE_UNAVAILABLE'),
                               ('detail', '可比较的日期不足 2 个，无法核对时点一致性')]))
    return mm


def build_stock_linkage(records, codes=None):
    """records: display-chain records (with optional info_layer). codes: optional
    restriction; default = codes appearing in info records UNION formal snapshot rows.
    Returns {code: view}. Read-only; missing sources are explicit, never fabricated."""
    info_by_code = OrderedDict()
    for r in records:
        info = r.get('info_layer')
        if not info:
            continue
        for code in (r.get('stocks') or []):
            if code and (codes is None or code in codes):
                c = str(code).zfill(6)
                info_by_code.setdefault(c, []).append(OrderedDict([
                    ('corpus_id', r.get('source_id') or r.get('id')),
                    ('title', r.get('title')),
                    ('announcement_date', info.get('announcement_date')),
                    ('l0_status', info.get('l0_status')),
                    ('primary_event_status', next(
                        (e.get('status') for e in (info.get('event_statuses') or [])
                         if e.get('is_primary')), None)),
                    ('unknowns_count', len(info.get('unknowns') or [])),
                    ('importance_slot', r.get('importance')),
                    ('body_version', r.get('body_version', 1)),
                ]))
    snapshot = None
    snap_err = None
    try:
        snapshot = _snapshot_provider()
    except Exception as e:
        snap_err = str(e)[:160]
    evidence = {}
    market_lines = []
    if snapshot:
        try:
            evidence = _evidence_provider(snapshot) or {}
        except Exception as e:
            evidence = {}
            snap_err = (snap_err or '') + '|EVIDENCE_UNAVAILABLE:' + str(e)[:120]
        try:
            market_codes = [row['code'] for row in snapshot.get('rows', [])]
            market_lines = _market_provider(market_codes, evidence) or []
        except Exception as e:
            market_lines = ['大盘环境读取失败(%s)：缺失显式标注，不脑补' % str(e)[:80]]
    snap_codes = [str(row['code']).zfill(6) for row in (snapshot or {}).get('rows', [])]
    # Default scope = information codes ∪ current plan codes (user review fix #2:
    # the previous default dropped all 42 information codes; stocks covered by
    # announcements but not in the roster stay visible, explicitly NOT in it).
    codes = list(dict.fromkeys(
        [str(c).zfill(6) for c in (codes or [])] + snap_codes
        + list(info_by_code.keys())))
    confirmed = _confirmed_plan_provider(snapshot) if snapshot else \
        dict(available=False, reason='NO_CURRENT_SNAPSHOT')
    rp_out = {}
    if _CFG.get('include_research_pool') and codes:
        rp_out = research_pool_provider(codes)
    view = OrderedDict()
    for c in codes:
        ev = evidence.get(c) or {}
        sel = (ev.get('selection') or {}) if snapshot else {}
        risk = (ev.get('risk') or {}) if snapshot else {}
        if not snapshot:
            risk = dict(status='MISSING',
                        reason='当前选股快照不可用(%s)；不借用历史分数' % (snap_err or '未知'))
        row = next((r for r in (snapshot or {}).get('rows', [])
                    if str(r['code']).zfill(6) == c), None)
        if snapshot and row is None:
            risk = dict(status='MISSING',
                        reason='该代码不在当前正式选股快照(数据日 %s)；无同日风险证据，不借用历史分数'
                               % (snapshot.get('data_date') or '未知'))
        infos = info_by_code.get(c) or []
        dates = {'announcement': (infos[0]['announcement_date'] if infos else None)}
        entry = OrderedDict([
            ('info', infos),
            ('selection', OrderedDict([
                ('available', bool(row)),
                ('source', 'FORMAL_SNAPSHOT:PAGE_OF_PENTACLES_DEMO_V1' if row
                 else 'NOT_IN_CURRENT_SNAPSHOT'),
                ('snapshot_data_date', (snapshot or {}).get('data_date')),
                ('name', (row or {}).get('name')),
                ('setup', (row or {}).get('setup')),
                ('reference_price', (row or {}).get('reference_price')),
                ('entry_rule_levels', ((row or {}).get('entry_rule') or {}).get('levels')),
                ('selection_model', (row or {}).get('selection_model')),
                ('plan_source', (row or {}).get('plan_source')),
                ('note', '当前正式选股快照行；观察价位以 entry_rule.levels 为准' if row
                 else '该代码不在当前正式名单；不因有公告而虚构入围'),
            ])),
            ('confirmed_plan', confirmed),
            ('risk', OrderedDict([
                ('status', risk.get('status')),
                ('available', risk.get('status') == 'AVAILABLE'),
                ('asof_ts', risk.get('asof_ts')),
                ('reason', risk.get('reason')),
                ('ref', risk.get('ref')),
                ('lines', None),  # 填充于下
            ])),
            ('market_env', OrderedDict([
                ('available', bool(market_lines)),
                ('lines', market_lines),
                ('note', '大盘环境取自已绑定的同日 Pyxis 证据（一次性有日期讨论），非逐公告重算'),
            ])),
            ('mismatches', _date_mismatches(dates, snapshot, risk, confirmed)),
        ])
        try:
            _, PE = _desktop_sources()
            entry['risk']['lines'] = PE.risk_lines(risk)
        except Exception:
            entry['risk']['lines'] = None
        if _CFG.get('include_research_pool') and rp_out:
            entry['research_pool'] = OrderedDict([
                ('label', '历史研究池（GREEN-WATCH pool_daily_v11）— 非当前正式选股，仅供参考'),
                ('data', rp_out.get(c)),
            ])
        view[c] = entry
    return view


def render_linkage_lines(linkage):
    """Render the linkage view as DeepSeek context lines (explanations, NOT scoring)."""
    lines = ['【按股票三源联动（正式选股/信息/风险，读取既有生产结果，只读）】',
             '说明：三源独立；SHADOW 不是真实重要性等级；日期不一致处已标注，不得混用；缺失即缺失。']
    for code, v in linkage.items():
        lines.append('— 股票 %s' % code)
        s = v['selection']
        if s.get('available'):
            lines.append('  选股(正式快照): 数据日=%s | setup=%s | 参考价=%s | entry规则层级=%s' % (
                _fmt_date(s.get('snapshot_data_date')), s.get('setup'),
                s.get('reference_price'), s.get('entry_rule_levels')))
        else:
            lines.append('  选股: %s（不在当前正式名单）' % s.get('source'))
        cp = v['confirmed_plan']
        if cp.get('available'):
            lines.append('  确认计划: %s（plan_id=%s, 数据日=%s）' % (
                '与当前快照匹配' if cp.get('confirmed') else '与当前快照不匹配（旧计划）',
                cp.get('assistant_plan_id'), _fmt_date(cp.get('data_date'))))
        else:
            lines.append('  确认计划: 缺失(%s)' % cp.get('reason'))
        if v['info']:
            for inf in v['info']:
                lines.append('  信息(SHADOW规则层): %s | L0=%s | 主事件状态=%s | 未知项=%d | 公告日=%s%s' % (
                    inf.get('corpus_id'), inf.get('l0_status'),
                    inf.get('primary_event_status'), inf.get('unknowns_count'),
                    _fmt_date(inf.get('announcement_date')) or '未知',
                    '' if (inf.get('body_version') or 1) == 1
                    else ' | 正文版本=v%d（同公告更正版本）' % inf['body_version']))
        else:
            lines.append('  信息(SHADOW规则层): 无该股票的公告信息记录')
        risk = v['risk']
        if risk.get('available'):
            for l in (risk.get('lines') or ['风险证据可用（截截止 %s）' % risk.get('asof_ts')]):
                lines.append('  风险: ' + l)
        else:
            lines.append('  风险: 缺失(%s)%s' % (risk.get('status'),
                        (' ' + str(risk.get('reason'))[:100]) if risk.get('reason') else ''))
        m = v['market_env']
        if m.get('available'):
            for l in m['lines'][:4]:
                lines.append('  大盘环境: ' + l)
        else:
            lines.append('  大盘环境: 缺失')
        for mm in v['mismatches']:
            lines.append('  ⚠ %s: %s%s' % (mm['type'], mm['detail'],
                        (' ' + str(mm.get('dates'))) if mm.get('dates') else ''))
        if v.get('research_pool'):
            d = v['research_pool']['data']
            lines.append('  %s: %s' % (v['research_pool']['label'],
                        ('date=%s setup=%s tier=%s' % (_fmt_date(d.get('date')),
                         d.get('primary_setup'), d.get('primary_tier')))
                        if d and d.get('available') else '缺失(%s)' % (d or {}).get('reason')))
    return lines
