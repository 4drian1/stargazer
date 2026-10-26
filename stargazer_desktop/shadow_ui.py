"""shadow_ui.py — 影子建议 UI 供给层（Stargazer 1.0 影子建议接线③）。

纪律：UI 不自己算特征——证据包全部来自数据线的 dist/data/shadow_packs.jsonl
（每行一个 EvidencePack dict，由证据流构建器产出；缺文件/缺行=该候选不可评分）。
一切评分调用经 shadow_journal.record_event(trigger='MANUAL') 入影子日记；
本模块任何异常不外抛，只返回明确状态码，绝不影响既有选股/提醒/风控。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import paths
import jev_adapter_v1 as jev
import shadow_journal

# 证据包输入文件（数据线产出；UI 只读不写）
PACKS_FILE = Path(paths.data_file(
    'data/shadow_packs.jsonl' if paths.FROZEN else 'dist/data/shadow_packs.jsonl',
    seed_from_bundle=False))
VALID_MINUTES = 30  # 人工查询默认建议有效期（自当前时刻起）

# EvidencePack dataclass 的合法字段（行内多余键直接忽略，避免数据线加字段即崩）
_PACK_FIELDS = ('evidence_id', 'asof_ts', 'market_date', 'query_ts',
                'candidates', 'risk_context', 'risk_missing', 'corvus_gap',
                'confirmed_plan', 'notes')

_index_cache: dict = {}


def _packs_index() -> dict:
    """(code, market_date) -> EvidencePack。按文件 mtime 缓存；坏行跳过。

    同一行包可能含多个候选，索引逐候选登记，取包时再按 code 过滤出单候选副本。
    """
    try:
        stamp = (str(PACKS_FILE), PACKS_FILE.stat().st_mtime)
    except OSError:
        return {}
    if stamp in _index_cache:
        return _index_cache[stamp]
    idx = {}
    try:
        lines = PACKS_FILE.read_text(encoding='utf-8').splitlines()
    except OSError:
        _index_cache[stamp] = idx
        return idx
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            pack = jev.EvidencePack(
                **{k: raw.get(k) for k in _PACK_FIELDS}).bind()
        except Exception:
            continue  # 坏行=该候选不可评分，不拖垮其它行
        if not isinstance(pack.candidates, list):
            continue
        for c in pack.candidates:
            code = str((c or {}).get('code') or '')
            if code:
                idx[(code, str(pack.market_date))] = pack
    _index_cache[stamp] = idx
    return idx


def _clear_cache() -> None:
    """测试用：文件被替换后强制重建索引。"""
    _index_cache.clear()


def evidence_catalog() -> dict:
    """Read-only available dates/codes; no provider calls or observation writes."""
    index = _packs_index()
    return {'items': [{'code': code, 'market_date': date}
                      for code, date in sorted(index)], 'count': len(index)}


def get_shadow_evidence(code: str, date: str) -> dict:
    """Historical input inspection, never a newly generated trading suggestion."""
    pack, why = find_pack(code, date)
    if pack is None:
        return dict(status='UNSCORABLE', reason=why, advice=None)
    pack.bind()
    gate = jev.gate_checks(pack, 'UNPOSITIONED')
    return dict(status='UNSCORABLE' if gate else 'EVIDENCE_READY',
                reason=gate.notes if gate else '', mode='HISTORICAL_REPLAY',
                market_date=pack.market_date, asof_ts=pack.asof_ts,
                query_ts=pack.query_ts, pack_hash=pack.pack_hash,
                candidates=pack.candidates, risk_context=pack.risk_context,
                risk_missing=pack.risk_missing, corvus_gap=pack.corvus_gap,
                notes=pack.notes, advice=None)


def find_pack(code: str, date: str):
    """按 code+日期查包。返回 (EvidencePack 单候选副本, None) 或 (None, 原因)。"""
    pack = _packs_index().get((str(code), str(date)))
    if pack is None:
        return None, 'PACK_MISSING'
    cand = [c for c in pack.candidates if str((c or {}).get('code')) == str(code)]
    if not cand:
        return None, 'PACK_MISSING'
    # 副本只带所查候选：评分对象是单个候选，不影响缓存里的原包
    return jev.EvidencePack(
        evidence_id=pack.evidence_id, asof_ts=pack.asof_ts,
        market_date=pack.market_date, query_ts=pack.query_ts,
        candidates=cand, risk_context=pack.risk_context,
        risk_missing=pack.risk_missing, corvus_gap=pack.corvus_gap,
        confirmed_plan=pack.confirmed_plan, notes=pack.notes), None


def _default_valid_until(pack) -> str:
    """默认有效期=当前时刻+30分钟；必须晚于证据包查询时刻（jev 三时刻校验）。"""
    vu = datetime.now() + timedelta(minutes=VALID_MINUTES)
    q = jev._parse_ts14(str(pack.query_ts))
    if q is not None and vu <= q:
        vu = q + timedelta(minutes=5)
    return vu.strftime('%Y%m%d%H%M%S')


def _journal_ref(rows: list) -> str | None:
    if not rows:
        return None
    r = rows[0]
    return '{}:{}:{}'.format(r.get('event_id'), r.get('vendor'),
                             r.get('correctness') or r.get('error') or 'UNKNOWN')


def _advice_from_row(row: dict) -> dict | None:
    """journal 行 → render_advice 展示字段（display 缺失时从 advice 回填）。"""
    display = row.get('display')
    if isinstance(display, dict):
        return display
    raw = row.get('advice')
    if isinstance(raw, dict):
        try:
            return jev.render_advice(jev.JEVAdvice(**{
                k: raw.get(k) for k in
                ('scenario', 'option', 'confidence', 'supporting', 'opposing',
                 'watch', 'invalidations', 'evidence_refs', 'valid_until_ts',
                 'information_gap', 'notes')}))
        except Exception:
            return None
    return None


def get_shadow_advice(code: str, date: str, *, scenario: str = 'UNPOSITIONED',
                      valid_until_ts: str | None = None) -> dict:
    """影子建议唯一入口。返回 {status, reason, advice, journal_ref}。

    status: OK=可取建议（advice=render_advice 字段，另带 expired 标记）；
    UNSCORABLE=缺包/风险证据缺失或过期（reason 透出 jev.gate_checks 原因）；
    ERROR=本次建议不可用（供应商失败/超时/额度尽，reason 注明）。
    本函数绝不外抛。
    """
    out = dict(status='ERROR', reason='', advice=None, journal_ref=None)
    code, date = str(code or '').strip(), str(date or '').strip()
    if not code or not date:
        out.update(status='UNSCORABLE', reason='code/date 必填')
        return out
    if scenario not in ('UNPOSITIONED', 'POSITIONED'):
        out.update(status='UNSCORABLE', reason='UNKNOWN_SCENARIO')
        return out
    try:
        pack, why = find_pack(code, date)
        if pack is None:
            out.update(status='UNSCORABLE',
                       reason='缺证据包（{}）：数据线未产出 {} 在 {} 的 EvidencePack'
                              .format(why, code, date))
            return out
        vu = str(valid_until_ts or '') or _default_valid_until(pack)
        # 一切调用入影子日记（额度/去重/记账/硬门短路都在 record_event 内）
        rows = shadow_journal.record_event(pack, scenario, 'MANUAL',
                                           valid_until_ts=vu,
                                           vendors=('deepseek',))
        ref = _journal_ref(rows)
        out['journal_ref'] = ref
        # 硬门：本地风险证据缺失/过期，未发送供应商请求（复用 jev 原因透出）
        gate = jev.gate_checks(pack, scenario)
        if gate is not None:
            out.update(status='UNSCORABLE',
                       reason='风险证据缺失或过期：' +
                              gate.notes.replace('HARD_GATE: ', ''))
            return out
        row = rows[0] if rows else {}
        if row.get('error') == 'OUT_OF_WINDOW':
            out.update(status='UNSCORABLE',
                       reason='证据包日期在影子观察窗口外：{}'.format(
                           row.get('window')))
            return out
        corr = row.get('correctness')
        if corr in ('PASS', 'REUSED', 'EXPIRED'):
            advice = _advice_from_row(row)
            if advice is None:
                out['reason'] = '本次建议不可用（日记缺少展示字段）'
                return out
            out.update(status='OK', advice=advice,
                       expired=(corr == 'EXPIRED'))
            if corr == 'REUSED':
                out['reused'] = True
            return out
        # 供应商/额度/超时失败：明确原因，绝不外抛
        if corr == 'BUDGET_EXHAUSTED':
            out['reason'] = '本次建议不可用（额度已用尽：{}）'.format(
                row.get('gate_error'))
        elif corr == 'FAIL':
            out['reason'] = '本次建议不可用（门禁拒绝：{}）'.format(
                row.get('gate_error'))
        elif corr == 'EVENT_TIMEOUT':
            out['reason'] = '本次建议不可用（等待超时，迟到结果已弃用）'
        elif corr == 'HARD_GATE':
            out.update(status='UNSCORABLE', reason='风险证据缺失或过期')
        else:
            detail = row.get('msg') or row.get('gate_error') or row.get('error') \
                or corr or 'UNKNOWN'
            out['reason'] = '本次建议不可用（{}）'.format(detail)
        return out
    except Exception as exc:  # 最后一道防线：任何异常不炸主进程
        out['reason'] = '本次建议不可用（{}）'.format(type(exc).__name__)
        return out
