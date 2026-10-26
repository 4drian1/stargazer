"""oracles_v1.py — Stargazer 1.0 LLM 外包层（接口 V1.2；用户第二轮反例修订）。

修订要点（对应用户反例）：
1. **引用≠结论**：字段注册表绑定允许的引用/类型/单位；facts 只允许
   {ref, field, unit} 三键（注入 value=True 等额外键拒绝）；结论 evidence_ids
   必须在注册表；**极性校验**——claim 含"成立/未成立"且引用状态字段时，必须与
   代码值一致（divergence=False 时声称"背离已成立"→拒）；幅度字段拒绝
   unit="probability"，claim 引用幅度时禁止"概率/亏损概率"措辞。
   关键状态（成立/未成立）由代码值认证：payload 标 certified 字段，
   LLM 自由文本保持未认证、不直接成为决策事实。
2. （jev_adapter）占位绕过修复。
3. （jev_adapter）买入确认机会门 + valid_until 严格解析。
4. **线程不累积**：模块级共享 ThreadPoolExecutor(max_workers=2) + 在飞上限
   （排队超限→UNAVAILABLE(BUSY)）；超时迟到结果仍永久弃用。真实传输超时的
   socket 级设置在接入真实客户端时叠加。
5. Pyxis 版本绑定升级为**完整哈希**（模型+预处理）并引用已验收指纹来源。
6. 幅度/置信度允许合法 0：幅度=有限且≥0；置信度=有限且在[0,1]。
"""
from __future__ import annotations

import json
import math
import re
import hashlib
import hmac
import itertools
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

SCHEMA = 'STARGAZER_ORACLES_V1_3'
CORVUS_GAP = True
# 解释结果防伪造密钥（进程内常量；真实部署换用户级密钥管理）
_VERIFY_SECRET = b'stargazer-oracles-v1-process-key'


def _canon(o) -> bytes:
    return json.dumps(o, ensure_ascii=False, sort_keys=True,
                      default=str).encode('utf-8')


def _vtoken(payload: dict, pack_hash: str) -> str:
    return hmac.new(_VERIFY_SECRET, _canon(payload) + pack_hash.encode(),
                    hashlib.sha256).hexdigest()[:24]

# 版本冻结：完整哈希（2026-10-20 实测），来源=各 fit 目录 manifest + replay 验收
PYXIS_FROZEN = {
    '232主版': dict(
        model_sha256='e589ea52c60c1a14344b50b871c49d8476b0c0c1779f192fe368d26f90972f8f',
        preprocessing_sha256='42e902b862109147b0f6e744c4ae37e2df23569d8ecb366e4059b39a6f66b201',
        fit_dir='auxiliary_120m_fixed_fit_20261018_v1',
        acceptance='R0连续动能固定训练验收20261018+rebuild_features重放maxΔ=0.0'),
    '266对照版': dict(
        model_sha256='45672661d8bda9f15e2105f573866a890ebd04bc37f9c8188675ffb4759e7e13',
        preprocessing_sha256='25abea5a0458ceeb77b20d4ecf5325bdb1a432ea34aa1f800404241cb9d6f65f',
        fit_dir='common_candidate_fixed_fit_20261019_v1',
        acceptance='共同成员固定训练验收20261019+rebuild_features重放maxΔ=0.0'),
}

# 字段注册表：路径后缀 → 类别/允许单位。kind: state(极性校验)/price/score/amplitude
FIELD_SPEC = {
    'downside_state': dict(kind='state', units={'state', '状态'}),
    'trigger': dict(kind='state', units={'state', '状态'}),
    'f618': dict(kind='price', units={'price', '价格', '元'}),
    'atr_ref': dict(kind='price', units={'price', '价格', '元'}),
    'p_close_score': dict(kind='score', units={'score', '评分'}),
    'golden_zone': dict(kind='score', units={'atr', 'ATR'}),
    'plan_dist_atr': dict(kind='score', units={'atr', 'ATR'}),
    'amp_232': dict(kind='amplitude', units={'amplitude', '幅度', 'pct'}),
    'amp_266': dict(kind='amplitude', units={'amplitude', '幅度', 'pct'}),
    'horizon': dict(kind='text', units={'text', '文本'}),
}
_FACT_KEYS = {'ref', 'field', 'unit'}


@dataclass
class EvidencePack:
    evidence_id: str
    asof_ts: str                     # 风险特征截止（最近已完成15分截止）
    market_date: str
    query_ts: str = ''               # 本次查询时刻（消费侧时钟；重放=回放查询时刻）
    candidates: list[dict] = field(default_factory=list)
    risk_context: dict | None = None
    risk_missing: list[str] = field(default_factory=list)
    corvus_gap: bool = CORVUS_GAP
    confirmed_plan: dict | None = None
    notes: str = ''

    def bind(self) -> 'EvidencePack':
        object.__setattr__(self, 'evidence_id',
                           self.evidence_id or uuid.uuid4().hex[:12])
        return self

    @property
    def pack_hash(self) -> str:
        return hashlib.sha256(_canon(dict(
            evidence_id=self.evidence_id, asof_ts=self.asof_ts,
            query_ts=self.query_ts, market_date=self.market_date,
            candidates=self.candidates,
            risk_context=self.risk_context,
            risk_missing=self.risk_missing))).hexdigest()[:24]

    def ref_registry(self) -> set[str]:
        refs = {self.evidence_id}
        for c in self.candidates:
            er = c.get('evidence_refs') or {}
            refs.update(str(v) for v in er.values())
            refs.add(str(c.get('code')))
        if self.risk_context:
            refs.add(str(self.risk_context.get('version', '')))
            refs.add(str(self.risk_context.get('model_sha256', ''))[:16])
        # 包级锚点章节（合法引用目标：章节名即证据坐标）
        refs.update({'risk_context', 'confirmed_plan', 'corvus_gap',
                     'asof_ts', 'query_ts'})
        refs.discard('')
        return refs

    def field_lookup(self, field_path: str):
        _MISSING = _Missing = object()
        parts = field_path.split(':')
        try:
            if parts[0] == 'cand' and len(parts) == 3:
                for c in self.candidates:
                    if str(c.get('code')) == parts[1]:
                        return c.get(parts[2], _MISSING)
                return _MISSING
            if parts[0] == 'risk' and len(parts) == 2:
                return (self.risk_context or {}).get(parts[1], _MISSING)
        except Exception:
            return _MISSING
        return _MISSING

    def field_spec(self, field_path: str):
        parts = field_path.split(':')
        key = parts[-1]
        return FIELD_SPEC.get(key)


def tarot_candidate(code: str, *, p_close_score, golden_zone, f618,
                    plan_dist_atr, atr_ref, downside_state: dict,
                    trigger: dict | None, evidence_refs: dict) -> dict:
    return dict(code=code, p_close_score=p_close_score, golden_zone=golden_zone,
                f618=f618, plan_dist_atr=plan_dist_atr, atr_ref=atr_ref,
                downside_state=downside_state, trigger=trigger,
                evidence_refs=evidence_refs)


def pyxis_context(*, version: str, amp_232, amp_266, market_state: dict,
                  board_states: list[dict], leader_states: list[dict],
                  missing: list[str], asof_ts: str,
                  horizon='未来120交易分钟最大不利下探幅度'):
    if version not in PYXIS_FROZEN:
        raise ValueError('PYXIS_VERSION_NOT_FROZED')
    return dict(version=version, amp_232=amp_232, amp_266=amp_266,
                horizon=horizon, market_state=market_state,
                board_states=board_states, leader_states=leader_states,
                missing=missing, asof_ts=asof_ts,
                model_sha256=PYXIS_FROZEN[version]['model_sha256'],
                preprocessing_sha256=PYXIS_FROZEN[version]['preprocessing_sha256'],
                horizon_note='幅度非概率')


OPPORTUNITY_ROLE = """你是 Stargazer 1.0 的机会解释角色。只能解释给定 JSON 证据包。
铁律：
- claim/notes 不得含数字；数值必须经 facts 引用：{"ref","field","unit"} 三键，
  不得增加任何其他键；值由代码按 field 路径从证据包取出并负责展示；
- field 只能是证据包存在的字段；成立/未成立等状态以代码值为准——你的 claim
  若声称某状态成立，引用的状态字段代码值必须为真，否则整条被拒绝；
- 无归因证据只能描述共现，禁止因果措辞；Corvus 未接入，缺口不等于无利空。
回答三问：为何入名单/当前黄金位置；下杀、止跌、背离是否成立；下一步观察与失效条件。
输出 JSON：{"conclusions":[{"claim","facts":[{"ref","field","unit"}],
"evidence_ids":[...]}],"invalidations":[同结构],"notes":""}"""

RISK_ROLE = """你是 Stargazer 1.0 的风险解释角色。只能解释给定 JSON 证据包。
铁律：claim/notes 不得含数字；facts 只允许 {"ref","field","unit"} 三键；
幅度是未来120交易分钟最大不利下探幅度（amplitude），引用幅度字段时 unit 必须为
amplitude/幅度，claim 不得称其为概率或亏损概率；
缺失=不确定，不得描述为安全；结论必须绑定引用；无归因证据只描述共现；
Corvus 缺口显式保留。
回答三问：风险主要来自大盘/板块/领涨股还是个股；是否出现已计算信号；missing
意味着什么、判断能解释到什么程度。
输出 JSON：{"conclusions":[...],"uncertainties":[...],"notes":""}"""

_DIGIT = re.compile(r'\d')
_POLAR_POS = ('成立', '满足')
_POLAR_NEG = ('未成立', '不成立', '没有成立', '未满足', '不满足')
_PROB_WORDS = ('概率', '亏损概率', '胜率')


def _truthy_state(v) -> bool:
    if isinstance(v, dict):
        return all(bool(x) for x in v.values())
    return bool(v)


def _validate_payload(payload: dict, pack: EvidencePack) -> list[str]:
    errs = []
    if not isinstance(payload, dict):
        return ['PAYLOAD_NOT_DICT']
    concl = payload.get('conclusions')
    if not isinstance(concl, list) or not concl:
        errs.append('CONCLUSIONS_REQUIRED')
        concl = []
    registry = pack.ref_registry()
    certified = True
    sections = [('conclusions', concl),
                ('invalidations', payload.get('invalidations') or []),
                ('uncertainties', payload.get('uncertainties') or [])]
    for sec, items in sections:
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                errs.append(f'{sec}[{i}]_NOT_DICT')
                certified = False
                continue
            claim = it.get('claim', '')
            if not claim or not isinstance(claim, str):
                errs.append(f'{sec}[{i}]_CLAIM_EMPTY')
                certified = False
            elif _DIGIT.search(claim):
                errs.append(f'{sec}[{i}]_CLAIM_HAS_DIGIT')
                certified = False
            eids = it.get('evidence_ids') or []
            for e in eids:
                if str(e) not in registry:
                    errs.append(f'{sec}[{i}]_UNKNOWN_EVIDENCE_ID')
                    certified = False
            facts = it.get('facts')
            if sec == 'conclusions' and not facts:
                errs.append(f'{sec}[{i}]_NO_FACTS')
                certified = False
            state_vals, amp_cited = [], False
            cited_kinds = set()
            for j, fct in enumerate(facts or []):
                if not isinstance(fct, dict):
                    errs.append(f'{sec}[{i}].facts[{j}]_SHAPE')
                    certified = False
                    continue
                if set(fct.keys()) - _FACT_KEYS:
                    errs.append(f'{sec}[{i}].facts[{j}]_EXTRA_KEYS')
                    certified = False
                if not fct.get('ref') or not fct.get('field'):
                    errs.append(f'{sec}[{i}].facts[{j}]_SHAPE')
                    certified = False
                    continue
                if str(fct['ref']) not in registry:
                    errs.append(f'{sec}[{i}].facts[{j}]_UNKNOWN_REF')
                    certified = False
                spec = pack.field_spec(str(fct['field']))
                if spec is None:
                    errs.append(f'{sec}[{i}].facts[{j}]_UNREGISTERED_FIELD')
                    certified = False
                    continue
                if str(fct.get('unit', '')) not in spec['units']:
                    errs.append(f'{sec}[{i}].facts[{j}]_BAD_UNIT')
                    certified = False
                val = pack.field_lookup(str(fct['field']))
                if type(val) is object:
                    errs.append(f'{sec}[{i}].facts[{j}]_UNKNOWN_FIELD')
                    certified = False
                    continue
                cited_kinds.add(spec['kind'])
                if spec['kind'] == 'state':
                    state_vals.append(_truthy_state(val))
                elif spec['kind'] == 'amplitude':
                    amp_cited = True
            claim_polar = isinstance(claim, str) and (
                any(w in claim for w in _POLAR_POS) or
                any(w in claim for w in _POLAR_NEG))
            if claim_polar:
                if not state_vals:
                    # 状态声明却未引用状态字段：不得认证（反例：背离=False 引
                    # 价格字段声称"背离已成立"）
                    errs.append(f'{sec}[{i}]_STATE_CLAIM_WITHOUT_STATE_FACT')
                    certified = False
                else:
                    pos = any(w in claim for w in _POLAR_POS) and \
                        not any(w in claim for w in _POLAR_NEG)
                    neg = any(w in claim for w in _POLAR_NEG)
                    if pos and not all(state_vals):
                        errs.append(f'{sec}[{i}]_POLARITY_MISMATCH')
                        certified = False
                    if neg and any(state_vals):
                        errs.append(f'{sec}[{i}]_POLARITY_MISMATCH')
                        certified = False
            if isinstance(claim, str) and amp_cited and \
                    any(w in claim for w in _PROB_WORDS):
                errs.append(f'{sec}[{i}]_AMPLITUDE_AS_PROBABILITY')
                certified = False
    if _DIGIT.search(str(payload.get('notes', ''))):
        errs.append('NOTES_HAS_DIGIT')
        certified = False
    payload['certified'] = certified and not errs
    payload['_pack_hash'] = pack.pack_hash
    payload['_vtoken'] = _vtoken(
        {k: v for k, v in payload.items() if not k.startswith('_')},
        pack.pack_hash)
    return errs


# —— 有界共享线程池：防线程累积（反例4）——
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='oracles')
_INFLIGHT = threading.Semaphore(4)
_MAX_QUEUE = 6
_queued = itertools.count(0)
_queued_lock = threading.Lock()
_queued_n = [0]


def _q_inc():
    with _queued_lock:
        _queued_n[0] += 1


def _q_dec():
    with _queued_lock:
        _queued_n[0] -= 1


def _q_get():
    with _queued_lock:
        return _queued_n[0]


def _role_call(role_prompt: str, pack: EvidencePack, ask: str,
               complete: Callable[[list[dict]], Any], *, timeout_s=30.0):
    pack_blob = json.dumps(
        dict(schema=SCHEMA, asof_ts=pack.asof_ts,
             market_date=pack.market_date, ask=ask,
             corvus_gap=pack.corvus_gap, candidates=pack.candidates,
             risk_context=pack.risk_context, risk_missing=pack.risk_missing,
             confirmed_plan=pack.confirmed_plan), ensure_ascii=False)
    messages = [dict(role='system', content=role_prompt),
                dict(role='user', content=pack_blob)]
    if _q_get() >= _MAX_QUEUE:
        return dict(status='UNAVAILABLE', reason='BUSY')
    if not _INFLIGHT.acquire(blocking=False):
        return dict(status='UNAVAILABLE', reason='BUSY')
    _q_inc()

    def _release(_fut):
        _q_dec()
        _INFLIGHT.release()

    try:
        fut = _EXECUTOR.submit(complete, messages)
        try:
            raw = fut.result(timeout=timeout_s)
        except FutTimeout:
            # 取消尚未运行的排队任务；已运行任务由 done 回调在真正结束时释放名额
            fut.cancel()
            fut.add_done_callback(_release)
            return dict(status='UNAVAILABLE', reason='TIMEOUT',
                        note='late_result_discarded')
        text = raw if isinstance(raw, str) else getattr(raw, 'content', str(raw))
        m = re.search(r'\{.*\}', text, re.S)
        payload = json.loads(m.group(0)) if m else None
        if not isinstance(payload, dict):
            _release(fut)
            return dict(status='PARSE_FAIL', raw=text[:500])
        errs = _validate_payload(payload, pack)
        _release(fut)
        if errs:
            return dict(status='SCHEMA_REJECT', errors=errs[:12], payload=payload)
        return dict(status='OK', payload=payload, pack_hash=pack.pack_hash)
    except Exception as exc:
        _release(None)
        return dict(status='UNAVAILABLE', reason=type(exc).__name__)


def explain_opportunity(pack: EvidencePack, ask: str,
                        complete: Callable, timeout_s=30.0) -> dict:
    return _role_call(OPPORTUNITY_ROLE, pack, ask, complete, timeout_s=timeout_s)


def explain_risk(pack: EvidencePack, ask: str,
                 complete: Callable, timeout_s=30.0) -> dict:
    return _role_call(RISK_ROLE, pack, ask, complete, timeout_s=timeout_s)


def build_pack_from_research(snapshot_row: dict, *,
                             version: str = '232主版') -> EvidencePack:
    cand = tarot_candidate(
        snapshot_row['code'],
        p_close_score=snapshot_row.get('pB2'),
        golden_zone=snapshot_row.get('golden_dist_618_atr'),
        f618=snapshot_row.get('F618'),
        plan_dist_atr=snapshot_row.get('plan_dist'),
        atr_ref=snapshot_row.get('atr_ref'),
        downside_state=dict(approach=None, stop=None, divergence=None),
        trigger=None, evidence_refs=dict(snapshot='pday_snapshot_v2'))
    risk = pyxis_context(
        version=version, amp_232=snapshot_row.get('amp_232_raw'),
        amp_266=snapshot_row.get('amp_266_raw'), market_state={},
        board_states=[], leader_states=[],
        missing=snapshot_row.get('missing_fields') or [],
        asof_ts=str(snapshot_row.get('cutoff_ts',
                                     snapshot_row.get('asof', ''))))
    return EvidencePack(evidence_id='', asof_ts=risk['asof_ts'],
                        market_date=str(snapshot_row.get('P', '')),
                        candidates=[cand], risk_context=risk,
                        risk_missing=risk['missing']).bind()
