"""jev_adapter_v1.py — JEV 综合判断适配器（Stargazer 1.0；接口 V1.2 反例修订）。

V1.2 修订：
- **占位绕过修复**：必需叶子字段深校验（None/空串/空dict/空list 不算有效叶子）；
  market/boards/leaders 各有≥1个有效叶子才放行；risk_context 必须自带 asof_ts
  且与包一致（缺失时间不再默认补成包时间）。按场景定义必需字段，可选缺失不阻断。
- **买入确认机会门**：READY_BUY_CONFIRM_HUMAN 需代码侧冻结的机会触发条件全部
  成立（trigger 在场且 下杀/止跌/背离 全真），否则响应被拒/模拟降级；
  REDUCE_EXIT 同理要求已持仓场景（scenario 已约束）。
- **valid_until 严格解析**：14位时间戳强解析，必须晚于 asof；消费时再次核对。
- **解释信任边界**：传入的解释若非 status=='OK'（含 certified==False），
  不作为可信解释传入请求（置 None 并注明）。
- 幅度/置信度允许合法 0：幅度=有限且≥0；置信度=有限且0≤c≤1。
- mock_judge 改为**从证据包推导**状态（触发/走弱），布尔参数仅测试用。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from oracles_v1 import PYXIS_FROZEN, EvidencePack, _vtoken

SCHEMA = 'STARGAZER_JEV_V1_4'

OPTIONS_UNPOSITIONED = ('CONTINUE_OBSERVE',
                        'READY_BUY_CONFIRM_HUMAN',
                        'DEFER',
                        'EVIDENCE_INSUFFICIENT')
OPTIONS_POSITIONED = ('HOLD_OBSERVE',
                      'HEIGHTEN_RISK_WATCH',
                      'REDUCE_EXIT_CONFIRM_HUMAN',
                      'EVIDENCE_INSUFFICIENT')

_TS14 = re.compile(r'^\d{14}$')


class JEVNotConfigured(RuntimeError):
    pass


@dataclass
class JEVAdvice:
    scenario: str
    option: str
    confidence: float
    supporting: list[str] = field(default_factory=list)
    opposing: list[str] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    invalidations: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    valid_until_ts: str = ''
    information_gap: bool = True
    notes: str = ''

    def is_expired(self, now_ts: str) -> bool:
        """消费侧按实际时刻复核有效期（返回后检查）。"""
        now, vu = _parse_ts14(str(now_ts)), _parse_ts14(str(self.valid_until_ts))
        return now is None or vu is None or now > vu

    def as_dict(self) -> dict:
        return dict(schema=SCHEMA, scenario=self.scenario, option=self.option,
                    confidence=self.confidence, supporting=self.supporting,
                    opposing=self.opposing, watch=self.watch,
                    invalidations=self.invalidations,
                    evidence_refs=self.evidence_refs,
                    valid_until_ts=self.valid_until_ts,
                    information_gap=self.information_gap, notes=self.notes)


def _parse_ts14(s) -> datetime | None:
    if not isinstance(s, str) or not _TS14.match(s):
        return None
    try:
        return datetime.strptime(s, '%Y%m%d%H%M%S')
    except ValueError:
        return None


# 各上下文组的必需状态字段：名称/代码类键不算证据（反例3）
_REQUIRED_LEAVES = {
    'market_state': ('trend', 'state', 'direction', 'strength', 'breadth'),
    'board_states': ('breadth', 'strength', 'state', 'direction', 'weakening',
                     'narrowing'),
    'leader_states': ('weakening', 'strength', 'state', 'drawdown',
                      'retreat'),
}
_MISSING_GROUPS = {'market_state', 'board_states', 'leader_states',
                   'amp_232', 'amp_266'}


def _known(v) -> bool:
    """已知状态：False 也是已知；None/空串不算。"""
    if v is None:
        return False
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    if isinstance(v, dict):
        return any(_known(x) for x in v.values())
    return True


def _group_has_state(group_key, obj) -> bool:
    leaves = _REQUIRED_LEAVES[group_key]
    items = obj if isinstance(obj, list) else [obj]
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(_known(item.get(k)) for k in leaves):
            return True
    return False

def _finite_ge0(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x) and x >= 0


def opportunity_triggered(pack: EvidencePack,
                          asof: str | None = None) -> bool:
    """代码侧冻结的机会触发条件：trigger 在场、三态全真、触发时刻属于当前执行日，
    且风险快照不晚于触发时刻（asof≤t：用最近已完成快照，禁止未来数据回填）。"""
    asof = asof or str(pack.asof_ts)
    asof_dt = _parse_ts14(asof)
    if asof_dt is None:
        return False
    if not pack.candidates:
        return False
    for c in pack.candidates:
        trig = c.get('trigger')
        if not isinstance(trig, dict) or not trig.get('t'):
            return False
        t_dt = _parse_ts14(str(trig['t']))
        if t_dt is None or t_dt < asof_dt or                 str(trig['t'])[:8] != asof[:8]:
            return False
        ds = c.get('downside_state') or {}
        if not all(ds.get(k) is True for k in ('approach', 'stop', 'divergence')):
            return False
    return True


def risk_context_weakening(pack: EvidencePack) -> bool:
    rc = pack.risk_context or {}
    for grp in ('leader_states', 'board_states'):
        for item in rc.get(grp) or []:
            if isinstance(item, dict) and item.get('weakening') is True:
                return True
    return False


def _risk_evidence_incomplete(pack: EvidencePack) -> list[str]:
    reasons = []
    rc = pack.risk_context
    if not isinstance(rc, dict) or not rc:
        return ['RISK_CONTEXT_EMPTY']
    ver = rc.get('version')
    if ver not in PYXIS_FROZEN:
        reasons.append('PYXIS_VERSION_UNBOUND')
    else:
        fz = PYXIS_FROZEN[ver]
        if rc.get('model_sha256') != fz['model_sha256'] or \
                rc.get('preprocessing_sha256') != fz['preprocessing_sha256']:
            reasons.append('PYXIS_HASH_MISMATCH')
    amp_main = rc.get('amp_232') if ver == '232主版' else rc.get('amp_266')
    if not _finite_ge0(amp_main):
        reasons.append('AMP_INVALID')
    if not _group_has_state('market_state', rc.get('market_state')):
        reasons.append('MARKET_STATE_UNKNOWN')
    if not _group_has_state('board_states', rc.get('board_states') or []):
        reasons.append('BOARD_STATES_UNKNOWN')
    if not _group_has_state('leader_states', rc.get('leader_states') or []):
        reasons.append('LEADER_STATES_UNKNOWN')
    # 必要缺失清单参与判断：登记了关键组缺失即不充分（占位不能抵消）
    declared = set(pack.risk_missing or []) & _MISSING_GROUPS
    if declared:
        reasons.append('REQUIRED_DECLARED_MISSING:' + ','.join(sorted(declared)))
    rc_asof = _parse_ts14(str(rc.get('asof_ts', '')))
    pack_asof = _parse_ts14(str(pack.asof_ts))
    if rc_asof is None:
        reasons.append('RISK_ASOF_MISSING')
    elif pack_asof is not None and rc_asof != pack_asof:
        reasons.append('RISK_ASOF_MISMATCH')
    return reasons


def gate_checks(pack: EvidencePack, scenario: str) -> JEVAdvice | None:
    if scenario not in ('UNPOSITIONED', 'POSITIONED'):
        raise ValueError('UNKNOWN_SCENARIO')
    reasons = _risk_evidence_incomplete(pack)
    if reasons:
        return JEVAdvice(scenario=scenario, option='EVIDENCE_INSUFFICIENT',
                         confidence=0.0,
                         opposing=['风险证据缺口: ' + ','.join(reasons)],
                         watch=['补齐后重估'], evidence_refs=[pack.evidence_id],
                         notes='HARD_GATE: ' + '|'.join(reasons))
    return None


def _trusted(explain: dict | None, pack: EvidencePack) -> dict | None:
    """入口重验：自报 status/certified 一律不信；重算 vtoken 与包哈希绑定。"""
    if not isinstance(explain, dict):
        return None
    payload = explain.get('payload')
    if not isinstance(payload, dict):
        return None
    if payload.get('certified') is not True:
        return None
    if payload.get('_pack_hash') != pack.pack_hash:
        return None
    if payload.get('_vtoken') != _vtoken(
            {k: v for k, v in payload.items()
             if not k.startswith('_')}, pack.pack_hash):
        return None
    return payload


def judge(pack: EvidencePack, scenario: str,
          opportunity_explain: dict | None = None,
          risk_explain: dict | None = None,
          api_caller=None, valid_until_ts='') -> JEVAdvice:
    gate = gate_checks(pack, scenario)
    if gate is not None:
        return gate
    # 三时刻分开校验（不共用 asof）：
    # ①风险特征截止≤触发（opportunity_triggered 内，防回填）
    # ②触发≤本次查询时刻（触发必须已发生）
    # ③建议有效期>本次查询时刻
    query_dt = _parse_ts14(str(pack.query_ts))
    if query_dt is None:
        raise ValueError('JEV_QUERY_TS_INVALID')
    for c in pack.candidates:
        trig = (c.get('trigger') or {}).get('t')
        t_dt = _parse_ts14(str(trig)) if trig else None
        if t_dt is not None and t_dt > query_dt:
            raise ValueError('JEV_TRIGGER_AFTER_QUERY')
    vu_dt = _parse_ts14(str(valid_until_ts))
    if vu_dt is None or vu_dt <= query_dt:
        raise ValueError('JEV_VALID_UNTIL_INVALID')
    if api_caller is None:
        raise JEVNotConfigured('JEV API 未配置：等待供应商接口')
    resp = api_caller(_build_request(
        pack, scenario, _trusted(opportunity_explain, pack),
        _trusted(risk_explain, pack), valid_until_ts))
    advice = _validate_response(resp, pack, scenario, valid_until_ts)
    advice_pack_hash = pack.pack_hash
    advice.notes = (advice.notes + f' | pack_hash={advice_pack_hash}')[:300]
    return advice


def _build_request(pack: EvidencePack, scenario: str, opp, risk, valid_until):
    return dict(schema=SCHEMA, scenario=scenario, asof_ts=pack.asof_ts,
                query_ts=pack.query_ts, market_date=pack.market_date,
                evidence_id=pack.evidence_id,
                candidates=pack.candidates, risk_context=pack.risk_context,
                risk_missing=pack.risk_missing, corvus_gap=pack.corvus_gap,
                confirmed_plan=pack.confirmed_plan,
                opportunity_explain=opp, risk_explain=risk,
                valid_until_ts=valid_until,
                legal_refs=sorted(pack.ref_registry()),
                options=(OPTIONS_UNPOSITIONED if scenario == 'UNPOSITIONED'
                         else OPTIONS_POSITIONED))


def _validate_response(resp, pack: EvidencePack, scenario: str,
                       valid_until_ts: str) -> JEVAdvice:
    if not isinstance(resp, dict):
        raise ValueError('JEV_RESPONSE_NOT_DICT')
    opts = (OPTIONS_UNPOSITIONED if scenario == 'UNPOSITIONED'
            else OPTIONS_POSITIONED)
    if resp.get('option') not in opts:
        raise ValueError('JEV_RESPONSE_INVALID_OPTION')
    if resp['option'] == 'READY_BUY_CONFIRM_HUMAN' and \
            not opportunity_triggered(pack):
        raise ValueError('JEV_OPPORTUNITY_GATE')
    conf = resp.get('confidence')
    if not (isinstance(conf, (int, float)) and not isinstance(conf, bool)
            and math.isfinite(conf) and 0.0 <= conf <= 1.0):
        raise ValueError('JEV_CONFIDENCE_INVALID')
    for key in ('supporting', 'opposing', 'watch', 'invalidations',
                'evidence_refs'):
        if key in resp and not _str_list(resp[key]):
            raise ValueError(f'JEV_FIELD_TYPE:{key}')
    refs = resp.get('evidence_refs') or [pack.evidence_id]
    registry = pack.ref_registry()
    if any(str(r) not in registry for r in refs):
        raise ValueError('JEV_UNKNOWN_EVIDENCE_REF')
    return JEVAdvice(
        scenario=scenario, option=resp['option'], confidence=float(conf),
        supporting=list(resp.get('supporting', [])),
        opposing=list(resp.get('opposing', [])),
        watch=list(resp.get('watch', [])),
        invalidations=list(resp.get('invalidations', [])),
        evidence_refs=[str(r) for r in refs],
        valid_until_ts=valid_until_ts, information_gap=True,
        notes=str(resp.get('notes', ''))[:300])


OPTION_DISPLAY = {
    'CONTINUE_OBSERVE': ('继续观察', '等待哪些条件'),
    'READY_BUY_CONFIRM_HUMAN': ('条件已满足', '提交人工买入确认'),
    'DEFER': ('暂缓计划', '反证是什么、何时重新评估'),
    'HOLD_OBSERVE': ('继续持有观察', '等待哪些条件'),
    'HEIGHTEN_RISK_WATCH': ('加强风险关注', '关注哪些信号'),
    'REDUCE_EXIT_CONFIRM_HUMAN': ('风险转弱', '提交人工减仓/退出确认'),
    'EVIDENCE_INSUFFICIENT': ('证据不足', '缺失什么、补齐后重估'),
}


def render_advice(advice) -> dict:
    """代码侧渲染展示（V1.4）：标题+下一步说明来自冻结映射，不依赖 LLM 文本。"""
    title, hint = OPTION_DISPLAY.get(advice.option, (advice.option, ''))
    return dict(option=advice.option, title=title, next_hint=hint,
                confidence=advice.confidence,
                wait_conditions=advice.watch,
                counterevidence=advice.opposing,
                supporting=advice.supporting,
                invalidations=advice.invalidations,
                evidence_refs=advice.evidence_refs,
                valid_until_ts=advice.valid_until_ts,
                information_gap=advice.information_gap)


def _str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


# —— 模拟判断器：状态从证据包推导（布尔参数仅供测试注入）——

def mock_judge(pack: EvidencePack, scenario: str, *,
               triggered: bool | None = None,
               board_leader_weakening: bool | None = None,
               risk_explain_status: str = 'OK') -> JEVAdvice:
    gate = gate_checks(pack, scenario)
    if gate:
        return gate
    if triggered is None:
        triggered = opportunity_triggered(pack)
    if board_leader_weakening is None:
        board_leader_weakening = risk_context_weakening(pack)
    if risk_explain_status != 'OK':
        return JEVAdvice(scenario=scenario, option='EVIDENCE_INSUFFICIENT',
                         confidence=0.0, opposing=['风险解释不可用'],
                         evidence_refs=[pack.evidence_id])
    if scenario == 'UNPOSITIONED':
        if triggered and board_leader_weakening:
            return JEVAdvice(scenario=scenario, option='DEFER', confidence=0.55,
                             supporting=['机会侧已触发'],
                             opposing=['板块及领涨股正在减弱'],
                             watch=['等待板块/领涨股企稳证据'],
                             evidence_refs=[pack.evidence_id])
        if triggered:
            return JEVAdvice(scenario=scenario,
                             option='READY_BUY_CONFIRM_HUMAN', confidence=0.6,
                             supporting=['机会侧触发成立', '风险证据完整'],
                             watch=['触发后回踩是否守住关键位'],
                             invalidations=['跌回关键位下方且无止跌'],
                             evidence_refs=[pack.evidence_id])
        return JEVAdvice(scenario=scenario, option='CONTINUE_OBSERVE',
                         confidence=0.5, watch=['等待下杀到位/止跌/背离'],
                         evidence_refs=[pack.evidence_id])
    if board_leader_weakening:
        return JEVAdvice(scenario=scenario,
                         option='REDUCE_EXIT_CONFIRM_HUMAN', confidence=0.55,
                         supporting=['风险上下文转弱'], watch=['反弹力度'],
                         evidence_refs=[pack.evidence_id])
    return JEVAdvice(scenario=scenario, option='HOLD_OBSERVE', confidence=0.5,
                     watch=['风险幅度变化'], evidence_refs=[pack.evidence_id])
