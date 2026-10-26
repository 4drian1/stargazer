"""test_oracles_v1.py — Stargazer 1.0 外包层 V1.2 验证（两轮用户反例回归）。

覆盖：V1.1 五项 + 第二轮六项（极性/单位/注入键/evidence_ids 校验、占位绕过、
asof 缺失、买入机会门、valid_until 强解析、线程不累积、零值合法性）。
"""
import json
import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracles_v1 import (EvidencePack, build_pack_from_research,
                        explain_opportunity, explain_risk,
                        pyxis_context, tarot_candidate)
from jev_adapter_v1 import (JEVNotConfigured, gate_checks, judge, mock_judge,
                            OPTIONS_UNPOSITIONED, opportunity_triggered)
import oracles_v1  # noqa: E402  (V1.3 内部计数检查)

ASOF = '20260709150000'
VU = '20260709160000'


def pack_full() -> EvidencePack:
    cand = tarot_candidate(
        '300639', p_close_score=0.55, golden_zone=-0.12, f618=25.43,
        plan_dist_atr=0.31, atr_ref=0.407,
        downside_state=dict(approach=True, stop=True, divergence=True),
        trigger=dict(t='20260709150000', close=25.48),
        evidence_refs=dict(snapshot='pday_snapshot_v2'))
    risk = pyxis_context(
        version='232主版', amp_232=0.0146, amp_266=0.0141,
        market_state=dict(index='sse', trend='flat'),
        board_states=[dict(board='semiconductor', breadth='narrowing')],
        leader_states=[dict(code='688012', weakening=False)],
        missing=[], asof_ts=ASOF)
    return EvidencePack(evidence_id='test1', asof_ts=ASOF,
                        market_date='20260709', query_ts=ASOF,
                        candidates=[cand],
                        risk_context=risk, risk_missing=[]).bind()


def ok_client(messages):
    return json.dumps(dict(
        conclusions=[dict(
            claim='下杀、止跌、背离均成立',
            facts=[dict(ref='pday_snapshot_v2',
                        field='cand:300639:downside_state', unit='state')],
            evidence_ids=['pday_snapshot_v2'])],
        invalidations=[dict(
            claim='跌回关键位下方且无止跌',
            facts=[dict(ref='pday_snapshot_v2',
                        field='cand:300639:f618', unit='price')])],
        notes='无新增'), ensure_ascii=False)


def claim_with_number(messages):
    return json.dumps(dict(
        conclusions=[dict(claim='评分为25.43',
                          facts=[dict(ref='pday_snapshot_v2',
                                      field='cand:300639:p_close_score',
                                      unit='score')])], notes=''), ensure_ascii=False)


def empty_payload(messages):
    return '{}'


def fake_ref_payload(messages):
    return json.dumps(dict(
        conclusions=[dict(claim='背离已成立',
                          facts=[dict(ref='FAKE_REFERENCE',
                                      field='cand:300639:downside_state',
                                      unit='state')])], notes=''), ensure_ascii=False)


def polarity_lie_payload(messages):
    """divergence 实为 False（pack_full_undiv），声称成立。"""
    return json.dumps(dict(
        conclusions=[dict(claim='背离已成立',
                          facts=[dict(ref='pday_snapshot_v2',
                                      field='cand:300639:downside_state',
                                      unit='state')])], notes=''), ensure_ascii=False)


def injected_value_payload(messages):
    return json.dumps(dict(
        conclusions=[dict(claim='背离已成立',
                          facts=[dict(ref='pday_snapshot_v2',
                                      field='cand:300639:downside_state',
                                      unit='state', value=True)])],
        notes=''), ensure_ascii=False)


def amp_as_probability_payload(messages):
    return json.dumps(dict(
        conclusions=[dict(claim='这是亏损概率',
                          facts=[dict(ref='e589ea52c60c1a14',
                                      field='risk:amp_232',
                                      unit='probability')])], notes=''), ensure_ascii=False)


def forged_evid_payload(messages):
    return json.dumps(dict(
        conclusions=[dict(claim='触发有效',
                          facts=[dict(ref='pday_snapshot_v2',
                                      field='cand:300639:downside_state',
                                      unit='state')],
                          evidence_ids=['FORGED_EVIDENCE'])],
        notes=''), ensure_ascii=False)


def slow_client(messages):
    time.sleep(0.081)
    return ok_client(messages)


def pack_undiv() -> EvidencePack:
    p = pack_full()
    p.candidates[0]['downside_state']['divergence'] = False
    return p


def test_ok_with_citations():
    r = explain_opportunity(pack_full(), '评估300639', ok_client)
    assert r['status'] == 'OK' and r['payload']['certified'] is True, r
    assert explain_risk(pack_full(), 'x', ok_client)['status'] == 'OK'


def test_c1_claim_digits_rejected():
    r = explain_opportunity(pack_full(), 'x', claim_with_number)
    assert r['status'] == 'SCHEMA_REJECT'
    assert any('CLAIM_HAS_DIGIT' in e for e in r['errors'])


def test_c2_structure_refs_and_injection():
    assert explain_opportunity(pack_full(), 'x', empty_payload)['status'] in (
        'SCHEMA_REJECT', 'PARSE_FAIL')
    r = explain_opportunity(pack_full(), 'x', fake_ref_payload)
    assert r['status'] == 'SCHEMA_REJECT' and any(
        'UNKNOWN_REF' in e for e in r['errors'])
    r2 = explain_opportunity(pack_full(), 'x', forged_evid_payload)
    assert r2['status'] == 'SCHEMA_REJECT' and any(
        'UNKNOWN_EVIDENCE_ID' in e for e in r2['errors'])
    r3 = explain_opportunity(pack_full(), 'x', injected_value_payload)
    assert r3['status'] == 'SCHEMA_REJECT' and any(
        'EXTRA_KEYS' in e for e in r3['errors'])


def test_polarity_lie_rejected():
    r = explain_opportunity(pack_undiv(), 'x', polarity_lie_payload)
    assert r['status'] == 'SCHEMA_REJECT', r
    assert any('POLARITY_MISMATCH' in e for e in r['errors'])


def test_amp_not_probability():
    r = explain_risk(pack_full(), 'x', amp_as_probability_payload)
    assert r['status'] == 'SCHEMA_REJECT', r
    assert any(e.endswith('_BAD_UNIT') or 'AMPLITUDE_AS_PROBABILITY' in e
               for e in r['errors'])


def test_c3_real_timeout_and_bounded_threads():
    before = len([t for t in threading.enumerate()
                  if t.name.startswith('oracles')])
    rs = [explain_opportunity(pack_full(), 'x', slow_client, timeout_s=0.001)
          for _ in range(5)]
    assert all(r['status'] == 'UNAVAILABLE' and r['reason'] == 'TIMEOUT'
               for r in rs), rs
    time.sleep(0.3)
    after = len([t for t in threading.enumerate()
                 if t.name.startswith('oracles')])
    assert after <= 2, f'oracles线程累积: {before}->{after}'
    assert explain_opportunity(pack_full(), 'x', slow_client,
                               timeout_s=1.0)['status'] == 'OK'


def test_c4_placeholder_and_asof_gate():
    p = pack_full()
    p.risk_context = {'version': '232主版',
                      'amp_232': 0.01, 'asof_ts': ASOF,
                      'market_state': {'trend': None},
                      'board_states': [{}], 'leader_states': []}
    adv = gate_checks(p, 'UNPOSITIONED')
    assert adv and adv.option == 'EVIDENCE_INSUFFICIENT'
    for frag in ('MARKET_STATE_UNKNOWN', 'BOARD_STATES_UNKNOWN',
                 'LEADER_STATES_UNKNOWN'):
        assert frag in adv.notes
    # 名称/代码占位（状态全None）同样拦截
    p_n = pack_full()
    p_n.risk_context['market_state'] = {'index': 'sse', 'trend': None}
    p_n.risk_context['board_states'] = [{'board': 'semi', 'strength': None}]
    p_n.risk_context['leader_states'] = [{'code': '688012', 'weakening': None}]
    adv_n = gate_checks(p_n, 'UNPOSITIONED')
    assert adv_n and adv_n.option == 'EVIDENCE_INSUFFICIENT'
    # 必要缺失清单参与判断：登记市场缺失→不充分（占位不能抵消）
    p_d = pack_full()
    p_d.risk_missing = ['market_state']
    adv_d = gate_checks(p_d, 'UNPOSITIONED')
    assert adv_d and 'REQUIRED_DECLARED_MISSING' in adv_d.notes
    # 合法False=已知状态：weakening False 不应拦截
    assert gate_checks(pack_full(), 'UNPOSITIONED') is None
    # asof 缺失（默认补包时间=已修）
    p2 = pack_full()
    del p2.risk_context['asof_ts']
    adv2 = gate_checks(p2, 'UNPOSITIONED')
    assert adv2 and 'RISK_ASOF_MISSING' in adv2.notes
    # 合法零值幅度放行（反例6）
    p3 = pack_full()
    p3.risk_context['amp_232'] = 0.0
    assert gate_checks(p3, 'UNPOSITIONED') is None


def test_future_trigger_blocked():
    p = pack_full()
    p.candidates[0]['trigger'] = dict(t='20260710100000', close=25.5)  # 次日
    assert not opportunity_triggered(p, ASOF)
    bad = dict(option='READY_BUY_CONFIRM_HUMAN', confidence=0.6,
               evidence_refs=['test1'])
    try:
        judge(p, 'UNPOSITIONED', api_caller=lambda req: bad,
              valid_until_ts=VU)
        raise AssertionError('未来触发用于当日确认应拒')
    except ValueError as e:
        assert ('OPPORTUNITY_GATE' in str(e)) or ('TRIGGER_AFTER_QUERY' in str(e))


def test_backfilled_snapshot_blocked():
    """快照晚于触发=用未来风险数据回填，必须拦截（真实300639场景回归）。"""
    from jev_adapter_v1 import opportunity_triggered as _ot
    p = pack_full()
    # asof 11:00、trigger 10:55 → 快照晚于触发（正常是 asof≤t）
    p.asof_ts = '20260709110000'
    p.candidates[0]['trigger'] = dict(t='20260709105500', close=25.4)
    assert not _ot(p, p.asof_ts)
    # asof 11:00、trigger 11:10 → 快照早于触发（合法：最近已完成快照）
    p2 = pack_full()
    p2.asof_ts = '20260709110000'
    p2.candidates[0]['trigger'] = dict(t='20260709111000', close=25.4)
    assert _ot(p2, p2.asof_ts)


def test_forged_explain_rejected():
    p = pack_full()
    forged = dict(status='OK', payload=dict(
        certified=True, conclusions=[dict(claim='任意')]))
    seen = {}

    def spy(req):
        seen.update(req)
        return dict(option='CONTINUE_OBSERVE', confidence=0.5)

    judge(p, 'UNPOSITIONED', opportunity_explain=forged, api_caller=spy,
          valid_until_ts=VU)
    assert seen.get('opportunity_explain') is None


def test_queue_drains_and_slots_follow_completion():
    import oracles_v1 as ov
    rs = [explain_opportunity(pack_full(), 'x', slow_client, timeout_s=0.001)
          for _ in range(12)]
    assert all(r['status'] == 'UNAVAILABLE' for r in rs)
    time.sleep(0.4)
    assert ov._q_get() == 0, f'内部计数未归零: {ov._q_get()}'
    after = len([t for t in threading.enumerate()
                 if t.name.startswith('oracles')])
    assert after <= 2, f'线程累积: {after}'


def test_three_clock_semantics():
    """用户精确反例：风险截止/触发/查询三时刻分开校验。
    场景A：风险11:00、触发15:00、查询11:12、有效期11:30 → 触发尚未发生，拒。
    场景B（重放）：风险11:00、触发11:10、查询11:12 → 合法。
    场景C：有效期≤查询时刻 → 拒。"""
    p = pack_full()
    p.risk_context['asof_ts'] = '20260709110000'
    p.asof_ts = '20260709110000'
    p.candidates[0]['trigger'] = dict(t='20260709150000', close=25.4)
    p.query_ts = '20260709111200'
    try:
        judge(p, 'UNPOSITIONED',
              api_caller=lambda req: dict(
                  option='READY_BUY_CONFIRM_HUMAN', confidence=0.6,
                  evidence_refs=['test1']),
              valid_until_ts='20260709113000')
        raise AssertionError('触发晚于查询应拒')
    except ValueError as e:
        assert 'TRIGGER_AFTER_QUERY' in str(e)
    # 场景B：重放合法
    p2 = pack_full()
    p2.risk_context['asof_ts'] = '20260709110000'
    p2.asof_ts = '20260709110000'
    p2.candidates[0]['trigger'] = dict(t='20260709111000', close=25.4)
    p2.query_ts = '20260709111200'
    ok = judge(p2, 'UNPOSITIONED',
               api_caller=lambda req: dict(
                   option='READY_BUY_CONFIRM_HUMAN', confidence=0.5,
                   evidence_refs=['test1']),
               valid_until_ts='20260709120000')
    assert ok.option == 'READY_BUY_CONFIRM_HUMAN'
    assert not ok.is_expired('20260709111200')
    assert ok.is_expired('20260709120001')   # 消费侧过期复核
    # 场景C：有效期不晚于查询
    try:
        judge(p2, 'UNPOSITIONED',
              api_caller=lambda req: dict(option='CONTINUE_OBSERVE',
                                          confidence=0.5),
              valid_until_ts='20260709111200')
        raise AssertionError('有效期≤查询应拒')
    except ValueError as e:
        assert 'VALID_UNTIL' in str(e)


def test_state_claim_price_only_rejected():
    """用户精确反例：背离=False，声称"背离已成立"，只引用F618价格字段。"""
    pu = pack_undiv()
    def client(msgs):
        return json.dumps(dict(
            conclusions=[dict(claim='背离已成立',
                facts=[dict(ref='pday_snapshot_v2',
                            field='cand:300639:f618', unit='price')])],
            notes=''), ensure_ascii=False)
    r = explain_opportunity(pu, 'x', client)
    assert r['status'] == 'SCHEMA_REJECT', r
    assert any('STATE_CLAIM_WITHOUT_STATE_FACT' in e for e in r['errors'])


def test_c5_buy_gate_and_valid_until():
    p = pack_full()
    p.candidates[0]['trigger'] = None
    assert not opportunity_triggered(p)
    bad = dict(option='READY_BUY_CONFIRM_HUMAN', confidence=0.6,
               evidence_refs=['test1'])
    try:
        judge(p, 'UNPOSITIONED', api_caller=lambda req: bad,
              valid_until_ts=VU)
        raise AssertionError('无触发却给买入确认应拒')
    except ValueError as e:
        assert 'OPPORTUNITY_GATE' in str(e)
    # garbage valid_until
    try:
        judge(pack_full(), 'UNPOSITIONED',
              api_caller=lambda req: dict(option='CONTINUE_OBSERVE',
                                          confidence=0.5),
              valid_until_ts='garbage')
        raise AssertionError('garbage valid_until 应拒')
    except ValueError as e:
        assert 'VALID_UNTIL' in str(e)
    # 合法：触发在位 + 合法valid_until + 零置信度允许（反例6）
    ok = judge(pack_full(), 'UNPOSITIONED',
               api_caller=lambda req: dict(
                   option='READY_BUY_CONFIRM_HUMAN', confidence=0.0,
                   evidence_refs=['test1']), valid_until_ts=VU)
    assert ok.option == 'READY_BUY_CONFIRM_HUMAN' and ok.confidence == 0.0


def test_untrusted_explain_not_forwarded():
    seen = {}

    def spy_caller(req):
        seen.update(req)
        return dict(option='CONTINUE_OBSERVE', confidence=0.5)

    untrusted = dict(status='OK', payload=dict(certified=False))
    judge(pack_full(), 'UNPOSITIONED', opportunity_explain=untrusted,
          api_caller=spy_caller, valid_until_ts=VU)
    assert seen.get('opportunity_explain') is None


def test_jev_not_configured_and_mock_derives():
    try:
        judge(pack_full(), 'UNPOSITIONED', valid_until_ts=VU)
        raise AssertionError('应抛 JEVNotConfigured')
    except JEVNotConfigured:
        pass
    a = mock_judge(pack_full(), 'UNPOSITIONED')          # 触发由包推导
    assert a.option == 'READY_BUY_CONFIRM_HUMAN'
    p2 = pack_full()
    p2.candidates[0]['trigger'] = None
    a2 = mock_judge(p2, 'UNPOSITIONED')                  # 无触发→降级观察
    assert a2.option == 'CONTINUE_OBSERVE'
    p3 = pack_full()
    p3.risk_context['leader_states'] = [dict(code='X', weakening=True)]
    a3 = mock_judge(p3, 'UNPOSITIONED')                  # 走弱由包推导→暂缓
    assert a3.option == 'DEFER'


def test_version_hash_binding():
    try:
        pyxis_context(version='当日挑选版', amp_232=0.1, amp_266=0.1,
                      market_state=dict(a=1), board_states=[dict(b=1)],
                      leader_states=[dict(c=1)], missing=[], asof_ts=ASOF)
        raise AssertionError('非法版本应拒')
    except ValueError as e:
        assert 'NOT_FROZED' in str(e)
    p = pack_full()
    p.risk_context['model_sha256'] = 'tampered'
    adv = gate_checks(p, 'UNPOSITIONED')
    assert adv and 'PYXIS_HASH_MISMATCH' in adv.notes


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception as exc:
            bad += 1
            print('FAIL', fn.__name__, type(exc).__name__, str(exc)[:150])
    print(f'== {len(fns) - bad}/{len(fns)} 项 V1.2 反例验证通过 ==')
    sys.exit(1 if bad else 0)
