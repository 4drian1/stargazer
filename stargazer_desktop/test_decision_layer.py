"""test_decision_layer.py — 决策层可替换接口离线验证（零网络：mock 供应商）。

验证：同一请求结构两家共用、硬门短路（不消耗供应商调用）、
代码校验统一作用于两家响应、盲评标签可解、费用延迟字段存在。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decision_layer as dl
import jev_adapter_v1 as jev
from oracles_v1 import EvidencePack, pyxis_context, tarot_candidate

ASOF = '20260709110000'
QUERY = '20260709111200'
VU = '20260709120000'


def pack(case: str) -> EvidencePack:
    trig = dict(t='20260709111000', close=25.4) if 'triggered' in case else None
    ds = dict(approach=True, stop=True, divergence=True) \
        if 'triggered' in case else \
        dict(approach=False, stop=False, divergence=False)
    weak = 'weakening' in case
    if 'missing' in case:
        risk = None
        rm = ['market_state']
    else:
        risk = pyxis_context(
            version='232主版', amp_232=0.014, amp_266=0.013,
            market_state=dict(index='sse', trend='flat'),
            board_states=[dict(board='semi', strength='narrowing')],
            leader_states=[dict(code='688012',
                                weakening=weak if 'triggered' in case else False)],
            missing=[], asof_ts=ASOF)
        rm = []
    cand = tarot_candidate(
        '300639', p_close_score=0.55, golden_zone=-0.12, f618=25.43,
        plan_dist_atr=0.31, atr_ref=0.407, downside_state=ds, trigger=trig,
        evidence_refs=dict(snapshot='snap'))
    return EvidencePack(evidence_id=case, asof_ts=ASOF,
                        market_date='20260709', query_ts=QUERY,
                        candidates=[cand], risk_context=risk,
                        risk_missing=rm).bind()


class MockProvider(dl.JudgeProvider):
    def __init__(self, name, payload):
        self.name = name
        self.payload = payload
        self.calls = 0

    def judge(self, request, *, timeout_s=30.0):
        self.calls += 1
        if 'hard_gate' in request:
            return dict(hard_gate=request['hard_gate'], provider=self.name)
        return dict(provider=self.name, raw=json.dumps(
            self.payload, ensure_ascii=False), latency_s=0.01,
            usage=dict(total_tokens=10))


def test_same_request_both_vendors():
    p = pack('case_triggered')
    req = dl.build_judge_request(p, 'UNPOSITIONED', None, None, VU)
    assert 'options' in req and req['options'] == jev.OPTIONS_UNPOSITIONED
    assert req['candidates'][0]['code'] == '300639'
    assert 'opportunity_explain' in req  # 同一结构


def test_hard_gate_shortcircuit_no_vendor_call():
    p = pack('case_missing')
    prov = MockProvider('Fake', {})
    recs = dl.run_case(p, 'UNPOSITIONED', [prov], valid_until_ts=VU)
    assert recs[0].get('gate') == 'HARD' and recs[0]['correctness'] == 'PASS'
    assert prov.calls == 0          # 本地硬门短路：未发送供应商请求（归因修正）


def test_time_gate_active_in_real_entry():
    """用户反例：查询11:12、触发15:00 → 真实联调入口（run_case/judge）必须拦。"""
    p = pack('case_triggered')
    p.query_ts = '20260709111200'
    p.candidates[0]['trigger'] = dict(t='20260709150000', close=25.4)
    prov = MockProvider('Fake', dict(option='READY_BUY_CONFIRM_HUMAN',
                                     confidence=0.6, evidence_refs=['snap']))
    recs = dl.run_case(p, 'UNPOSITIONED', [prov], valid_until_ts=VU)
    r = recs[0]
    assert r['correctness'] == 'FAIL' and 'TRIGGER_AFTER_QUERY' in r['gate_error'], r
    assert prov.calls == 0          # 前置校验拦截，未消耗供应商调用


def test_code_validation_applies_to_both():
    p = pack('case_triggered')
    good = MockProvider('V_A', dict(option='READY_BUY_CONFIRM_HUMAN',
                                    confidence=0.6, evidence_refs=['snap']))
    bad = MockProvider('V_B', dict(option='READY_BUY_CONFIRM_HUMAN',
                                   confidence=float('nan'),
                                   evidence_refs=['snap']))
    recs = dl.run_case(p, 'UNPOSITIONED', [good, bad], valid_until_ts=VU)
    by = {r['provider']: r for r in recs}
    assert by['V_A']['correctness'] == 'PASS'
    assert by['V_B']['correctness'] == 'FAIL' and 'CONFIDENCE' in \
        by['V_B']['gate_error']


def test_buy_gate_blocks_untriggered():
    p = pack('case_normal')
    prov = MockProvider('V_A', dict(option='READY_BUY_CONFIRM_HUMAN',
                                    confidence=0.6, evidence_refs=['snap']))
    recs = dl.run_case(p, 'UNPOSITIONED', [prov], valid_until_ts=VU)
    assert recs[0]['correctness'] == 'FAIL'
    assert 'OPPORTUNITY_GATE' in recs[0]['gate_error']


def test_conflicting_case_shape():
    p = pack('case_triggered_weakening')
    req = dl.build_judge_request(p, 'UNPOSITIONED', None, None, VU)
    assert req['risk_context']['leader_states'][0]['weakening'] is True
    assert req['query_ts'] == QUERY


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
    print(f'== {len(fns) - bad}/{len(fns)} 项决策层离线验证通过 ==')
    sys.exit(1 if bad else 0)
