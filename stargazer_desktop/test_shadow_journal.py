"""test_shadow_journal.py — 影子记录器 V1.1 验证（用户五项反例回归，零网络）。

覆盖：去重复用不占额度、原子额度+并发、实际尝试计数（失败也结算）、
事件类别限额、token/费用入账、过期单列、窗口外拒记。
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shadow_journal as sj
from oracles_v1 import EvidencePack, pyxis_context, tarot_candidate

ASOF = '20261021110000'
QUERY = '20261021111200'
VU = '20261021120000'


def pack(date='20261021', missing=False) -> EvidencePack:
    cand = tarot_candidate(
        '300639', p_close_score=0.55, golden_zone=-0.12, f618=25.43,
        plan_dist_atr=0.31, atr_ref=0.407,
        downside_state=dict(approach=True, stop=True, divergence=True),
        trigger=dict(t='20261021111000', close=25.4),
        evidence_refs=dict(snapshot='snap'))
    risk = None if missing else pyxis_context(
        version='232主版', amp_232=0.014, amp_266=0.013,
        market_state=dict(index='sse', trend='flat'),
        board_states=[dict(board='semi', strength='narrowing')],
        leader_states=[dict(code='688012', weakening=False)],
        missing=[], asof_ts=ASOF)
    return sj.jev.EvidencePack(
        evidence_id='t1', asof_ts=ASOF, market_date=date,
        query_ts=QUERY, candidates=[cand], risk_context=risk,
        risk_missing=['market_state'] if missing else []).bind()


class FakeProvider:
    USAGE = dict(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    def __init__(self, *a, **k):
        self._last_out = {}

    def judge(self, request, *, timeout_s=30.0):
        self._last_out = dict(provider='fake', latency_s=0.01,
                              usage=dict(self.USAGE))
        return dict(provider='fake', raw=json.dumps(dict(
            option='CONTINUE_OBSERVE', confidence=0.5, evidence_refs=['snap']),
            ensure_ascii=False), latency_s=0.01, usage=dict(self.USAGE))


def setup(tmp: Path):
    sj.JOURNAL = tmp / 'journal.jsonl'
    sj.BUDGET = tmp / 'budget.json'
    sj.LOCK = tmp / 'budget.lock'
    sj._PROVIDERS = {'deepseek': FakeProvider, 'kimi': FakeProvider}


def _budget(tmp):
    return json.loads((tmp / 'budget.json').read_text(encoding='utf-8'))


def test_out_of_window(tmp_path):
    setup(tmp_path)
    rows = sj.record_event(pack('20261019'), 'UNPOSITIONED', 'MANUAL',
                           valid_until_ts=VU)
    assert rows[0].get('error') == 'OUT_OF_WINDOW'
    assert not sj.JOURNAL.exists()


def test_hard_gate_no_attempt(tmp_path):
    setup(tmp_path)
    rows = sj.record_event(pack(missing=True), 'UNPOSITIONED', 'FIRST_TRIGGER',
                           valid_until_ts=VU)
    assert all(r['correctness'] == 'HARD_GATE' for r in rows)
    assert not (tmp_path / 'budget.json').exists()


def test_dedup_reuse_no_charge(tmp_path):
    setup(tmp_path)
    sj.record_event(pack(), 'UNPOSITIONED', 'MANUAL', valid_until_ts=VU)
    rows2 = sj.record_event(pack(), 'UNPOSITIONED', 'MANUAL',
                            valid_until_ts=VU)
    assert all(r['correctness'] == 'REUSED' for r in rows2), rows2
    b = _budget(tmp_path)
    assert b['deepseek']['total'] == 1 and b['kimi']['total'] == 1


def test_category_caps(tmp_path):
    setup(tmp_path)
    sj.DAILY_CAP = 99
    for _ in range(6):
        sj.record_event(pack(), 'UNPOSITIONED', 'FIRST_TRIGGER',
                        valid_until_ts=VU, event_id=f'e{_}')
    cats = [json.loads(l) for l in sj.JOURNAL.read_text(encoding='utf-8')
            .splitlines()]
    ds = [c for c in cats if c['vendor'] == 'deepseek']
    passed = sum(1 for c in ds if c['correctness'] == 'PASS')
    denied = sum(1 for c in ds if c['correctness'] == 'BUDGET_EXHAUSTED')
    assert passed == 5 and denied == 1, (passed, denied)
    # 未知事件类型拒记
    bad = sj.record_event(pack(), 'UNPOSITIONED', 'EVERY_MINUTE',
                          valid_until_ts=VU)
    assert bad[0].get('error') == 'UNKNOWN_TRIGGER'


def test_attempt_counted_on_failure(tmp_path):
    setup(tmp_path)

    class Boom(FakeProvider):
        def judge(self, request, *, timeout_s=30.0):
            self._last_out = {}
            raise ConnectionError('boom')

    sj._PROVIDERS = {'deepseek': Boom, 'kimi': FakeProvider}
    rows = sj.record_event(pack(), 'UNPOSITIONED', 'MANUAL', valid_until_ts=VU)
    ds = [r for r in rows if r['vendor'] == 'deepseek'][0]
    assert ds['correctness'] == 'ERROR'
    b = _budget(tmp_path)
    assert b['deepseek']['total'] == 1     # 失败也按实际尝试计数
    assert b['deepseek']['days'][sj._day()] == 1


def test_tokens_cost_and_expired(tmp_path):
    setup(tmp_path)
    rows = sj.record_event(pack(), 'UNPOSITIONED', 'MANUAL', valid_until_ts=VU)
    r = rows[0]
    assert r['tokens'] == 150 and 'usage' in r and 'cost' in r
    assert r['correctness'] == 'PASS'
    # 过期：valid_until 过 judge 校验，但消费时刻（补丁时钟）已晚于它 → EXPIRED 单列
    orig_now = sj._now14
    sj._now14 = lambda *a, **k: '20261021113500'
    try:
        rows2 = sj.record_event(pack(), 'UNPOSITIONED', 'MANUAL',
                                valid_until_ts='20261021113000', event_id='exp')
    finally:
        sj._now14 = orig_now
    assert any(x['correctness'] == 'EXPIRED' for x in rows2),         [(x['correctness'], x.get('gate_error')) for x in rows2]
    s = sj.summarize()
    assert s['deepseek'].get('EXPIRED', 0) >= 1
    assert s['deepseek']['tokens_total'] >= 150


def test_concurrent_budget_atomic(tmp_path):
    setup(tmp_path)
    sj.DAILY_CAP = 99

    def worker(n):
        for i in range(4):
            sj.record_event(pack(), 'UNPOSITIONED', 'MANUAL',
                            valid_until_ts=VU, event_id=f'w{n}_{i}')

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    b = _budget(tmp_path)
    assert b['deepseek']['total'] == 16, b['deepseek']
    tmp_budget = tmp_path / 'budget.json'
    raw = json.loads(tmp_budget.read_text(encoding='utf-8'))
    assert 'reserved' not in raw.get('deepseek', {}) or \
        raw['deepseek']['reserved'] == 0


if __name__ == '__main__':
    bad = 0
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_')]
    for n, f in fns:
        try:
            with tempfile.TemporaryDirectory() as td:
                f(Path(td))
            print('PASS', n)
        except Exception as exc:
            bad += 1
            print('FAIL', n, type(exc).__name__, str(exc)[:200])
    print(f'== {len(fns) - bad}/{len(fns)} 项影子记录器 V1.1 验证通过 ==')
    sys.exit(1 if bad else 0)
