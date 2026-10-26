"""test_shadow_ui.py — 影子建议 UI 供给层验证（接线③；mock judge + 假 packs 文件，零网络）。

覆盖：可评分路径（含入影子日记）、缺包、硬门原因透出（复用 jev.gate_checks）、
供应商失败不炸主流程、证据包窗口外不可评分。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shadow_journal as sj
import shadow_ui
from oracles_v1 import pyxis_context, tarot_candidate

ASOF = '20261021110000'
QUERY = '20261021111200'
VU = '20261021230000'
CODE = '300639'
DATE = '20261021'


def pack_dict(risk=True) -> dict:
    cand = tarot_candidate(
        CODE, p_close_score=0.55, golden_zone=-0.12, f618=25.43,
        plan_dist_atr=0.31, atr_ref=0.407,
        downside_state=dict(approach=True, stop=True, divergence=True),
        trigger=dict(t='20261021111000', close=25.4),
        evidence_refs=dict(snapshot='snap'))
    risk_ctx = None if not risk else pyxis_context(
        version='232主版', amp_232=0.014, amp_266=0.013,
        market_state=dict(index='sse', trend='flat'),
        board_states=[dict(board='semi', strength='narrowing')],
        leader_states=[dict(code='688012', weakening=False)],
        missing=[], asof_ts=ASOF)
    return dict(evidence_id='ui1', asof_ts=ASOF, market_date=DATE,
                query_ts=QUERY, candidates=[cand], risk_context=risk_ctx,
                risk_missing=[] if risk else ['market_state'])


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


class BoomProvider(FakeProvider):
    def judge(self, request, *, timeout_s=30.0):
        self._last_out = {}
        raise ConnectionError('boom')


def setup(tmp: Path, lines=None, providers=None):
    sj.JOURNAL = tmp / 'journal.jsonl'
    sj.BUDGET = tmp / 'budget.json'
    sj.LOCK = tmp / 'budget.lock'
    sj.WINDOW = (20000101, 29991231)          # 测试时钟任意，窗口放开
    sj._PROVIDERS = {'deepseek': FakeProvider, 'kimi': FakeProvider}
    if providers:
        sj._PROVIDERS.update(providers)
    packs = tmp / 'shadow_packs.jsonl'
    if lines is not None:
        packs.write_text('\n'.join(json.dumps(x, ensure_ascii=False)
                                   for x in lines) + '\n', encoding='utf-8')
    shadow_ui.PACKS_FILE = packs
    shadow_ui._clear_cache()


def test_scorable_path(tmp_path):
    setup(tmp_path, lines=[pack_dict()])
    out = shadow_ui.get_shadow_advice(CODE, DATE, valid_until_ts=VU)
    assert out['status'] == 'OK', out
    assert out['advice']['option'] == 'CONTINUE_OBSERVE'
    assert out['advice']['title'] == '继续观察'
    assert out['advice']['confidence'] == 0.5
    assert out['journal_ref'] and 'deepseek' in out['journal_ref']
    # 一切调用入影子日记：MANUAL + trigger 记录可查
    rows = [json.loads(l) for l in
            sj.JOURNAL.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1 and rows[0]['trigger'] == 'MANUAL'
    assert rows[0]['correctness'] in ('PASS', 'EXPIRED')
    assert rows[0]['code'] == CODE


def test_missing_pack(tmp_path):
    setup(tmp_path, lines=None)               # 无 packs 文件
    out = shadow_ui.get_shadow_advice(CODE, DATE)
    assert out['status'] == 'UNSCORABLE', out
    assert '缺证据包' in out['reason']
    assert not sj.JOURNAL.exists()            # 无包不写日记、不扣额度
    # 有文件但缺该候选行
    setup(tmp_path, lines=[pack_dict()])
    out = shadow_ui.get_shadow_advice('000001', DATE)
    assert out['status'] == 'UNSCORABLE' and '缺证据包' in out['reason']


def test_hard_gate_reason_surfaces(tmp_path):
    setup(tmp_path, lines=[pack_dict(risk=False)])
    out = shadow_ui.get_shadow_advice(CODE, DATE, valid_until_ts=VU)
    assert out['status'] == 'UNSCORABLE', out
    # 复用 jev.gate_checks 的 reason 透出（缺风险证据的具体原因）
    assert 'RISK_CONTEXT_EMPTY' in out['reason'], out['reason']
    assert out['advice'] is None
    rows = [json.loads(l) for l in
            sj.JOURNAL.read_text(encoding='utf-8').splitlines()]
    assert rows[0]['correctness'] == 'HARD_GATE'
    assert not sj.BUDGET.exists()             # 硬门短路：未尝试供应商，不耗额度


def test_vendor_failure_no_crash(tmp_path):
    setup(tmp_path, lines=[pack_dict()], providers={'deepseek': BoomProvider})
    out = shadow_ui.get_shadow_advice(CODE, DATE, valid_until_ts=VU)
    assert out['status'] == 'ERROR', out
    assert '本次建议不可用' in out['reason']
    assert out['advice'] is None and out['journal_ref']
    rows = [json.loads(l) for l in
            sj.JOURNAL.read_text(encoding='utf-8').splitlines()]
    assert rows[0]['correctness'] == 'ERROR'
    # 失败仍按实际尝试记账（shadow_journal 纪律），但主流程不受影响
    b = json.loads(sj.BUDGET.read_text(encoding='utf-8'))
    assert b['deepseek']['total'] == 1
    # 再次调用（换正常供应商）照常工作——不炸主进程
    sj._PROVIDERS['deepseek'] = FakeProvider
    out = shadow_ui.get_shadow_advice(CODE, DATE, valid_until_ts=VU)
    assert out['status'] == 'OK', out


def test_out_of_window_unscorable(tmp_path):
    setup(tmp_path, lines=[pack_dict()])
    sj.WINDOW = (20260101, 20260131)          # 证据包日期落在窗口外
    out = shadow_ui.get_shadow_advice(CODE, DATE, valid_until_ts=VU)
    assert out['status'] == 'UNSCORABLE', out
    assert '窗口' in out['reason']


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
            print('FAIL', n, type(exc).__name__, str(exc)[:300])
    print(f'== {len(fns) - bad}/{len(fns)} 项影子建议 UI 供给层验证通过 ==')
    sys.exit(1 if bad else 0)


def load_tests(loader, suite, pattern):
    """兼容 python -m unittest 发现：把裸函数包装进 TestCase。"""
    import unittest

    class FnCase(unittest.TestCase):
        def __init__(self, fn):
            super().__init__('runTest')
            self._fn = fn

        def runTest(self):
            with tempfile.TemporaryDirectory() as td:
                self._fn(Path(td))

        def shortDescription(self):
            return self._fn.__name__

    for n, f in sorted(globals().items()):
        if n.startswith('test_') and callable(f):
            suite.addTest(FnCase(f))
    return suite
