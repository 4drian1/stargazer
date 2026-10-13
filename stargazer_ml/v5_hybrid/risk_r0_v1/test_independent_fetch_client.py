"""test_independent_fetch_client.py — 零网络受控客户端测试（响应校验/计数/限速/重试预算）。

全部使用 mock 传输层；本文件运行不产生任何网络请求（请求单批准前置门槛）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_fetch_client import (FetchBudgetExceeded, FetchCounter,
                                      RawResponseRejected, TencentDailyClient,
                                      _validate_raw, case_windows)


def make_payload(symbol, key='day', n=3):
    return json.dumps({'data': {symbol: {key: [[f'2026-01-0{i+1}', 10, 10.5, 11, 9.5, 1000]
                                               for i in range(n)]}}}).encode()


def test_accept_raw_day():
    rows = _validate_raw(json.loads(make_payload('sh600000')), 'sh600000')
    assert len(rows) == 3


def test_reject_qfq():
    for bad in ('qfqday', 'hfqday'):
        try:
            _validate_raw(json.loads(make_payload('sh600000', key=bad)), 'sh600000')
            raise AssertionError(f'{bad} 未被拒绝')
        except RawResponseRejected:
            pass


def test_counter_hard_cap():
    c = FetchCounter(3)
    c.consume(1); c.consume(2)
    try:
        c.consume(1)
        raise AssertionError('超预算未被拦截')
    except FetchBudgetExceeded:
        pass


def test_fetch_counts_every_attempt():
    """失败重试每次都计数；hard_cap=3、2 次失败后第 3 次尝试直接预算拦截。"""
    calls = []

    def flaky(url, timeout):
        calls.append(url)
        raise ConnectionError('boom')

    c = FetchCounter(2)
    client = TencentDailyClient(c, max_retries=3, transport=flaky, sleep=lambda s: None)
    try:
        client.fetch_raw_daily('600000', '20260101', '20260110')
        raise AssertionError('应已失败')
    except FetchBudgetExceeded:
        pass
    assert len(calls) == 2, f'预算内尝试次数 {len(calls)} != 2（含重试计数）'


def test_retry_then_success_within_budget():
    state = {'n': 0}

    def flaky(url, timeout):
        state['n'] += 1
        if state['n'] < 3:
            raise TimeoutError()
        return make_payload('sz000001')

    c = FetchCounter(5)
    client = TencentDailyClient(c, max_retries=3, transport=flaky, sleep=lambda s: None)
    rows = client.fetch_raw_daily('000001', '20260101', '20260110')
    assert len(rows) == 3 and c.used == 3 and state['n'] == 3


def test_rate_limit_sleeps():
    sleeps = []

    def ok(url, timeout):
        return make_payload('sh600000')

    client = TencentDailyClient(FetchCounter(10), rate_limit_s=1.0,
                                transport=ok, sleep=sleeps.append)
    client._last_ts = 100.0  # 强制需要等待
    import independent_fetch_client as m
    real_time = m.time.time
    m.time.time = lambda: 100.2
    try:
        client.fetch_raw_daily('600000', '20260101', '20260110')
    finally:
        m.time.time = real_time
    assert sleeps and abs(sleeps[0] - 0.8) < 1e-6, f'限速等待 {sleeps}'


def test_case_windows_shape():
    wins = case_windows([dict(code='600733', P=20180919, W_end=20180927)])
    code, s, e = wins[0]
    assert code == '600733' and s == '20180912' and e == '20180927'


def test_full_batch_budget_25():
    """18 案例 + 2 冒烟 = 20 正常请求；hard_cap=25 给 5 次重试余量（与请求单 §5 一致）。"""
    c = FetchCounter(25)
    # 模拟 20 次成功调用（mock 不计真实网络）
    calls = []

    def ok(url, timeout):
        calls.append(1)
        return make_payload('sh600000')

    client = TencentDailyClient(c, transport=ok, sleep=lambda s: None)
    for _ in range(20):
        client.fetch_raw_daily('600000', '20260101', '20260110')
    assert c.used == 20 <= 25


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print(f'== {len(fns)} 项零网络测试全过 ==')
