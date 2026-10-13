"""test_independent_fetch_client.py — 零网络受控客户端测试（响应校验/计数/限速/重试预算）。

全部使用 mock 传输层；本文件运行不产生任何网络请求（请求单批准前置门槛）。
"""
import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter, FetchBudgetExceeded, FetchCounter,
                                      RawResponseRejected, RedirectRefused, TencentDailyClient, _NoRedirect,
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

    def flaky(url, timeout, read_cap=10**9):
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

    def flaky(url, timeout, read_cap=10**9):
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

    def ok(url, timeout, read_cap=10**9):
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

    def ok(url, timeout, read_cap=10**9):
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


# ---------- REV3 反例（字节预算/重定向/原始留证/冒烟零重试） ----------

def test_oversize_response_rejected():
    """5,000,096 字节响应在 5MB 硬限下必须被拒（用户反例）：超限即 ByteBudgetExceeded。"""
    big = b'x' * 5_000_096
    bc = ByteCounter(5 * 1024 * 1024)
    client = TencentDailyClient(FetchCounter(10), bc,
                                transport=lambda u, t, rc: big, sleep=lambda s: None)
    try:
        client.fetch_raw_daily('600000', '20260101', '20260110')
        raise AssertionError('超字节响应被接受')
    except ByteBudgetExceeded:
        pass
    assert bc.used == len(big)  # 被拒前已按实际字节计入


def test_cumulative_bytes_trip():
    """小响应累计触线：3×2MB 在 5MB 硬限下第三次必停。"""
    # 合法 JSON 的 ~2.1MB 原始 day 序列（5MB 硬限下恰好 2 次成功、第 3 次触线）
    rows = [[f'2020-01-01', 10, 10, 11, 9.5, 1000] for _ in range(60000)]
    payload = json.dumps({'data': {'sh600000': {'day': rows}}}).encode()
    assert 2 * 1024 * 1024 < len(payload) < 5 * 1024 * 1024, len(payload)
    bc = ByteCounter(5 * 1024 * 1024)
    client = TencentDailyClient(FetchCounter(10), bc,
                                transport=lambda u, t, rc: payload,
                                sleep=lambda s: None)
    ok = 0
    try:
        for _ in range(5):
            client.fetch_raw_daily('600000', '20260101', '20260110')
            ok += 1
    except ByteBudgetExceeded:
        pass
    assert ok == 2, f'ok={ok} used={bc.used}'


def test_redirect_refused_no_retry():
    """重定向即停：RedirectRefused 原样抛出、不重试、计数器只有 1。"""
    c = FetchCounter(10)

    def redir(url, timeout, read_cap=10**9):
        raise RedirectRefused('重定向拒绝（HTTP 302）')

    client = TencentDailyClient(c, transport=redir, sleep=lambda s: None)
    try:
        client.fetch_raw_daily('600000', '20260101', '20260110')
        raise AssertionError('重定向未被拒绝')
    except RedirectRefused:
        pass
    assert c.used == 1


def test_no_redirect_handler_present():
    """默认传输挂 _NoRedirect：3xx 不跟随（隐藏请求永不发生）。"""
    from independent_fetch_client import _NoRedirect
    h = _NoRedirect()
    try:
        h.redirect_request(None, None, 302, 'Found', {}, 'https://evil.example/')
        raise AssertionError('handler 未拒绝')
    except RedirectRefused:
        pass


def test_raw_archive_roundtrip_and_rejected_preserved():
    """原始响应可读回核验：成功与拒收响应都留 .bin+.json，sha256 与 meta 一致。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c = FetchCounter(20)
        client = TencentDailyClient(c, transport=lambda u, t, rc: make_payload('sh600000'),
                                    sleep=lambda s: None)
        client.fetch_raw_daily('600000', '20260101', '20260110', archive_dir=td)
        # 拒收响应（qfq）也留证
        bad = json.dumps({'data': {'sh600001': {'qfqday': [['2026-01-01', 1, 1, 1, 1, 1]]}}}).encode()
        client2 = TencentDailyClient(FetchCounter(20),
                                     transport=lambda u, t, rc: bad, sleep=lambda s: None)
        try:
            client2.fetch_raw_daily('600001', '20260101', '20260110', archive_dir=td)
        except RawResponseRejected:
            pass
        bins = sorted(Path(td).glob('*.bin'))
        metas = sorted(Path(td).glob('*.json'))
        assert len(bins) == 2 and len(metas) == 2, f'{len(bins)} bin / {len(metas)} meta'
        for b in bins:
            meta = json.loads(b.with_suffix('.json').read_text(encoding='utf-8'))
            assert meta['sha256'] == hashlib.sha256(b.read_bytes()).hexdigest()
            assert meta['fetched_at'] and meta['param'] and meta['url']
        outcomes = [json.loads(m.read_text(encoding='utf-8'))['outcome']
                    for m in metas if m.name != 'index.jsonl']
        assert 'ok' in outcomes and any(o.startswith('rejected') for o in outcomes), outcomes


def test_smoke_failure_no_retry():
    """冒烟零重试：smoke=True 失败时恰好 1 次尝试。"""
    c = FetchCounter(10)
    state = {'n': 0}

    def always_fail(url, timeout, read_cap=10**9):
        state['n'] += 1
        raise ConnectionError('boom')

    client = TencentDailyClient(c, max_retries=3, transport=always_fail,
                                sleep=lambda s: None)
    try:
        client.fetch_raw_daily('600000', '20260101', '20260110', smoke=True)
        raise AssertionError('应失败')
    except ConnectionError:
        pass
    assert state['n'] == 1 and c.used == 1
