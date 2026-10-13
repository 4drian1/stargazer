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




# ---------- REV3 反例（字节预算/重定向/原始留证/冒烟零重试） ----------

ROW = ['2020-01-01', 10, 10, 11, 9.5, 1]


def sized_day_payload(target: int) -> bytes:
    """构造恰好 target 字节的合法 JSON 原始 day 序列（解析式，一次成型）。
    payload_len(n) = C + n*r + 2*(n-1)（r=单行序列化字节，C=空表结构字节）；
    余数用末行 volume 位数吸收（+rem 字节 = volume 取 rem+1 位），rem ≤ 40 位远低于 4300 上限。"""
    C = len(json.dumps({'data': {'sh600000': {'day': []}}}).encode())
    r = len(json.dumps(ROW).encode())
    n = (target - C + 2) // (r + 2)
    base = C + n * r + 2 * (n - 1)
    rem = target - base
    assert 0 <= rem <= r + 1, (target, n, rem)
    rows = [list(ROW) for _ in range(max(n, 1))]
    if rem:
        rows[-1][5] = 10 ** rem          # rem+1 位 → 恰好 +rem 字节
    payload = json.dumps({'data': {'sh600000': {'day': rows}}}).encode()
    assert len(payload) == target, (len(payload), target)
    return payload


def test_oversize_first_response_rejected():
    """>5MiB 响应首次即拒（用户反例修正：5,000,096B < 5MiB 会被接受，测试必须用真超限值），
    且不发生重试（请求计数=1）。"""
    cap = 5 * 1024 * 1024                       # 5 MiB = 5,242,880 B
    big = sized_day_payload(cap + 500_000)
    c, bc = FetchCounter(10), ByteCounter(cap)
    client = TencentDailyClient(c, bc,
                                transport=lambda u, t, rc: big, sleep=lambda s: None)
    try:
        client.fetch_raw_daily('600000', '20260101', '20260110')
        raise AssertionError('超字节响应被接受')
    except ByteBudgetExceeded:
        pass
    assert c.used == 1 and bc.used + len(big) > bc.hard_cap_bytes  # 触线即停、未超采、零重试


def test_mib_boundary_units():
    """单位统一为 MiB：5,000,082B（< 5MiB）合法响应必须被接受；5,242,881B（=5MiB+1）首次即拒。"""
    cap = 5 * 1024 * 1024
    ok_payload = sized_day_payload(5_000_082)
    c = TencentDailyClient(FetchCounter(10), ByteCounter(cap),
                           transport=lambda u, t, rc: ok_payload, sleep=lambda s: None)
    rows = c.fetch_raw_daily('600000', '20260101', '20260110')
    assert rows, '5,000,082B（<5MiB）应被接受——单位是 MiB 不是 MB'
    over = sized_day_payload(cap + 1)
    c2 = TencentDailyClient(FetchCounter(10), ByteCounter(cap),
                            transport=lambda u, t, rc: over, sleep=lambda s: None)
    try:
        c2.fetch_raw_daily('600000', '20260101', '20260110')
        raise AssertionError('5MiB+1 未在首次拒绝')
    except ByteBudgetExceeded:
        pass
    assert c2.counter.used == 1


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
        bins = sorted(Path(td).glob('run_*/*.bin'))
        metas = sorted(Path(td).glob('run_*/*.json'))
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


# ---------- REV4 反例（落盘预检/重启不覆盖） ----------

def test_disk_budget_pre_checked_no_write():
    """落盘预算先算后写：限额 1 字节时，任何文件都不得落盘（用户反例：曾先写 776B/3 文件才报错）。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c = FetchCounter(10)
        client = TencentDailyClient(c, transport=lambda u, t, rc: make_payload('sh600000'),
                                    sleep=lambda s: None, disk_counter=ByteCounter(1))
        try:
            client.fetch_raw_daily('600000', '20260101', '20260110', archive_dir=td)
            raise AssertionError('超落盘预算未被拦截')
        except ByteBudgetExceeded:
            pass
        assert list(Path(td).rglob('*')) == [], '预检失败：仍有文件写入'
        assert client.disk.used == 0


def test_restart_no_overwrite():
    """重启不覆盖：两个客户端（模拟两次运行）同 archive_dir → 两个 run_* 目录，全部证据可区分保留。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p1 = make_payload('sh600000', n=2)
        p2 = make_payload('sh600000', n=5)
        TencentDailyClient(FetchCounter(10),
                           transport=lambda u, t, rc: p1,
                           sleep=lambda s: None).fetch_raw_daily(
            '600000', '20260101', '20260110', archive_dir=td)
        TencentDailyClient(FetchCounter(10),
                           transport=lambda u, t, rc: p2,
                           sleep=lambda s: None).fetch_raw_daily(
            '600000', '20260101', '20260110', archive_dir=td)
        runs = sorted(d for d in Path(td).iterdir() if d.is_dir())
        bins = sorted(Path(td).glob('run_*/*.bin'))
        assert len(runs) == 2 and len(bins) == 2, f'{len(runs)} runs / {len(bins)} bins'
        sizes = sorted(b.stat().st_size for b in bins)
        assert sizes == sorted([len(p1), len(p2)]), '证据内容应可区分保留'


def test_evidence_collision_refused():
    """文件身份碰撞即拒：xb 独占创建，同 seq 重名不可能发生；碰撞路径抛 EvidenceCollision。"""
    from independent_fetch_client import EvidenceCollision
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / 'run_x'
        run.mkdir()
        (run / '0001_sh600000.bin').write_bytes(b'old-evidence')
        client = TencentDailyClient(FetchCounter(10), transport=lambda u, t, rc: make_payload('sh600000'),
                                    sleep=lambda s: None)
        client._run_dir = run
        try:
            client._archive(td, 'sh600000', 'p', 'u', 't', b'new', 'ok')
            raise AssertionError('碰撞未被拒绝')
        except EvidenceCollision:
            pass
        assert (run / '0001_sh600000.bin').read_bytes() == b'old-evidence'


if __name__ == '__main__':
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception as exc:
            failed += 1
            print('FAIL', fn.__name__, type(exc).__name__, exc)
    print(f'== {len(fns) - failed}/{len(fns)} 项零网络测试通过 ==')
    _sys.exit(1 if failed else 0)
