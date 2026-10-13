"""test_sina_div_client.py — 新浪适配器零网络测试（mock 传输 + 固件 HTML）。"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter,
                                      RedirectRefused)
from sina_div_client import SinaDivClient, parse_bonus_page

FIXTURE = """
<html><body>
<table><thead><tr><th>公告日期</th><th>送股(股)</th><th>转增(股)</th>
<th>派息(税前)(元)</th><th>除权除息日</th><th>股权登记日</th></tr></thead>
<tbody><tr><td>2025-06-05</td><td>0</td><td>0</td><td>3.5</td><td>2025-06-11</td><td>2025-06-10</td></tr>
<tr><td>2025-09-30</td><td>0</td><td>0</td><td>2.0</td><td>2025-10-15</td><td>2025-10-14</td></tr></tbody></table>
<table><thead><tr><th>公告日期</th><th>配股比例</th><th>配股价格</th>
<th>配股除权日</th><th>股权登记日</th></tr></thead>
<tbody><tr><td>2023-11-27</td><td>10配3</td><td>5.0</td><td>2023-12-01</td><td>2023-11-30</td></tr></tbody></table>
</body></html>
""".encode('gbk')


def mk():
    return FetchCounter(18), ByteCounter(3 * 1024 * 1024)


def test_one_page_two_tables():
    """单页双表解析：分红+配股一次请求覆盖（用户更正①）。"""
    out = parse_bonus_page(FIXTURE)
    assert out['bonus'] and out['allotment'], out['headers']
    assert out['bonus'][0]['除权除息日'] == '2025-06-11'
    assert out['allotment'][0]['配股除权日'] == '2023-12-01'


def test_request_counted_and_get():
    c, b = mk()
    seen = {}

    def spy(url, timeout, read_cap):
        seen['url'] = url
        return FIXTURE

    client = SinaDivClient(c, b, transport=spy, sleep=lambda s: None)
    client.fetch_bonus_page('000001')
    assert c.used == 1 and 'vISSUE_ShareBonus' in seen['url']


def test_budget_no_retry():
    c, b = FetchCounter(18), ByteCounter(len(FIXTURE))
    client = SinaDivClient(c, b, max_retries=3, transport=lambda u, t, rc: FIXTURE,
                           sleep=lambda s: None)
    client.fetch_bonus_page('000001')
    try:
        client.fetch_bonus_page('000049')
        raise AssertionError('未触线')
    except ByteBudgetExceeded:
        pass
    assert c.used == 2, f'预算异常后重试：used={c.used}'


def test_hard_cap_and_redirect():
    c, b = FetchCounter(1), ByteCounter(3 * 1024 * 1024)
    client = SinaDivClient(c, b, transport=lambda u, t, rc: FIXTURE, sleep=lambda s: None)
    client.fetch_bonus_page('000001')
    for exc in (FetchBudgetExceeded,):
        try:
            client.fetch_bonus_page('000049')
            raise AssertionError('未拦截')
        except exc:
            pass
    def redir(u, t, rc):
        raise RedirectRefused('302')
    c2, _ = mk()
    client2 = SinaDivClient(c2, b, transport=redir, sleep=lambda s: None)
    try:
        client2.fetch_bonus_page('000001')
        raise AssertionError('重定向未拒')
    except RedirectRefused:
        pass


def test_archive_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        c, b = mk()
        disk = ByteCounter(3 * 1024 * 1024)
        client = SinaDivClient(c, b, transport=lambda u, t, rc: FIXTURE,
                               sleep=lambda s: None, disk_counter=disk)
        client.fetch_bonus_page('000001', archive_dir=td)
        bins = list(Path(td).glob('run_*/*.bin'))
        metas = list(Path(td).glob('run_*/*.json'))
        assert len(bins) == 1 and len(metas) == 1
        import hashlib
        meta = json.loads(metas[0].read_text(encoding='utf-8'))
        assert meta['sha256'] == hashlib.sha256(bins[0].read_bytes()).hexdigest()
        assert meta['method'] == 'GET'


def test_disk_precheck_no_write():
    with tempfile.TemporaryDirectory() as td:
        c, b = mk()
        disk = ByteCounter(1)
        client = SinaDivClient(c, b, transport=lambda u, t, rc: FIXTURE,
                               sleep=lambda s: None, disk_counter=disk)
        try:
            client.fetch_bonus_page('000001', archive_dir=td)
            raise AssertionError('落盘预检未拦截')
        except ByteBudgetExceeded:
            pass
        assert list(Path(td).rglob('*')) == []


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
            print('FAIL', fn.__name__, type(exc).__name__, str(exc)[:80])
    print(f'== {len(fns) - bad}/{len(fns)} 项新浪适配器零网络测试通过 ==')
    sys.exit(1 if bad else 0)
