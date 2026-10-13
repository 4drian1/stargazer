"""test_cninfo_client.py — 巨潮受控适配器零网络测试（mock 传输层）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cninfo_ann_client import CninfoAnnClient
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchCounter, FetchBudgetExceeded)

LIST_RESP = json.dumps({'totalAnnouncement': 2, 'announcements': [
    {'secCode': '600733', 'announcementTitle': '关于公司股票复牌的公告',
     'announcementTime': 1537977600000, 'adjunctUrl': 'finalpage/2018-09-26/...PDF'},
]}).encode()


def mk():
    c, b = FetchCounter(6), ByteCounter(1024 * 1024)
    return c, b


def test_list_one_request_counted():
    c, b = mk()
    calls = []
    client = CninfoAnnClient(c, b,
                             transport=lambda u, d, t, rc: calls.append(u) or LIST_RESP,
                             sleep=lambda s: None)
    out = client.query_ann_list('600733', '2018-01-01~2018-12-31')
    assert out['totalAnnouncement'] == 2 and c.used == 1 and len(calls) == 1


def test_no_hidden_pagination():
    """显式页码：页 1 页 2 各计 1 请求（无自动翻页、无重复首页）。"""
    c, b = mk()
    client = CninfoAnnClient(c, b, transport=lambda u, d, t, rc: LIST_RESP,
                             sleep=lambda s: None)
    client.query_ann_list('600733', '2018-01-01~2018-12-31', page_num=1)
    client.query_ann_list('600733', '2018-01-01~2018-12-31', page_num=2)
    assert c.used == 2


def test_hard_cap_stops():
    c, b = FetchCounter(1), ByteCounter(1024 * 1024)
    client = CninfoAnnClient(c, b, transport=lambda u, d, t, rc: LIST_RESP,
                             sleep=lambda s: None)
    client.query_ann_list('600733', '2018-01-01~2018-12-31')
    try:
        client.query_ann_list('300586', '2018-01-01~2018-12-31')
        raise AssertionError('超请求预算未拦截')
    except FetchBudgetExceeded:
        pass


def test_byte_cap_stops():
    c, b = FetchCounter(6), ByteCounter(100)
    client = CninfoAnnClient(c, b, transport=lambda u, d, t, rc: LIST_RESP,
                             sleep=lambda s: None)
    try:
        client.query_ann_list('600733', '2018-01-01~2018-12-31')
        raise AssertionError('超字节预算未拦截')
    except ByteBudgetExceeded:
        pass


def test_archive_roundtrip_with_disk_budget():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c, b = FetchCounter(6), ByteCounter(1024 * 1024)
        disk = ByteCounter(1024 * 1024)
        client = CninfoAnnClient(c, b, transport=lambda u, d, t, rc: LIST_RESP,
                                 sleep=lambda s: None, disk_counter=disk)
        client.query_ann_list('600733', '2018-01-01~2018-12-31', archive_dir=td)
        bins = list(Path(td).glob('run_*/*.bin'))
        metas = list(Path(td).glob('run_*/*.json'))
        assert len(bins) == 1 and len(metas) == 1
        import hashlib
        meta = json.loads(metas[0].read_text(encoding='utf-8'))
        assert meta['sha256'] == hashlib.sha256(bins[0].read_bytes()).hexdigest()
        assert meta['form']['searchkey'] == '600733'


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
            print('FAIL', fn.__name__, type(exc).__name__, exc)
    print(f'== {len(fns) - bad}/{len(fns)} 项巨潮适配器零网络测试通过 ==')
    sys.exit(1 if bad else 0)
