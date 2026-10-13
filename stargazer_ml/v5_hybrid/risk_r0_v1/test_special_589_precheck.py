"""test_special_589_precheck.py — 批次e（600589）运行前零网络预检（用户强制门槛）。

覆盖：URL 编码（无空白/控制字符、clean_url 幂等）、GET 方式、
失败终局（不续跑下一篇+失败 stop_reason+再跑拒绝）、预算异常不重试、
额度数字与请求单一致。全部通过后才允许执行 special_589_run.py。
"""
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter)
from special_589_run import CAP_B, CAP_R, RAW_URLS, TARGETS


def test_cap_matches_signed_order():
    assert CAP_R == 6, CAP_R
    assert CAP_B == 2 * 1024 * 1024, CAP_B


def test_urls_normalized_and_frozen():
    """URL 无空白/控制字符；规范化幂等；清单三份与批准一致。"""
    from special_564_run import clean_url
    assert len(TARGETS) == 3
    ids = [t['id'] for t in TARGETS]
    assert ids == ['1218734245', '1218734247', '1218747449'], ids
    for t in TARGETS:
        u = t['url']
        assert u == clean_url(u)
        assert ' ' not in u and '\t' not in u and '\n' not in u, u
        assert u.startswith('http://www.cninfo.com.cn/new/disclosure/detail')
        assert len(u) < 250


def test_get_method():
    """正文下载必须是 GET（data=None），与标记一致。"""
    from cninfo_ann_client import CninfoAnnClient
    c, b = FetchCounter(6), ByteCounter(1024 * 1024)
    seen = {}

    def spy(url, data, timeout, read_cap):
        seen['data'] = data
        return b'%PDF-fake'

    client = CninfoAnnClient(c, b, transport=spy, sleep=lambda s: None)
    client.fetch_ann_text(TARGETS[0]['url'])
    assert seen['data'] is None, f'实际 data={seen["data"]!r}（应为 None=GET）'


def test_budget_exception_no_retry():
    from cninfo_ann_client import CninfoAnnClient
    c, b = FetchCounter(6), ByteCounter(1024)

    def big(url, data, timeout, read_cap):
        return b'x' * 2048

    client = CninfoAnnClient(c, b, max_retries=0, transport=big, sleep=lambda s: None)
    try:
        client.fetch_ann_text(TARGETS[0]['url'])
        raise AssertionError('未触线')
    except ByteBudgetExceeded:
        pass
    assert c.used == 1, f'预算异常后重试：used={c.used}'


def test_failure_finalizes_and_refuses_rerun():
    """失败即停：首篇详情页失败 → 不请求下一篇 → 失败终局 → 再次运行拒绝。"""
    import special_589_run

    with tempfile.TemporaryDirectory() as td:
        os.environ['BATCH_589_DIR'] = td
        importlib.reload(special_589_run)
        calls = []

        class BoomClient:
            def __init__(self, *a, **k):
                pass

            def fetch_ann_text(self, url, archive_dir=None):
                calls.append(url)
                raise ConnectionError('boom')

        special_589_run.CninfoAnnClient = BoomClient
        try:
            rc = special_589_run.main()
            assert rc == 0
            assert len(calls) == 1, f'失败后仍请求下一篇：实际 {len(calls)} 次'
            stop = json.loads((Path(td) / 'stop_record.json').read_text(encoding='utf-8'))
            assert stop['stop_reason'].startswith('fetch_fail'), stop
            try:
                special_589_run.main()
                raise AssertionError('失败终局后再次运行未被拒绝')
            except SystemExit:
                pass
        finally:
            os.environ.pop('BATCH_589_DIR', None)
            importlib.reload(special_589_run)


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
            print('FAIL', fn.__name__, type(exc).__name__, str(exc)[:120])
    print(f'== {len(fns) - bad}/{len(fns)} 项批次e预检（零网络）通过 ==')
    sys.exit(1 if bad else 0)
