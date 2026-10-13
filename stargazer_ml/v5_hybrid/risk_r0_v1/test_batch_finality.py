"""test_batch_finality.py — 批次停止/终局路径零网络验收（用户 REV6 裁决）。

覆盖：finalize 完整写入（UTF-8 字节）、读回一致、二次 finalize 拒绝覆盖、
batch_finalized 守卫、stop 文件非零字节。不触碰已终结批次目录。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import independent_fetch_run as R
from independent_fetch_client import ByteCounter, FetchCounter


def _counters():
    return FetchCounter(25), ByteCounter(5 * 1024 * 1024), ByteCounter(5 * 1024 * 1024)


def test_finalize_writes_valid_utf8_and_readback():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        req, nb, dk = _counters()
        req.used, nb.used, dk.used = 3, 100, 200
        R.finalize([dict(event='x')], req, nb, dk, 'smoke_failed:X', batch_dir=d)
        stop = d / 'stop_record.json'
        led = d / 'request_ledger.json'
        assert stop.stat().st_size > 0 and led.stat().st_size > 0   # 非零字节
        obj = json.loads(stop.read_text(encoding='utf-8'))           # UTF-8 可读回
        assert obj['stop_reason'] == 'smoke_failed:X' and obj['requests_used'] == 3
        assert json.loads(led.read_text(encoding='utf-8'))[0]['event'] == 'x'


def test_finalize_refuses_overwrite():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        req, nb, dk = _counters()
        R.finalize([], req, nb, dk, 'completed', batch_dir=d)
        try:
            R.finalize([], req, nb, dk, 'hacker', batch_dir=d)
            raise AssertionError('终局被覆盖')
        except FileExistsError:
            pass
        obj = json.loads((d / 'stop_record.json').read_text(encoding='utf-8'))
        assert obj['stop_reason'] == 'completed', '原终局被篡改'


def test_batch_finalized_guard():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        assert not R.batch_finalized(d)
        (d / 'stop_record.json').write_bytes(b'{}')
        assert R.batch_finalized(d)


def test_full_report_write_readback():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        req, nb, dk = _counters()
        pts = [dict(code='000001', P=20260101, remote='ok', kind='cross_shard',
                    days=[dict(date=20260101, field='close', local=1.0, remote=1.0,
                               diff=0.0, verdict='MATCH')])]
        R.finalize([], req, nb, dk, 'completed', points=pts, batch_dir=d)
        text = (d / 'report.md').read_text(encoding='utf-8')
        assert 'MATCH:1' in text and '000001' in text


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
    print(f'== {len(fns) - bad}/{len(fns)} 项批次终局测试通过 ==')
    sys.exit(1 if bad else 0)
