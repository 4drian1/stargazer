import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_minute_snapshot_11_v1 import Batch, NoRedirect, decode_rows, encoded
import local_minute_snapshot_11_v1 as snapshot


Q = dict(code='605358', date=20250102, scope='STOCK', key='分钟k:605358:20250102*')


def payload(**changes):
    row = dict(code='605358', date=20250102093100, open=10., high=11., low=9., close=10., volume=100.)
    row.update(changes)
    return [[f"分钟k:605358:{row['date']}", row]]


class SnapshotTests(unittest.TestCase):
    def batch(self, folder, io, **kwargs):
        b = Batch(Path(folder)/'batch', io=io, sleeper=lambda t: None, registered=lambda: 0, **kwargs)
        b.start(dict(test=True))
        return b

    def test_raw_pairs_and_empty(self):
        self.assertEqual(len(decode_rows(encoded(payload()), Q)), 1)
        self.assertEqual(decode_rows(b'[]', Q), [])

    def test_msgpack(self):
        import msgpack
        self.assertEqual(len(decode_rows(msgpack.packb(payload(), use_bin_type=True), Q)), 1)

    def test_bad_code(self):
        with self.assertRaisesRegex(ValueError, 'CODE_IDENTITY'):
            decode_rows(encoded(payload(code='000001')), Q)

    def test_bad_date_or_price(self):
        for change in (dict(date=20250103093100), dict(close=20.), dict(volume=-1.), dict(date=202501020931)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                decode_rows(encoded(payload(**change)), Q)

    def test_nonfinite(self):
        body = json.dumps(payload(close=float('nan'))).encode()
        with self.assertRaises(ValueError):
            decode_rows(body, Q)

    def test_duplicate_conflict(self):
        with self.assertRaisesRegex(ValueError, 'CONFLICTING_MINUTE'):
            decode_rows(encoded(payload()+payload(volume=50.)), Q)

    def test_index_cannot_be_bank(self):
        q = dict(Q, code='sh000001', scope='SSE_PRICE_INDEX')
        p = payload(code='sh000001', name='平安银行')
        p[0][0] = '分钟k:sh000001:20250102093100'
        with self.assertRaisesRegex(ValueError, 'INDEX_NAME'):
            decode_rows(encoded(p), q)
        p[0][1]['name'] = '上证指数'
        self.assertEqual(len(decode_rows(encoded(p), q)), 1)

    def test_failed_first_request_terminates_and_archives(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            b = self.batch(folder, lambda u, c: (calls.append(u) or b'bad', 500))
            out = b.execute([Q, Q])
            self.assertEqual(len(calls), 1)
            self.assertEqual(out['status'], 'failed')
            self.assertEqual((b.out/'01_raw.bin').read_bytes(), b'bad')
            with self.assertRaisesRegex(ValueError, 'TERMINATED'):
                b.execute([Q])

    def test_generic_exception_no_retry(self):
        def fail(u, c):
            raise OSError('mock failure')
        with tempfile.TemporaryDirectory() as folder:
            b = self.batch(folder, fail)
            out = b.execute([Q, Q])
            self.assertEqual(out['calls_started'], 1)
            self.assertEqual(out['status'], 'failed')

    def test_11_cap_includes_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            b = self.batch(folder, lambda u, c: (b'[]', 200))
            out = b.execute([Q]*12)
            self.assertEqual(out['calls_started'], 11)
            self.assertEqual(out['status'], 'failed')

    def test_byte_cap_archives_truncated_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            b = self.batch(folder, lambda u, c: (b'x'*c, 200), response_cap=50)
            out = b.execute([Q, Q])
            self.assertEqual(out['response_bytes_received'], 50)
            self.assertEqual(out['calls_started'], 1)
            self.assertEqual((b.out/'01_raw.bin').stat().st_size, 50)
            self.assertIn('COMPLETENESS_UNKNOWN', out['results'][0]['error'])

    def test_disk_preflight_no_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'batch'
            b = Batch(p, disk_cap=1, registered=lambda: 0)
            with self.assertRaisesRegex(ValueError, 'OPENING_BUDGET'):
                b.start({})
            self.assertFalse(p.exists())

    def test_restarts_never_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            b = self.batch(folder, lambda u, c: (b'[]', 200))
            out = b.execute([Q])
            original = (b.out/'stop_record.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'NO_RESTART'):
                Batch(b.out, registered=lambda: 0).start({})
            self.assertEqual(original, (b.out/'stop_record.json').read_bytes())
            self.assertEqual(out['status'], 'completed')

    def test_rate_and_fixed_get_url(self):
        waits, urls = [], []
        with tempfile.TemporaryDirectory() as folder:
            b = Batch(Path(folder)/'batch', io=lambda u, c: (urls.append(u) or b'[]', 200),
                      registered=lambda: 0, sleeper=waits.append, clock=lambda: 0.)
            b.start({}); b.execute([Q, Q])
            self.assertEqual(waits, [1.])
            self.assertTrue(all(u.startswith('http://127.0.0.1:7899/?cmd=get&t=') for u in urls))

    def test_redirect_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'REDIRECT_REFUSED'):
            NoRedirect().redirect_request(None, None, 302, '', {}, 'http://example.com')

    def test_auth_accepts_resolved_path_spellings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inv = root/'inventory.json'
            inv.write_bytes(encoded(dict(proposed_new_snapshot=dict(stock_queries=[Q]*8,
                index_queries=[Q]*3, http_request_cap=11, retries=0))))
            paths = [inv, Path(snapshot.__file__), snapshot.HERE/'test_local_minute_snapshot_11_v1.py']
            auth = root/'auth.json'
            auth.write_bytes(encoded(dict(scope='LOCAL_MINUTE_SNAPSHOT_11', status='CHAT_APPROVED',
                inventory_sha256=snapshot.sha(inv), inputs=[dict(path=p.resolve().as_posix(),
                    sha256=snapshot.sha(p)) for p in paths])))
            with patch.object(snapshot, 'INVENTORY', inv), patch.object(snapshot, 'AUTH', auth), \
                    patch.object(snapshot, 'OUT', root/'not_started'), patch.object(snapshot, 'bytes_used', lambda: 0):
                self.assertEqual(len(snapshot.check()[1]), 11)


if __name__ == '__main__':
    unittest.main()
