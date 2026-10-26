"""Offline only: temporary batches and injected transports, no HTTP."""
import json
from pathlib import Path
import tempfile
import unittest

import msgpack
import probe_ths_native8_v1 as p


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)/'batch'
        self.calls = []

    def run_probe(self, send=None, **kwargs):
        def empty(query, limit):
            self.calls.append(query)
            return msgpack.packb([])
        return p.execute(self.out, send=send or empty, pause=lambda _: None,
                         global_check=lambda _: True, **kwargs)

    def test_eight_empty_keys_complete_and_raw_hash(self):
        report = self.run_probe()
        self.assertEqual((report['status'], report['attempts']), ('COMPLETE', 8))
        self.assertEqual(self.calls, p.fixed_queries())
        for entry in report['entries']:
            body = (self.out/f"{entry['seq']:02d}_raw.msgpack").read_bytes()
            self.assertEqual(p.digest(body), entry['sha256'])
            self.assertEqual(entry['rows'], 0)
        self.assertFalse(report['training_ready'])

    def test_complete_batch_cannot_restart(self):
        self.run_probe()
        with self.assertRaisesRegex(RuntimeError, 'BATCH_EXISTS'):
            self.run_probe()
        self.assertEqual(len(self.calls), 8)

    def test_failure_stops_first_and_restart_refused(self):
        def fail(query, limit):
            self.calls.append(query)
            raise TimeoutError('offline timeout')
        report = self.run_probe(fail)
        self.assertEqual((report['status'], len(self.calls)), ('FAILED', 1))
        self.assertTrue((self.out/'stop_record.json').is_file())
        with self.assertRaisesRegex(RuntimeError, 'BATCH_EXISTS'):
            self.run_probe()

    def test_existing_unfinished_or_corrupt_batch_refused(self):
        self.out.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'BATCH_EXISTS'):
            self.run_probe()

    def test_malformed_body_preserved(self):
        body = b'not msgpack'
        report = self.run_probe(lambda q, n: body)
        self.assertEqual(report['attempts'], 1)
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual((self.out/'01_raw.msgpack').read_bytes(), body)

    def test_budget_preflight_no_writes_no_http(self):
        with self.assertRaisesRegex(RuntimeError, 'PREFLIGHT'):
            self.run_probe(disk_cap=1)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.calls, [])

    def test_cap_reaching_response_preserved_and_stop(self):
        report = self.run_probe(lambda q, n: b'x'*n, response_cap=100)
        self.assertEqual(report['attempts'], 1)
        self.assertIn('EOF_UNPROVEN', report['error'])
        self.assertEqual((self.out/'01_raw.msgpack').stat().st_size, 100)

    def test_identity_date_and_price_rejections(self):
        query = p.fixed_queries()[0]
        row = dict(code='306380', name='储能', date=20250102,
                   open=10, high=11, low=9, close=10)
        result = p.summarize(msgpack.packb([row]), query)
        self.assertFalse(result['native_index_admitted'])
        for field, value, reason in [('code', '600000', 'CODE'),
                                     ('name', '浦发银行', 'NAME'),
                                     ('date', 20250103, 'DATE'),
                                     ('low', float('nan'), 'PRICE')]:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, reason):
                    p.summarize(msgpack.packb([dict(row, **{field: value})]), query)

    def test_no_redirect(self):
        self.assertIsNone(p.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com'))

    def test_bound_plan_and_byte_totals(self):
        report = self.run_probe()
        binding = json.loads((self.out/'binding.json').read_bytes())
        self.assertEqual(binding['plan_sha256'], p.digest(p.PLAN.read_bytes()))
        self.assertEqual(report['response_bytes'], 8)
        actual = sum(f.stat().st_size for f in self.out.iterdir())
        self.assertEqual(actual, report['disk_bytes_before_stop']+(self.out/'stop_record.json').stat().st_size)


if __name__ == '__main__':
    unittest.main()
