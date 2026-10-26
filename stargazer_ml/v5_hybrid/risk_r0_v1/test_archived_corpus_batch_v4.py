import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch

from build_archived_corpus_v4 import fetch_local, REQUEST_CAP, RESPONSE_CAP, run


class LocalFetchTests(unittest.TestCase):
    def opener(self, body):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = body
        return opener

    def test_count_and_raw_archive(self):
        body = b'[]'
        opener = self.opener(body)
        ledger, saved = [], {}
        result = fetch_local('000001', '日k', opener, ledger, lambda n, d: saved.update({n: d}))
        self.assertEqual(result, [])
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]['bytes'], 2)
        self.assertEqual(gzip.decompress(saved['000001_daily.json.gz']), body)
        self.assertTrue(opener.open.call_args.args[0].startswith('http://127.0.0.1:7899/'))
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 10)

    def test_no_retry_on_failure(self):
        opener = Mock()
        opener.open.side_effect = TimeoutError('test')
        ledger = []
        with self.assertRaises(TimeoutError):
            fetch_local('000001', '日k', opener, ledger, Mock())
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(len(ledger), 1)
        self.assertIn('TimeoutError', ledger[0]['error'])

    def test_request_cap_no_send(self):
        opener = Mock()
        with self.assertRaisesRegex(ValueError, 'LOCAL_REQUEST_CAP'):
            fetch_local('000001', '日k', opener, [{}] * REQUEST_CAP, Mock())
        opener.open.assert_not_called()

    def test_response_cap_no_send(self):
        opener = Mock()
        with self.assertRaisesRegex(ValueError, 'LOCAL_RESPONSE_CAP'):
            fetch_local('000001', '日k', opener, [dict(bytes=RESPONSE_CAP)], Mock())
        opener.open.assert_not_called()

    def test_truncated_body_archived_then_stop(self):
        opener, saved = self.opener(b'xxxx'), {}
        ledger = [dict(bytes=RESPONSE_CAP - 4)]
        with self.assertRaisesRegex(ValueError, 'LOCAL_RESPONSE_LIMIT_UNCERTIFIED'):
            fetch_local('000001', '日k', opener, ledger, lambda n, d: saved.update({n: d}))
        self.assertEqual(opener.open.return_value.__enter__.return_value.read.call_args.args, (4,))
        self.assertEqual(ledger[-1]['bytes'], 4)
        self.assertEqual(gzip.decompress(next(iter(saved.values()))), b'xxxx')

    def test_invalid_json_archived_no_retry(self):
        opener, save = self.opener(b'not json'), Mock()
        with self.assertRaises(json.JSONDecodeError):
            fetch_local('000001', '复权', opener, [], save)
        save.assert_called_once()
        self.assertEqual(opener.open.call_count, 1)

    def test_scope_rejected_before_send(self):
        opener = Mock()
        with self.assertRaisesRegex(ValueError, 'LOCAL_QUERY_SCOPE'):
            fetch_local('000001', 'other', opener, [], Mock())
        opener.open.assert_not_called()

    def test_real_entry_failure_finalizes_and_no_restart(self):
        # Real frozen local preflight + archived first three responses, no HTTP transport.
        opener = Mock()
        opener.open.side_effect = TimeoutError('entry regression')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'isolated_batch'
            with patch('build_archived_corpus_v4.OUT', output), patch(
                    'build_archived_corpus_v4.urllib.request.build_opener', return_value=opener):
                self.assertEqual(run(), 1)
                terminal = json.loads((output / 'stop_record.json').read_bytes())
                self.assertEqual(terminal['status'], 'failed')
                self.assertEqual(terminal['local_requests'], 1)
                self.assertFalse(terminal['fit_performed'])
                with self.assertRaisesRegex(ValueError, 'BATCH_ALREADY_CLAIMED'):
                    run()
                self.assertEqual(opener.open.call_count, 1)


if __name__ == '__main__':
    unittest.main()
