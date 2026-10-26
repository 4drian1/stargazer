"""Synthetic entry tests. All HTTP is mocked; no signed production authorization."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import price_verification_batch_44win as api
import price_verification_guarded_run as guard
import independent_fetch_client as client_module


class ComparatorTests(unittest.TestCase):
    def test_rounding_cannot_turn_difference_into_match(self):
        self.assertEqual(api.compare_point(10,10.01004), ('DIFF',-.01004))

    def test_one_cent_inclusive(self):
        self.assertEqual(api.compare_point(10,10.01)[0],'MATCH')

    def test_invalid_nonfinite_and_boolean(self):
        for value in (float('nan'),float('inf'),-1,0,True):
            self.assertEqual(api.compare_point(value,10)[0],'INVALID_PRICE')

    def test_reference_price_perturbation_changes_boundary(self):
        win = ['2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-12']
        local = {('000001',20260105):{'close':1}}
        local.update({('000001',int(d.replace('-',''))):{'low':.937} for d in win})
        self.assertTrue(api.boundary_sensitivity(local,'000001',20260105,win)['sensitive'])

    def test_remote_duplicate_conflict_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            api.tencent_rows_to_map([['2026-01-05',10,10,11,9],['2026-01-05',10,10.5,11,9]])

    def test_remote_identical_duplicate_is_not_a_conflict(self):
        self.assertEqual(len(api.tencent_rows_to_map([['2026-01-05',10,10,11,9]]*2)),1)

    def test_remote_wrong_date_nonfinite_and_ohlc_rejected(self):
        for row in (['2026-02-30',10,10,11,9],['2026-01-05',10,float('nan'),11,9],
                    ['2026-01-05',10,12,11,9],['2026-01-05',True,10,11,9]):
            with self.assertRaises(ValueError):
                api.tencent_rows_to_map([row])

    def test_local_conflict_blocks_needed_price(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            files = []
            for i,price in enumerate((10.,10.5)):
                file = Path(folder) / f'part_{i}.parquet'
                pq.write_table(pa.table({'code':['000001'],'date':[20260105],
                                        'close':[price],'low':[9.]}),file)
                files.append(str(file))
            with self.assertRaisesRegex(ValueError,'LOCAL_DUPLICATE_CONFLICT'):
                api.load_local_prices([{'code':'000001','date':20260105,'field':'close'}],files)

    def test_invalid_unneeded_field_does_not_delete_valid_reference_close(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'part_0.parquet'
            pq.write_table(pa.table({'code':['000001'],'date':[20260105],
                                    'close':[10.],'low':[None]}),file)
            local = api.load_local_prices([{'code':'000001','date':20260105,'field':'close'}],[str(file)])
            self.assertEqual(local[('000001',20260105)]['close'],10.)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'independent_fetch').mkdir()
        self.fake = types.SimpleNamespace(**{key:value for key,value in vars(api).items() if not key.startswith('__')})
        self.fake.HERE, self.fake.BATCH = self.root, self.root / 'independent_fetch/legacy'
        self.output = self.root / 'independent_fetch/synthetic_batch'
        self.input = self.root / 'input.txt'
        self.input.write_text('frozen')
        self.manifest = self.root / 'manifest.json'
        self.m = {'inputs':[{'path':str(self.input),'sha256':guard.sha(self.input)}]}
        self.manifest.write_bytes(guard.encoded(self.m))
        self.auth = self.root / 'authorization.json'
        now = datetime.now(timezone.utc)
        self.a = {'status':'SIGNED','scope':'INDEPENDENT_FETCH',
                  'approved_by':'SYNTHETIC_TEST_ONLY','approval_reference':'mock: no real network',
                  'authorized_at':(now-timedelta(minutes=1)).isoformat(),
                  'expires_at':(now+timedelta(hours=1)).isoformat(),
                  'batch_id':self.output.name,'output_dir':str(self.output),
                  'manifest_sha256':guard.sha(self.manifest),
                  'limits':{'requests':api.REQUEST_CAP,'response_bytes':api.BYTE_CAP,'disk_bytes':api.BYTE_CAP}}
        self.save_auth()
        self.codes = ['600001','000001','000002']
        self.win = ['2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-12']
        self.windows = [{'code':code,'P':20260105,'E':20260106,'window':self.win} for code in self.codes]
        self.points = [{'code':code,'date':20260105,'field':'close'} for code in self.codes] + [
            {'code':code,'date':int(day.replace('-','')),'field':'low'} for code in self.codes for day in self.win]
        self.local = {(point['code'],point['date']):{'close':10.,'low':9.5} for point in self.points}
        self.calls = []
        self.mode = 'ok'

    def tearDown(self):
        self.temp.cleanup()

    def save_auth(self):
        self.auth.write_bytes(guard.encoded(self.a))

    def transport(self,url,timeout,read_cap):
        symbol = parse_qs(urlsplit(url).query)['param'][0].split(',')[0]
        self.calls.append(symbol)
        if self.mode == 'smoke_fail' or self.mode == 'ordinary_fail' and symbol == 'sz000002':
            raise ConnectionError('synthetic failure')
        if self.mode == 'redirect':
            raise api.RedirectRefused('synthetic redirect')
        if self.mode == 'invalid_json':
            return b'not-json-original-evidence'
        if self.mode == 'oversize':
            return b'x' * (api.BYTE_CAP+1)
        dates = ['2026-01-05'] + self.win
        rows = [[day,10,10,10.5,9.5,100] for day in dates]
        if self.mode == 'smoke_missing':
            rows.pop()
        if self.mode == 'duplicate_conflict':
            rows.append(['2026-01-05',10,10.2,10.5,9.5,100])
        if self.mode == 'qfq':
            return json.dumps({'data':{symbol:{'qfqday':rows}}}).encode()
        return json.dumps({'data':{symbol:{'day':rows}}}).encode()

    def factory(self,*args,**kwargs):
        return api.TencentDailyClient(*args,**kwargs,transport=self.transport,sleep=lambda n:None)

    def execute(self):
        with patch.object(guard,'verify_bundle',return_value=(self.m,self.windows,self.points,self.local)):
            return guard.execute(self.output,self.auth,guard.sha(self.auth),self.manifest,self.fake,self.factory)

    def test_missing_authorization_is_zero_requests_zero_writes(self):
        with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):
            guard.execute(self.output,None,None,self.manifest,self.fake,self.factory)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls,[])

    def test_wrong_scope_with_valid_hash_still_rejected(self):
        self.a['scope'] = 'NOT_INDEPENDENT_FETCH'
        self.save_auth()
        with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):
            self.execute()
        self.assertEqual(self.calls,[])

    def test_draft_authorization_refused(self):
        self.a['status'] = 'DRAFT'
        self.save_auth()
        with self.assertRaises(ValueError):
            self.execute()

    def test_auth_hash_tampering_refused(self):
        before = guard.sha(self.auth)
        self.a['batch_id'] = 'changed'
        self.save_auth()
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):
            guard.execute(self.output,self.auth,before,self.manifest,self.fake,self.factory)

    def test_expired_or_naive_authorization_refused(self):
        for expiry in ('2000-01-01T00:00:00+00:00','2099-01-01T00:00:00'):
            self.a['expires_at'] = expiry
            self.save_auth()
            with self.assertRaises(ValueError):
                self.execute()

    def test_output_cannot_be_changed_or_old_batch_reused(self):
        self.output = self.fake.BATCH
        with self.assertRaisesRegex(ValueError,'OUTPUT_SCOPE'):
            self.execute()

    def test_manifest_binding_cannot_be_changed(self):
        self.manifest.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'BINDING'):
            self.execute()

    def test_preflight_hash_failure_is_zero_requests_zero_writes(self):
        with patch.object(guard,'verify_bundle',side_effect=ValueError('INPUT_HASH_MISMATCH')):
            with self.assertRaisesRegex(ValueError,'INPUT_HASH'):
                guard.execute(self.output,self.auth,guard.sha(self.auth),self.manifest,self.fake,self.factory)
        self.assertFalse(self.output.exists())

    def test_success_full_report_and_measured_disk_bytes(self):
        result = self.execute()
        self.assertEqual(result['stop_reason'],'completed')
        self.assertEqual(result['requests_used'],3)  # two smoke successes reused, not five
        self.assertEqual(result['directory_actual_bytes'],sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()))
        report = json.loads((self.output / 'window_price_report_44win.json').read_bytes())
        self.assertTrue(all(r['all_six_prices_match'] for r in report))
        self.assertTrue(all(not r['label_eligible'] and not r['training_eligible'] for r in report))

    def test_completed_batch_refuses_restart(self):
        self.execute()
        count = len(self.calls)
        with self.assertRaisesRegex(ValueError,'NO_RESTART'):
            self.execute()
        self.assertEqual(len(self.calls),count)

    def test_incomplete_or_empty_terminal_batch_also_refuses_restart(self):
        self.output.mkdir()
        (self.output / 'stop_record.json').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'NO_RESTART'):
            self.execute()
        self.assertEqual(self.calls,[])

    def test_smoke_first_failure_terminates_and_does_not_retry(self):
        self.mode = 'smoke_fail'
        result = self.execute()
        self.assertEqual(len(self.calls),1)
        self.assertNotEqual(result['stop_reason'],'completed')
        self.assertTrue((self.output / 'stop_record.json').exists())

    def test_ordinary_failure_stops_after_three_attempts_no_retry(self):
        self.mode = 'ordinary_fail'
        result = self.execute()
        self.assertEqual(len(self.calls),3)
        self.assertNotEqual(result['stop_reason'],'completed')

    def test_missing_smoke_price_terminates(self):
        self.mode = 'smoke_missing'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)

    def test_row_conflict_stops_with_raw_evidence_preserved(self):
        self.mode = 'duplicate_conflict'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)
        self.assertTrue(list(self.output.glob('run_*/*.bin')))

    def test_invalid_json_raw_body_is_preserved_not_replaced_by_empty(self):
        self.mode = 'invalid_json'
        self.execute()
        self.assertEqual(next(self.output.glob('run_*/*.bin')).read_bytes(),b'not-json-original-evidence')
        self.assertEqual(len(self.calls),1)

    def test_redirect_and_adjusted_responses_terminate(self):
        self.mode = 'redirect'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)

    def test_adjusted_response_terminates(self):
        self.mode = 'qfq'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)

    def test_received_overlimit_mock_bytes_not_falsely_reported_as_zero(self):
        self.mode = 'oversize'
        result = self.execute()
        self.assertEqual(result['response_body_bytes'],api.BYTE_CAP+1)
        self.assertNotEqual(result['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)

    def test_fetch_cli_without_auth_refuses_before_mkdir(self):
        done = subprocess.run([sys.executable,'-B','-X','utf8',str(api.HERE / 'price_verification_batch_44win.py'),
                               '--fetch',str(self.output)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(done.returncode,2)
        self.assertIn('AUTHORIZATION_REQUIRED',done.stdout)
        self.assertFalse(self.output.exists())


class FrozenBundleTests(unittest.TestCase):
    def modified(self, change):
        manifest = json.loads(api.MANIFEST.read_bytes())
        change(manifest)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'manifest.json'
            file.write_bytes(guard.encoded(manifest))
            return guard.verify_bundle(file, api)

    def test_changed_full_hash_refuses_actual_preflight(self):
        with self.assertRaisesRegex(ValueError,'INPUT_HASH_MISMATCH'):
            self.modified(lambda m:m['inputs'][0].update(sha256='0'*64))

    def test_missing_required_role_refuses_actual_preflight(self):
        with self.assertRaisesRegex(ValueError,'INPUTS_INCOMPLETE'):
            self.modified(lambda m:m['inputs'].pop(0))

    def test_changed_budget_refuses_actual_preflight(self):
        with self.assertRaisesRegex(ValueError,'LIMITS_MISMATCH'):
            self.modified(lambda m:m['limits'].update(requests=1000))

    def test_calendar_truncation_not_hidden_by_old_pass_flag(self):
        mapping = json.loads((api.BATCH / 'five_day_window_map_v3.json').read_bytes())
        first = mapping['codes'][0]['windows'][0]
        first['window'] = first['window'][:-1]
        with self.assertRaisesRegex(ValueError,'CALENDAR_MISMATCH'):
            guard.materialize_points(mapping,json.loads(api.MANIFEST.read_bytes())['calendar_dates'])


class TransportTests(unittest.TestCase):
    def test_default_transport_does_not_read_a_hidden_extra_probe_byte(self):
        response = types.SimpleNamespace(read=lambda n:b'x'*n)
        from unittest.mock import MagicMock
        manager = MagicMock()
        manager.__enter__.return_value = response
        opener = types.SimpleNamespace(open=lambda *a,**kw:manager)
        with patch.object(client_module.urllib.request,'build_opener',return_value=opener):
            with self.assertRaises(api.ByteBudgetExceeded) as caught:
                client_module._default_transport('https://example.test',10,10)
        self.assertEqual(len(caught.exception.received_bytes),10)


if __name__ == '__main__':
    unittest.main()
