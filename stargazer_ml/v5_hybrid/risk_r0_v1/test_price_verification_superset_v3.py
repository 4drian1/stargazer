"""REV3 offline tests: synthetic transport and one already archived real response."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import price_verification_batch_44win as v2
import price_verification_batch_44win_v3 as api
import price_verification_guarded_run as old_guard
import price_verification_guarded_run_v3 as guard
import test_price_verification_entry as original_tests


class SupersetEntryTests(original_tests.EntryTests):
    def execute(self):
        with patch.object(guard,'verify_bundle',return_value=(self.m,self.windows,self.points,self.local)):
            return guard.execute(self.output,self.auth,guard.sha(self.auth),self.manifest,self.fake,self.factory)

    def transport(self,url,timeout,read_cap):
        if self.mode == 'replay_real_first':
            from urllib.parse import parse_qs, urlsplit
            symbol = parse_qs(urlsplit(url).query)['param'][0].split(',')[0]
            if symbol == 'sh600313':
                self.calls.append(symbol)
                return self.real_body
        raw = super().transport(url,timeout,read_cap)
        if not self.mode.startswith('superset'):
            return raw
        value = json.loads(raw)
        symbol,node = next(iter(value['data'].items()))
        rows = node['day']
        extra = [['2020-01-02',10,10,11,9,100],['2099-01-05',10,10,11,9,100]]
        node['day'] = extra + rows
        if self.mode == 'superset_missing':
            node['day'] = [r for r in node['day'] if r[0] != self.win[-1]]
        elif self.mode == 'superset_duplicate':
            node['day'].append(['2020-01-02',10,10.2,11,9,100])
        elif self.mode == 'superset_invalid':
            node['day'].append(['2019-01-02',10,12,11,9,100])
        elif self.mode == 'superset_qfq':
            node['qfqday'] = rows
        elif self.mode == 'superset_wrong_symbol':
            value['data'] = {'sh999999':node}
        elif self.mode == 'superset_ordinary_missing' and symbol == 'sz000002':
            node['day'] = [r for r in node['day'] if r[0] != self.win[-1]]
        return json.dumps(value).encode()

    def test_superset_complete_succeeds_only_frozen_points_compared(self):
        self.mode = 'superset_ok'
        result = self.execute()
        self.assertEqual(result['stop_reason'],'completed')
        self.assertEqual(result['requests_used'],3)
        matches = json.loads((self.output / 'price_match_table_44win.json').read_bytes())
        self.assertEqual(len(matches),len(self.points))
        self.assertEqual({(p['code'],p['date'],p['field']) for p in matches},
                         {(p['code'],p['date'],p['field']) for p in self.points})
        ledger = json.loads((self.output / 'request_ledger.json').read_bytes())
        for entry in ledger:
            scope = entry['response_scope']
            self.assertEqual(scope['rows_before_start'],1)
            self.assertEqual(scope['rows_after_end'],1)
            self.assertEqual(scope['excluded_from_comparison_date_count'],2)
            self.assertEqual(scope['missing_required_points'],[])

    def test_archived_real_response_passes_full_mocked_entry_not_just_parser(self):
        batch = v2.HERE / 'independent_fetch/batch_price_44win_rev2_20261014'
        blob = next(batch.glob('run_*/*.bin'))
        self.real_body = blob.read_bytes()
        metadata = json.loads(blob.with_suffix('.json').read_bytes())
        self.assertEqual(hashlib.sha256(self.real_body).hexdigest(),metadata['sha256'])
        _,windows,points,local = old_guard.verify_bundle(v2.MANIFEST,v2)
        self.windows = [w for w in self.windows if w['code'] != '600001'] + [w for w in windows if w['code'] == '600313']
        self.points = [p for p in self.points if p['code'] != '600001'] + [p for p in points if p['code'] == '600313']
        self.local.update({key:value for key,value in local.items() if key[0] == '600313'})
        self.mode = 'replay_real_first'
        result = self.execute()
        self.assertEqual(result['stop_reason'],'completed')
        self.assertEqual(result['requests_used'],3)
        ledger = json.loads((self.output / 'request_ledger.json').read_bytes())
        self.assertEqual(ledger[0]['code'],'600313')
        self.assertEqual(ledger[0]['response_scope']['raw_rows'],640)
        self.assertEqual(ledger[0]['response_scope']['rows_before_start'],628)
        self.assertEqual(ledger[0]['response_scope']['missing_required_points'],[])
        self.assertEqual(ledger[1]['outcome'],'ok')
        # Only this tempfile mock run completed; the original real batch stays failed.
        self.assertEqual(json.loads((batch / 'stop_record.json').read_bytes())['stop_reason'],'failed:ValueError')

    def test_superset_original_bytes_not_filtered_in_archive(self):
        self.mode = 'superset_ok'
        self.execute()
        blob = next(self.output.glob('run_*/*.bin'))
        raw = blob.read_bytes()
        meta = json.loads(blob.with_suffix('.json').read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),meta['sha256'])
        rows = next(iter(json.loads(raw)['data'].values()))['day']
        self.assertIn('2020-01-02',[r[0] for r in rows])
        self.assertIn('2099-01-05',[r[0] for r in rows])

    def test_superset_missing_point_stops_no_denominator_reduction(self):
        self.mode = 'superset_missing'
        result = self.execute()
        self.assertNotEqual(result['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)
        matches = json.loads((self.output / 'price_match_table_44win.json').read_bytes())
        self.assertEqual(len(matches),len(self.points))
        self.assertIn('MISS_REMOTE',{m['verdict'] for m in matches})
        ledger = json.loads((self.output / 'request_ledger.json').read_bytes())
        self.assertEqual(len(ledger[0]['response_scope']['missing_required_points']),1)

    def test_non_smoke_missing_point_also_stops(self):
        self.mode = 'superset_ordinary_missing'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),3)

    def test_conflicting_extra_duplicate_still_rejected(self):
        self.mode = 'superset_duplicate'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)
        self.assertTrue(list(self.output.glob('run_*/*.bin')))

    def test_invalid_extra_ohlc_not_hidden_by_filtering(self):
        self.mode = 'superset_invalid'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)

    def test_superset_cannot_hide_adjusted_price_keys(self):
        self.mode = 'superset_qfq'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)

    def test_superset_wrong_symbol_rejected(self):
        self.mode = 'superset_wrong_symbol'
        self.assertNotEqual(self.execute()['stop_reason'],'completed')
        self.assertEqual(len(self.calls),1)

    def test_new_cli_without_authorization_refuses(self):
        done = subprocess.run([sys.executable,'-B','-X','utf8',str(api.HERE / 'price_verification_batch_44win_v3.py'),
                               '--fetch',str(self.output)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(done.returncode,2)
        self.assertIn('AUTHORIZATION_REQUIRED',done.stdout)
        self.assertFalse(self.output.exists())


class ScopeAndBindingTests(unittest.TestCase):
    def test_requested_point_outside_request_cannot_be_excused_as_extra(self):
        with self.assertRaisesRegex(ValueError,'FROZEN_POINT_OUTSIDE'):
            guard.select_required({20260105:{'close':10}},'000001',
                [{'code':'000001','date':20260105,'field':'close'}],
                {'start':'20260106','end':'20260110'},1)

    def test_old_manifest_does_not_authorize_new_policy(self):
        with self.assertRaisesRegex(ValueError,'REV3_POLICY'):
            guard.verify_bundle(v2.MANIFEST,api)

    def test_unbound_rev3_implementation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'unbound.json'
            file.write_text(json.dumps({'runtime_revision':3,'response_scope_policy':guard.POLICY,'inputs':[]}))
            with self.assertRaisesRegex(ValueError,'IMPLEMENTATION_NOT_BOUND'):
                guard.verify_bundle(file,api)

    def test_real_archived_response_zero_network_replay(self):
        batch = v2.HERE / 'independent_fetch/batch_price_44win_rev2_20261014'
        stop_before = (batch / 'stop_record.json').read_bytes()
        blob = next(batch.glob('run_*/*.bin'))
        raw = blob.read_bytes()
        metadata = json.loads(blob.with_suffix('.json').read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),metadata['sha256'])
        source = json.loads(raw)['data'][metadata['symbol']]['day']
        points = json.loads((v2.BATCH / 'price_comparison_points_44win_v2.json').read_bytes())['points']
        parsed = api.tencent_rows_to_map(source)
        selected,scope = guard.select_required(parsed,'600313',points,api.plan_from_points(points)['600313'],len(source))
        self.assertEqual(scope['rows_before_start'],628)
        self.assertEqual(scope['rows_after_end'],0)
        self.assertEqual(scope['raw_rows'],640)
        self.assertEqual(scope['required_point_count'],8)
        self.assertEqual(scope['missing_required_points'],[])
        self.assertEqual(len(selected),7)
        _,_,_,local = old_guard.verify_bundle(v2.MANIFEST,v2)
        for point in (p for p in points if p['code'] == '600313'):
            self.assertEqual(api.compare_point(local[('600313',point['date'])][point['field']],
                                              selected[point['date']][point['field']])[0],'MATCH')
        self.assertEqual((batch / 'stop_record.json').read_bytes(),stop_before)


if __name__ == '__main__':
    unittest.main()
