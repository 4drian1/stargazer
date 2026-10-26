import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import msgpack
import numpy as np
import pandas as pd

import postclose_daily_job_v1 as job
from postclose_research_scorer_v1 import HERE,ResearchScorer,strict_json,encode,sha
from postclose_local_source_bridge_v1 import load_local_features
from read_postclose_incremental_v2 import load_refreshed,ROOT


class DailyJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer=ResearchScorer();cls.old,_=load_local_features(cls.scorer)
        cls.calendar=sorted(cls.old.day.unique().tolist())+[20261019]
        cls.refreshed,_=load_refreshed()
        row=cls.refreshed[cls.refreshed.day.eq(20261016)].iloc[0]
        cls.background={f:None if pd.isna(row[f]) else float(row[f]) for f in job.background_fields(cls.scorer.fields)}

    def inputs(self,root,day=20261019):
        source=Path(root)/'source.txt';source.write_bytes(b'SYNTHETIC_TEST_INPUT')
        refs=[dict(path=str(source),sha256=sha(source.read_bytes()))]
        calendar=Path(root)/'calendar.json';calendar.write_bytes(encode(dict(kind='TRADING_CALENDAR',source_refs=refs,days=self.calendar)))
        background=Path(root)/'background.json';background.write_bytes(encode(dict(kind='POSTCLOSE_MARKET_GOLDEN',source_refs=refs,day=day,features=self.background)))
        return calendar,background

    def test_01_cached_check_and_run_no_fetch(self):
        with patch('urllib.request.build_opener',side_effect=AssertionError('NO_NETWORK')):
            plan,_,_=job.plan(20261016)
            self.assertEqual(plan['status'],'CACHED_READY_NO_FETCH')
            result=job.run(20261016)
            self.assertEqual(result['requests'],0)
            self.assertEqual(sum(r['status']=='RESEARCH_SCORE' for r in result['rows']),39)

    def test_02_future_missing_inputs_blocks_without_fetch(self):
        with patch('urllib.request.build_opener',side_effect=AssertionError('NO_NETWORK')):
            result=job.run(20261019)
        self.assertFalse(result['ready'])
        self.assertIn('CALENDAR_INPUT_REQUIRED',result['blockers'])
        self.assertIn('SAME_DAY_MARKET_GOLDEN_INPUT_REQUIRED',result['blockers'])

    def test_03_complete_plan_bounds_one_day(self):
        with tempfile.TemporaryDirectory() as root:
            calendar,background=self.inputs(root)
            plan,_,bound=job.plan(20261019,calendar,background)
            self.assertTrue(plan['ready']);self.assertEqual(plan['max_requests'],44)
            self.assertEqual(bound['previous_day'],20261016)

    def test_04_wrong_background_date_refuses_old_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            calendar,background=self.inputs(root,20261016)
            with self.assertRaisesRegex(ValueError,'NO_FALLBACK'):job.plan(20261019,calendar,background)

    def test_05_source_hash_drift(self):
        with tempfile.TemporaryDirectory() as root:
            calendar,background=self.inputs(root);(Path(root)/'source.txt').write_bytes(b'CHANGED')
            with self.assertRaisesRegex(ValueError,'HASH_DRIFT'):job.plan(20261019,calendar,background)

    def test_06_background_missing_or_foreign_field(self):
        for field,value in [('golden_position',None),('label',1)]:
            with tempfile.TemporaryDirectory() as root:
                calendar,background=self.inputs(root);data=strict_json(background.read_bytes())
                data['features'][field]=value;background.write_bytes(encode(data))
                if field=='label':
                    with self.assertRaisesRegex(ValueError,'SCHEMA'):job.plan(20261019,calendar,background)
                else:self.assertIn('REQUIRED_BACKGROUND_MISSING',job.plan(20261019,calendar,background)[0]['blockers'])

    def test_07_bad_calendar_and_oversized_catchup(self):
        for days in [[20261019,20261016],[20261016,20261016],[20260230],[20261016.0]]:
            with self.assertRaises(ValueError):job.calendar_days(days)

    def test_08_generic_context_equals_real_two_day_builder(self):
        import pyarrow.parquet as pq
        data={}
        for d,root,entry in job.prior_daily_archives():
            raw=gzip.decompress((root/entry['file']).read_bytes())
            data[d]=job.valid_daily(job.structured_pairs(msgpack.unpackb(raw,raw=False,strict_map_key=False),'日k','*'),d)
        catalog=strict_json((HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
        members=sorted([r for r in pq.read_table(catalog['source']).to_pylist() if r['source']=='ths' and r['type']=='concept'],key=lambda r:r['code'])
        actual=job.context_day(data[20261015],data[20261016],members,self.scorer.manifest['research_codes'],20261016).set_index('code')
        expected=self.refreshed[self.refreshed.day.eq(20261016)].set_index('code')
        fields=[f for f in actual if f.startswith('context_')]
        np.testing.assert_allclose(actual[fields],expected.loc[actual.index,fields],rtol=0,atol=1e-12,equal_nan=True)

    def test_09_job_failure_terminal_no_restart(self):
        with tempfile.TemporaryDirectory() as root:
            calendar,background=self.inputs(root);jobs=Path(root)/'jobs'
            with patch.object(job,'JOBS',jobs),patch('urllib.request.build_opener') as opener:
                opener.return_value.open.side_effect=TimeoutError('synthetic timeout')
                with self.assertRaises(TimeoutError):job.run(20261019,calendar,background)
                stop=strict_json((jobs/'20261019/stop_record.json').read_bytes())
                self.assertEqual(stop['status'],'FAILED');self.assertEqual(stop['requests'],1)
                blocked=job.run(20261019,calendar,background)
                self.assertIn('DAY_JOB_EXISTS_NO_RESTART',blocked['blockers'])
                self.assertEqual(opener.return_value.open.call_count,1)

    def test_10_successful_mock_end_to_end_is_not_validation(self):
        from urllib.parse import urlparse,parse_qs
        entry=next(e for d,_,e in job.prior_daily_archives() if d==20261016)
        raw=gzip.decompress((ROOT/entry['file']).read_bytes())
        parsed=job.structured_pairs(msgpack.unpackb(raw,raw=False,strict_map_key=False),'日k','*')
        future=[]
        for key,row in parsed:
            row=dict(row,date=20261019)
            future.append([f'日k:{row["code"]}:20261019',row])
        def response(url,timeout):
            params=parse_qs(urlparse(url).query)
            self.assertIn('k1',params);self.assertIn('k2',params)
            return io.BytesIO(msgpack.packb(future if params['t']==['日k'] else [],use_bin_type=True))
        with tempfile.TemporaryDirectory() as root:
            calendar,background=self.inputs(root);jobs=Path(root)/'jobs'
            with patch.object(job,'JOBS',jobs),patch('urllib.request.build_opener') as opener:
                opener.return_value.open.side_effect=response
                result=job.run(20261019,calendar,background)
                self.assertEqual(opener.return_value.open.call_count,44)
                self.assertEqual(len(result['rows']),43)
                self.assertFalse(result['forward_validation_eligible'])
                stop=strict_json((jobs/'20261019/stop_record.json').read_bytes())
                self.assertEqual(stop['status'],'COMPLETE_RESEARCH_ONLY')

    def test_11_risk_not_profit_boundary(self):
        spec=strict_json((HERE/'contracts/RISK_ONLY_EVALUATION_BOUNDARY_20261018.json').read_bytes())
        self.assertIn('trading_profit',spec['excluded_as_standalone_risk_model_gates'])
        self.assertIn('strategy_max_drawdown',spec['excluded_as_standalone_risk_model_gates'])
        self.assertIn('empirical_quantile_coverage',spec['risk_evaluation'])
        self.assertFalse(spec['q90_is_probability'])


if __name__=='__main__':unittest.main()
