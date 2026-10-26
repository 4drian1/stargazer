"""Real frozen weights with old feature examples and synthetic clocks; zero collection."""
import copy
import tempfile
import unittest
from pathlib import Path

from postclose_forward_offline_v1 import OfflineAdapter, OfflineStore, MODE, join_labels
from postclose_research_scorer_v1 import DEFAULT_PACKAGE, strict_json, encode, sha
from postclose_next_period_evaluation_v1 import descriptive_evaluation


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = OfflineAdapter()
        examples = strict_json((DEFAULT_PACKAGE.parent/'example_input.json').read_bytes())
        cls.inputs = [dict(code=r['code'],features=r['features'],eligibility_known_ts=20261019150000,
                          data_arrival_ts=20261019160000,feature_max_available_at=20261019160100,
                          scored_at=20261019160200) for r in examples['rows']]
        cls.forecast = cls.adapter.forecast(20261019,cls.inputs,20261020,20261019160300,mode=MODE)

    def labels(self):
        return dict(mode=MODE,forecast_sha256=sha(encode(self.forecast)),rows=[
            dict(code=r['code'],day=r['day'],label_state='PENDING',data_complete_time=None,
                 next_session_adverse_excursion=None) for r in self.forecast['rows']])

    def join(self, labels, forecast=None):
        return join_labels(forecast or self.forecast,labels,20261021180000,
            exposure_end=20261016,freeze_ts=20261017180000,next_sessions={20261019:20261020},
            expected_codes=self.adapter.scorer.manifest['research_codes'])

    def test_01_full_population_missing_inputs_preserved(self):
        rows = self.forecast['rows']
        self.assertEqual(len(rows),43)
        self.assertEqual(sum(r['score_state']=='SCORED' for r in rows),2)
        self.assertEqual(sum(r['reasons']==['MISSING_INPUT'] for r in rows),41)

    def test_02_live_mode_refused(self):
        with self.assertRaises(PermissionError):
            self.adapter.forecast(20261019,self.inputs,20261020,20261019160300,mode='LIVE')

    def test_03_declared_clock_order_enforced(self):
        for field, value in [('eligibility_known_ts',20261019160500),
                              ('data_arrival_ts',20261019145900),('scored_at',20261020093000)]:
            data=copy.deepcopy(self.inputs); data[0][field]=value
            with self.assertRaises(ValueError):
                self.adapter.forecast(20261019,data,20261020,20261019160300,mode=MODE)

    def test_04_missing_time_not_low_risk(self):
        data=copy.deepcopy(self.inputs); data[0]['data_arrival_ts']=None
        out=self.adapter.forecast(20261019,data,20261020,20261019160300,mode=MODE)
        row=next(r for r in out['rows'] if r['code']==data[0]['code'])
        self.assertEqual(row['reasons'],['UNKNOWN_TIME'])
        self.assertIsNone(row['q90_adverse'])
        self.assertEqual(len(out['rows']),43)

    def test_05_missing_required_family_not_low_risk(self):
        data=copy.deepcopy(self.inputs); data[0]['features']['golden_position']=None
        out=self.adapter.forecast(20261019,data,20261020,20261019160300,mode=MODE)
        row=next(r for r in out['rows'] if r['code']==data[0]['code'])
        self.assertEqual(row['score_state'],'UNSCORABLE')
        self.assertIsNone(row['q90_adverse'])

    def test_06_duplicate_or_extra_candidates_refused(self):
        for data in [self.inputs+[self.inputs[0]], [dict(self.inputs[0],code='999999')]]:
            with self.assertRaises(ValueError):
                self.adapter.forecast(20261019,data,20261020,20261019160300,mode=MODE)

    def test_07_frozen_model_numerical_bridge(self):
        import pandas as pd
        path=Path(self.adapter.scorer.manifest['artifacts']['predictions']['path'])
        old=pd.read_parquet(path).set_index(['code','day'])
        example=strict_json((DEFAULT_PACKAGE.parent/'example_input.json').read_bytes())
        for r in example['rows']:
            record=next(x for x in self.forecast['rows'] if x['code']==r['code'])
            saved=old.loc[(r['code'],r['day'])]
            self.assertAlmostEqual(record['q90_adverse'],saved.q90_adverse,places=12)
            self.assertAlmostEqual(record['q90_linear'],saved.q90_linear,places=12)
        self.assertTrue(self.forecast['synthetic_clock'])
        self.assertFalse(self.forecast['provenance_verified'])

    def test_08_restart_readback_and_idempotency(self):
        with tempfile.TemporaryDirectory() as root:
            store=OfflineStore(root)
            name=store.freeze('forecast','20261019',self.forecast)
            before=(Path(root)/name).read_bytes()
            restarted=OfflineStore(root)
            self.assertEqual(restarted.read(name),self.forecast)
            restarted.freeze('forecast','20261019',self.forecast)
            self.assertEqual((Path(root)/name).read_bytes(),before)

    def test_09_immutable_forecast_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            store=OfflineStore(root); store.freeze('forecast','20261019',self.forecast)
            changed=copy.deepcopy(self.forecast); changed['rows'][0]['q90_adverse']=.99
            with self.assertRaisesRegex(ValueError,'CONFLICT'):
                store.freeze('forecast','20261019',changed)

    def test_10_tampering_detected(self):
        with tempfile.TemporaryDirectory() as root:
            store=OfflineStore(root); name=store.freeze('forecast','20261019',self.forecast)
            path=Path(root)/name; item=strict_json(path.read_bytes())
            item['content']['day']=20261020; path.write_bytes(encode(item))
            with self.assertRaisesRegex(ValueError,'INTEGRITY'):
                store.read(name)

    def test_11_budget_preflight_no_write(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'BUDGET'):
                OfflineStore(root,cap=1).freeze('forecast','20261019',self.forecast)
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_12_missing_or_extra_outcomes_rejected(self):
        for extra in [False,True]:
            labels=self.labels()
            if extra: labels['rows'].append(dict(labels['rows'][0],code='999999'))
            else: labels['rows'].pop()
            with self.assertRaisesRegex(ValueError,'POPULATION'):
                self.join(labels)

    def test_13_wrong_forecast_reference(self):
        labels=self.labels();labels['forecast_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'REFERENCE'):
            self.join(labels)

    def test_14_outcome_cannot_override_prediction(self):
        labels=self.labels(); labels['rows'][0]['q90_adverse']=0
        with self.assertRaisesRegex(ValueError,'OVERWRITE'):
            self.join(labels)

    def test_15_zero_filled_pending_refused(self):
        labels=self.labels(); labels['rows'][0]['next_session_adverse_excursion']=0
        with self.assertRaisesRegex(ValueError,'NULL'):
            self.join(labels)

    def test_16_immature_label_refused(self):
        labels=self.labels(); labels['rows'][0].update(label_state='COMPLETE',
            data_complete_time=20261022180000,next_session_adverse_excursion=.03)
        with self.assertRaisesRegex(ValueError,'MATURE'):
            self.join(labels)

    def test_17_labels_append_and_full_join_end_to_end(self):
        labels=self.labels()
        with tempfile.TemporaryDirectory() as root:
            store=OfflineStore(root); name=store.freeze('forecast','20261019',self.forecast)
            old=store.freeze('labels',sha(encode(labels)),labels)
            new=copy.deepcopy(labels)
            codes={x['code'] for x in self.inputs}
            for row in new['rows']:
                if row['code'] in codes:
                    row.update(label_state='COMPLETE',data_complete_time=20261020180000,
                               next_session_adverse_excursion=.03)
            updated=store.freeze('labels',sha(encode(new)),new)
            replay=OfflineStore(root)
            self.assertEqual(replay.read(old),labels)
            self.assertEqual(replay.read(name),self.forecast)
            joined=self.join(replay.read(updated),replay.read(name))
            result=descriptive_evaluation(joined,.04,[.02,.04,.06,.08])
            self.assertEqual(result['population']['rows'],43)
            self.assertEqual(result['metrics']['rows'],2)
            self.assertAlmostEqual(result['evaluated_fraction'],2/43)
            self.assertFalse(result['independent_validation_passed'])

    def test_18_population_cannot_be_shrunk_by_caller(self):
        forecast=copy.deepcopy(self.forecast)
        forecast['rows']=forecast['rows'][:2]
        forecast['population_codes']=[r['code'] for r in forecast['rows']]
        labels=self.labels();labels['forecast_sha256']=sha(encode(forecast))
        with self.assertRaisesRegex(ValueError,'FROZEN_POPULATION'):
            self.join(labels,forecast)


if __name__=='__main__':
    unittest.main()
