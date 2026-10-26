"""Offline rejection and exact native-missing inference tests; original files untouched."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_postclose_scoring_handoff_v1 import make_manifest, input_for
from postclose_repaired_training_v1 import prepare
from postclose_research_scorer_v1 import ResearchScorer, encode, strict_json


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fields, _, parts, _ = prepare()
        cls.frame = parts['test'].iloc[:2]
        cls.manifest = make_manifest(fields, cls.frame.code.unique().tolist())
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'manifest.json'
        cls.path.write_bytes(encode(cls.manifest))
        cls.scorer = ResearchScorer(cls.path)
        cls.payload = input_for(cls.frame, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def changed(self):
        return copy.deepcopy(self.payload)

    def test_01_valid_output_not_probability_or_action(self):
        result = self.scorer.score(self.payload)
        self.assertTrue(all(r['status'] == 'RESEARCH_SCORE' for r in result['rows']))
        self.assertFalse(result['provenance_verified'])
        self.assertFalse(result['deployment_authorized'])
        self.assertFalse(result['independent_validation_passed'])
        self.assertNotIn('buy', result['rows'][0])

    def test_02_required_families_null_is_unscorable(self):
        for family, fields in self.scorer.families.items():
            with self.subTest(family=family):
                p = self.changed()
                for field in fields:
                    p['rows'][0]['features'][field] = None
                row = self.scorer.score(p)['rows'][0]
                self.assertEqual(row['status'], 'UNSCORABLE')
                self.assertIsNone(row['q90_next_session_adverse_excursion'])
                self.assertIn('MISSING_REQUIRED_FAMILY:' + family, row['reasons'])

    def test_03_optional_null_retained(self):
        p = self.changed()
        required = {f for fields in self.scorer.families.values() for f in fields}
        field = next(f for f in self.scorer.fields if f not in required)
        p['rows'][0]['features'][field] = None
        result = self.scorer.score(p)['rows'][0]
        self.assertEqual(result['status'], 'RESEARCH_SCORE')
        self.assertIn(field, result['missing_fields'])

    def test_04_extra_label_or_missing_key_refused(self):
        for extra in (True, False):
            p = self.changed()
            if extra:
                p['rows'][0]['features']['label'] = 1
            else:
                del p['rows'][0]['features'][self.scorer.fields[0]]
            self.assertEqual(self.scorer.score(p)['rows'][0]['status'], 'UNSCORABLE')

    def test_05_nonfinite_not_silently_filled(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            p = self.changed()
            p['rows'][0]['features'][self.scorer.fields[0]] = value
            with self.assertRaises(ValueError):
                self.scorer.score(p)

    def test_06_numeric_string_refused(self):
        p = self.changed()
        p['rows'][0]['features'][self.scorer.fields[0]] = '1.0'
        self.assertEqual(self.scorer.score(p)['rows'][0]['status'], 'UNSCORABLE')

    def test_07_unknown_scope(self):
        p = self.changed()
        p['rows'][0]['code'] = '999999'
        self.assertIsNone(self.scorer.score(p)['rows'][0]['q90_next_session_adverse_excursion'])

    def test_08_time_not_arrival_proof(self):
        for value in (0, 20260230150000, self.payload['rows'][0]['feature_asof_ts'] - 1):
            p = self.changed()
            p['rows'][0]['feature_asof_ts'] = value
            self.assertEqual(self.scorer.score(p)['rows'][0]['status'], 'UNSCORABLE')
        p = self.changed()
        p['rows'][0]['decision_available_ts'] = p['rows'][0]['feature_asof_ts']
        self.assertEqual(self.scorer.score(p)['rows'][0]['status'], 'UNSCORABLE')

    def test_09_batch_mode_and_schema(self):
        for key, value in [('usage_mode', 'LIVE'), ('feature_schema_sha256', 'wrong'),
                           ('candidate_id', 'D3')]:
            p = self.changed()
            p[key] = value
            with self.assertRaises(ValueError):
                self.scorer.score(p)

    def test_10_duplicate_sample_no_overwrite(self):
        p = self.changed()
        p['rows'].append(copy.deepcopy(p['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_SAMPLE'):
            self.scorer.score(p)

    def test_11_input_limits(self):
        p = self.changed()
        p['rows'] = []
        with self.assertRaises(ValueError):
            self.scorer.score(p)
        p['rows'] = [self.payload['rows'][0]] * 513
        with self.assertRaisesRegex(ValueError, 'ROW_LIMIT'):
            self.scorer.score(p)
        p = self.changed()
        p['oversize'] = 'x' * (4 * 1024**2)
        with self.assertRaisesRegex(ValueError, 'INPUT_BYTE_LIMIT'):
            self.scorer.score(p)

    def test_12_artifact_drift_rejected_before_load(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['artifacts']['model']['sha256'] = 'wrong'
        path = Path(self.temp.name) / 'bad_manifest.json'
        path.write_bytes(encode(manifest))
        with self.assertRaisesRegex(ValueError, 'ARTIFACT_HASH_DRIFT:model'):
            ResearchScorer(path)

    def test_13_json_duplicate_and_nonfinite(self):
        for body in ('{"x": NaN}', '{"x": Infinity}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                strict_json(body)

    def test_14_input_not_mutated(self):
        before = encode(self.payload)
        self.scorer.score(self.payload)
        self.assertEqual(before, encode(self.payload))

    def test_15_feature_order_is_registry_order(self):
        p = self.changed()
        p['rows'][0]['features'] = dict(reversed(list(p['rows'][0]['features'].items())))
        self.assertEqual(self.scorer.score(p)['rows'], self.scorer.score(self.payload)['rows'])

    def test_16_one_leader_allowed_but_both_market_required(self):
        p = self.changed()
        p['rows'][0]['features']['context_captured_leader_mean_return_loo_raw'] = None
        if p['rows'][0]['features']['context_current_leader_return_loo_raw'] is None:
            p['rows'][0]['features']['context_current_leader_return_loo_raw'] = 0.01
        self.assertEqual(self.scorer.score(p)['rows'][0]['status'], 'RESEARCH_SCORE')
        p['rows'][0]['features']['cyb_daily_rsi14'] = None
        self.assertEqual(self.scorer.score(p)['rows'][0]['status'], 'UNSCORABLE')

    def test_17_invalid_calendar_day(self):
        p = self.changed()
        p['rows'][0]['day'] = 20260230
        p['rows'][0]['feature_asof_ts'] = 20260230150000
        self.assertEqual(self.scorer.score(p)['rows'][0]['status'], 'UNSCORABLE')

    def test_18_drift_after_loading(self):
        manifest = copy.deepcopy(self.manifest)
        model = Path(self.temp.name) / 'model_copy.txt'
        model.write_bytes(Path(manifest['artifacts']['model']['path']).read_bytes())
        manifest['artifacts']['model']['path'] = str(model)
        path = Path(self.temp.name) / 'copied_manifest.json'
        path.write_bytes(encode(manifest))
        scorer = ResearchScorer(path)
        model.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'ARTIFACT_HASH_DRIFT:model'):
            scorer.score(self.payload)

    def test_19_cli_success_and_read_only(self):
        from postclose_research_scorer_v1 import HERE
        path = Path(self.temp.name) / 'input.json'
        path.write_bytes(encode(self.payload))
        before = {p.name:p.read_bytes() for p in Path(self.temp.name).iterdir() if p.is_file()}
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8',
                                 str(HERE / 'postclose_research_scorer_v1.py'),
                                 '--manifest', str(self.path), '--input', str(path)],
                                capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(strict_json(result.stdout)['status'], 'RESEARCH_ONLY')
        after = {p.name:p.read_bytes() for p in Path(self.temp.name).iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_20_cli_bad_json_exits_two(self):
        from postclose_research_scorer_v1 import HERE
        path = Path(self.temp.name) / 'invalid.json'
        path.write_bytes(b'{"x": NaN}')
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8',
                                 str(HERE / 'postclose_research_scorer_v1.py'),
                                 '--manifest', str(self.path), '--input', str(path)],
                                capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(strict_json(result.stdout)['status'], 'REJECTED')


if __name__ == '__main__':
    unittest.main()
