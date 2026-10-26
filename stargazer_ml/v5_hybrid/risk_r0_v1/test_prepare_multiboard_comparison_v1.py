"""No-network, no-fit checks for the three-arm preparation and shared eligibility."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import prepare_multiboard_comparison_v1 as prep
from joint_fit_preparation_v1 import fit_lr_preprocessing, transform_lr, native_missing_view


def proposals():
    old = json.loads((prep.HERE / 'joint_momentum_review_20261017_v1/fit_review_config_v2.json').read_bytes())
    return old, prep.metadata()[1]


class ComparisonTests(unittest.TestCase):
    def test_exact_three_arms_and_separate_golden_update(self):
        old, spec = proposals()
        arms = prep.make_arms(old, spec)
        self.assertEqual([len(v) for v in arms.values()], [100, 100, 200])
        self.assertEqual(sum(a != b for a, b in zip(arms['B0'], arms['B1'])), 6)
        self.assertEqual(arms['B1'], arms['M1'][:100])
        self.assertFalse(set(old['feature_groups']['golden_position']) & set(arms['M1']))

    def test_golden_order_drift_refused(self):
        old, spec = proposals()
        spec['shared'].reverse()
        with self.assertRaisesRegex(ValueError, 'GOLDEN_MAPPING'):
            prep.make_arms(old, spec)

    def test_metadata_and_duplicate_whitelists_refused(self):
        old, spec = proposals()
        old['feature_groups']['stock'][0] = 'y_tail_1200m'
        with self.assertRaisesRegex(ValueError, 'FORBIDDEN'):
            prep.make_arms(old, spec)
        old, spec = proposals()
        old['feature_groups']['stock'][0] = old['feature_groups']['stock'][1]
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            prep.make_arms(old, spec)

    def test_common_population_ignores_target_and_coverage(self):
        frame = pd.DataFrame({'return_a': [np.nan, 0., 1.], 'return_b': [np.nan, 1., 0.],
                              'coverage': [1., 1., 1.], 'y_tail_1200m': [0, 1, 0]})
        groups = {'a': ['return_a'], 'b': ['return_b']}
        mask, _ = prep.common_population(frame, groups)
        self.assertEqual(mask.tolist(), [False, True, True])
        frame['y_tail_1200m'] = 1 - frame.y_tail_1200m
        np.testing.assert_array_equal(mask, prep.common_population(frame, groups)[0])

    def test_one_known_current_or_captured_leader_suffices(self):
        frame = pd.DataFrame({'current': [np.nan, -0.1, np.nan], 'captured': [-0.1, np.nan, np.nan]})
        self.assertEqual(prep.common_population(frame, {'leader': list(frame)})[0].tolist(), [True, True, False])

    def test_unknown_or_infinite_core_not_safety(self):
        with self.assertRaisesRegex(ValueError, 'INFINITE_CORE'):
            prep.common_population(pd.DataFrame({'v': [np.inf]}), {'core': ['v']})
        self.assertFalse(prep.common_population(pd.DataFrame({'v': [np.nan]}), {'core': ['v']})[0][0])

    def test_lr_train_only_and_future_values_do_not_change_state(self):
        train = pd.DataFrame({'v': [1., np.nan, 3.], 'constant': [5., 5., 5.]})
        state = fit_lr_preprocessing(train)
        before = state['median'].copy()
        transform_lr(pd.DataFrame({'v': [9999.], 'constant': [5.]}), state)
        np.testing.assert_array_equal(before, state['median'])
        self.assertEqual(state['scale'][1], 1.)
        self.assertEqual(transform_lr(train, state).shape, (3, 4))
        self.assertEqual(native_missing_view(train).shape, (3, 4))
        with self.assertRaisesRegex(ValueError, 'TRAIN_ONLY'):
            fit_lr_preprocessing(train, phase='test')

    def test_all_null_and_infinite_training_inputs_refused(self):
        with self.assertRaisesRegex(ValueError, 'ALL_NULL'):
            fit_lr_preprocessing(pd.DataFrame({'empty': [np.nan]}))
        with self.assertRaisesRegex(ValueError, 'INFINITE'):
            fit_lr_preprocessing(pd.DataFrame({'bad': [np.inf]}))

    def test_neutral_ties_independent_of_row_order(self):
        codes = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
        times, scores = np.repeat(20250102100000, 6), np.zeros(6)
        keep, supported = prep.neutral_keep(codes, times, scores)
        self.assertEqual(int((~keep).sum()), 1)
        self.assertTrue(supported.all())
        reverse, _ = prep.neutral_keep(codes[::-1], times[::-1], scores[::-1])
        np.testing.assert_array_equal(keep, reverse[::-1])

    def test_cutoffs_separate_tiny_groups_not_filterable(self):
        codes = list('ABCDE') + list('ABC')
        keep, supported = prep.neutral_keep(codes, [1]*5 + [2]*3, np.arange(8))
        self.assertEqual(supported.tolist(), [True]*5 + [False]*3)
        self.assertTrue(keep[5:].all())
        self.assertFalse(keep[4])

    def test_duplicate_or_nonfinite_scores_refused(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            prep.neutral_keep(['A', 'A'], [1, 1], [0., 0.])
        with self.assertRaisesRegex(ValueError, 'INVALID_RANKING'):
            prep.neutral_keep(['A'], [1], [np.nan])

    def test_time_label_purge_and_threshold(self):
        bounds = {s: {'start': start, 'end': end} for s, start, end in [
            ('train', 20250102, 20250930), ('select', 20251001, 20251231),
            ('calibration', 20260101, 20260331), ('test', 20260401, 20260928)]}
        good = pd.DataFrame({'code': ['A'], prep.TIME: [20250102100000],
            'label_end_time': [20250103100000], 'proposed_segment_name': ['train'],
            'y_tail_1200m': [1], 'mae_1200m': [-0.08]})
        prep.validate_times_labels(good, 'train', bounds)
        for key, val, reason in [('label_end_time', 20251001000000, 'CROSSES'),
                                 ('label_end_time', 20250102100000, 'STRICTLY_FUTURE'),
                                 ('label_end_time', 2025010310000, 'TIMESTAMP'),
                                 ('y_tail_1200m', 0, 'THRESHOLD_CONFLICT'),
                                 ('mae_1200m', np.nan, 'INVALID_TARGET'),
                                 ('proposed_segment_name', 'test', 'SEGMENT_MISMATCH')]:
            broken = good.copy()
            broken[key] = val
            with self.subTest(key=key, value=val), self.assertRaisesRegex(ValueError, reason):
                prep.validate_times_labels(broken, 'train', bounds)

    def test_frozen_config_drift_refused_by_readonly_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_bytes(b'{}')
            report = {'inputs': {}, 'outputs': {'config.json': {'sha256': prep.sha(root / 'config.json')}},
                      'status': 'PASS_FIXED_COMPARISON_PREPARATION_NOT_FIT', 'technical_blockers': [], 'segments': {}}
            (root / 'report.json').write_bytes(prep.encoded(report))
            (root / 'stop_record.json').write_bytes(prep.encoded({'status': 'COMPLETE'}))
            with patch.object(prep, 'OUT', root):
                result = prep.check()
                self.assertTrue(result['preparation_ready'])
                self.assertFalse(result['training_ready'])
                self.assertFalse(result['fit_authorized'])
                (root / 'config.json').write_bytes(b'{"fit_authorized":true}')
                with self.assertRaisesRegex(ValueError, 'ARTIFACT_DRIFT'):
                    prep.check()

    def test_terminal_output_cannot_be_restarted(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(prep, 'OUT', Path(temporary)):
            with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS_NO_RESTART'):
                prep.prepare()

    def test_real_cohort_no_observation_changes(self):
        old, spec = proposals()
        arms = prep.make_arms(old, spec)
        total = 0
        for segment in prep.SEGMENTS:
            frame = prep.read_segment(segment)
            mask, _ = prep.common_population(frame)
            modified = frame.copy()
            modified['y_tail_1200m'] = 1 - modified.y_tail_1200m
            modified['mae_1200m'] = -100.
            np.testing.assert_array_equal(mask, prep.common_population(modified)[0])
            self.assertTrue(set(arms['M1']) <= set(frame))
            total += len(frame)
        self.assertEqual(total, 100704)


if __name__ == '__main__':
    unittest.main()
