import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import common_candidate_aggregation_v1 as agg


class CandidatePreparationTests(unittest.TestCase):
    def setUp(self):
        self.population = pd.DataFrame({'segment': ['train', 'train'], 'code': ['A', 'B'],
                                       agg.TIME: [20250103101500, 20250103101500]})
        rows = []
        for code, board, value in [('A', 'x', 1.), ('A', 'y', 3.), ('B', 'z', 8.)]:
            row = self.population[self.population.code.eq(code)].iloc[0].to_dict()
            row.update(board=board, **dict.fromkeys(agg.OWN_METRICS, value))
            rows.append(row)
        self.rows = pd.DataFrame(rows)
        self.memberships = {'A': ['x', 'y'], 'B': ['z']}

    def aggregate(self, rows=None):
        return agg.aggregate_own(self.population, self.rows if rows is None else rows,
                                 5, self.memberships)

    def test_equal_candidate_weight_not_board_row_expansion(self):
        result, audit = self.aggregate()
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result['ccm_v1_5_own_deceleration'], [2, 8])
        self.assertEqual(result['ccm_v1_5_own_deceleration'].mean(), 5)
        self.assertEqual(audit['ccm_v1_5_own_expected_boards'].tolist(), [2, 1])

    def test_missing_unknown_not_zero(self):
        rows = self.rows.copy()
        rows.loc[rows.board.eq('x'), list(agg.OWN_METRICS)] = np.nan
        result, audit = self.aggregate(rows)
        self.assertEqual(result['ccm_v1_5_own_deceleration'].iloc[0], 3)
        self.assertEqual(audit['ccm_v1_5_own_deceleration_known_boards'].iloc[0], 1)
        rows.loc[rows.code.eq('A'), list(agg.OWN_METRICS)] = np.nan
        self.assertTrue(np.isnan(self.aggregate(rows)[0]['ccm_v1_5_own_deceleration'].iloc[0]))

    def test_duplicate_missing_unexpected_and_infinite_assignments_rejected(self):
        bad = [pd.concat([self.rows, self.rows.iloc[:1]]), self.rows.iloc[1:]]
        changed = self.rows.copy(); changed.loc[0, 'board'] = 'unknown'; bad.append(changed)
        changed = self.rows.copy(); changed.loc[0, 'deceleration'] = np.inf; bad.append(changed)
        for rows in bad:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.aggregate(rows)

    def test_attach_preserves_labels_old_features_keys_order_and_index(self):
        frame = self.population[['code', agg.TIME]].copy()
        frame['old'] = [1., np.nan]; frame['label'] = [0., .1]; frame.index = [8, 2]
        side = self.population.copy()
        for field in agg.learning_fields():
            side[field] = [2., 4.]
        result = agg.attach_sidecars({'train': frame}, side.iloc[::-1], ['old'], expected_rows=2)
        pd.testing.assert_frame_equal(result['train'][frame.columns], frame)
        self.assertEqual(result['train'][agg.learning_fields()[0]].tolist(), [2, 4])
        for field in ('coverage', 'label'):
            with self.assertRaisesRegex(ValueError, 'SCHEMA'):
                agg.attach_sidecars({'train': frame}, side.assign(**{field: 1}), ['old'], 2)
        side[agg.learning_fields()[0]] = np.nan
        with self.assertRaisesRegex(ValueError, 'ALL_NULL_TRAIN'):
            agg.attach_sidecars({'train': frame}, side, ['old'], 2)

    def test_learning_schema(self):
        fields = agg.learning_fields()
        self.assertEqual(len(fields), 34)
        self.assertEqual(len(set(fields)), 34)
        self.assertFalse(any('coverage' in f or 'count' in f or 'denominator' in f for f in fields))

    def test_existing_output_refused_before_any_replay(self):
        import verify_continuous_motion_common_coverage_v1 as verifier
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'evidence'; out.mkdir()
            sentinel = out / 'report.json'; sentinel.write_bytes(b'unchanged')
            with patch.object(verifier, 'OUT_DIR', out), patch.object(verifier, 'arrays') as replay:
                with self.assertRaises(FileExistsError):
                    verifier.run()
                replay.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'unchanged')


if __name__ == '__main__':
    unittest.main()
