from copy import deepcopy
import unittest

import numpy as np

from risk_failure_diagnostics_v1 import (FIELDS, SCOPE, base_valid, feature_drift, fold_inventory,
                                         group_diagnostics, market_thresholds, regime, tail_stats)


def row(code='000001', p=20200102, y=0, end=20200109):
    return dict(code=code, P=p, label_end_date=end, segment='train', training_eligible=True,
        label_status='LABEL_COMPLETE', y_tail_5d=y, mae_5d=-.1 if y else -.02,
        admission_scope=SCOPE, coverage_verified=False, features={f:.02 for f in FIELDS},
        atr_score=.02, lr=.3, lr_raw=.4, lightgbm=.2, lightgbm_raw=.25,
        lr_retained_primary=True, lightgbm_retained_primary=True, atr_rule_retained_primary=True)


F2 = dict(id='F2', mode='NEW_FIT_PENDING_AUTHORIZATION', train=[20190101,20221231],
          select=[20230101,20230630], calibration=[20230701,20231231], test=[20240101,20241231])


class FoldInventoryTests(unittest.TestCase):
    def test_original_time_purge_can_be_eligible_in_new_fold(self):
        r = row(p=20240628, end=20240705)
        r.update(segment='select', training_eligible=False)
        inventory = fold_inventory([r], F2)
        self.assertEqual(inventory['segments']['test']['n'], 1)
        self.assertEqual(inventory['segments']['test']['old_admitted_false_but_this_fold_eligible'], 1)
        self.assertFalse(inventory['sample_gate_passed'])

    def test_new_boundary_purges_without_reassigning_p(self):
        r = row(p=20221229, end=20230106)
        report = fold_inventory([r], F2)
        self.assertEqual(report['segments']['train']['boundary_purged'], 1)
        self.assertEqual(report['segments']['select']['population_rows'], 0)

    def test_missing_label_never_revives(self):
        r = row(p=20240628, end=20240705)
        r.update(training_eligible=False, label_status='PRICE_EVIDENCE_PENDING', y_tail_5d=None, mae_5d=None)
        self.assertEqual(fold_inventory([r], F2)['segments']['test']['n'], 0)

    def test_non_temporal_quarantine_not_revived(self):
        r = row(p=20240502, end=20240509)
        r['training_eligible'] = False  # Within old select period: not merely old time-boundary purge.
        with self.assertRaisesRegex(ValueError, 'NON_TEMPORAL'):
            fold_inventory([r], F2)

    def test_finite_and_scope_required(self):
        for value in (None, float('nan'), float('inf')):
            r = row()
            r['features']['ret_1d'] = value
            self.assertFalse(base_valid(r))
        r = row()
        r['coverage_verified'] = True
        self.assertFalse(base_valid(r))


class RegimeAndDriftTests(unittest.TestCase):
    def test_cutoffs_equal_weight_dates_not_stock_counts(self):
        rows = []
        for d, vol, repeats in ((20200102,1.,100), (20200103,2.,1), (20200106,3.,1)):
            for i in range(repeats):
                r = row(code=f'{i:06}', p=d)
                r['features']['market_vol_20d'] = vol
                rows.append(r)
        np.testing.assert_allclose(market_thresholds(rows), [5/3,7/3])

    def test_future_and_labels_do_not_change_regimes(self):
        rows = [row(p=d) for d in (20200102,20200103,20200106)]
        for i, r in enumerate(rows):
            r['features']['market_vol_20d'] = .01 + .01*i
        before = market_thresholds(rows)
        other = row(p=20260102)
        other['segment'] = 'description'
        other['features']['market_vol_20d'] = 1000.
        changed = deepcopy(rows)
        for r in changed:
            r['y_tail_5d'] = 1
        self.assertEqual(market_thresholds(changed+[other]), before)
        self.assertEqual(regime(before[0],before), 'LOW')
        self.assertEqual(regime(before[1],before), 'MID')

    def test_conflicting_market_value_blocks(self):
        a, b = row(), row(code='000002')
        b['features']['market_vol_20d'] = .03
        with self.assertRaisesRegex(ValueError, 'CONFLICT'):
            market_thresholds([a,b])

    def test_identical_distribution_psi_zero_constant_null(self):
        rows = [row(code=f'{i:06}') for i in range(20)]
        for i, r in enumerate(rows):
            r['features']['ret_1d'] = i/100
        result = feature_drift(rows, rows)['fields']
        self.assertAlmostEqual(result['ret_1d']['psi'], 0)
        self.assertIsNone(result['market_ret_5d']['psi'])


class RankingAndTailTests(unittest.TestCase):
    def test_stored_masks_inherited_not_reranked(self):
        rows = [row(code=f'{i:06}', y=int(i>=8)) for i in range(10)]
        for i, r in enumerate(rows):
            r['lightgbm'] = .9 if r['y_tail_5d'] else .1
            r['lightgbm_retained_primary'] = i>=2
            r['atr_rule_retained_primary'] = i<8
            r['lr_retained_primary'] = i<8
        d = group_diagnostics(rows)
        self.assertEqual(d['retained']['lightgbm']['tail_count'], 2)
        self.assertEqual(d['retained']['atr_rule']['tail_count'], 0)
        self.assertTrue(d['same_day_retained_counts_paired'])
        self.assertEqual(d['gb_atr_swap_sets']['gb_only']['n'], 2)
        self.assertIsNone(d['retained']['lightgbm']['worst5_mae_mean'])

    def test_exposure_subgroup_not_mislabeled_paired(self):
        a, b = row(), row(code='000002')
        b['lightgbm_retained_primary'] = False
        self.assertFalse(group_diagnostics([a,b])['same_day_retained_counts_paired'])

    def test_pooled_auc_can_hide_bad_same_day_ranking(self):
        rows = []
        for d, npos, nneg, ps, ns in ((20200102,1,10,.05,.1),(20200103,10,1,.8,.9)):
            for i in range(npos+nneg):
                r = row(code=f'{i:06}', p=d, y=int(i<npos))
                r['lightgbm'] = ps if i<npos else ns
                rows.append(r)
        result = group_diagnostics(rows)
        self.assertGreater(result['probability']['lightgbm']['pooled_auc'], .8)
        self.assertEqual(result['ranking']['lightgbm']['same_day_pair_weighted_auc'], 0)

    def test_one_class_day_kept_in_counts_not_auc(self):
        result = group_diagnostics([row(y=0),row(code='000002',y=0)])
        self.assertEqual(result['n'],2)
        self.assertIsNone(result['ranking']['lightgbm']['same_day_pair_weighted_auc'])
        self.assertEqual(result['ranking']['lightgbm']['dates_with_both_classes'],0)

    def test_small_and_empty_tail_not_zero_imputation(self):
        self.assertIsNone(tail_stats([])['tail_rate'])
        self.assertIsNone(tail_stats([row()])['worst5_mae_mean'])


if __name__ == '__main__':
    unittest.main()
