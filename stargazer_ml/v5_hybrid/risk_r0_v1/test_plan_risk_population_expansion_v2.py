from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from plan_risk_population_expansion_v2 import quotas,mechanical_selection,study_universe,inventory_dates,snapshot_file_set


class ExpansionPlanTests(unittest.TestCase):
    def test_iterator_consumed_once_new_files_are_reported(self):
        self.assertEqual(snapshot_file_set(iter([Path('old')]),iter([Path('old'),Path('new')])),[Path('new')])

    def test_added_files_not_read_missing_frozen_file_blocks(self):
        self.assertEqual(snapshot_file_set([Path('old')],[Path('old'),Path('new')]),[Path('new')])
        with self.assertRaisesRegex(ValueError,'FROZEN_DAILY_FILE_MISSING'):
            snapshot_file_set([Path('old')],[Path('new')])

    def test_largest_remainder_exact_total_and_ties(self):
        self.assertEqual(quotas(dict(GEM=3,SH_MAIN=3,STAR=3,SZ_MAIN=3),5),
                         dict(GEM=2,SH_MAIN=1,STAR=1,SZ_MAIN=1))
        q=quotas(dict(GEM=1,SH_MAIN=2,STAR=7,SZ_MAIN=10),9)
        self.assertEqual(sum(q.values()),9)
        self.assertTrue(all(q[b]<=n for b,n in dict(GEM=1,SH_MAIN=2,STAR=7,SZ_MAIN=10).items()))

    def test_zero_stratum_not_fabricated_and_oversize_rejected(self):
        self.assertEqual(quotas(dict(SZ_MAIN=3),2)['STAR'],0)
        with self.assertRaisesRegex(ValueError,'INVALID_TARGET'):
            quotas(dict(SZ_MAIN=3),4)

    def test_order_invariance_and_no_evidence_or_label_input(self):
        codes=['000001','000002','000008','300001','300002','600001','600002','688001','688002']
        selected=mechanical_selection(codes,5)
        self.assertEqual(selected,mechanical_selection(list(reversed(codes)),5))
        # Mapping values can include arbitrary outcomes/source availability; selection only uses code keys.
        self.assertEqual(selected,mechanical_selection({c:dict(net=100,has_archive=False) for c in codes},5))

    def test_full_target_keeps_codes_with_no_archive(self):
        codes=['000001','300001','600001','688001']
        self.assertEqual(mechanical_selection(codes,4)[0],sorted(codes))

    def test_duplicate_or_excluded_prefix_not_accepted(self):
        for codes in (['000001','000001'],['510300'],['920001']):
            with self.assertRaisesRegex(ValueError,'INVALID_UNIVERSE'):
                mechanical_selection(codes,1)

    def test_2026_only_not_selected_and_old_exit_visible_retained(self):
        stats={'000001':dict(year_rows={'2019':100}), '000002':dict(year_rows={'2026':100}),
               '000008':dict(year_rows={'2018':100})}
        original=deepcopy(stats)
        self.assertEqual(study_universe(stats),['000001'])
        self.assertEqual(stats,original)

    def test_code_date_inventory_not_listing_or_price_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'part.parquet'
            pq.write_table(pa.Table.from_pylist([dict(code='000001',date=20190102),
                dict(code='000001',date=20190102),dict(code='000001',date=20260105),
                dict(code='510300',date=20190102)]),p)
            result,physical,excluded=inventory_dates([p],[20190102,20260105])
            self.assertEqual(physical,4)
            self.assertEqual(result['000001']['year_rows']['2019'],2)  # raw duplicates counted, not certified valid days
            self.assertEqual(result['000001']['last_visible_date_through_2025'],20190102)
            self.assertEqual(result['000001']['last_visible_date'],20260105)
            self.assertEqual(excluded,{'510':1})

    def test_calendar_drift_blocks_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'part.parquet'
            pq.write_table(pa.Table.from_pylist([dict(code='000001',date=20190103)]),p)
            with self.assertRaisesRegex(ValueError,'DATE_NOT_IN_FROZEN_CALENDAR'):
                inventory_dates([p],[20190102])


if __name__=='__main__':
    unittest.main()
