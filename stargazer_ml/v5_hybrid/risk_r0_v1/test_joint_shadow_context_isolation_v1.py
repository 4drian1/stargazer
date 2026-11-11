"""Per-stock anomaly isolation in joint_shadow_context_v1.build_features.

Uses the frozen local research snapshot (historical mode). A deliberately
broken bar is injected into a copy of a real stock frame, so the failure is
INVALID_SOURCE_PRICE and independent of which cached stocks happen to be
suspect today.

Run: python -X utf8 -m unittest test_joint_shadow_context_isolation_v1 -v
"""
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

import joint_shadow_context_v1 as ctx
from joint_shadow_compute_v2 import ComputeUnavailable
from joint_shadow_snapshot_v1 import produce

CUTOFF = 20260907150000
DAYS_WINDOW = 12
INDEX_SOURCES = {Path('D:/a股数据/沪指/sh999999.lc5'), Path('D:/a股数据/创业板指/sz399006.lc5')}
REQUIRED = ['stock_day_return_raw', 'context_sector_mean_return_loo_raw',
            'sse_15m_day_return_prev_close', 'cyb_15m_day_return_prev_close',
            'new_sse_golden_position']


def _available():
    return all(p.is_file() for p in INDEX_SOURCES)


@unittest.skipUnless(_available(), 'frozen local research snapshot is not present')
class StockIsolationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.snap = produce(CUTOFF, historical=True)[0]
        days = list(cls.snap.days)[-DAYS_WINDOW:]
        assert days[-1] == CUTOFF // 1000000, days[-3:]
        cls.days = days
        cls.index = {k: v[v.date // 1000000 >= days[0]].copy()
                     for k, v in cls.snap.index_5m.items()}
        bound = sorted({c for m in cls.snap.boards.values() for c in m
                        if c in cls.snap.stock_minutes})
        # First 12 bound codes that validate cleanly at this cutoff.
        cls.codes = []
        for code in bound:
            if len(cls.codes) == 12:
                break
            frame = cls.snap.stock_minutes[code]
            if not (frame.date == CUTOFF).any():
                continue
            try:
                ctx._prefix(frame, CUTOFF, cls.snap.receipts.get(code), stock=True,
                            research=True, observed_at=cls.snap.observed_at)
            except ComputeUnavailable:
                continue
            cls.codes.append(code)
        if len(cls.codes) < 11:
            raise unittest.SkipTest('not enough clean bound stocks in snapshot')
        cls.frames = {c: cls.snap.stock_minutes[c] for c in cls.codes}

    def snapshot(self, boards, frames=None):
        return ctx.RiskSnapshot(
            CUTOFF, self.snap.available_at, self.days, frames if frames is not None else self.frames,
            self.index, boards, self.snap.receipts, self.snap.membership_known_at,
            self.snap.membership_scope, observed_at=self.snap.observed_at)

    def candidates(self, codes):
        return [dict(code=c, P=self.days[-2], E=self.days[-1], cutoff_ts=CUTOFF)
                for c in codes]

    def broken(self, code):
        """Copy of a valid frame with one zero-price bar, as a suspension filler."""
        frame = self.frames[code].copy()
        row = frame.index[-1]
        frame.loc[row, ['open', 'high', 'low', 'close']] = 0.0
        return frame

    def build(self, boards, codes, frames=None):
        return ctx.build_features(self.snapshot(boards, frames), self.candidates(codes))

    # 1. whole universe valid
    def test_all_valid_universe(self):
        board = {'b': self.codes}
        frame, x232, x266, audit = self.build(board, self.codes[:2])
        self.assertEqual(audit['invalid_stock_count'], 0)
        self.assertEqual(audit['invalid_stock_sources'], [])
        self.assertEqual(audit['computable_stock_count'], len(self.codes))
        self.assertEqual(frame['mb15_member_coverage'].iloc[0], 1.0)
        self.assertEqual(list(frame.code), self.codes[:2])

    # 2. one non-candidate stock invalid
    def test_one_non_candidate_invalid(self):
        bad = self.codes[-1]
        frames = dict(self.frames)
        frames[bad] = self.broken(bad)
        board = {'b': self.codes}
        frame, x232, x266, audit = self.build(board, self.codes[:2], frames)
        self.assertEqual([r['code'] for r in audit['invalid_stock_sources']], [bad])
        self.assertEqual(audit['invalid_stock_sources'][0]['reason'], 'INVALID_SOURCE_PRICE')
        # Denominator stays the original member count: (n-1)/n, not 1.0.
        cover = frame['mb15_member_coverage'].iloc[0]
        self.assertAlmostEqual(cover, (len(self.codes) - 1) / len(self.codes))
        self.assertNotEqual(cover, 1.0)
        # Normal candidates keep their own stock features untouched.
        base = self.build(board, self.codes[:2])
        for col in [c for c in frame if c.startswith('stock_')
                    and pd.api.types.is_numeric_dtype(frame[c])]:
            np.testing.assert_allclose(frame[col].iloc[:2].to_numpy(dtype=float),
                                       base[0][col].iloc[:2].to_numpy(dtype=float),
                                       equal_nan=True, err_msg=col)

    # 3. several non-candidate stocks invalid
    def test_many_non_candidate_invalid(self):
        bad = self.codes[-2:]
        frames = dict(self.frames)
        for code in bad:
            frames[code] = self.broken(code)
        frame, _, _, audit = self.build({'b': self.codes}, self.codes[:2], frames)
        self.assertEqual(sorted(r['code'] for r in audit['invalid_stock_sources']), sorted(bad))
        self.assertAlmostEqual(frame['mb15_member_coverage'].iloc[0],
                               (len(self.codes) - 2) / len(self.codes))
        self.assertEqual(audit['computable_stock_count'], len(self.codes) - 2)

    # 4. the candidate itself is invalid -> not computable, no normal-looking score
    def test_candidate_invalid_not_computable(self):
        bad = self.codes[1]
        frames = dict(self.frames)
        frames[bad] = self.broken(bad)
        frame, x232, x266, audit = self.build({'b': self.codes}, self.codes[:2], frames)
        self.assertEqual([r['code'] for r in audit['invalid_stock_sources']], [bad])
        self.assertEqual(list(frame.code), self.codes[:2])          # order preserved
        missing = [k for k in REQUIRED if not np.isfinite(float(frame.loc[1, k]))]
        self.assertIn('stock_day_return_raw', missing)
        self.assertEqual(missing, ['stock_day_return_raw'])
        self.assertTrue(np.isfinite(float(frame.loc[0, 'stock_day_return_raw'])))
        self.assertEqual(int(x232.iloc[1].isna().sum()) > int(x232.iloc[0].isna().sum()), True)

    # 5. minute data absent entirely
    def test_missing_minute_data(self):
        absent = self.codes[-1]
        frames = {c: f for c, f in self.frames.items() if c != absent}
        frame, _, _, audit = self.build({'b': self.codes}, self.codes[:2], frames)
        self.assertEqual(audit['missing_stock_sources'], [absent])
        self.assertEqual(audit['invalid_stock_sources'], [])
        self.assertAlmostEqual(frame['mb15_member_coverage'].iloc[0],
                               (len(self.codes) - 1) / len(self.codes))

    # 6. empty board
    def test_empty_board_rejected(self):
        with self.assertRaises(ComputeUnavailable) as caught:
            self.build({'b': self.codes, 'empty': []}, self.codes[:2])
        self.assertIn('INVALID_BOARD_MEMBERSHIP', str(caught.exception))

    # 7. duplicate board members
    def test_duplicate_board_members_rejected(self):
        duplicated = self.codes + [self.codes[0]]
        with self.assertRaises(ComputeUnavailable) as caught:
            self.build({'b': duplicated}, self.codes[:2])
        self.assertIn('INVALID_BOARD_MEMBERSHIP', str(caught.exception))

    # 8. every stock invalid
    def test_all_stocks_invalid(self):
        frames = {c: self.broken(c) for c in self.codes}
        with self.assertRaises(ComputeUnavailable) as caught:
            self.build({'b': self.codes}, self.codes[:2], frames)
        self.assertIn('NO_STOCK_INPUT', str(caught.exception))

    # 9. unknown program faults and contract breaches must still propagate
    def test_unknown_compute_error_propagates(self):
        real = ctx._prefix

        def boom(frame, code, *a, **kw):
            return real(frame, code, *a, **kw)

        with mock.patch.object(ctx, '_prefix',
                               side_effect=ComputeUnavailable('UNREGISTERED_SOURCE_FAULT')):
            with self.assertRaises(ComputeUnavailable) as caught:
                self.build({'b': self.codes}, self.codes[:2])
        self.assertIn('UNREGISTERED_SOURCE_FAULT', str(caught.exception))
        self.assertIs(real, ctx._prefix)

    def test_snapshot_contract_error_propagates(self):
        with mock.patch.object(ctx, '_prefix',
                               side_effect=ComputeUnavailable('SOURCE_RECEIPT_MISSING')):
            with self.assertRaises(ComputeUnavailable) as caught:
                self.build({'b': self.codes}, self.codes[:2])
        self.assertIn('SOURCE_RECEIPT_MISSING', str(caught.exception))

    def test_unknown_exception_propagates(self):
        with mock.patch.object(ctx, '_prefix', side_effect=RuntimeError('kernel bug')):
            with self.assertRaises(RuntimeError):
                self.build({'b': self.codes}, self.codes[:2])

    # classification helper contract
    def test_reason_classification(self):
        self.assertEqual(ctx._stock_data_reason(ComputeUnavailable('INVALID_SOURCE_OHLC')), 'INVALID_SOURCE_OHLC')
        self.assertEqual(ctx._stock_data_reason(
            ComputeUnavailable('INVALID_SOURCE_PRICE:/tmp/a.parquet')), 'INVALID_SOURCE_PRICE')
        self.assertIsNone(ctx._stock_data_reason(ComputeUnavailable('SOURCE_CLOCK_UNBOUND')))
        self.assertIsNone(ctx._stock_data_reason(ComputeUnavailable('RAW_SNAPSHOT_HASH_MISMATCH:x')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
