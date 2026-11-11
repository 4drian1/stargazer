"""#18.188/W1 isolated acceptance: realtime intraday feed never touches 7899."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for sub in ('stargazer_desktop', 'stargazer_ml/v5_hybrid/green_common_profile',
            'stargazer_ml/v5_hybrid/risk_r0_v1'):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import hashlib
import io
import json as _json
import gzip as _gzip

import pandas as pd

import intraday_risk_feed_v1 as F
import live_postclose_source_v1 as S
from live_postclose_source_v1 import LocalCapture, FIELDS

DAY = 20261103
CUTOFF = DAY * 1000000 + 100000
HISTORY_ROWS = [dict(code='002585', date=20261102150000, open=8.0, high=8.1, low=7.9, close=8.05, volume=1, amount=8)]
LIVE_ROWS = HISTORY_ROWS + [dict(code='002585', date=DAY * 1000000 + 93100, open=8.1, high=8.2, low=8.0, close=8.15, volume=1, amount=8)]


def _capture(tmp):
    cap = F.RealtimeCapture(Path(tmp) / 'run', budget_root=Path(tmp))
    cap.cache_day = DAY
    cap.required_cutoff = CUTOFF
    return cap


class FeedIsolationTest(unittest.TestCase):
    def test_today_uses_live_lane_not_7899(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            seen = {}

            def fake_snapshot(root, code, P, asof, previous_close, history, budget):
                seen.update(code=code, P=P, asof=asof, previous_close=previous_close, history=history)
                return LIVE_ROWS, dict(sha256='x'), []

            with mock.patch.object(LocalCapture, 'fetch',
                                   side_effect=AssertionError('LEGACY_7899_PATH_CALLED')), \
                    mock.patch.object(F, 'trading_days', return_value=[20261102, DAY]), \
                    mock.patch.object(F, '_native', return_value=(HISTORY_ROWS, {})), \
                    mock.patch('live_minute_source_v1.snapshot', side_effect=fake_snapshot):
                rows, ref = cap.fetch('002585', str(DAY) + '*')
            self.assertEqual(rows, LIVE_ROWS)
            self.assertEqual(ref['provenance'], 'PUBLIC_REALTIME_LANE')
            self.assertEqual(seen['P'], 20261102)
            self.assertAlmostEqual(seen['previous_close'], 8.05)
            self.assertEqual(seen['history'], HISTORY_ROWS)
            self.assertEqual(seen['asof'], CUTOFF)

    def test_month_uses_tdx_local_not_7899(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            calls = []

            def fake_native(code, day):
                calls.append(day)
                return [dict(code=code, date=day * 1000000 + 93100, open=1, high=1, low=1, close=1, volume=1, amount=1)], {}

            with mock.patch.object(LocalCapture, 'fetch',
                                   side_effect=AssertionError('LEGACY_7899_PATH_CALLED')), \
                    mock.patch.object(F, 'trading_days', return_value=[20261030, 20261031, DAY]), \
                    mock.patch.object(F, '_native', side_effect=fake_native):
                rows, ref = cap.fetch('002585', '202610*')
            self.assertEqual(calls, [20261030, 20261031])
            self.assertEqual(ref['provenance'], 'TDX_NATIVE_LOCAL')
            self.assertEqual(len(rows), 2)

    def test_months_helper_output_feeds_fetch_directly(self):
        """months() real output format ('YYYYMM*') must be accepted as-is."""
        from live_postclose_source_v1 import months
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            calls = []

            def fake_native(code, day):
                calls.append(day)
                return [dict(code=code, date=day * 1000000 + 93100, open=1, high=1, low=1, close=1, volume=1, amount=1)], {}

            produced = list(months(20260924, DAY))
            self.assertEqual(produced, ['202609*', '202610*', '202611*'])
            with mock.patch.object(LocalCapture, 'fetch',
                                   side_effect=AssertionError('LEGACY_7899_PATH_CALLED')), \
                    mock.patch.object(F, 'trading_days', return_value=[20260930, 20261030, 20261102, DAY]), \
                    mock.patch.object(F, '_native', side_effect=fake_native):
                for period in produced:
                    rows, ref = cap.fetch('002585', period)
                    self.assertEqual(ref['provenance'], 'TDX_NATIVE_LOCAL')
            self.assertEqual(calls, [20260930, 20261030, 20261102])

    def test_star_period_refused_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            with self.assertRaises(ValueError) as ctx:
                cap.fetch('002585', '*')
            self.assertIn('REALTIME_HISTORY_UNBOUND', str(ctx.exception))

    def test_produce_shared_realtime_routes_and_skips_legacy_cache(self):
        marker = {}

        def fake_live(day, codes, output, progress, *, cutoff=None, index_source=None, capture_factory=None):
            marker['factory'] = capture_factory
            return 'SNAPSHOT', 'CAPTURE', {}

        with mock.patch('shared_risk_context_v1.prepared_snapshot',
                        side_effect=AssertionError('LEGACY_PREPARED_CACHE_USED')), \
                mock.patch('live_postclose_source_v1.produce_live', side_effect=fake_live):
            import shared_risk_context_v1 as S
            out = S.produce_shared(DAY, ['002585'], '/tmp/x', lambda **k: None,
                                   cutoff=CUTOFF, feed='realtime')
        self.assertIs(marker['factory'], F.RealtimeCapture)
        self.assertEqual(out[0], 'SNAPSHOT')

    def test_default_feed_keeps_legacy_path(self):
        with mock.patch('shared_risk_context_v1.prepared_snapshot', return_value='CACHED'):
            import shared_risk_context_v1 as S
            out = S.produce_shared(DAY, ['002585'], '/tmp/x', lambda **k: None, cutoff=CUTOFF)
        self.assertEqual(out, 'CACHED')


class FeedHistoricalDayTest(unittest.TestCase):
    def test_historical_day_uses_tdx_local_not_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            calls = []

            def fake_native(code, day):
                calls.append(day)
                return [dict(code=code, date=day * 1000000 + 93100, open=1, high=1, low=1, close=1, volume=1, amount=1)], {}

            with mock.patch.object(LocalCapture, 'fetch',
                                   side_effect=AssertionError('LEGACY_7899_PATH_CALLED')), \
                    mock.patch('live_minute_source_v1.snapshot',
                               side_effect=AssertionError('LIVE_LANE_CALLED')), \
                    mock.patch.object(F, '_native', side_effect=fake_native):
                rows, ref = cap.fetch('002585', '20261102*')
            self.assertEqual(calls, [20261102])
            self.assertEqual(ref['provenance'], 'TDX_NATIVE_LOCAL')
            self.assertEqual(len(rows), 1)

    def test_illegal_calendar_day_is_format_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            with mock.patch.object(F, '_native', side_effect=AssertionError('NATIVE_CALLED')):
                with self.assertRaises(ValueError) as ctx:
                    cap.fetch('002585', '20261345*')
            self.assertIn('REALTIME_PERIOD_FORMAT', str(ctx.exception))

    def test_february_30_is_format_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            with self.assertRaises(ValueError) as ctx:
                cap.fetch('002585', '20260230*')
            self.assertIn('REALTIME_PERIOD_FORMAT', str(ctx.exception))

    def test_future_day_refused_before_any_local_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            with mock.patch.object(F, '_native', side_effect=AssertionError('NATIVE_CALLED')), \
                    mock.patch('live_minute_source_v1.snapshot',
                               side_effect=AssertionError('LIVE_LANE_CALLED')):
                with self.assertRaises(ValueError) as ctx:
                    cap.fetch('002585', '20261104*')
            self.assertIn('REALTIME_PERIOD_NOT_FINISHED', str(ctx.exception))

    def test_illegal_month_is_format_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = _capture(tmp)
            with mock.patch.object(F, '_native', side_effect=AssertionError('NATIVE_CALLED')):
                with self.assertRaises(ValueError) as ctx:
                    cap.fetch('002585', '202613*')
            self.assertIn('REALTIME_PERIOD_FORMAT', str(ctx.exception))


class ProduceLiveRealtimeScopeTest(unittest.TestCase):
    """Real produce_live through the realtime feed: history fill is local-only,
    the public lane serves scoring codes only, post-cutoff bars never enter."""

    def _frozen_mirror(self, static):
        here = Path(F.__file__).resolve().parent
        mirror = static / 'index_minutes_v1'
        mirror.mkdir(parents=True)
        for name in ('sse', 'cyb'):
            frame = pd.read_parquet(here / 'index_minutes_v1' / (name + '_5min.parquet'))
            frame = frame[frame['ts'] <= CUTOFF]
            buf = io.BytesIO()
            frame.to_parquet(buf, index=False)
            body = buf.getvalue()
            (mirror / (name + '_5min.parquet')).write_bytes(body)
            meta = _json.loads((here / 'index_minutes_v1' / (name + '_5min.meta.json')).read_bytes())
            meta.update(sha256=hashlib.sha256(body).hexdigest(), parquet_bytes=len(body),
                        rows=len(frame), last_complete_ts=CUTOFF)
            (mirror / (name + '_5min.meta.json')).write_text(_json.dumps(meta), encoding='utf-8')
        return mirror

    def test_scoped_history_fill_and_zero_network(self):
        here = Path(F.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            static = tmp / 'static'
            self._frozen_mirror(static)
            (static / 'ths_full_concept_scope_20261017_v1.json').write_bytes(
                (here / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
            (static / 'relative_stall_5m_20261018_v1').mkdir()
            (static / 'relative_stall_5m_20261018_v1' / 'raw_input_ledger.json').write_text('[]', encoding='utf-8')
            scope = _json.loads((static / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
            catalog = pd.read_parquet(scope['source'])
            catalog = catalog[(catalog.source == 'ths') & (catalog.type == 'concept')]
            boards = {r.code: list(r.symbols) for r in catalog.itertuples()}
            universe = {c for members in boards.values() for c in members}
            candidate, inherited = sorted(universe)[0], sorted(universe)[1]

            runtime = tmp / 'runtime'
            prior = runtime / 'run_20261029_091600_test'
            prior.mkdir(parents=True)
            entries = {}
            for code in (candidate, inherited):
                seed = pd.DataFrame([dict(code=code, date=20261029150000, open=1., high=1., low=1.,
                                          close=1., volume=1., amount=1.)], columns=FIELDS)
                buf = io.BytesIO()
                seed.to_parquet(buf, index=False)
                seed_bytes = buf.getvalue()
                seed_path = runtime / ('seed_%s.parquet' % code)
                seed_path.write_bytes(seed_bytes)
                entries[code] = dict(seed=dict(path=str(seed_path), sha256=hashlib.sha256(seed_bytes).hexdigest()),
                                     extensions=[])
            (prior / 'source_manifest.json').write_text(
                _json.dumps(dict(entries=entries, failures={})), encoding='utf-8')
            (prior / 'terminal.json').write_text(_json.dumps(
                dict(status='COMPLETE', data_date=20261029, run_dir=str(prior))), encoding='utf-8')
            output = runtime / ('run_%d_scoped_test' % DAY)
            output.mkdir(parents=True)

            live_rows = [dict(code=candidate, date=DAY * 1000000 + 100000, open=1., high=1., low=1., close=1., volume=1., amount=1.),
                         dict(code=candidate, date=DAY * 1000000 + 100100, open=1., high=1., low=1., close=1., volume=1., amount=1.)]
            # Mimic the real public-lane archive: gzip of the provider's raw JSON
            # (day strings), not standardized rows. Downstream re-read must come
            # from the standardized archive the feed writes itself.
            raw_json = _json.dumps(
                [{'day': '2026-11-03 10:00', 'open': '1.0', 'high': '1.0', 'low': '1.0', 'close': '1.0', 'volume': '1'},
                 {'day': '2026-11-03 10:01', 'open': '1.0', 'high': '1.0', 'low': '1.0', 'close': '1.0', 'volume': '1'}],
                ensure_ascii=False).encode()
            body = _gzip.compress(raw_json, mtime=0)
            archive = output / ('live_%s_%d.json.gz' % (candidate, DAY))
            archive.write_bytes(body)
            live_calls = []

            def fake_snapshot(root, code, P, asof, previous_close, history, budget):
                live_calls.append((code, int(asof)))
                return list(live_rows), dict(path=str(archive), sha256=hashlib.sha256(body).hexdigest()), []

            def fake_native(code, day):
                return [dict(code=code, date=day * 1000000 + 93100, open=1., high=1., low=1.,
                             close=1., volume=1., amount=1.)], {}

            with mock.patch.object(S, 'HERE', static), \
                    mock.patch.object(F, 'MIRROR', static / 'index_minutes_v1'), \
                    mock.patch.object(F, 'trading_days', return_value=[20261029, 20261030, 20261102, DAY]), \
                    mock.patch.object(F, '_native', side_effect=fake_native), \
                    mock.patch('live_minute_source_v1.snapshot', side_effect=fake_snapshot), \
                    mock.patch.object(LocalCapture, 'fetch',
                                      side_effect=AssertionError('LEGACY_7899_PATH_CALLED')):
                snapshot, capture, failures = S.produce_live(
                    DAY, [candidate], output, lambda **k: None,
                    cutoff=CUTOFF, capture_factory=F.RealtimeCapture)

            self.assertEqual(live_calls, [(candidate, CUTOFF)])
            self.assertEqual(failures.get(inherited), 'REALTIME_CONTEXT_SUPPLY_PENDING')
            self.assertIn(candidate, snapshot.stock_minutes.entries)
            self.assertNotIn(inherited, snapshot.stock_minutes.entries)
            entry = snapshot.stock_minutes.entries[candidate]
            ref = entry['extensions'][0]
            self.assertEqual(ref['raw_response_contract'], 'PUBLIC_LANE_RAW_JSON')
            self.assertNotEqual(ref['path'], ref['raw_response_path'])
            self.assertTrue(Path(ref['raw_response_path']).exists())  # raw evidence kept
            frame = snapshot.stock_minutes[candidate]  # re-read via standardized archive
            self.assertEqual(int(frame.date.max()), CUTOFF)
            self.assertLessEqual(int(frame.date.max()), CUTOFF)


if __name__ == '__main__':
    unittest.main()
