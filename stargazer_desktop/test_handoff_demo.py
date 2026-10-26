"""Handoff regressions: historical browsing never calls a paid provider."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import index_minutes_v1
import paths
import shadow_ui
from test_shadow_ui import pack_dict, CODE, DATE


class HandoffTests(unittest.TestCase):
    def test_historical_http_is_read_only(self):
        import server
        with tempfile.TemporaryDirectory() as folder:
            packs = Path(folder) / 'packs.jsonl'
            raw = pack_dict()
            raw.update(market_date='20260908', asof_ts='20260908150000',
                       query_ts='20260908150000', notes='HISTORICAL_REPLAY')
            raw['risk_context']['asof_ts'] = raw['asof_ts']
            raw['candidates'][0]['trigger'] = None
            packs.write_text(json.dumps(raw) + '\n', encoding='utf-8')
            with patch.object(shadow_ui, 'PACKS_FILE', packs), patch.object(
                    shadow_ui.shadow_journal, 'record_event', side_effect=AssertionError('PAID_CALL')):
                shadow_ui._clear_cache()
                client = server.app.test_client()
                catalog = client.get('/api/shadow_evidence').get_json()
                self.assertEqual(catalog['count'], 1)
                result = client.get('/api/shadow_evidence?code=' + CODE + '&date=20260908').get_json()
                self.assertEqual(result['mode'], 'HISTORICAL_REPLAY')
                self.assertIsNone(result['advice'])
                self.assertEqual(result['asof_ts'], '20260908150000')
                self.assertEqual(client.get('/api/shadow_evidence?code=000001&date=20260908').get_json()['status'], 'UNSCORABLE')
                self.assertEqual(client.get('/api/shadow_evidence?code=000001').status_code, 400)
                shadow_ui._clear_cache()

    def test_missing_index_never_fresh(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(index_minutes_v1, 'MIRROR', Path(folder)):
            Path(folder, 'freshness.json').write_text('{"stale": false, "checked_at": "2026-10-20"}')
            for name in ('sse', 'cyb'):
                Path(folder, name + '_5min.parquet').touch()
                Path(folder, name + '_5min.meta.json').write_text(json.dumps({'last_date': 20261016 if name == 'sse' else None}))
            result = index_minutes_v1.get_index_freshness()
            self.assertTrue(result['stale'])
            self.assertIsNone(result['last_date'])

    def test_packaged_paths_stay_beside_exe(self):
        import importlib
        import trade_plan_v1
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(paths, 'FROZEN', True), patch.object(paths, 'APP_DIR', folder):
                importlib.reload(shadow_ui)
                importlib.reload(trade_plan_v1)
                self.assertEqual(shadow_ui.PACKS_FILE, Path(folder, 'data', 'shadow_packs.jsonl'))
                self.assertEqual(trade_plan_v1.TRADE_PLAN_FILE, Path(folder, 'data', 'trade_plan.json'))
            importlib.reload(shadow_ui)
            importlib.reload(trade_plan_v1)


if __name__ == '__main__':
    unittest.main()
