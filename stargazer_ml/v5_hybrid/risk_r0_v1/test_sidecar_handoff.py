import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import sidecar_rebuild_v1 as rebuild


class SidecarHandoffTests(unittest.TestCase):
    def test_cutoff_values_names_and_own_output(self):
        def markets(frame, days, frequency):
            n = len(days) * (240 // frequency)
            return list(range(n - 1)) + [rebuild.CUTOFF_TS], {
                'sse': np.arange(n, dtype=float) + 100,
                'cyb': np.arange(n, dtype=float) + 200}

        def summary(board, market, frequency, prefix, own=False):
            n = len(market['sse'])
            fields = {prefix + 'positive_share': np.ones(n)}
            for name in ('sse', 'cyb'):
                for flag in ('nonpositive', 'falling'):
                    fields[prefix + name + '_positive_' + flag + '_rate'] = np.linspace(0, 1, n)
                    fields[prefix + name + '_positive_' + flag + '_coverage'] = np.ones(n)
            return fields

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'golden_anchor_replay_20261017_v2').mkdir()
            (path / 'golden_anchor_replay_20261017_v2' / 'replay.json').write_text(
                json.dumps({'states': [{'asof_day': 20260904, 'low': 100, 'high': 200}]}))
            with patch.object(rebuild, 'HERE', path), patch.object(pd, 'read_parquet', return_value=pd.DataFrame()), patch.object(
                    rebuild, 'market_arrays', side_effect=markets), patch.object(
                    rebuild, 'matrix_features', return_value={}), patch.object(rebuild, 'summarize', side_effect=summary):
                shared, nh, own, nh_own = rebuild.shared_and_nh(
                    {5: np.zeros((1, 96)), 15: np.zeros((1, 32))}, [[0]], {'a': 0}, ['a'])
            self.assertEqual(nh['nh_mb5_sse_positive_common_falling_rate'], 1.)
            self.assertEqual(nh_own['a']['nh_own15_cyb_positive_common_nonpositive_rate'], 1.)
            self.assertEqual(own['a']['own15_positive_share'], 1.)
            self.assertEqual(shared['new_sse_golden_position'], .31)


if __name__ == '__main__':
    unittest.main()
