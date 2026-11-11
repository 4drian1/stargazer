"""Selector has a separate binding contract; risk-manifest checks alone miss it."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

GCP=Path(__file__).resolve().parent.parent/'stargazer_ml/v5_hybrid/green_common_profile'
sys.path.insert(0,str(GCP))
import tarot_live_v2 as selector


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


class SelectorBindingTests(unittest.TestCase):
    def test_actual_selection_manifest_and_weights_match(self):
        current=selector.verified_manifest(SimpleNamespace(sha=sha))
        old=json.loads((GCP/'tarot_live_manifest_v4_20261023_flow.json').read_bytes())
        root=selector._project_root()
        def normalize(items):
            return {str((Path(p) if Path(p).is_absolute() else root/p).resolve()):h for p,h in items.items()}
        predecessor=root/current['supersedes']
        prior=json.loads(gzip.decompress(predecessor.read_bytes()))
        actual,expected=normalize(current['frozen_inputs']),normalize(prior['frozen_inputs'])
        allowed={str((GCP/'golden_resonance_v4.py').resolve()),str((root/'stargazer_desktop/web/pentacles.js').resolve()),
                 str((root/'stargazer_desktop/sector_rule_scan.py').resolve()),str(Path(__file__).resolve()),
                 str((root/'stargazer_desktop/golden_watch_groups.py').resolve())}
        self.assertEqual(set(actual),set(expected)|allowed)
        self.assertEqual(actual[str((root/'stargazer_desktop/sector_rule_scan.py').resolve())],
                         sha(root/'stargazer_desktop/sector_rule_scan.py'))
        original=normalize(old['frozen_inputs'])
        self.assertEqual(actual[str(selector.MODEL.resolve())],original[str(selector.MODEL.resolve())])
        self.assertEqual(actual[str((selector.FIT/'run.py').resolve())],original[str((selector.FIT/'run.py').resolve())])
        self.assertFalse(current['models_retrained'])

    def test_changed_runtime_or_model_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);runtime=folder/'server.py';model=folder/'model.pkl'
            runtime.write_bytes(b'approved runtime');model.write_bytes(b'frozen model')
            manifest=folder/'manifest.json.gz'
            manifest.write_bytes(gzip.compress(json.dumps(dict(frozen_inputs={str(f):sha(f) for f in [runtime,model]})).encode()))
            with patch.object(selector,'MANIFEST',manifest):
                selector.verified_manifest(SimpleNamespace(sha=sha))
                for target in [runtime,model]:
                    saved=target.read_bytes();target.write_bytes(b'unregistered change')
                    with self.assertRaisesRegex(ValueError,'TAROT_FROZEN_INPUT_CHANGED:'+target.name):
                        selector.verified_manifest(SimpleNamespace(sha=sha))
                    target.write_bytes(saved)

    def test_missing_input_never_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);manifest=folder/'manifest.json.gz'
            manifest.write_bytes(gzip.compress(json.dumps(dict(frozen_inputs={str(folder/'missing.pkl'):'0'*64})).encode()))
            with patch.object(selector,'MANIFEST',manifest),self.assertRaises(FileNotFoundError):
                selector.verified_manifest(SimpleNamespace(sha=sha))

    def test_build_verifies_before_data_read_or_model_deserialization(self):
        with patch.object(selector,'frozen'),patch.object(selector,'verified_manifest',side_effect=ValueError('TAROT_FROZEN_INPUT_CHANGED:server.py')),patch.object(selector.pd,'read_parquet') as read,patch.object(selector.pickle,'load') as load:
            with self.assertRaisesRegex(ValueError,'server.py'):
                selector.build('unused',through=20261023)
        read.assert_not_called();load.assert_not_called()


if __name__=='__main__':unittest.main()
