import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from read_postclose_incremental_v2 import load_refreshed,ROOT
from postclose_research_scorer_v1 import encode,strict_json,HERE


class RefreshedReaderTests(unittest.TestCase):
    def test_real_accepted_sidecar(self):
        frame,_=load_refreshed()
        self.assertEqual(len(frame),86)
        self.assertEqual(frame.groupby('day').size().to_dict(),{20261015:43,20261016:43})

    def test_tampered_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'feature_sidecar.parquet').write_bytes(b'tampered')
            (root/'readback_acceptance.json').write_bytes(encode(dict(
                status='PASS_INCREMENTAL_FEATURE_AND_SCORE_READBACK',artifact_sha256={'feature_sidecar.parquet':'wrong'})))
            with self.assertRaisesRegex(ValueError,'ARTIFACT_DRIFT'):load_refreshed(root)

    def test_cli_scores_current_refreshed_day(self):
        before={p.name:p.stat().st_size for p in ROOT.iterdir()}
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(HERE/'read_postclose_incremental_v2.py'),
                               '--day','20261016','--score'],capture_output=True,text=True,encoding='utf-8',timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        scored=strict_json(result.stdout)
        self.assertEqual(sum(r['status']=='RESEARCH_SCORE' for r in scored['rows']),39)
        self.assertFalse(scored['forward_validation_eligible'])
        self.assertEqual(before,{p.name:p.stat().st_size for p in ROOT.iterdir()})

    def test_cli_no_old_day_fallback(self):
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(HERE/'read_postclose_incremental_v2.py'),
                               '--day','20261017'],capture_output=True,text=True,encoding='utf-8',timeout=30)
        self.assertEqual(result.returncode,2)
        self.assertIn('NO_FALLBACK',strict_json(result.stdout)['error'])


if __name__=='__main__':unittest.main()
